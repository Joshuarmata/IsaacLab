# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bag, strap and grasp quantities shared by the observation, reward and termination terms.

The scene is the one from :mod:`isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene`: an SO-101 arm
over a tray, a rigid bag lying on its side, and a deformable strap handle (``USE_DEFORMABLE_HANDLE``) on the bag's +Z
face.

The grasp is detected geometrically, and only that way: the strap's simulation nodes against the closed mouth of the
gripper (``closed_mouth`` in ``assets/jaw_points.json``), see :func:`strap_in_mouth` and :func:`holding`. The contact
sensor on the arm links is used only for the lower-body safety force (:func:`lower_body_force`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.core.official_handle_grab.scene import handle_grab_scene as scene_defaults

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Scene entity names used by the task.
ROBOT, BAG, CONTACT, CAMERA, HANDLE = "robot", "bag", "contact", "camera", "handle"

# Arm geometry, from the scene module so the task and the scene cannot drift apart.
EE_BODY_NAME = scene_defaults.EE_BODY_NAME
EE_TIP_OFFSET = scene_defaults.EE_TIP_OFFSET
JAW_INSERT_AXIS = scene_defaults.JAW_INSERT_AXIS
MOVING_JAW_BODY_NAME = scene_defaults.MOVING_JAW_BODY_NAME
GRIPPER_JOINT = scene_defaults.ARM_GRIPPER_JOINT
GRIPPER_OPEN_POS = scene_defaults.GRIPPER_OPEN_POS
GRIPPER_CLOSED_POS = scene_defaults.GRIPPER_CLOSED_POS
ARM_JOINTS = ["revolute_[1-6]"]
"""Joints 1-6; the gripper (``revolute_7``) is driven by its own action term."""

# Bag-frame +Z runs from the bottom to the opening; the strap sits above that. Only the lower fraction of the body
# counts as a collision, so the top rim and the strap are free.
BAG_BODY_Z_MIN = -0.076  # [m]
BAG_BODY_Z_MAX = 0.076  # [m]
COLLISION_LOWER_FRAC = 0.95
LOWER_BODY_Z_CUT = BAG_BODY_Z_MIN + COLLISION_LOWER_FRAC * (BAG_BODY_Z_MAX - BAG_BODY_Z_MIN)


GRIPPER_HOLD_POS = GRIPPER_CLOSED_POS + 0.08  # [rad]
"""The jaw counts as closed below this measured angle (-0.30 rad), within 0.08 rad of the -0.38 rad stop; open is
0.6 rad. A strap pinched at the tips stalls the jaw near -0.19 rad, so only a strap inside the mouth counts."""

ARM_BASE_ROT = scene_defaults.ARM_BASE_ROT
"""Arm base rotation (x, y, z, w), the same value as the scene robot's ``init_state.rot``.

180 deg yaw composed with the 90 deg roll that stands the Y-up CAD arm upright. Isaac Lab writes this onto the
articulation root; a roll on ``/Arm/Mount`` is not kept. ``grasp_goal.json`` was recorded with the arm upright, so
with this rotation the recorded joints put the tip on the strap to within a centimetre."""

GRASP_GOAL_FILE = Path(scene_defaults.__file__).with_name("grasp_goal.json")
"""Recorded pinch in the bag frame, written by ``scene/grasp_pose_tool.py --record`` into this folder."""
FALLBACK_PINCH_TARGET = (0.0, -0.012, 0.1135)
"""Strap apex, 12 mm below the strap's centre line (5.5 mm past its lower edge), bag frame. Used only when there is
no recorded grasp goal."""


@dataclass(frozen=True)
class GraspGoal:
    """The hand-recorded pinch on the strap, all in the bag frame."""

    tip_pos_bag: tuple[float, float, float]
    """Rigid-jaw tip position."""
    jaw_quat_bag: tuple[float, float, float, float] | None
    """Orientation (x, y, z, w) of the link that carries the rigid jaw, or None when no recording exists."""
    arm_joints: dict[str, float]
    """Recorded arm joint angles [rad], the comfortable seed for inverse kinematics."""


@lru_cache(maxsize=1)
def load_grasp_goal() -> GraspGoal:
    """Read ``grasp_goal.json``, falling back to the nominal pinch point when it is absent."""
    if not GRASP_GOAL_FILE.exists():
        return GraspGoal(FALLBACK_PINCH_TARGET, None, {})
    goal = json.loads(GRASP_GOAL_FILE.read_text())
    joints = {name: math.radians(float(deg)) for name, deg in goal.get("joints_deg", {}).items()}
    return GraspGoal(tuple(goal["tip_pos_bag"]), tuple(goal["jaw_quat_bag"]), joints)


@dataclass
class _Indices:
    gripper_joint: int
    ee_body: int


def _indices(env: ManagerBasedRLEnv) -> _Indices:
    """Robot indices, resolved once per environment."""
    cached = getattr(env, "_hg_indices", None)
    if cached is None:
        robot = env.scene[ROBOT]
        cached = _Indices(
            gripper_joint=robot.find_joints(GRIPPER_JOINT)[0][0],
            ee_body=robot.find_bodies(EE_BODY_NAME)[0][0],
        )
        env._hg_indices = cached
    return cached


def link_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Contact force [N] each arm link exerts on the bag, shape ``(num_envs, num_links)``."""
    forces = env.scene.sensors[CONTACT].data.normal_force_matrix_w.torch
    return torch.linalg.vector_norm(forces, dim=-1).sum(dim=-1)


def _bag_frame(env: ManagerBasedRLEnv, points_w: torch.Tensor) -> torch.Tensor:
    """Express world points of shape ``(num_envs, k, 3)`` in the bag frame."""
    bag = env.scene[BAG]
    n, k = points_w.shape[:2]
    pos = bag.data.root_pos_w.torch.unsqueeze(1).expand(n, k, 3).reshape(-1, 3)
    quat = bag.data.root_quat_w.torch.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
    return quat_apply_inverse(quat, points_w.reshape(-1, 3) - pos).reshape(n, k, 3)


def contact_in_bag_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Average contact point of every arm link on the bag, in the bag frame; NaN where a link does not touch."""
    points_w = env.scene.sensors[CONTACT].data.contact_pos_w.torch[:, :, 0]
    return _bag_frame(env, points_w)


def has_deformable_strap(env: ManagerBasedRLEnv) -> bool:
    """Whether the strap is its own deformable mesh (``USE_DEFORMABLE_HANDLE``) rather than part of the rigid bag."""
    return HANDLE in env.scene.deformable_objects


@dataclass
class JawGap:
    """Boxes in a frame fixed to the rigid jaw: ``axes @ (p - origin)`` within any box's bounds for a link-frame
    point p."""

    origin: torch.Tensor
    """Box frame origin in the rigid-jaw link frame, shape ``(3,)``."""
    axes: torch.Tensor
    """Box frame axes (rows) in the rigid-jaw link frame, shape ``(3, 3)``."""
    bounds: torch.Tensor
    """(min, max) of each box in the box frame, shape ``(boxes, 2, 3)``."""

    @classmethod
    def from_json(cls, data: dict, device: str | torch.device) -> JawGap:
        """Read a ``closed_gap`` (one box) or ``closed_mouth`` (``boxes``) entry of ``jaw_points.json``."""

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=device)

        bounds = tensor([[box["min"], box["max"]] for box in data.get("boxes", [data])])
        return cls(tensor(data["origin"]), tensor(data["axes"]), bounds)

    @property
    def back_point(self) -> torch.Tensor:
        """Centre of the back face of the box nearest the hinge, in the box frame, shape ``(3,)``.

        Axis 1 runs along the jaws and is negative toward the tips, so the box nearest the hinge (the palm) has the
        largest axis-1 maximum.
        """
        back = self.bounds[self.bounds[:, 1, 1].argmax()]
        point = back.mean(dim=0)
        point[1] = back[1, 1]
        return point


@dataclass
class _StrapGeometry:
    rest_bag: torch.Tensor
    """Strap nodes at rest in the bag frame, shape ``(nodes, 3)``."""
    mouth: JawGap
    """Space inside the jaws with the gripper at its closed stop (``closed_mouth`` in ``jaw_points.json``)."""
    columns: int
    """Strap nodes across the ribbon. Node ``i * columns + j`` is row ``i`` along the strap, column ``j``."""
    rigid_body: int


def _strap_geometry(env: ManagerBasedRLEnv) -> _StrapGeometry:
    """Rest shape and closed mouth, read once per environment from the files ``build_handle_assets.py`` writes."""
    cached = getattr(env, "_hg_strap", None)
    if cached is None:
        info = json.loads(Path(scene_defaults.HANDLE_INFO_JSON).read_text())
        jaws = json.loads(Path(scene_defaults.JAW_POINTS_JSON).read_text())
        robot = env.scene[ROBOT]
        cached = _StrapGeometry(
            rest_bag=torch.tensor(info["rest_points"], dtype=torch.float32, device=env.device),
            mouth=JawGap.from_json(jaws["closed_mouth"], env.device),
            columns=info["n_w"] + 1,
            rigid_body=robot.find_bodies(EE_BODY_NAME)[0][0],
        )
        nodes = env.scene[HANDLE].data.nodal_pos_w.torch.shape[1]
        if nodes != cached.rest_bag.shape[0] or nodes != info["n_len"] * cached.columns:
            raise RuntimeError(
                f"The strap has {nodes} simulation nodes but {scene_defaults.HANDLE_INFO_JSON} lists"
                f" {cached.rest_bag.shape[0]}. Rerun scene/assets/build_handle_assets.py."
            )
        env._hg_strap = cached
    return cached


def strap_rest_nodes(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Strap nodes at rest in the bag frame, shape ``(nodes, 3)``; no nodes with the rigid strap."""
    if not has_deformable_strap(env):
        return torch.zeros(0, 3, device=env.device)
    return _strap_geometry(env).rest_bag


def strap_nodes_bag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current strap nodes in the bag frame, shape ``(num_envs, nodes, 3)``; no nodes with the rigid strap."""
    if not has_deformable_strap(env):
        return torch.zeros(env.num_envs, 0, 3, device=env.device)
    return _bag_frame(env, env.scene[HANDLE].data.nodal_pos_w.torch)


def write_strap(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, bag_pose_w: torch.Tensor, nodes_bag: torch.Tensor | None = None
) -> None:
    """Lay the strap onto bag poses, at rest or in a given bag-frame shape, with zero velocity.

    Every bag teleport needs this: the strap's feet are tied to the body with an infinitely stiff attachment, so a
    bag that jumps away from its strap tears it. Does nothing with the rigid strap.

    Args:
        env_ids: Environments to write, shape ``(n,)``.
        bag_pose_w: Bag root pose ``(x, y, z, qx, qy, qz, qw)`` in the world frame, shape ``(n, 7)``.
        nodes_bag: Strap nodes in the bag frame, shape ``(n, nodes, 3)``. ``None`` or zero nodes lays it at rest.
    """
    if not has_deformable_strap(env):
        return
    rest = _strap_geometry(env).rest_bag
    n, v = bag_pose_w.shape[0], rest.shape[0]
    if nodes_bag is None or nodes_bag.shape[1] == 0:
        nodes_bag = rest.expand(n, v, 3)
    quat = bag_pose_w[:, 3:7].unsqueeze(1).expand(n, v, 4).reshape(-1, 4)
    nodes_w = bag_pose_w[:, :3].unsqueeze(1) + quat_apply(quat, nodes_bag.reshape(-1, 3)).reshape(n, v, 3)
    handle = env.scene[HANDLE]
    handle.write_nodal_pos_to_sim_index(nodes_w, env_ids=env_ids)
    handle.write_nodal_velocity_to_sim_index(torch.zeros_like(nodes_w), env_ids=env_ids)


def _gap_local(
    nodes_w: torch.Tensor, jaw_pos_w: torch.Tensor, jaw_quat_w: torch.Tensor, gap: JawGap, columns: int
) -> torch.Tensor:
    """Strap nodes in the gap frame, shape ``(num_envs, columns, rows, 3)``, each column ordered along the strap."""
    n, v = nodes_w.shape[:2]
    quat = jaw_quat_w.unsqueeze(1).expand(n, v, 4).reshape(-1, 4)
    link = quat_apply_inverse(quat, (nodes_w - jaw_pos_w.unsqueeze(1)).reshape(-1, 3))
    return ((link - gap.origin) @ gap.axes.T).reshape(n, v // columns, columns, 3).transpose(1, 2)


def strap_through_gap(
    nodes_w: torch.Tensor, jaw_pos_w: torch.Tensor, jaw_quat_w: torch.Tensor, gap: JawGap, columns: int
) -> torch.Tensor:
    """Whether any strap segment crosses or touches any of the gap's boxes, shape ``(num_envs,)``.

    The direct task's grasp test (``Isaac-HandleGrab-Direct``); this task uses :func:`strap_in_mouth`. A box can be
    thinner than the node spacing, so a strap crossing it need not leave a node inside.

    Args:
        nodes_w: Strap nodes in the world frame, shape ``(num_envs, rows * columns, 3)``, row-major along the strap.
        jaw_pos_w: Rigid-jaw link position, shape ``(num_envs, 3)``.
        jaw_quat_w: Rigid-jaw link orientation (x, y, z, w), shape ``(num_envs, 4)``.
        gap: The gap boxes, fixed to the rigid jaw.
        columns: Strap nodes across the ribbon.
    """
    local = _gap_local(nodes_w, jaw_pos_w, jaw_quat_w, gap, columns)
    start, step = local[:, :, :-1], local[:, :, 1:] - local[:, :, :-1]  # (num_envs, columns, segments, 3)
    start, step = start.unsqueeze(-2), step.unsqueeze(-2)  # (num_envs, columns, segments, 1, 3) against every box

    # Slab test: the part of each segment (0 <= t <= 1) inside every pair of box faces.
    lo, hi = gap.bounds[:, 0], gap.bounds[:, 1]
    moving = step.abs() > 1e-9
    safe = torch.where(moving, step, torch.ones_like(step))
    t0, t1 = (lo - start) / safe, (hi - start) / safe
    enter = torch.where(moving, torch.minimum(t0, t1), torch.full_like(step, -float("inf")))
    leave = torch.where(moving, torch.maximum(t0, t1), torch.full_like(step, float("inf")))
    parallel_out = ~moving & ((start < lo) | (start > hi))
    t_in = enter.amax(dim=-1).clamp(min=0.0)
    t_out = leave.amin(dim=-1).clamp(max=1.0)
    crosses = (t_in <= t_out) & ~parallel_out.any(dim=-1)
    return crosses.flatten(1).any(dim=-1)


@dataclass
class StrapGrasp:
    """The strap against the closed mouth, each field of shape ``(num_envs,)``."""

    through: torch.Tensor
    """The strap crosses the mouth from one side of the gripper to the other (:func:`strap_in_mouth`)."""
    along: torch.Tensor
    """Where along the strap it crosses: 0 and 1 at the feet, 0.5 at the top of the arch; NaN without a crossing."""
    back_dist: torch.Tensor
    """Distance [m] from the mouth's back point (:attr:`JawGap.back_point`) to the nearest point on the strap."""


def strap_in_mouth(
    nodes_w: torch.Tensor, jaw_pos_w: torch.Tensor, jaw_quat_w: torch.Tensor, mouth: JawGap, columns: int
) -> StrapGrasp:
    """Whether the strap runs through the closed mouth side to side, where along the strap, and how deep.

    Each column of the strap ribbon is a polyline along the strap. It crosses a box when it goes from beyond one of the
    box's hinge faces (axis 0, the gripper's sides) to beyond the other, and everything between those two faces stays
    within the box's along-jaw and across-jaw bounds (axes 1 and 2). So it enters through one side of the gripper and
    leaves through the other, never through the front, the back, or a jaw face. Any number of nodes may lie between
    the two faces, including none (one segment spanning the box). The strap is in the mouth when any column crosses
    any box.

    Args:
        nodes_w: Strap nodes in the world frame, shape ``(num_envs, rows * columns, 3)``, row-major along the strap.
        jaw_pos_w: Rigid-jaw link position, shape ``(num_envs, 3)``.
        jaw_quat_w: Rigid-jaw link orientation (x, y, z, w), shape ``(num_envs, 4)``.
        mouth: The closed-mouth boxes, fixed to the rigid jaw.
        columns: Strap nodes across the ribbon.
    """
    local = _gap_local(nodes_w, jaw_pos_w, jaw_quat_w, mouth, columns)  # (num_envs, columns, rows, 3)
    rows = local.shape[2]
    lo, hi = mouth.bounds[:, 0], mouth.bounds[:, 1]  # (boxes, 3)
    points = local.unsqueeze(-2)  # (num_envs, columns, rows, 1, 3) against every box
    side = (points[..., 0] > hi[:, 0]).long() - (points[..., 0] < lo[:, 0]).long()  # -1, 0 or +1 per node and box

    # The part of each segment between the two hinge faces must stay inside the box's other two bounds. The
    # cross-section is convex, so checking that part's two ends is enough.
    start, step = points[:, :, :-1], points[:, :, 1:] - points[:, :, :-1]
    s0, d0 = start[..., 0], step[..., 0]
    moving = d0.abs() > 1e-9
    safe = torch.where(moving, d0, torch.ones_like(d0))
    ta, tb = (lo[:, 0] - s0) / safe, (hi[:, 0] - s0) / safe
    t_in = torch.where(moving, torch.minimum(ta, tb), torch.zeros_like(ta)).clamp(min=0.0)
    t_out = torch.where(moving, torch.maximum(ta, tb), torch.ones_like(ta)).clamp(max=1.0)
    between = (moving | ((s0 >= lo[:, 0]) & (s0 <= hi[:, 0]))) & (t_in <= t_out)

    def in_cross_section(t: torch.Tensor) -> torch.Tensor:
        p = start + t.unsqueeze(-1) * step
        return ((p[..., 1:] >= lo[:, 1:]) & (p[..., 1:] <= hi[:, 1:])).all(dim=-1)

    clean = ~between | (in_cross_section(t_in) & in_cross_section(t_out))  # (num_envs, columns, segments, boxes)

    # From every node beyond a hinge face, the next node beyond one: a crossing when it is the opposite face and
    # every segment in between is clean.
    index = torch.arange(rows, device=local.device).view(1, 1, rows, 1)
    beyond = torch.where(side != 0, index, rows)
    after = beyond.flip(2).cummin(dim=2).values.flip(2)[:, :, 1:]  # first such node after each segment's start
    end = after.clamp(max=rows - 1)
    dirty = torch.cat((torch.zeros_like(side[:, :, :1]), (~clean).long().cumsum(dim=2)), dim=2)
    clean_run = dirty.gather(2, end) == dirty[:, :, :-1]
    begin_side = side[:, :, :-1]
    crosses = (begin_side != 0) & (after < rows) & (begin_side * side.gather(2, end) == -1) & clean_run

    middle = (index[:, :, :-1] + end).float() / (2 * (rows - 1))
    flat = crosses.flatten(1)
    along = (middle * crosses).flatten(1).sum(dim=-1) / flat.sum(dim=-1)  # NaN where nothing crosses

    back = mouth.back_point
    seg_start, seg_step = local[:, :, :-1], local[:, :, 1:] - local[:, :, :-1]
    t = ((back - seg_start) * seg_step).sum(dim=-1) / seg_step.square().sum(dim=-1).clamp(min=1e-12)
    nearest = seg_start + t.clamp(0.0, 1.0).unsqueeze(-1) * seg_step
    back_dist = torch.linalg.vector_norm(nearest - back, dim=-1).flatten(1).amin(dim=-1)
    return StrapGrasp(flat.any(dim=-1), along, back_dist)


def strap_grasp(env: ManagerBasedRLEnv) -> StrapGrasp:
    """The strap against the closed mouth (``closed_mouth`` in ``jaw_points.json``, boxes 25 to 65 mm in front of the
    hinge), carried by the rigid jaw; see :func:`strap_in_mouth`."""
    if not has_deformable_strap(env):
        raise RuntimeError(
            "Grasp detection needs the deformable strap (USE_DEFORMABLE_HANDLE in handle_grab_scene.py)."
        )
    strap = _strap_geometry(env)
    robot = env.scene[ROBOT]
    return strap_in_mouth(
        env.scene[HANDLE].data.nodal_pos_w.torch,
        robot.data.body_pos_w.torch[:, strap.rigid_body],
        robot.data.body_quat_w.torch[:, strap.rigid_body],
        strap.mouth,
        strap.columns,
    )


def lower_body_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Force [N] summed over every arm link on the lower part of the bag body, shape ``(num_envs,)``.

    The jaws on the strap and on the top rim are free; any link on the lower body counts, including the jaws.
    """
    on_lower = contact_in_bag_frame(env)[..., 2] < LOWER_BODY_Z_CUT  # NaN -> False
    return (link_forces(env) * on_lower.float()).sum(dim=-1)


def gripper_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured gripper joint position [rad]."""
    return env.scene[ROBOT].data.joint_pos.torch[:, _indices(env).gripper_joint]


def gripper_closed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the measured jaw angle is below :data:`GRIPPER_HOLD_POS`."""
    return gripper_joint_pos(env) < GRIPPER_HOLD_POS


def gripper_closed_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """0 with the jaw fully open, 1 with it fully closed."""
    span = GRIPPER_OPEN_POS - GRIPPER_CLOSED_POS
    return ((GRIPPER_OPEN_POS - gripper_joint_pos(env)) / span).clamp(0.0, 1.0)


def holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The grasp boolean, shape ``(num_envs,)``: the jaw is closed (:func:`gripper_closed`) and the strap crosses the
    closed mouth from one side of the gripper to the other (:func:`strap_in_mouth`)."""
    return gripper_closed(env) & strap_grasp(env).through


def tip_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rigid-jaw tip in the world frame [m]."""
    robot = env.scene[ROBOT]
    body = _indices(env).ee_body
    pos, quat = robot.data.body_pos_w.torch[:, body], robot.data.body_quat_w.torch[:, body]
    return pos + quat_apply(quat, pos.new_tensor(EE_TIP_OFFSET).expand_as(pos))


def strap_target_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The recorded pinch point on the strap in the world frame [m]; follows the bag's pose."""
    bag = env.scene[BAG]
    pos, quat = bag.data.root_pos_w.torch, bag.data.root_quat_w.torch
    return pos + quat_apply(quat, pos.new_tensor(load_grasp_goal().tip_pos_bag).expand_as(pos))
