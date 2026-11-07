# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bag, strap-zone and contact quantities shared by the observation, reward and termination terms.

The scene is the one from :mod:`isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene`: an SO-101 arm
over a tray, a rigid bag lying on its side, and a strap handle on the bag's +Z face. Contact comes from one sensor
on every arm link, filtered to the bag, with the average contact point of each link tracked.
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
ROBOT, BAG, CONTACT, CAMERA = "robot", "bag", "contact", "camera"

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

# Strap zone in the bag frame (x along the strap, y its 13 mm width, z outward from the body). The body's face is at
# z = 0.076 and the strap spans z 0.076 .. 0.114, so a contact point inside this box is on the strap.
HANDLE_ZONE_HALF_X = 0.040  # [m]
HANDLE_ZONE_HALF_Y = 0.015  # [m]
HANDLE_ZONE_MIN_Z = 0.078  # [m]
# Bag-frame +Z runs from the bottom to the opening; the strap sits above that. Only the lower fraction of the body
# counts as a collision, so the top rim and the strap are free.
BAG_BODY_Z_MIN = -0.076  # [m]
BAG_BODY_Z_MAX = 0.076  # [m]
COLLISION_LOWER_FRAC = 0.95
LOWER_BODY_Z_CUT = BAG_BODY_Z_MIN + COLLISION_LOWER_FRAC * (BAG_BODY_Z_MAX - BAG_BODY_Z_MIN)
GRIPPER_CLOSED_MARGIN = (
    0.10  # [rad] jaw this close to its closed limit counts as closed (holding the strap stalls ~0.05)
)
"""Holding the 1.3 cm strap stalls the jaw near -0.33 rad, 0.05 rad off the -0.38 rad limit; the margin leaves room."""

ARM_BASE_ROT = scene_defaults.ARM_BASE_ROT
"""Arm base rotation (x, y, z, w), the same value as the scene robot's ``init_state.rot``.

180 deg yaw. The 90 deg roll that stands the Y-up CAD arm upright is on ``/Arm/Mount`` in
``so101_physics.usda``, and this yaw is applied to ``/Arm`` above that mount. ``grasp_goal.json`` was recorded
with this mount; with it, the recorded joints put the tip on the strap to within a centimetre."""

GRASP_GOAL_FILE = Path(scene_defaults.__file__).with_name("grasp_goal.json")
"""Recorded pinch in the bag frame, written by ``scene/grasp_pose_tool.py --record`` into this folder."""
FALLBACK_PINCH_TARGET = (0.0, -0.012, 0.1135)
"""Strap apex 12 mm below its lower edge, bag frame. Used only when there is no recorded grasp goal."""


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
    rigid_jaw: int
    moving_jaw: int
    gripper_joint: int
    ee_body: int


def _indices(env: ManagerBasedRLEnv) -> _Indices:
    """Contact-sensor columns and robot indices, resolved once per environment."""
    cached = getattr(env, "_hg_indices", None)
    if cached is None:
        names = list(env.scene.sensors[CONTACT].body_names)
        robot = env.scene[ROBOT]
        cached = _Indices(
            rigid_jaw=names.index(EE_BODY_NAME),
            moving_jaw=names.index(MOVING_JAW_BODY_NAME),
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


def strap_contacts(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether the rigid jaw and the moving jaw each touch the bag inside the strap zone."""
    rel = contact_in_bag_frame(env)
    on_strap = (
        (rel[..., 0].abs() < HANDLE_ZONE_HALF_X)
        & (rel[..., 1].abs() < HANDLE_ZONE_HALF_Y)
        & (rel[..., 2] > HANDLE_ZONE_MIN_Z)
    )  # NaN (no contact) compares False
    idx = _indices(env)
    return on_strap[:, idx.rigid_jaw], on_strap[:, idx.moving_jaw]


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
    """Whether the jaw is within ``GRIPPER_CLOSED_MARGIN`` of its closed limit."""
    return gripper_joint_pos(env) < GRIPPER_CLOSED_POS + GRIPPER_CLOSED_MARGIN


def gripper_closed_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """0 with the jaw fully open, 1 with it fully closed."""
    span = GRIPPER_OPEN_POS - GRIPPER_CLOSED_POS
    return ((GRIPPER_OPEN_POS - gripper_joint_pos(env)) / span).clamp(0.0, 1.0)


def holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Strap in the jaws: closed jaw, with both jaws touching the bag inside the strap zone."""
    rigid_on, moving_on = strap_contacts(env)
    return gripper_closed(env) & rigid_on & moving_on


def bag_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bag root height above the environment origin [m]."""
    return env.scene[BAG].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


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
