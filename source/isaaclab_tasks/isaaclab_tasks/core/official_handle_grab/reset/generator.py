# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the physics-validated reset dataset of the handle-grab task.

The attempt follows the vial generator. The arm settles at home, then the bag is placed in the spawn box and settled:
that is the home row. Each later phase is continued from rows of the phase before it that physics already accepted
(a seed bank), and each phase is filled to its own quota. The long approach runs with the bag pinned out of the
workspace; the bag is put back before the jaw moves onto the strap, and the closing grasp shuts the jaw most of the
way. The task is grasp-only for now, so the vial's lift phases are not generated (``LIFT_GRAVEYARD.md``), and there is
no rack, so neither are its reorient, transport, insert, and release phases::

    uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator

Writes ``handle_grab_reset_dataset.pt`` next to this file. Rows hold measured joint positions and the bag's pose in the
environment frame, plus the deformable strap's shape in the bag frame, so no row starts interpenetrating or off the
strap.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch

from isaaclab.utils.math import normalize, quat_apply, quat_box_minus, quat_from_euler_xyz, quat_inv, quat_mul

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import (
    ARM_JOINT_NAMES,
    BAG_FOOTPRINT_X,
    BAG_FOOTPRINT_Y,
    BAG_REST_ROT,
    BAG_REST_Z,
    BAG_SPAWN_XY,
    TRAY_INNER_X,
    TRAY_INNER_Y,
)

from ..mdp.task import (
    ARM_JOINTS,
    BAG,
    EE_BODY_NAME,
    EE_TIP_OFFSET,
    GRIPPER_CLOSED_POS,
    GRIPPER_HOLD_POS,
    GRIPPER_OPEN_POS,
    JAW_INSERT_AXIS,
    ROBOT,
    gripper_closed,
    gripper_joint_pos,
    holding,
    load_grasp_goal,
    lower_body_force,
    strap_grasp,
    strap_nodes_bag,
    strap_target_w,
    tip_pos_w,
    write_strap,
)
from .curriculum import GRASP, GRASP_CLOSING, HOME, PHASES, PREGRASP
from .dataset import DEFAULT_DATASET_PATH, ResetDataset

TASK = "Isaac-HandleGrab"

JOINT_SPEED = 0.4
"""Scripted joint speed [rad/s]; slow enough that the 3 N*m drives keep up with the command."""
SETTLE_STEPS = 30
"""Policy steps the bag and arm rest before home is recorded."""
HOLD_STEPS = 15
"""Policy steps the arm rests at each open-jaw waypoint before it is recorded."""
CLOSING_JAW_POS = GRIPPER_HOLD_POS + 0.2
"""Jaw angle [rad] of a ``grasp_closing`` row: 0.2 rad short of the hold angle, so the row is not yet a grasp."""
CLOSING_SETTLE_STEPS = 30
"""Policy steps the partly closed jaw rests before the row is recorded; the drive lags the command."""
CLOSE_STEPS = 150
"""Policy steps the jaw is then commanded shut to check that the row can be grasped. The drive lags the command
(0.75 rad in 45 steps), and the closing jaw sweeps the strap into the mouth over the last part of the travel."""
PARK_HEIGHT = 1.0
"""Height [m] the bag is pinned to during the home-to-pregrasp sweep, above the arm's reach."""
MAX_BAG_SPEED = 0.02
"""A row whose bag still moves faster than this [m/s] is not settled."""
PREGRASP_POSITION_TOLERANCE = 0.015
"""Position tolerance [m] of the pregrasp IK, the vial generator's tolerance for intermediate waypoints."""
MAX_SEED_TIP_ERROR = 0.05
"""The recorded joints must put the tip within this [m] of the recorded pinch, or the recording is stale."""
WALL_MARGIN = 0.005
"""Clearance [m] kept between the bag's footprint and the tray's inner walls."""


##
# Pure geometry (torch only, no simulation).
##


def sample_bag_poses(
    count: int, x_half: float, y_half: float, yaw_half: float, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Sample bag root poses ``(x, y, z, qx, qy, qz, qw)`` in the environment frame.

    The position is uniform in the box ``x_half`` by ``y_half`` around the nominal spawn, and the heading is uniform in
    ``+-yaw_half`` on top of the lying-flat rest rotation. The nominal spot is only 14 mm from the near wall, so poses
    whose rotated footprint would reach within :data:`WALL_MARGIN` of a tray wall are resampled instead of clamped.
    """
    spawn = torch.tensor(BAG_SPAWN_XY, device=device)
    half = torch.tensor((0.5 * BAG_FOOTPRINT_X, 0.5 * BAG_FOOTPRINT_Y), device=device)
    low = torch.tensor((TRAY_INNER_X[0], TRAY_INNER_Y[0]), device=device) + WALL_MARGIN
    high = torch.tensor((TRAY_INNER_X[1], TRAY_INNER_Y[1]), device=device) - WALL_MARGIN
    box = torch.tensor((x_half, y_half), device=device)

    xy = torch.zeros(count, 2, device=device)
    yaw = torch.zeros(count, device=device)
    todo = torch.arange(count, device=device)
    for _ in range(200):
        k = todo.numel()
        cand_xy = spawn + (2.0 * torch.rand(k, 2, device=device) - 1.0) * box
        cand_yaw = (2.0 * torch.rand(k, device=device) - 1.0) * yaw_half
        c, s = cand_yaw.cos().abs(), cand_yaw.sin().abs()
        extent = torch.stack((half[0] * c + half[1] * s, half[0] * s + half[1] * c), dim=-1)
        ok = ((cand_xy - extent) >= low).all(dim=-1) & ((cand_xy + extent) <= high).all(dim=-1)
        xy[todo[ok]] = cand_xy[ok]
        yaw[todo[ok]] = cand_yaw[ok]
        todo = todo[~ok]
        if todo.numel() == 0:
            break
    else:
        raise RuntimeError("The spawn box has no bag pose that clears the tray walls; reduce --x_half/--y_half/--yaw.")

    zeros = torch.zeros_like(yaw)
    rest = torch.tensor(BAG_REST_ROT, device=device).expand(count, 4)
    quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), rest)
    z = torch.full((count, 1), BAG_REST_Z, device=device)
    return torch.cat((xy, z, quat), dim=-1)


def vertical_jaw_quat_bag(
    jaw_quat_bag: tuple[float, float, float, float],
    bag_rest_quat: tuple[float, float, float, float] = BAG_REST_ROT,
) -> tuple[float, float, float, float]:
    """The recorded jaw orientation with its insert axis made exactly vertical, in the bag frame.

    The vial generator derives the grasp orientation from the object's geometry (pads along the vial, closing axis
    horizontal) instead of replaying a recording. The bag lies flat, so its strap geometry reduces to one fact: the jaw
    comes straight down onto it. This takes the recorded orientation, which tilts the insert axis a few degrees off the
    vertical, and applies the smallest rotation that points the axis straight down, keeping the recorded roll about it.
    The bag lies at rest with only a heading, so the result is one constant in the bag frame: a wrist heading that
    follows the bag, an insert axis that never leaves the vertical.
    """
    rest = torch.tensor(bag_rest_quat, dtype=torch.float64)[None]
    jaw = torch.tensor(jaw_quat_bag, dtype=torch.float64)[None]
    jaw_world = quat_mul(rest, jaw)  # the jaw with the bag at rest and no heading
    axis = quat_apply(jaw_world, torch.tensor(JAW_INSERT_AXIS, dtype=torch.float64)[None])
    down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
    # Shortest rotation from ``axis`` to ``down``: (a x b, 1 + a . b) in (x, y, z, w) order.
    align = normalize(torch.cat((torch.linalg.cross(axis, down), 1.0 + (axis * down).sum(-1, keepdim=True)), dim=-1))
    snapped = quat_mul(quat_inv(rest), quat_mul(align, jaw_world))
    return tuple(float(v) for v in snapped[0])


def wrist_targets(
    bag_pos: torch.Tensor,
    bag_quat: torch.Tensor,
    tip_pos_bag: tuple[float, float, float],
    jaw_quat_bag: tuple[float, float, float, float],
    backoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrist poses that put the rigid-jaw tip on the recorded pinch, and ``backoff`` metres back from it.

    The recording gives the tip position and the wrist link's orientation in the bag frame; both follow the bag. The
    wrist link sits behind the tip by :data:`EE_TIP_OFFSET` (in the link frame), and the pregrasp slides the wrist back
    along the opposite of the jaw's insert axis.

    Returns:
        ``(grasp_wrist_pos, pregrasp_wrist_pos, wrist_quat)``, in the frame of ``bag_pos``.
    """
    n = bag_pos.shape[0]
    quat = quat_mul(bag_quat, bag_quat.new_tensor(jaw_quat_bag).expand(n, 4))
    tip = bag_pos + quat_apply(bag_quat, bag_pos.new_tensor(tip_pos_bag).expand(n, 3))
    grasp = tip - quat_apply(quat, bag_pos.new_tensor(EE_TIP_OFFSET).expand(n, 3))
    insert = quat_apply(quat, bag_pos.new_tensor(JAW_INSERT_AXIS).expand(n, 3))
    return grasp, grasp - backoff * insert, quat


##
# Inverse kinematics (the vial generator's recipe, on the live PhysX articulation).
##

IK_POSITION_WEIGHT = 100.0
"""Weight of the position rows in the IK cost; the vial generator's ``position_weight``."""
IK_ROTATION_WEIGHT = 1.0
"""Weight of the rotation rows; the vial generator's ``rotation_weight`` ("position dominates")."""
IK_MARGIN_SLACK = 1e-5
"""Slack [rad] on the joint-margin check, so a start clamped onto the inset search limit still counts as valid."""


class IKSolution(NamedTuple):
    """Chosen seed per environment from a multi-start solve; every tensor has one leading entry per environment."""

    index: torch.Tensor
    """Index of the chosen seed. For environments with no valid seed, the seed with the smallest position error."""
    valid: torch.Tensor
    """True where at least one seed met the position, rotation and joint-margin tolerances."""
    valid_count: torch.Tensor
    """How many seeds were valid."""
    delta: torch.Tensor
    """Joint-space distance [rad] from the starting joints to the chosen seed's solution."""
    clearance: torch.Tensor
    """Smallest distance [rad] from any arm joint of the chosen solution to its limit."""


def select_ik_solution(
    q: torch.Tensor,
    pos_err: torch.Tensor,
    rot_err: torch.Tensor,
    seed: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    pos_tol: float,
    rot_tol: float,
    margin: float,
    randomize: bool = False,
    generator: torch.Generator | None = None,
) -> IKSolution:
    """Pick one solution per environment out of several IK starts, the way the vial generator does.

    A start is valid when its joints are finite, its position error is within ``pos_tol`` [m], its rotation error is
    within ``rot_tol`` [rad] and every arm joint is at least ``margin`` [rad] from its limit. Of the valid starts the
    one nearest the seed joints wins, which stays on the same IK branch and avoids wrist flips. With ``randomize`` a
    valid start is drawn uniformly instead, the vial generator's ``randomize_solution``: the dataset then holds several
    arm configurations for the same target. With no valid start the best position error is reported, so the failure
    can be read.

    Args:
        q: Solved arm joints, shape ``(envs, starts, joints)``.
        pos_err: Position error [m] of each start, shape ``(envs, starts)``.
        rot_err: Rotation error [rad] of each start, shape ``(envs, starts)``.
        seed: The joints every start was perturbed from, shape ``(envs, joints)``.
        low: Lower joint limits, shape ``(joints,)`` or ``(envs, joints)``.
        high: Upper joint limits, same shape as ``low``.
        randomize: Draw among the valid starts instead of taking the nearest.
        generator: Random generator for ``randomize``.
    """
    low, high = low.unsqueeze(-2), high.unsqueeze(-2)
    clearance = torch.minimum(q - low, high - q).amin(dim=-1)
    delta = torch.linalg.vector_norm(q - seed.unsqueeze(1), dim=-1)
    clear = clearance >= margin - IK_MARGIN_SLACK
    ok = torch.isfinite(q).all(dim=-1) & (pos_err <= pos_tol) & (rot_err <= rot_tol) & clear
    valid = ok.any(dim=-1)
    nearest = delta.masked_fill(~ok, torch.inf).argmin(dim=-1)
    best_position = torch.nan_to_num(pos_err, nan=torch.inf, posinf=torch.inf).argmin(dim=-1)
    if randomize:
        weights = torch.where(valid[:, None], ok.float(), torch.ones_like(pos_err))  # a zero row cannot be sampled
        nearest = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
    index = torch.where(valid, nearest, best_position)
    rows = torch.arange(q.shape[0], device=q.device)
    return IKSolution(index, valid, ok.sum(dim=-1), delta[rows, index], clearance[rows, index])


def levenberg_marquardt(
    evaluate: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    q0: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    iterations: int,
    position_weight: float,
    lam0: float = 0.1,
    max_step: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Minimise a weighted pose error over joints, with a per-environment damping factor.

    ``evaluate(q)`` returns the weighted residual ``(envs, 6)`` (position rows first) that a joint step ``dq`` should
    cancel, and the weighted Jacobian ``(envs, 6, joints)``. A step is kept only if it lowers that environment's cost;
    the damping halves after a kept step and doubles after a rejected one. Joints are clamped to ``[low, high]``.

    Returns:
        The joints, the weighted residual at them, and the median position error [m] before each iteration.
    """
    q = q0.clamp(low, high)
    r, jac = evaluate(q)
    cost = (r**2).sum(dim=-1)
    lam = torch.full_like(cost, lam0)
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype)
    history: list[float] = []
    for _ in range(iterations):
        history.append(float((r[:, :3].norm(dim=-1) / position_weight).median()))
        jt = jac.transpose(-1, -2)
        dq = torch.linalg.solve(jt @ jac + lam[:, None, None] * eye, jt @ r.unsqueeze(-1)).squeeze(-1)
        q_try = (q + dq.clamp(-max_step, max_step)).clamp(low, high)
        r_try, jac_try = evaluate(q_try)
        cost_try = (r_try**2).sum(dim=-1)
        kept = cost_try < cost  # False for NaN, so a broken step is never kept
        q = torch.where(kept[:, None], q_try, q)
        r = torch.where(kept[:, None], r_try, r)
        jac = torch.where(kept[:, None, None], jac_try, jac)
        cost = torch.where(kept, cost_try, cost)
        lam = torch.where(kept, lam * 0.5, lam * 2.0).clamp(1e-6, 1e6)
        if bool(((cost < 1e-8) | (lam >= 1e5)).all()):  # solved, or stalled at the best it can do
            break
    return q, r, history


##
# Simulation helpers.
##


class WristIK:
    """Multi-start inverse kinematics of the wrist link on the live articulation.

    This follows the vial generator's recipe: Levenberg-Marquardt on a position-first pose cost
    (:data:`IK_POSITION_WEIGHT` to :data:`IK_ROTATION_WEIGHT`), started from the seed and from Gaussian perturbations of
    it, then the valid solution nearest the seed (see :func:`select_ik_solution`). The vial generator runs a separate
    Newton model; here each iteration writes the candidate joints into PhysX and reads the wrist pose and Jacobian back,
    so the solution uses the exact kinematics the physics uses. Starts run one after another over all environments.
    The jaw joint is left alone: the rigid jaw is on the wrist link.
    """

    def __init__(self, env, seed: int = 0):
        self.env = env
        self.robot = env.scene[ROBOT]
        arm_ids, _ = self.robot.find_joints(ARM_JOINTS)
        self.arm_ids = list(arm_ids)
        self.ee = self.robot.find_bodies(EE_BODY_NAME)[0][0]
        self.jacobi_ee = self.ee - (1 if self.robot.is_fixed_base else 0)
        self.cols = [i + self.robot.num_base_dofs for i in self.arm_ids]
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.generator = torch.Generator(device=env.device).manual_seed(seed)
        self.weights = torch.tensor([IK_POSITION_WEIGHT] * 3 + [IK_ROTATION_WEIGHT] * 3, device=env.device)
        self.history: list[float] = []
        """Median position error [m] before each LM iteration of the first (unperturbed) start."""

    def write(self, q: torch.Tensor) -> None:
        """Put the arm at ``q`` with zero velocity and refresh the kinematics."""
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=self.ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=self.ids)
        self.env.sim.forward()

    def _evaluate(
        self, q_arm: torch.Tensor, base: torch.Tensor, pos_des: torch.Tensor, quat_des: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = base.clone()
        q[:, self.arm_ids] = q_arm
        self.write(q)
        pos = self.robot.data.body_pos_w.torch[:, self.ee]
        quat = self.robot.data.body_quat_w.torch[:, self.ee]
        goal = torch.where((quat_des * quat).sum(dim=-1, keepdim=True) < 0.0, -quat_des, quat_des)
        error = torch.cat((pos_des - pos, quat_box_minus(goal, quat)), dim=-1)
        jac = self.robot.data.body_link_jacobian_w.torch[:, self.jacobi_ee][:, :, self.cols]
        return error * self.weights, jac * self.weights[None, :, None]

    def solve(
        self,
        pos_des: torch.Tensor,
        quat_des: torch.Tensor,
        seed: torch.Tensor,
        *,
        seeds: int,
        noise_std: float,
        iterations: int,
        pos_tol: float,
        rot_tol: float,
        margin: float,
        randomize: bool = False,
    ) -> tuple[torch.Tensor, IKSolution, dict[str, torch.Tensor]]:
        """Joints ``(num_envs, 7)`` that put the wrist at the target, which environments found one, and diagnostics.

        Environments with no valid solution come back at ``seed``. The first start is the seed itself; the rest add
        Gaussian noise of ``noise_std`` [rad] to the arm joints. ``randomize`` draws among the valid starts instead
        of taking the one nearest the seed. ``info["quat"]`` is the wrist orientation the chosen joints achieve.
        """
        limits = self.robot.data.soft_joint_pos_limits.torch
        low, high = limits[:, self.arm_ids, 0], limits[:, self.arm_ids, 1]
        arm_seed = seed[:, self.arm_ids]
        # Deviation from the vial: it only rejects solutions within ``margin`` of a limit. The recorded grasp leaves
        # revolute_4 just 0.22 rad from its stop, so a pregrasp above it lands on the stop and every start is rejected.
        # Searching inside the inset limits lets the optimizer spend the rotation tolerance to stay clear instead.
        search_low, search_high = low + margin, high - margin
        solved, pos_errs, rot_errs = [], [], []
        for start in range(seeds):
            q0 = arm_seed.clone()
            if start > 0:
                q0 += noise_std * torch.randn(q0.shape, device=q0.device, generator=self.generator)
            q_arm, r, history = levenberg_marquardt(
                lambda qa: self._evaluate(qa, seed, pos_des, quat_des),
                q0,
                search_low,
                search_high,
                iterations=iterations,
                position_weight=IK_POSITION_WEIGHT,
            )
            if start == 0:
                self.history = history
            solved.append(q_arm)
            pos_errs.append(r[:, :3].norm(dim=-1) / IK_POSITION_WEIGHT)
            rot_errs.append(r[:, 3:].norm(dim=-1) / IK_ROTATION_WEIGHT)
        q_all, pos_all, rot_all = torch.stack(solved, dim=1), torch.stack(pos_errs, dim=1), torch.stack(rot_errs, dim=1)
        choice = select_ik_solution(
            q_all,
            pos_all,
            rot_all,
            arm_seed,
            low,
            high,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            margin=margin,
            randomize=randomize,
            generator=self.generator,
        )
        rows = torch.arange(q_all.shape[0], device=q_all.device)
        q = seed.clone()
        q[:, self.arm_ids] = torch.where(choice.valid[:, None], q_all[rows, choice.index], arm_seed)
        self.write(q)
        info = {
            "pos_err": pos_all[rows, choice.index],
            "rot_err": rot_all[rows, choice.index],
            "best_pos_err": pos_all.amin(dim=-1),
            "quat": torch.where(choice.valid[:, None], self.robot.data.body_quat_w.torch[:, self.ee].clone(), quat_des),
        }
        return q, choice, info


class Driver:
    """Moves the arm through the task's action interface, so the dataset is reachable by policy actions.

    The task applies ``target = measured joint position + action * scale``. The driver turns a desired joint position
    into the action that makes that target, with a slowly integrated trim on the arm joints that cancels sag under load.
    The jaw gets no trim: a stalled jaw is commanded at most one action step past itself, the policy's own squeeze.
    """

    TRIM_GAIN = 0.1
    TRIM_LIMIT = 0.15

    def __init__(self, env, arm_scale: float, jaw_scale: float):
        self.env = env
        self.robot = env.scene[ROBOT]
        self.bag = env.scene[BAG]
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.scale = torch.tensor([arm_scale] * 6 + [jaw_scale], device=env.device)
        self.trim = torch.zeros(env.num_envs, 7, device=env.device)  # the jaw's entry stays zero
        self.peak_lower_force = torch.zeros(env.num_envs, device=env.device)
        self.streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._park: torch.Tensor | None = None

    def joints(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch.clone()

    def begin_monitoring(self) -> None:
        self.peak_lower_force.zero_()
        self.streak.zero_()

    def step(self, q_cmd: torch.Tensor) -> None:
        if self._park is not None:
            self.write_bag(self._park)
        err = q_cmd - self.robot.data.joint_pos.torch
        self.trim[:, :6] = (self.trim[:, :6] + self.TRIM_GAIN * err[:, :6]).clamp(-self.TRIM_LIMIT, self.TRIM_LIMIT)
        action = ((err + self.trim) / self.scale).clamp(-1.0, 1.0)
        self.env.step(action)
        if self._park is not None:
            self.write_bag(self._park)
        self.peak_lower_force = torch.maximum(self.peak_lower_force, lower_body_force(self.env))
        self.streak = torch.where(holding(self.env), self.streak + 1, torch.zeros_like(self.streak))

    def hold(self, q_cmd: torch.Tensor, steps: int) -> None:
        for _ in range(steps):
            self.step(q_cmd)

    def move(self, q_target: torch.Tensor, settle_steps: int = HOLD_STEPS) -> None:
        """Interpolate every joint to ``q_target`` at :data:`JOINT_SPEED`, then rest."""
        q0 = self.joints()
        longest = float((q_target - q0).abs().max())
        steps = max(1, math.ceil(longest / JOINT_SPEED / self.env.step_dt))
        for k in range(1, steps + 1):
            self.step(q0 + (q_target - q0) * (k / steps))
        self.hold(q_target, settle_steps)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Measured joints and the bag's state.

        The bag state is its pose in the environment frame (7 columns), followed by the deformable strap's nodes in
        the bag frame, flattened (none with the rigid strap). :meth:`write_bag` takes the same layout, so a parked
        or restored bag keeps the shape the jaws gave its strap.
        """
        origins = self.env.scene.env_origins
        state = torch.cat(
            (
                self.bag.data.root_pos_w.torch - origins,
                self.bag.data.root_quat_w.torch,
                strap_nodes_bag(self.env).flatten(1),
            ),
            dim=-1,
        )
        return self.joints(), state

    def write_bag(self, state_env: torch.Tensor) -> None:
        """Write a bag state (see :meth:`snapshot`) and zero its velocity. A bare pose lays the strap at rest."""
        n = state_env.shape[0]
        pose_w = state_env[:, :7].clone()
        pose_w[:, :3] += self.env.scene.env_origins
        self.bag.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=self.ids)
        self.bag.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(n, 6, device=self.env.device), env_ids=self.ids
        )
        write_strap(self.env, self.ids, pose_w, state_env[:, 7:].reshape(n, -1, 3))

    def pin_bag(self, pose_env: torch.Tensor | None) -> None:
        """Hold the bag at ``pose_env`` on every step, or release it when ``pose_env`` is ``None``.

        The approach sweep pins the bag above the arm, the way the vial generator parks the vial. A dynamic bag
        placed there once would fall through the sweep, so the pose is rewritten before and after each step.
        """
        self._park = None if pose_env is None else pose_env.clone()

    def restore(self, q: torch.Tensor, bag_pose: torch.Tensor, settle: int) -> None:
        """Put the arm and the bag back at a seed-bank row and hold that command."""
        self.pin_bag(None)
        self.trim.zero_()
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=self.ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=self.ids)
        self.robot.actuators.target_command.set_position_index(value=q, joint_ids=None, env_ids=self.ids)
        self.write_bag(bag_pose)
        self.env.sim.forward()
        self.begin_monitoring()
        self.hold(q, settle)


##
# Generation.
##


def configure_env_cfg(env_cfg, args: argparse.Namespace) -> None:
    """Strip the task config to a bare scene for scripted driving: no dataset resets, curriculum, or success."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
    from isaaclab.utils import configclass

    @configclass
    class _Rewards:
        unused = RewardTermCfg(func=mdp.action_rate_l2, weight=1.0e-6)

    @configclass
    class _Terminations:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # Friction and servo-gain randomization stay on, so the dataset only holds poses that grasp across them.
    env_cfg.events.reset_from_dataset = EventTermCfg(
        func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )
    env_cfg.rewards = _Rewards()
    env_cfg.terminations = _Terminations()
    env_cfg.curriculum = None
    env_cfg.episode_length_s = 120.0
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs


class _Bank:
    """Rows of one phase that physics accepted, the vial generator's seed bank."""

    def __init__(self) -> None:
        self.q: list[torch.Tensor] = []
        self.pose: list[torch.Tensor] = []
        self.attempt: list[torch.Tensor] = []

    def __len__(self) -> int:
        return sum(t.shape[0] for t in self.q)

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.cat(self.q), torch.cat(self.pose), torch.cat(self.attempt)

    def add(self, q: torch.Tensor, pose: torch.Tensor, attempt: torch.Tensor) -> int:
        """Append rows whose attempt id is not already in the bank. One parent yields one child."""
        if q.shape[0] == 0:
            return 0
        seen = set(self.stacked()[2].tolist()) if len(self) else set()
        keep: list[int] = []
        for i, raw in enumerate(attempt.tolist()):
            ident = int(raw)
            if ident in seen:
                continue
            seen.add(ident)
            keep.append(i)
        if not keep:
            return 0
        index = torch.tensor(keep, dtype=torch.long, device=q.device)
        attempt = attempt.to(q.device)
        self.q.append(q[index].detach().cpu())
        self.pose.append(pose[index].detach().cpu())
        self.attempt.append(attempt[index].detach().cpu())
        return len(keep)

    def sample(
        self, n: int, device: torch.device | str, child: _Bank | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw ``n`` parent rows, preferring ones that do not already have a child."""
        q, pose, attempt = self.stacked()
        if child is not None and len(child):
            taken = {int(v) for v in child.stacked()[2].tolist()}
            keep = [i for i, raw in enumerate(attempt.tolist()) if int(raw) not in taken]
            if keep:
                index = torch.tensor(keep, dtype=torch.long)
                q, pose, attempt = q[index], pose[index], attempt[index]
        pick = torch.randint(q.shape[0], (n,))
        return q[pick].to(device), pose[pick].to(device), attempt[pick].to(device)


def _gate(checks: tuple[tuple[str, torch.Tensor], ...]) -> tuple[torch.Tensor, Counter]:
    accepted = torch.ones(checks[0][1].shape[0], dtype=torch.bool, device=checks[0][1].device)
    rejected: Counter = Counter()
    for name, bad in checks:
        newly = accepted & bad
        rejected[name] += int(newly.sum())
        accepted &= ~bad
    return accepted, rejected


def _jaw_quat(goal, args: argparse.Namespace) -> tuple[float, float, float, float]:
    if goal.jaw_quat_bag is None:
        raise RuntimeError("The generator needs the recorded jaw orientation in scene/grasp_goal.json.")
    if args.orientation == "vertical":
        return vertical_jaw_quat_bag(goal.jaw_quat_bag)
    return goal.jaw_quat_bag


def _recorded_joints(robot, goal) -> torch.Tensor:
    recorded = robot.data.default_joint_pos.torch.clone()
    names = list(robot.joint_names)
    for name, value in goal.arm_joints.items():
        if name in names:
            recorded[:, names.index(name)] = value
    recorded[:, 6] = GRIPPER_OPEN_POS
    return recorded


def _ik_options(args: argparse.Namespace) -> dict:
    return dict(
        seeds=args.ik_seeds,
        noise_std=args.ik_noise_std,
        iterations=args.ik_iterations,
        rot_tol=math.radians(args.ik_rotation_tolerance_deg),
        margin=args.ik_joint_margin,
    )


def _targets(driver: Driver, goal, args: argparse.Namespace):
    """Wrist targets for the bag where it lies now, in the world frame."""
    pos = driver.bag.data.root_pos_w.torch
    quat = driver.bag.data.root_quat_w.torch
    return wrist_targets(pos, quat, goal.tip_pos_bag, _jaw_quat(goal, args), args.pregrasp_backoff)


def _held_grasp_target(driver: Driver, ik: WristIK, goal, args: argparse.Namespace):
    """Wrist position that puts the tip on the pinch with the orientation the wrist holds now, and that orientation.

    The pregrasp may only reach the recorded orientation within the IK tolerance. Keeping the tip on the pinch with
    the orientation actually held stops the 10 cm tip offset turning that tolerance into a centimetre-scale miss.
    """
    grasp_pos, _pregrasp_pos, wrist_quat = _targets(driver, goal, args)
    achieved = driver.robot.data.body_quat_w.torch[:, ik.ee].clone()
    tip_offset = grasp_pos.new_tensor(EE_TIP_OFFSET).expand_as(grasp_pos)
    return grasp_pos + quat_apply(wrist_quat, tip_offset) - quat_apply(achieved, tip_offset), achieved


def _tip_error(env) -> torch.Tensor:
    return (tip_pos_w(env) - strap_target_w(env)).norm(dim=-1)


def _check_recording(
    env, driver: Driver, ik: WristIK, goal, home_q: torch.Tensor, home_pose: torch.Tensor
) -> torch.Tensor:
    """Refuse to continue when the recorded joints no longer put the tip on the recorded pinch."""
    ik.write(_recorded_joints(driver.robot, goal))
    err = _tip_error(env)
    seed_tip = tip_pos_w(env)[0] - env.scene.env_origins[0]
    fmt = lambda v: "(" + ", ".join(f"{float(x):.3f}" for x in v) + ")"  # noqa: E731
    print(f"[generator]   env0 tip at recorded joints {fmt(seed_tip)}")
    if float(err.median()) > MAX_SEED_TIP_ERROR:
        raise RuntimeError(
            f"The recorded joints leave the tip {1e3 * float(err.median()):.0f} mm from the recorded pinch."
            " The arm is mounted differently from when grasp_goal.json was recorded (check the arm base rotation),"
            " or the recording is stale; inverse kinematics cannot fix that."
        )
    driver.restore(home_q, home_pose, HOLD_STEPS)
    return err


def _parked(pose: torch.Tensor) -> torch.Tensor:
    parked = pose.clone()
    parked[:, 2] = PARK_HEIGHT
    return parked


def _motion_stats(env, driver: Driver, pose: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    shift = (pose[:, :2] - reference[:, :2]).norm(dim=-1)
    lift = pose[:, 2] - reference[:, 2]
    speed = driver.bag.data.root_lin_vel_w.torch.norm(dim=-1)
    grasp = strap_grasp(env)
    return {
        "peak_lower_force_n": driver.peak_lower_force,
        "bag_shift_mm": 1e3 * shift,
        "bag_lift_mm": 1e3 * lift,
        "bag_speed_mps": speed,
        "jaw_pos_rad": gripper_joint_pos(env),
        "jaw_closed": gripper_closed(env).float(),
        "strap_through_mouth": grasp.through.float(),
        "strap_back_dist_mm": 1e3 * grasp.back_dist,
        "holding_streak": driver.streak.float(),
    }


def _print_batch(phase: str, batch: int, accepted: int, total: int, goal: int, why: Counter, stats: dict) -> None:
    print(f"[generator] {phase} batch {batch + 1}: accepted {accepted} ({total}/{goal})")
    print(f"[generator]   rejected: {dict(why)}")
    print("[generator]   median [min..max] over the batch:")
    for name, values in stats.items():
        low, mid, high = float(values.min()), float(values.median()), float(values.max())
        print(f"[generator]     {name:22s} {mid:9.3f} [{low:.3f}..{high:.3f}]")


def collect_home(env, driver: Driver, ik: WristIK, args, goal, checked: list[bool]):
    """Settle the arm at home, then place the bag and let it settle. That resting state is the home row."""
    n, device = env.num_envs, env.device
    env.reset()
    driver.pin_bag(None)
    names = list(driver.robot.joint_names)
    if names != list(ARM_JOINT_NAMES):
        raise RuntimeError(f"Expected the robot's joints to be {ARM_JOINT_NAMES}, got {names}.")
    home_cmd = driver.robot.data.default_joint_pos.torch.clone()
    home_cmd[:, 6] = GRIPPER_OPEN_POS
    driver.trim.zero_()
    driver.hold(home_cmd, SETTLE_STEPS)
    driver.write_bag(sample_bag_poses(n, args.x_half, args.y_half, math.radians(args.yaw_half_deg), device))
    driver.hold(home_cmd, SETTLE_STEPS)
    home_q, home_pose = driver.snapshot()
    if not checked[0]:
        checked[0] = True
        err = _check_recording(env, driver, ik, goal, home_q, home_pose)
        home_q, home_pose = driver.snapshot()
    else:
        err = _tip_error(env)
    speed = driver.bag.data.root_lin_vel_w.torch.norm(dim=-1)
    resting = (speed > MAX_BAG_SPEED) | (home_pose[:, 2] < BAG_REST_Z - 0.02)
    accepted, rejected = _gate((("bag_not_resting", resting),))
    stats = {"seed_tip_err_mm": 1e3 * err, "bag_speed_mps": speed, "bag_z_mm": 1e3 * home_pose[:, 2]}
    return home_q[accepted], home_pose[accepted], accepted, rejected, stats


def collect_pregrasp(env, driver: Driver, ik: WristIK, args, goal, bank: _Bank, child: _Bank):
    """Sweep home to the pregrasp with the bag pinned away, then put the bag back."""
    env.reset()
    q, pose, attempt = bank.sample(env.num_envs, env.device, child)
    driver.restore(q, pose, SETTLE_STEPS)
    settled_q, settled_pose = driver.snapshot()
    _grasp_pos, pregrasp_pos, wrist_quat = _targets(driver, goal, args)
    started = time.perf_counter()
    q_pre, choice, info = ik.solve(
        pregrasp_pos,
        wrist_quat,
        _recorded_joints(driver.robot, goal),
        pos_tol=PREGRASP_POSITION_TOLERANCE,
        randomize=True,
        **_ik_options(args),
    )
    history = [round(e, 3) for e in ik.history[::10]]
    print(f"[generator]   IK median position error [m] by iteration, unperturbed start: {history}")
    print(f"[generator]   pregrasp IK took {time.perf_counter() - started:.1f} s")
    ik.write(settled_q)
    driver.robot.actuators.target_command.set_position_index(value=settled_q, joint_ids=None, env_ids=driver.ids)
    driver.pin_bag(_parked(settled_pose))
    driver.move(q_pre, settle_steps=0)
    driver.pin_bag(None)
    driver.write_bag(settled_pose)
    driver.begin_monitoring()
    driver.hold(q_pre, HOLD_STEPS)
    snap_q, snap_pose = driver.snapshot()
    tip_err = _tip_error(env)
    stats = _motion_stats(env, driver, snap_pose, settled_pose)
    stats.update(
        {
            "pregrasp_tip_err_mm": 1e3 * tip_err,
            "pregrasp_ik_best_pos_err_mm": 1e3 * info["best_pos_err"],
            "pregrasp_ik_rot_err_rad": info["rot_err"],
            "pregrasp_ik_valid_starts": choice.valid_count.float(),
            "pregrasp_ik_joint_margin_rad": choice.clearance,
        }
    )
    shift = stats["bag_shift_mm"] / 1e3
    accepted, rejected = _gate(
        (
            ("ik_pregrasp_failed", ~choice.valid),
            ("lower_body_contact", driver.peak_lower_force > args.max_lower_force),
            ("bag_moved", shift > args.max_bag_shift),
            ("bag_still_moving", stats["bag_speed_mps"] > MAX_BAG_SPEED),
        )
    )
    return snap_q[accepted], snap_pose[accepted], attempt[accepted], rejected, stats


def collect_grasp(env, driver: Driver, ik: WristIK, args, goal, bank: _Bank, child: _Bank):
    """Move from a validated pregrasp onto the recorded pinch, jaw still open, bag in the tray."""
    env.reset()
    q, pose, attempt = bank.sample(env.num_envs, env.device, child)
    driver.restore(q, pose, HOLD_STEPS)
    settled_q, settled_pose = driver.snapshot()
    grasp_pos, achieved = _held_grasp_target(driver, ik, goal, args)
    started = time.perf_counter()
    q_grasp, choice, info = ik.solve(
        grasp_pos, achieved, settled_q, pos_tol=args.ik_position_tolerance, **_ik_options(args)
    )
    print(f"[generator]   grasp IK took {time.perf_counter() - started:.1f} s")
    ik.write(settled_q)
    driver.write_bag(settled_pose)
    driver.robot.actuators.target_command.set_position_index(value=settled_q, joint_ids=None, env_ids=driver.ids)
    driver.begin_monitoring()
    driver.move(q_grasp)
    snap_q, snap_pose = driver.snapshot()
    tip_err = _tip_error(env)
    stats = _motion_stats(env, driver, snap_pose, settled_pose)
    stats.update(
        {
            "grasp_tip_err_mm": 1e3 * tip_err,
            "ik_best_pos_err_mm": 1e3 * info["best_pos_err"],
            "ik_pos_err_mm": 1e3 * info["pos_err"],
            "ik_rot_err_rad": info["rot_err"],
            "ik_valid_starts": choice.valid_count.float(),
            "ik_joint_delta_rad": choice.delta,
            "ik_joint_margin_rad": choice.clearance,
            "track_err_rad": (snap_q - q_grasp)[:, :6].abs().amax(dim=-1),
        }
    )
    shift = stats["bag_shift_mm"] / 1e3
    accepted, rejected = _gate(
        (
            ("ik_grasp_failed", ~choice.valid),
            ("lower_body_contact", driver.peak_lower_force > args.max_lower_force),
            ("bag_moved", shift > args.max_bag_shift),
            ("bag_still_moving", stats["bag_speed_mps"] > MAX_BAG_SPEED),
        )
    )
    return snap_q[accepted], snap_pose[accepted], attempt[accepted], rejected, stats


def collect_grasp_closing(env, driver: Driver, ik: WristIK, args, goal, bank: _Bank, child: _Bank):
    """From a validated open-jaw grasp, slide the jaw ``args.close_insert`` further onto the strap and close it to
    :data:`CLOSING_JAW_POS`: that state is the row. Then finish the close, and keep the row only if it holds.

    The open-jaw grasp leaves the strap at the jaw tips, about 70 mm in front of the hinge and past the 65 mm front of
    the closed mouth, so a jaw closed there pinches the strap at the tips or misses it. The row is recorded before
    :func:`~..mdp.task.holding` can be true, so the policy still has to make the grasp.
    """
    env.reset()
    q, pose, attempt = bank.sample(env.num_envs, env.device, child)
    driver.restore(q, pose, HOLD_STEPS)
    settled_q, settled_pose = driver.snapshot()
    grasp_pos, achieved = _held_grasp_target(driver, ik, goal, args)
    insert = quat_apply(achieved, grasp_pos.new_tensor(JAW_INSERT_AXIS).expand_as(grasp_pos))
    q_deep, choice, _info = ik.solve(
        grasp_pos + args.close_insert * insert,
        achieved,
        settled_q,
        pos_tol=args.ik_position_tolerance,
        **_ik_options(args),
    )
    ik.write(settled_q)
    driver.write_bag(settled_pose)
    driver.robot.actuators.target_command.set_position_index(value=settled_q, joint_ids=None, env_ids=driver.ids)
    driver.begin_monitoring()
    driver.move(q_deep)
    q_part = q_deep.clone()
    q_part[:, 6] = CLOSING_JAW_POS
    driver.move(q_part, settle_steps=CLOSING_SETTLE_STEPS)
    snap_q, snap_pose = driver.snapshot()
    stats = _motion_stats(env, driver, snap_pose, settled_pose)
    already_holding = holding(env)
    q_closed = q_part.clone()
    q_closed[:, 6] = GRIPPER_CLOSED_POS
    driver.hold(q_closed, CLOSE_STEPS)
    stats["holding_streak_after_close"] = driver.streak.float()
    shift = stats["bag_shift_mm"] / 1e3
    accepted, rejected = _gate(
        (
            ("ik_insert_failed", ~choice.valid),
            ("already_holding", already_holding),
            ("not_holding_after_close", driver.streak < HOLD_STEPS),
            ("lower_body_contact", driver.peak_lower_force > args.max_lower_force),
            ("bag_moved", shift > args.max_bag_shift),
            ("bag_still_moving", stats["bag_speed_mps"] > MAX_BAG_SPEED),
        )
    )
    return snap_q[accepted], snap_pose[accepted], attempt[accepted], rejected, stats


def generate(env, args: argparse.Namespace) -> ResetDataset:
    """Fill each phase to ``args.attempts`` rows from the phase before it, then build the dataset.

    Home is sampled on its own. Every later phase draws seeds from the rows the previous phase kept, so a grasp only
    exists when the pregrasp before it did.
    """
    from ..env_cfg import ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE

    driver = Driver(env, ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE)
    ik = WristIK(env, args.seed)
    goal = load_grasp_goal()
    banks = {phase: _Bank() for phase in PHASES}
    parents = {PREGRASP: HOME, GRASP: PREGRASP, GRASP_CLOSING: GRASP}
    rejected: Counter = Counter()
    tried = 0
    checked = [False]
    next_id = 0

    for phase in PHASES:
        parent = parents.get(phase)
        empty_batches = 0
        for batch in range(args.max_batches):
            if len(banks[phase]) >= args.attempts:
                break
            if parent is not None and len(banks[parent]) == 0:
                print(f"[generator] {phase}: no {parent} rows to continue from.")
                break
            if phase == HOME:
                q, pose, _mask, why, stats = collect_home(env, driver, ik, args, goal, checked)
                attempt = torch.arange(next_id, next_id + q.shape[0])
                next_id += q.shape[0]
            else:
                q, pose, attempt, why, stats = {
                    PREGRASP: collect_pregrasp,
                    GRASP: collect_grasp,
                    GRASP_CLOSING: collect_grasp_closing,
                }[phase](env, driver, ik, args, goal, banks[parent], banks[phase])
            room = args.attempts - len(banks[phase])
            added = banks[phase].add(q[:room], pose[:room], attempt[:room])
            tried += env.num_envs
            rejected.update(why)
            _print_batch(phase, batch, added, len(banks[phase]), args.attempts, why, stats)
            if len(banks[phase]) >= args.attempts:
                break
            empty_batches = empty_batches + 1 if added == 0 else 0
            if empty_batches >= 2:
                print(f"[generator] {phase}: two batches in a row accepted nothing; stopping this phase.")
                break

    print(f"[generator] rejected by reason: {dict(rejected)}")
    missing = [phase for phase in PHASES if len(banks[phase]) == 0]
    if missing:
        raise SystemExit(f"[generator] no physics-validated rows for {missing}; see the reasons above.")
    short = [phase for phase in PHASES if len(banks[phase]) < args.attempts]
    if short:
        counts = {phase: len(banks[phase]) for phase in PHASES}
        print(f"[generator] WARNING: short of {args.attempts} rows per phase: {counts}")

    joint_pos, bag_state, phase_ids, attempt_ids = [], [], [], []
    for index, phase in enumerate(PHASES):
        q, state, attempt = banks[phase].stacked()
        joint_pos.append(q)
        bag_state.append(state)
        phase_ids.append(torch.full((q.shape[0],), index, dtype=torch.long))
        attempt_ids.append(attempt.long())
    meta = {
        "joint_names": list(env.scene[ROBOT].joint_names),
        "rows_per_phase": {phase: len(banks[phase]) for phase in PHASES},
        "tried": tried,
        "rejected": dict(rejected),
        "x_half": args.x_half,
        "y_half": args.y_half,
        "yaw_half_deg": args.yaw_half_deg,
        "pregrasp_backoff": args.pregrasp_backoff,
        "close_insert": args.close_insert,
        "max_lower_force": args.max_lower_force,
        "max_bag_shift": args.max_bag_shift,
        "orientation": args.orientation,
        "ik_seeds": args.ik_seeds,
        "ik_iterations": args.ik_iterations,
        "ik_noise_std": args.ik_noise_std,
        "ik_position_tolerance": args.ik_position_tolerance,
        "ik_rotation_tolerance_deg": args.ik_rotation_tolerance_deg,
        "ik_joint_margin": args.ik_joint_margin,
        "seed": args.seed,
    }
    state = torch.cat(bag_state)
    strap = state[:, 7:].reshape(state.shape[0], -1, 3)
    return ResetDataset(
        torch.cat(phase_ids), torch.cat(attempt_ids), torch.cat(joint_pos), state[:, :7].contiguous(), meta, strap
    )


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Generate the handle-grab reset dataset.")
    parser.add_argument("--attempts", type=int, default=128, help="Rows to collect for each phase.")
    parser.add_argument("--num_envs", type=int, default=128, help="Attempts scripted in parallel per batch.")
    parser.add_argument("--max_batches", type=int, default=40, help="Give up after this many batches.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="Dataset file to write.")
    parser.add_argument("--x_half", type=float, default=0.03, help="Half-width of the bag spawn box in x [m].")
    parser.add_argument("--y_half", type=float, default=0.04, help="Half-width of the bag spawn box in y [m].")
    parser.add_argument("--yaw_half_deg", type=float, default=20.0, help="Half-range of the bag heading [deg].")
    parser.add_argument("--pregrasp_backoff", type=float, default=0.06, help="Pregrasp distance from the pinch [m].")
    parser.add_argument(
        "--close_insert", type=float, default=0.015, help="Jaw slide onto the strap before a closing grasp [m]."
    )
    parser.add_argument("--max_lower_force", type=float, default=3.0, help="Peak lower-body contact allowed [N].")
    parser.add_argument("--max_bag_shift", type=float, default=0.03, help="Bag displacement allowed [m].")
    parser.add_argument(
        "--orientation",
        choices=("recorded", "vertical"),
        default="recorded",
        help=(
            "Jaw orientation target: as recorded, or the recording with its insert axis made vertical (the vial's"
            " geometric rule). The arm reaches 'vertical' mostly by drifting back to the recorded tilt, so it needs a"
            " looser --ik_rotation_tolerance_deg (about 30)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    ik = parser.add_argument_group("inverse kinematics", "Defaults are the vial generator's (reset/generator.py).")
    ik.add_argument("--ik_seeds", type=int, default=64, help="IK starts per pose: the recorded joints plus noisy ones.")
    ik.add_argument("--ik_iterations", type=int, default=120, help="Levenberg-Marquardt iterations per start.")
    ik.add_argument("--ik_noise_std", type=float, default=0.65, help="Std of the Gaussian start noise [rad].")
    ik.add_argument("--ik_position_tolerance", type=float, default=0.005, help="Grasp position tolerance [m].")
    ik.add_argument("--ik_rotation_tolerance_deg", type=float, default=18.0, help="Rotation tolerance [deg].")
    ik.add_argument("--ik_joint_margin", type=float, default=0.03, help="Required distance from joint limits [rad].")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    return setup_preset_cli(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args, hydra_args = _parse_args(argv)
    torch.manual_seed(args.seed)

    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.utils import validate

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(TASK, None, overrides=hydra_args)
    configure_env_cfg(env_cfg, args)
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        try:
            with torch.inference_mode():
                dataset = generate(env, args)
            # Closing the app can end the process, so the dataset has to be on disk first.
            dataset.save(args.output)
            print(f"[generator] wrote {dataset.phase.shape[0]} rows to {args.output}", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
