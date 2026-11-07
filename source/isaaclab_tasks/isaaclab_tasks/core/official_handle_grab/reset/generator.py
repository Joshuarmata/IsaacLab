# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the physics-validated reset dataset of the handle-grab task.

For every attempt the generator places the bag somewhere in the spawn box, solves inverse kinematics for the recorded
pinch (and for a pregrasp a few centimetres back along the approach axis), then drives the arm there in PhysX through
the task's own relative-joint action interface: home, pregrasp, grasp with the jaw open, and the jaw closed on the
strap. An attempt is kept only if the physics ends in a true zone grasp (the same ``holding`` test the task rewards)
without the arm leaning on the bag body or knocking the bag. Every accepted attempt contributes one row per phase::

    uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator

Writes ``handle_grab_reset_dataset.pt`` next to this file. Rows hold measured joint positions and the bag's pose in the
environment frame, so no row starts interpenetrating or off the strap.
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
    GRIPPER_OPEN_POS,
    JAW_INSERT_AXIS,
    ROBOT,
    gripper_closed,
    gripper_joint_pos,
    holding,
    load_grasp_goal,
    lower_body_force,
    strap_contacts,
    strap_target_w,
    tip_pos_w,
)
from .curriculum import GRASP, GRASP_CLOSED, HOME, PHASES, PREGRASP
from .dataset import DEFAULT_DATASET_PATH, ResetDataset

TASK = "Isaac-HandleGrab"

JOINT_SPEED = 0.4
"""Scripted joint speed [rad/s]; below the arm's 0.5 rad/s drive cap, so the arm keeps up with the command."""
SETTLE_STEPS = 30
"""Policy steps the bag and arm rest before home is recorded."""
HOLD_STEPS = 15
"""Policy steps the arm rests at each open-jaw waypoint before it is recorded."""
CLOSE_HOLD_STEPS = 20
"""Policy steps the closed jaw is held before the grasp is judged."""
HOLDING_STREAK = 10
"""The strap must be in the jaws for this many consecutive final steps."""
MAX_BAG_LIFT = 0.005
"""A closed grasp that raised the bag more than this [m] is not a resting grasp."""
MAX_BAG_SPEED = 0.02
"""A closed grasp whose bag still moves faster than this [m/s] is not settled."""
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
    ok = torch.isfinite(q).all(dim=-1) & (pos_err <= pos_tol) & (rot_err <= rot_tol) & (clearance >= margin)
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
    into the action that makes that target, with a slowly integrated trim on the arm joints that cancels gravity sag.
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

    def joints(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch.clone()

    def begin_monitoring(self) -> None:
        self.peak_lower_force.zero_()
        self.streak.zero_()

    def step(self, q_cmd: torch.Tensor) -> None:
        err = q_cmd - self.robot.data.joint_pos.torch
        self.trim[:, :6] = (self.trim[:, :6] + self.TRIM_GAIN * err[:, :6]).clamp(-self.TRIM_LIMIT, self.TRIM_LIMIT)
        action = ((err + self.trim) / self.scale).clamp(-1.0, 1.0)
        self.env.step(action)
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

    def close_jaw(self, q_arm: torch.Tensor) -> None:
        """Ramp the jaw shut at :data:`JOINT_SPEED` with the arm held at ``q_arm``, then keep squeezing."""
        q_cmd = q_arm.clone()
        start = float(q_cmd[:, 6].max())
        steps = math.ceil((start - GRIPPER_CLOSED_POS) / JOINT_SPEED / self.env.step_dt)
        for k in range(1, steps + 1):
            q_cmd[:, 6] = start + (GRIPPER_CLOSED_POS - start) * (k / steps)
            self.step(q_cmd)
        q_cmd[:, 6] = GRIPPER_CLOSED_POS
        self.hold(q_cmd, CLOSE_HOLD_STEPS)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Measured joints and the bag's pose in the environment frame."""
        origins = self.env.scene.env_origins
        pose = torch.cat(
            (self.bag.data.root_pos_w.torch - origins, self.bag.data.root_quat_w.torch),
            dim=-1,
        )
        return self.joints(), pose


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


def run_batch(
    env, driver: Driver, ik: WristIK, args: argparse.Namespace
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], Counter, dict[str, torch.Tensor]]:
    """Script one attempt in every environment.

    Returns:
        The accepted ``(joints, bag pose)`` rows per phase, why the rest were rejected, and per-environment
        measurements (tracking errors, contacts, forces) for diagnosing rejections.
    """
    robot, bag, device = driver.robot, driver.bag, env.device
    n = env.num_envs
    goal = load_grasp_goal()
    if goal.jaw_quat_bag is None:
        raise RuntimeError("The generator needs the recorded jaw orientation in scene/grasp_goal.json.")
    joint_names = list(robot.joint_names)
    if joint_names != list(ARM_JOINT_NAMES):
        raise RuntimeError(f"Expected the robot's joints to be {ARM_JOINT_NAMES}, got {joint_names}.")

    env.reset()
    pose = sample_bag_poses(n, args.x_half, args.y_half, math.radians(args.yaw_half_deg), device)
    pose_w = pose.clone()
    pose_w[:, :3] += env.scene.env_origins
    bag.write_root_pose_to_sim_index(root_pose=pose_w, env_ids=driver.ids)
    bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device), env_ids=driver.ids)
    driver.trim.zero_()

    home_cmd = robot.data.default_joint_pos.torch.clone()
    home_cmd[:, 6] = GRIPPER_OPEN_POS
    driver.hold(home_cmd, SETTLE_STEPS)
    home_q, home_pose = driver.snapshot()

    # Targets come from where the bag actually settled, in the world frame the kinematics use.
    bag_pos_w, bag_quat_w = bag.data.root_pos_w.torch.clone(), bag.data.root_quat_w.torch.clone()
    jaw_quat_bag = goal.jaw_quat_bag if args.orientation == "recorded" else vertical_jaw_quat_bag(goal.jaw_quat_bag)
    grasp_pos, pregrasp_pos, wrist_quat = wrist_targets(
        bag_pos_w, bag_quat_w, goal.tip_pos_bag, jaw_quat_bag, args.pregrasp_backoff
    )
    recorded = robot.data.default_joint_pos.torch.clone()
    for name, value in goal.arm_joints.items():
        if name in joint_names:
            recorded[:, joint_names.index(name)] = value
    recorded[:, 6] = GRIPPER_OPEN_POS

    def tip_error() -> torch.Tensor:
        """Distance [m] from the rigid-jaw tip to the recorded pinch point on the bag as it lies now."""
        return (tip_pos_w(env) - strap_target_w(env)).norm(dim=-1)

    home_tip_err = tip_error()
    ik.write(recorded)  # does the recorded pose, put at the recorded joints, really put the tip on the strap?
    seed_tip_err = tip_error()
    if float(seed_tip_err.median()) > MAX_SEED_TIP_ERROR:
        raise RuntimeError(
            f"The recorded joints leave the tip {1e3 * float(seed_tip_err.median()):.0f} mm from the recorded pinch."
            " The arm is mounted differently from when grasp_goal.json was recorded (check the arm base rotation),"
            " or the recording is stale; inverse kinematics cannot fix that."
        )
    seed_tip = tip_pos_w(env)[0] - env.scene.env_origins[0]
    ik_started = time.perf_counter()
    rot_tol = math.radians(args.ik_rotation_tolerance_deg)
    ik_options = dict(
        seeds=args.ik_seeds,
        noise_std=args.ik_noise_std,
        iterations=args.ik_iterations,
        rot_tol=rot_tol,
        margin=args.ik_joint_margin,
    )
    # The vial generator's order: the overhead pose first, then the grasp. The overhead (our pregrasp, backed off along
    # the insert axis) starts from the recorded joints and takes a random valid solution, which is what spreads the
    # arm configurations in the dataset. The grasp then reuses the orientation the overhead solution actually
    # achieved and starts from its joints, so both sit on one IK branch. Like the vial's intermediate waypoint the
    # overhead only needs 1.5 cm, since the arm then moves on to the grasp.
    q_pregrasp, pregrasp_ik, pregrasp_info = ik.solve(
        pregrasp_pos,
        wrist_quat,
        recorded,
        pos_tol=PREGRASP_POSITION_TOLERANCE,
        randomize=True,
        **ik_options,
    )
    ik_history = list(ik.history)
    q_grasp, grasp_ik, grasp_info = ik.solve(
        grasp_pos, pregrasp_info["quat"], q_pregrasp, pos_tol=args.ik_position_tolerance, **ik_options
    )
    ik_tip = tip_pos_w(env)[0] - env.scene.env_origins[0]
    grasp_ok, pregrasp_ok = grasp_ik.valid, pregrasp_ik.valid
    ik_seconds = time.perf_counter() - ik_started
    ik.write(home_q)  # IK moved the arm instantly; put it back before any physics runs
    robot.actuators.target_command.set_position_index(value=home_q, joint_ids=None, env_ids=driver.ids)

    origin = env.scene.env_origins[0]
    base = robot.data.root_pos_w.torch[0] - origin
    fmt = lambda v: "(" + ", ".join(f"{float(x):.3f}" for x in v) + ")"  # noqa: E731
    print(f"[generator]   env0 [m, env frame] arm base {fmt(base)}, bag {fmt(bag_pos_w[0] - origin)}")
    print(f"[generator]   env0 strap target tip {fmt(strap_target_w(env)[0] - origin)}")
    print(f"[generator]   env0 wrist target {fmt(grasp_pos[0] - origin)}")
    print(f"[generator]   env0 tip at recorded joints {fmt(seed_tip)}, tip after IK {fmt(ik_tip)}")
    history = [round(e, 3) for e in ik_history[::10]]
    print(f"[generator]   IK median position error [m] by iteration, unperturbed start: {history}")
    print(
        f"[generator]   IK took {ik_seconds:.1f} s for {args.ik_seeds} starts x {args.ik_iterations} iterations x 2 poses"
    )
    driver.begin_monitoring()
    driver.move(q_pregrasp)
    pregrasp_q, pregrasp_pose = driver.snapshot()
    pregrasp_tip_err = tip_error()
    driver.move(q_grasp)
    grasp_q, grasp_pose = driver.snapshot()
    grasp_tip_err = tip_error()
    driver.close_jaw(q_grasp)
    closed_q, closed_pose = driver.snapshot()

    shift = (closed_pose[:, :2] - home_pose[:, :2]).norm(dim=-1)
    lift = closed_pose[:, 2] - home_pose[:, 2]
    speed = bag.data.root_lin_vel_w.torch.norm(dim=-1)
    rigid_on, moving_on = strap_contacts(env)
    stats = {
        "home_tip_err_mm": 1e3 * home_tip_err,
        "seed_tip_err_mm": 1e3 * seed_tip_err,
        "ik_best_pos_err_mm": 1e3 * grasp_info["best_pos_err"],
        "ik_pos_err_mm": 1e3 * grasp_info["pos_err"],
        "ik_rot_err_rad": grasp_info["rot_err"],
        "ik_valid_starts": grasp_ik.valid_count.float(),
        "ik_joint_delta_rad": grasp_ik.delta,
        "ik_joint_margin_rad": grasp_ik.clearance,
        "pregrasp_ik_best_pos_err_mm": 1e3 * pregrasp_info["best_pos_err"],
        "pregrasp_ik_rot_err_rad": pregrasp_info["rot_err"],
        "pregrasp_ik_valid_starts": pregrasp_ik.valid_count.float(),
        "pregrasp_ik_joint_margin_rad": pregrasp_ik.clearance,
        "track_err_rad": (grasp_q - q_grasp)[:, :6].abs().amax(dim=-1),
        "pregrasp_tip_err_mm": 1e3 * pregrasp_tip_err,
        "grasp_tip_err_mm": 1e3 * grasp_tip_err,
        "closed_tip_err_mm": 1e3 * tip_error(),
        "peak_lower_force_n": driver.peak_lower_force,
        "bag_shift_mm": 1e3 * shift,
        "bag_lift_mm": 1e3 * lift,
        "bag_speed_mps": speed,
        "jaw_pos_rad": gripper_joint_pos(env),
        "jaw_closed": gripper_closed(env).float(),
        "rigid_jaw_on_strap": rigid_on.float(),
        "moving_jaw_on_strap": moving_on.float(),
        "holding_streak": driver.streak.float(),
    }
    checks = (
        ("ik_grasp_failed", ~grasp_ok),
        ("ik_pregrasp_failed", ~pregrasp_ok),
        ("lower_body_contact", driver.peak_lower_force > args.max_lower_force),
        ("bag_moved", shift > args.max_bag_shift),
        ("bag_lifted", lift > MAX_BAG_LIFT),
        ("not_holding", driver.streak < HOLDING_STREAK),
        ("bag_still_moving", speed > MAX_BAG_SPEED),
    )
    accepted = torch.ones(n, dtype=torch.bool, device=device)
    rejected: Counter = Counter()
    for name, bad in checks:
        newly = accepted & bad
        rejected[name] += int(newly.sum())
        accepted &= ~bad

    rows = {
        HOME: (home_q, home_pose),
        PREGRASP: (pregrasp_q, pregrasp_pose),
        GRASP: (grasp_q, grasp_pose),
        GRASP_CLOSED: (closed_q, closed_pose),
    }
    kept = {phase: (q[accepted].cpu(), p[accepted].cpu()) for phase, (q, p) in rows.items()}
    return kept, rejected, stats


def generate(env, args: argparse.Namespace) -> ResetDataset:
    """Run batches until ``args.attempts`` attempts are accepted, then build the dataset."""
    from ..env_cfg import ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE

    driver = Driver(env, ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE)
    ik = WristIK(env, args.seed)

    kept = {phase: ([], []) for phase in PHASES}
    rejected: Counter = Counter()
    accepted = tried = 0
    empty_batches = 0
    for batch in range(args.max_batches):
        rows, why, stats = run_batch(env, driver, ik, args)
        count = rows[HOME][0].shape[0]
        for phase in PHASES:
            kept[phase][0].append(rows[phase][0])
            kept[phase][1].append(rows[phase][1])
        accepted += count
        tried += env.num_envs
        rejected.update(why)
        print(f"[generator] batch {batch + 1}: accepted {count}/{env.num_envs} ({accepted}/{args.attempts} total)")
        print(f"[generator]   rejected: {dict(why)}")
        print("[generator]   median [min..max] over the batch:")
        for name, values in stats.items():
            low, mid, high = float(values.min()), float(values.median()), float(values.max())
            print(f"[generator]     {name:22s} {mid:9.3f} [{low:.3f}..{high:.3f}]")
        if accepted >= args.attempts:
            break
        empty_batches = empty_batches + 1 if count == 0 else 0
        if empty_batches >= 2:
            print("[generator] two batches in a row accepted nothing; stopping so the numbers above can be read.")
            break
    print(f"[generator] rejected by reason: {dict(rejected)}")
    if accepted == 0:
        raise SystemExit("[generator] no attempt produced a physics-validated grasp; see the reasons above.")
    if accepted < args.attempts:
        print(f"[generator] WARNING: only {accepted} of the requested {args.attempts} attempts were accepted.")

    keep = min(accepted, args.attempts)
    joint_pos, bag_pose, phase_ids, attempt_ids = [], [], [], []
    for index, phase in enumerate(PHASES):
        q = torch.cat(kept[phase][0])[:keep]
        pose = torch.cat(kept[phase][1])[:keep]
        joint_pos.append(q)
        bag_pose.append(pose)
        phase_ids.append(torch.full((keep,), index, dtype=torch.long))
        attempt_ids.append(torch.arange(keep))
    meta = {
        "joint_names": list(env.scene[ROBOT].joint_names),
        "attempts": keep,
        "tried": tried,
        "rejected": dict(rejected),
        "x_half": args.x_half,
        "y_half": args.y_half,
        "yaw_half_deg": args.yaw_half_deg,
        "pregrasp_backoff": args.pregrasp_backoff,
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
    return ResetDataset(torch.cat(phase_ids), torch.cat(attempt_ids), torch.cat(joint_pos), torch.cat(bag_pose), meta)


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Generate the handle-grab reset dataset.")
    parser.add_argument("--attempts", type=int, default=256, help="Accepted attempts to collect (four rows each).")
    parser.add_argument("--num_envs", type=int, default=128, help="Attempts scripted in parallel per batch.")
    parser.add_argument("--max_batches", type=int, default=40, help="Give up after this many batches.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="Dataset file to write.")
    parser.add_argument("--x_half", type=float, default=0.03, help="Half-width of the bag spawn box in x [m].")
    parser.add_argument("--y_half", type=float, default=0.04, help="Half-width of the bag spawn box in y [m].")
    parser.add_argument("--yaw_half_deg", type=float, default=20.0, help="Half-range of the bag heading [deg].")
    parser.add_argument("--pregrasp_backoff", type=float, default=0.06, help="Pregrasp distance from the pinch [m].")
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
        finally:
            env.close()
    dataset.save(args.output)
    print(f"[generator] wrote {dataset.phase.shape[0]} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
