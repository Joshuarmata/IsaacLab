# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms: dense approach and hold shaping, grasp placement, one-time milestones, and success."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import grab_progress
from .task import gripper_closed_fraction, lower_body_force, strap_grasp, strap_target_w, tip_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class approach_progress(ManagerTermBase):
    """Pay the decrease in jaw-tip-to-strap distance, in units of ``scale``, until the strap is grasped.

    A difference of potentials has no incentive to hover, and gives the same gradient 25 cm from the strap as 2 cm
    away, which is what lets a pixels-only policy discover the approach from the home pose. The strap point is the
    recorded pinch moved by the bag's current pose, so it follows every spawn.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_distance[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        distance = torch.linalg.vector_norm(tip_pos_w(env) - strap_target_w(env), dim=-1)
        progress = ((self.previous_distance - distance) / scale).clamp(-1.0, 1.0)
        self.previous_distance = distance
        # The first step after a reset has no previous distance (NaN) and pays nothing.
        return torch.where(grab_progress(env).grasped, 0.0, progress.nan_to_num(0.0))


class jaw_closing(ManagerTermBase):
    """Pay closing the jaw at the strap: each new episode best of ``closed fraction * nearness``, where nearness is
    ``1 - tanh(tip-to-strap distance / std)``.

    The holding rewards only start below :data:`~.task.GRIPPER_HOLD_POS`, 0.9 rad of jaw travel from open, which the
    policy's exploration noise does not cover. This is the gradient toward that state. Only improvements on the
    episode's best pay, so closing at the strap, backing off, opening and coming back earns nothing twice; one episode
    pays at most 1 before the weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.best = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.best[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, std: float = 0.02) -> torch.Tensor:
        distance = torch.linalg.vector_norm(tip_pos_w(env) - strap_target_w(env), dim=-1)
        value = gripper_closed_fraction(env) * (1.0 - torch.tanh(distance / std))
        # The first step after a reset sets the baseline (NaN) and pays nothing.
        gain = (value - self.best).clamp(min=0.0).nan_to_num(0.0)
        self.best = torch.fmax(self.best, value)
        return gain


def held_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pay every step the grasped strap is still in the closed jaws.

    It keeps paying through the whole hold, since the five-second success otherwise has no dense signal between the
    grasp milestone and the final bonus.
    """
    progress = grab_progress(env)
    return (progress.grasped & progress.holding).float()


def grasp_centered(env: ManagerBasedRLEnv) -> torch.Tensor:
    """While holding, how close to the middle of the strap the grasp is: 1 at the top of the arch, falling linearly
    to 0 at the feet.

    The position is where the strap crosses the closed mouth, as a fraction of the strap's length
    (:attr:`~.task.StrapGrasp.along`). A grasp on the top of the handle balances the bag; one on the side tips it.
    """
    centered = (1.0 - (2.0 * strap_grasp(env).along - 1.0).abs()).nan_to_num(0.0)
    return grab_progress(env).holding.float() * centered


def strap_depth(env: ManagerBasedRLEnv, std: float = 0.02) -> torch.Tensor:
    """While holding, how close the strap is to the back of the mouth: ``1 - tanh(distance / std)``.

    The distance is from the back point of the closed mouth (:attr:`~.task.JawGap.back_point`, 25 mm in front of the
    hinge, against the palm) to the nearest point on the strap. A strap deep in the jaws is held more securely than
    one near the tips. With ``std`` = 2 cm, a strap at the back pays about 0.75 and one 4 cm out pays about 0.04.
    """
    return grab_progress(env).holding.float() * (1.0 - torch.tanh(strap_grasp(env).back_dist / std))


class milestones(ManagerTermBase):
    """Pay the grasp once per episode.

    A grasp already latched on the first step after a reset is not paid.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        grasped = grab_progress(env).grasped
        reward = torch.where(self.started, (grasped & ~self.paid).float(), 0.0)
        self.paid = grasped.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step confirmed grab success."""
    return grab_progress(env).success.float()


def lower_body_contact(env: ManagerBasedRLEnv, free_force: float = 1.0, force_span: float = 5.0) -> torch.Tensor:
    """Penalize force on the lower part of the bag body beyond ``free_force`` [N], saturating after ``force_span``.

    Contact on the strap and on the top rim is free, since the jaws have to sit there. The scripted demo this task
    replaces brushed the body at about 1 N, which is why that force is free.
    """
    return ((lower_body_force(env) - free_force) / force_span).clamp(0.0, 1.0)
