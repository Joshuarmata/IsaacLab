# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms: dense approach and hold shaping, one-time milestones, and success."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import grab_progress
from .task import lower_body_force, strap_target_w, tip_pos_w

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


def held_goal(env: ManagerBasedRLEnv, require_lift: bool = True) -> torch.Tensor:
    """Pay for keeping the strap in the jaws, scaled by how far the bag has risen toward the lift height.

    The signal is gated on the bag's own height, never the hand's: a hand that rises while the strap slips out of the
    jaws lifts nothing and earns nothing. It keeps paying through the whole hold, since the five-second success
    otherwise has no dense signal between the lift and the final bonus. Without ``require_lift`` it pays a flat hold.
    """
    progress = grab_progress(env)
    reached = (progress.lift / progress.lift_height).clamp(0.0, 1.0) if require_lift else torch.ones_like(progress.lift)
    return torch.where(progress.grasped & progress.holding, reached, 0.0)


class milestones(ManagerTermBase):
    """Pay the grasp and the lift once per episode, weighted by ``weights`` (grasp, lift).

    Milestones present on the first step after a reset (restored from the reset dataset) are not paid.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.paid = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, weights: tuple[float, float] = (1.0, 2.0)) -> torch.Tensor:
        progress = grab_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted), dim=-1)
        reward = ((reached & ~self.paid).float() * reached.new_tensor(weights, dtype=torch.float32)).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
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
