# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-local task milestones with safe partial-reset semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GrabProgress:
    """Per-episode milestones. The flags latch the first time they are observed; the counters stay live.

    * ``grasped``: the strap has been in the closed jaws for ``grasp_steps`` consecutive steps,
    * ``lifted``: the bag has risen ``lift_height`` above where it settled while the jaws held the strap,
    * ``success``: the strap has stayed in the jaws (and, with ``require_lift``, the bag up) for ``hold_steps``
      consecutive steps,
    * ``dropped``: after the grasp, the strap has been out of the jaws for ``drop_steps`` consecutive steps.

    The bag's resting height is its lowest height this episode, so a bag the jaws drag across the tray does not count
    as lifted.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        lift_height: float = 0.01,
        hold_steps: int = 150,
        grasp_steps: int = 3,
        drop_steps: int = 15,
        require_lift: bool = True,
    ):
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.grasp_steps = grasp_steps
        self.drop_steps = drop_steps
        self.require_lift = require_lift
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.dropped = torch.zeros_like(self.grasped)
        self.holding = torch.zeros_like(self.grasped)
        """The strap-in-jaws test of the latest update; shared so every term sees one contact reading per step."""
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold_count = torch.zeros_like(self.grasp_count)
        self.lost_count = torch.zeros_like(self.grasp_count)
        self.time_to_grasp = torch.full_like(self.grasp_count, -1)
        self.time_to_success = torch.full_like(self.grasp_count, -1)
        self.rest_height = torch.full((num_envs,), float("inf"), device=device)
        self.lift = torch.zeros(num_envs, device=device)

    def configure(self, **settings) -> None:
        """Replace thresholds. Called by the success termination once the task configuration is known."""
        for name, value in settings.items():
            if not hasattr(self, name):
                raise AttributeError(f"GrabProgress has no setting '{name}'.")
            setattr(self, name, value)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset only the requested environments."""
        ids = slice(None) if env_ids is None else env_ids
        for flag in (self.grasped, self.lifted, self.success, self.dropped, self.holding):
            flag[ids] = False
        for count in (self.grasp_count, self.hold_count, self.lost_count):
            count[ids] = 0
        self.time_to_grasp[ids] = -1
        self.time_to_success[ids] = -1
        self.rest_height[ids] = float("inf")
        self.lift[ids] = 0.0

    def update(self, holding: torch.Tensor, bag_height: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Latch milestones from the current physical state and return confirmed success.

        Args:
            holding: Whether the closed jaws have the strap this step, shape ``(num_envs,)``.
            bag_height: Bag root height [m], shape ``(num_envs,)``.
            step: Policy steps since the episode began, shape ``(num_envs,)``.
        """
        self.holding = holding
        self.rest_height = torch.minimum(self.rest_height, bag_height)
        self.lift = bag_height - self.rest_height
        cleared = self.lift >= self.lift_height

        self.grasp_count = torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count))
        newly_grasped = ~self.grasped & (self.grasp_count >= self.grasp_steps)
        self.time_to_grasp = torch.where(newly_grasped, step, self.time_to_grasp)
        self.grasped |= newly_grasped
        # Lift counts only as physical evidence of a grasp: a bag flicked up by a jaw that is not holding it does not.
        self.lifted |= cleared & self.grasped & holding

        live = holding & cleared if self.require_lift else holding
        self.hold_count = torch.where(live, self.hold_count + 1, torch.zeros_like(self.hold_count))
        newly_successful = ~self.success & (self.hold_count >= self.hold_steps)
        self.time_to_success = torch.where(newly_successful, step, self.time_to_success)
        self.success |= newly_successful

        self.lost_count = torch.where(self.grasped & ~holding, self.lost_count + 1, torch.zeros_like(self.lost_count))
        self.dropped = self.grasped & (self.lost_count >= self.drop_steps)
        return self.success.clone()


def grab_progress(env: ManagerBasedRLEnv) -> GrabProgress:
    """Return the episode progress shared by the success termination, rewards and observations."""
    if not hasattr(env, "_hg_progress"):
        env._hg_progress = GrabProgress(env.num_envs, env.device)
    return env._hg_progress
