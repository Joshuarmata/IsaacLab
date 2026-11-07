# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms: confirmed grab success and a dropped bag.

Out-of-workspace bags and unstable robots use the lift task's ``out_of_bound`` and ``abnormal_robot_state``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .progress import grab_progress
from .task import bag_height, holding, lower_body_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

LIFT_HEIGHT = 0.01
"""Bag rise [m] above its resting height that counts as lifted."""
HOLD_SECONDS = 5.0
"""Time [s] the strap must stay in the jaws (and the bag up) for the episode to count as a success."""
GRASP_STEPS = 3
"""Consecutive steps with the strap in the jaws before the grasp latches."""
DROP_STEPS = 15
"""Consecutive steps without the strap in the jaws, after the grasp, before the bag counts as dropped (0.5 s)."""


class grab_success(ManagerTermBase):
    """Update the task milestones every step and terminate once the strap has been held long enough.

    Milestones a dataset reset starts with (a closed grasp) are restored on reset. The term also records the terminal
    statistics read by :mod:`isaaclab_tasks.core.official_handle_grab.evaluation`.

    With ``require_lift`` the bag must also stay ``lift_height`` above where it settled; without it, a held strap is
    enough (the ``pinch_only`` preset).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.progress = grab_progress(env)
        self.progress.configure(
            lift_height=params.get("lift_height", LIFT_HEIGHT),
            hold_steps=max(1, round(params.get("hold_s", HOLD_SECONDS) / env.step_dt)),
            grasp_steps=params.get("grasp_steps", GRASP_STEPS),
            drop_steps=params.get("drop_steps", DROP_STEPS),
            require_lift=params.get("require_lift", True),
        )
        self.max_lower_force = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.progress.reset(ids)
        self.max_lower_force[ids] = 0.0
        if hasattr(self._env, "_hg_reset_grasped"):
            grasped = self._env._hg_reset_grasped[ids]
            self.progress.grasped[ids] = grasped
            self.progress.grasp_count[ids] = torch.where(grasped, self.progress.grasp_steps, 0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height: float = LIFT_HEIGHT,
        hold_s: float = HOLD_SECONDS,
        require_lift: bool = True,
        grasp_steps: int = GRASP_STEPS,
        drop_steps: int = DROP_STEPS,
    ) -> torch.Tensor:
        success = self.progress.update(holding(env), bag_height(env), env.episode_length_buf)
        self.max_lower_force = torch.maximum(self.max_lower_force, lower_body_force(env))

        progress = self.progress
        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = progress.grasped.float().mean()
        log["Metrics/lift_rate"] = progress.lifted.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Safety/peak_lower_body_force_n"] = self.max_lower_force.mean()

        # Terminal statistics survive the reset that follows, so evaluation can read them.
        env._hg_terminal_progress = torch.stack((progress.grasped, progress.lifted, progress.success), dim=-1)
        env._hg_terminal_max_lower_force = self.max_lower_force.clone()
        env._hg_terminal_time_to_success_s = progress.time_to_success.float() * env.step_dt
        return success


def bag_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The strap was grasped and has been out of the jaws for the debounce time."""
    return grab_progress(env).dropped
