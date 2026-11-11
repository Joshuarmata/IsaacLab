# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms: the lower-body penalty ramp, the reset-phase fade, and success logged per reset phase."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from ..reset.curriculum import PHASES
from .progress import grab_progress

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def linear_weight_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    current_weight: float,
    final_weight: float,
    ramp_steps: int,
    start_steps: int = 0,
) -> float:
    """Grow a reward weight linearly from 0 to ``final_weight`` over ``ramp_steps`` environment steps.

    The ramp starts at ``start_steps``.

    ``env.common_step_counter`` counts environment steps, not PPO iterations: one iteration advances it by
    ``num_steps_per_env``.
    """
    fraction = min(max((env.common_step_counter - start_steps) / max(ramp_steps, 1), 0.0), 1.0)
    return fraction * final_weight


def fade_reset_phase(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    phase: str,
    start_weight: float,
    final_weight: float,
    fade_steps: int,
    term_name: str = "reset_from_dataset",
) -> float | None:
    """Move one phase's sampling weight in the dataset reset linearly from ``start_weight`` to ``final_weight`` over
    ``fade_steps`` environment steps, and return it (logged as ``Curriculum/<term>``).

    A mix without ``phase``, such as the home-only play and sim2real mixes, is left alone. Curriculum terms run before
    the reset events of the same reset, so the new weight applies to the episodes starting now.
    """
    weights = env.event_manager.get_term_cfg(term_name).params["phase_weights"]
    if phase not in weights:
        return None
    fraction = min(env.common_step_counter / max(fade_steps, 1), 1.0)
    weight = start_weight + fraction * (final_weight - start_weight)
    # A new dict: the default mix is a shared module constant.
    env.event_manager.get_term_cfg(term_name).params["phase_weights"] = {**weights, phase: weight}
    return weight


class success_by_start(ManagerTermBase):
    """Log the success rate of the episodes that just ended, separately for each reset phase they started from.

    Logged as ``Curriculum/<term>/<phase>``: a running average over about ``window`` finished episodes of each phase.
    ``Episode_Termination/success`` mixes every start; ``home`` here is the number play and evaluation measure.
    Changes nothing; it is a curriculum term only because those run at a reset before the reset event overwrites the
    phase each environment started from.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rate = torch.zeros(len(PHASES), device=env.device)
        self.seen = torch.zeros(len(PHASES), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], window: int = 200) -> dict[str, float]:
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        # The first reset, before any step, ends no episode.
        ids = ids[env.episode_length_buf[ids] > 0]
        phase = getattr(env, "_hg_reset_phase", None)
        if phase is not None and ids.numel():
            success = grab_progress(env).success[ids].float()
            for k in range(len(PHASES)):
                mine = phase[ids] == k
                count = mine.sum()
                if count:
                    self.seen[k] += count
                    # A plain mean until ``window`` episodes have been seen, then an exponential average.
                    step = count / torch.clamp(self.seen[k], max=float(window))
                    self.rate[k] += torch.clamp(step, max=1.0) * (success[mine].mean() - self.rate[k])
        return {name: float(self.rate[k]) for k, name in enumerate(PHASES)}
