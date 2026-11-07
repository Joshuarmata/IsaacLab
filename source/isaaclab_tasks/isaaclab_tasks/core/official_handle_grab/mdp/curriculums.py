# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for :class:`~isaaclab.envs.mdp.modify_term_cfg`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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
