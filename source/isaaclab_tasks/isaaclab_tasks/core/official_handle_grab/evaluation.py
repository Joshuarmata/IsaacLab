# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-out evaluation of a handle-grab policy from the canonical home starts.

Training mixes four reset phases, so its running success rate says little about the task the robot actually faces: start
at home and grab the strap. This module runs complete episodes from home instead, with the bag walking through the
dataset's home placements in order, and reports grasp, lift and success rates, why failed episodes ended, and the safety
numbers. Build the environment from the play configuration, which makes the resets sequential and pins the physical
parameters (``resolve_task_config(..., play_mode=True)``), and pass any callable mapping observations to actions::

    summary = evaluate(env, policy, episodes_per_env=2)
    print(summary.as_dict())
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class EvaluationSummary:
    """Outcome of :func:`evaluate`; rates are fractions of ``episodes``."""

    episodes: int
    success_rate: float
    """The strap was held (and, unless ``pinch_only``, the bag lifted) for the full hold time."""
    grasp_rate: float
    """The strap was in the closed jaws long enough to latch the grasp milestone."""
    lift_rate: float
    """The bag was lifted while the jaws held the strap."""
    drop_rate: float
    """The episode ended because the strap left the jaws after a grasp."""
    out_of_bounds_rate: float
    """The episode ended because the bag left the tray."""
    mean_time_to_success_s: float | None
    """Mean time [s] from reset to success over the successful episodes, ``None`` without any."""
    mean_peak_lower_body_force_n: float
    max_peak_lower_body_force_n: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.inference_mode()
def evaluate(
    env, policy: Callable[[Any], torch.Tensor], episodes_per_env: int = 1, max_steps: int | None = None
) -> EvaluationSummary:
    """Run ``episodes_per_env`` complete episodes in every environment and summarize them.

    Each environment contributes exactly its first ``episodes_per_env`` episodes, so episodes that end early (a success
    ends the episode) are not over-counted against ones that run to the time limit.

    Args:
        env: A gym or RSL-RL wrapped manager-based handle-grab environment, built from the play configuration.
        policy: Maps the observations returned by ``env`` to actions of shape ``(num_envs, action_dim)``.
        episodes_per_env: Episodes to count in each environment.
        max_steps: Step budget; defaults to enough for ``episodes_per_env`` full-length episodes.

    Raises:
        RuntimeError: If the environment does not report the handle-grab terminal statistics, or the budget runs out.
    """
    base = getattr(env, "unwrapped", env)
    if not hasattr(base, "termination_manager") or "success" not in base.termination_manager.active_terms:
        raise RuntimeError("evaluate() needs an Isaac-HandleGrab environment with its success termination.")
    num_envs = base.num_envs
    budget = max_steps if max_steps is not None else episodes_per_env * int(base.max_episode_length) + 1

    obs = env.reset()[0]
    finished = torch.zeros(num_envs, dtype=torch.long, device=base.device)
    records: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("grasped", "lifted", "success", "dropped", "bounds", "force", "time")
    }
    for _ in range(budget):
        out = env.step(policy(obs))
        obs = out[0]
        dones = out[2].bool() if len(out) == 4 else (out[2] | out[3]).bool()
        take = dones & (finished < episodes_per_env)
        if take.any():
            progress = base._hg_terminal_progress[take]
            terms = base.termination_manager
            records["grasped"].append(progress[:, 0])
            records["lifted"].append(progress[:, 1])
            records["success"].append(progress[:, 2])
            records["dropped"].append(terms.get_term("bag_dropped")[take])
            records["bounds"].append(terms.get_term("bag_out_of_bounds")[take])
            records["force"].append(base._hg_terminal_max_lower_force[take])
            records["time"].append(base._hg_terminal_time_to_success_s[take])
        finished += take.long()
        if bool((finished >= episodes_per_env).all()):
            break
    else:
        raise RuntimeError(f"Evaluation did not finish {episodes_per_env} episodes per environment in {budget} steps.")

    cat = {key: torch.cat(values) for key, values in records.items()}
    success = cat["success"]
    return EvaluationSummary(
        episodes=int(success.numel()),
        success_rate=float(success.float().mean()),
        grasp_rate=float(cat["grasped"].float().mean()),
        lift_rate=float(cat["lifted"].float().mean()),
        drop_rate=float(cat["dropped"].float().mean()),
        out_of_bounds_rate=float(cat["bounds"].float().mean()),
        mean_time_to_success_s=float(cat["time"][success].mean()) if bool(success.any()) else None,
        mean_peak_lower_body_force_n=float(cat["force"].mean()),
        max_peak_lower_body_force_n=float(cat["force"].max()),
    )
