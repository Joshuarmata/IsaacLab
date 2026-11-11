# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cycle a generated reset dataset in the viewer, without a policy.

This is the handle-grab counterpart of the vial tutorial's ``view-so101-resets``. Every row of the dataset is written
into the scene (arm joints, bag pose and strap shape), held for a few steps, and the next row follows, one attempt at
a time: home, pregrasp, grasp. Use it after the generator to see what the policy will start from::

    uv run python -m isaaclab_tasks.core.official_handle_grab.reset.viewer --viz kit --dataset /tmp/check.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..mdp.task import BAG, ROBOT, write_strap
from .curriculum import PHASES
from .dataset import DEFAULT_DATASET_PATH, ResetDataset
from .generator import TASK, configure_env_cfg

STEPS_PER_POSE = 45
"""Policy steps each row is held for; the vial viewer's default."""


def row_order(dataset: ResetDataset) -> torch.Tensor:
    """Row indices ordered by attempt and then phase, so one grab plays through before the next starts."""
    return torch.argsort(dataset.attempt * len(PHASES) + dataset.phase, stable=True)


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Cycle the handle-grab reset dataset in the viewer.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--steps_per_pose", type=int, default=STEPS_PER_POSE)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max_attempts", type=int, default=0, help="Show only the first N attempts (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    return setup_preset_cli(parser, argv)


def main(argv: list[str] | None = None) -> int:
    args, hydra_args = _parse_args(argv)
    args.num_envs = 1

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
                _cycle(env, ResetDataset.load(args.dataset, device=env.device), args)
        finally:
            env.close()
    return 0


def _cycle(env, dataset: ResetDataset, args: argparse.Namespace) -> None:
    from ..env_cfg import ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE
    from .generator import Driver

    driver = Driver(env, ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE)
    robot, bag = env.scene[ROBOT], env.scene[BAG]
    order = row_order(dataset)
    if args.max_attempts > 0:
        order = order[dataset.attempt[order] < args.max_attempts]
    ids = driver.ids
    env.reset()
    for _ in range(args.cycles):
        for row in order.tolist():
            q = dataset.joint_pos[row : row + 1]
            bag_pose = dataset.bag_pose[row : row + 1].clone()
            bag_pose[:, :3] += env.scene.env_origins
            robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
            robot.actuators.target_command.set_position_index(value=q, joint_ids=None, env_ids=ids)
            bag.write_root_pose_to_sim_index(root_pose=bag_pose, env_ids=ids)
            bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
            write_strap(env, ids, bag_pose, dataset.strap_nodes[row : row + 1])
            driver.trim.zero_()
            print(f"[viewer] row={row} attempt={int(dataset.attempt[row])} phase={PHASES[int(dataset.phase[row])]}")
            driver.hold(q, args.steps_per_pose)


if __name__ == "__main__":
    sys.exit(main())
