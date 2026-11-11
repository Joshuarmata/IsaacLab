# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check, in the simulator, that the real-robot bridge reproduces the student's training inputs, and that the ONNX
export acts like the checkpoint.

Runs the camera task with observation noise off, drives it with the exported TorchScript policy, and at every step
rebuilds the student's inputs from the raw simulated camera frames and joint state with :mod:`.bridge` (as
``deploy.py`` does from the real ones). Reports the largest difference from the environment's own observations, and
between the ONNX and TorchScript actions. Also checks the joint limits and home pose copied into :mod:`.joint_map`::

    uv run python -m isaaclab_tasks.core.official_handle_grab.real.check_bridge --run <distillation run dir> --viz none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import add_launcher_args

TASK = "Isaac-HandleGrab-Camera"
STEPS = 150


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="Distillation run with exported/policy.{pt,onnx}.")
    add_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    sys.argv = sys.argv[:1]

    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    from .bridge import ObservationBuilder, OnnxStudent, joint_targets
    from .joint_map import SIM_HOME_RAD, SIM_JOINT_NAMES, SIM_LIMITS_RAD

    env_cfg, _ = resolve_task_config(TASK, None, play_mode=True)
    env_cfg.scene.num_envs = 1
    for group in ("overhead_rgb", "overhead_depth", "proprioception"):
        getattr(env_cfg.observations, group).enable_corruption = False
    onnx_policy = OnnxStudent(args.run / "exported" / "policy.onnx", providers=["CPUExecutionProvider"])
    jit_policy = torch.jit.load(str(args.run / "exported" / "policy.pt"), map_location="cpu").eval()

    failures = []
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        robot, camera = env.scene["robot"], env.scene["camera"]
        if tuple(robot.joint_names) != SIM_JOINT_NAMES:
            failures.append(f"joint order {robot.joint_names} != {SIM_JOINT_NAMES}")
        limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
        if np.abs(limits - np.array(SIM_LIMITS_RAD)).max() > 1e-3:
            failures.append(f"joint limits differ from joint_map.SIM_LIMITS_RAD: {limits.round(4).tolist()}")
        home = robot.data.default_joint_pos.torch[0].cpu().numpy()
        if np.abs(home - np.array(SIM_HOME_RAD)).max() > 1e-4:
            failures.append(f"home pose differs from joint_map.SIM_HOME_RAD: {home.round(3).tolist()}")

        builder = ObservationBuilder()
        lower, upper = limits[:, 0], limits[:, 1]
        worst = {"obs": 0.0, "overhead_rgb": 0.0, "overhead_depth": 0.0, "onnx_vs_pt": 0.0}
        with torch.inference_mode():
            obs, _ = env.reset()
            last_action = np.zeros(7, dtype=np.float32)
            for step in range(STEPS):
                pos = robot.data.joint_pos.torch[0].cpu().numpy()
                rgb = camera.data.output["rgb"].torch[..., :3].cpu()
                depth = camera.data.output["distance_to_image_plane"].torch.cpu()
                target = robot.data.joint_pos_target.torch[0].cpu().numpy()
                vel = robot.data.joint_vel.torch[0].cpu().numpy()
                rebuilt = builder(pos, vel, target, last_action, rgb, depth)
                env_inputs = {
                    "obs": obs["proprioception"].cpu().numpy(),
                    "overhead_rgb": obs["overhead_rgb"].cpu().numpy(),
                    "overhead_depth": obs["overhead_depth"].cpu().numpy(),
                }
                for key in ("obs", "overhead_rgb", "overhead_depth"):
                    worst[key] = max(worst[key], float(np.abs(rebuilt[key] - env_inputs[key]).max()))
                onnx_action = onnx_policy(env_inputs)
                pt_action = jit_policy(
                    *(torch.from_numpy(env_inputs[k]) for k in ("obs", "overhead_rgb", "overhead_depth"))
                )[0].numpy()
                worst["onnx_vs_pt"] = max(worst["onnx_vs_pt"], float(np.abs(onnx_action - pt_action).max()))
                clipped, _ = joint_targets(pos, onnx_action, lower, upper)
                obs, *_ = env.step(torch.from_numpy(clipped)[None].to(env.device))
                last_action = clipped.astype(np.float32)
        env.close()

    tolerances = {"obs": 1e-4, "overhead_rgb": 1e-3, "overhead_depth": 1e-4, "onnx_vs_pt": 1e-4}
    for key, value in worst.items():
        status = "ok" if value <= tolerances[key] else "MISMATCH"
        print(f"  {key:15s} largest difference {value:.2e}  (tolerance {tolerances[key]:.0e})  {status}")
        if value > tolerances[key]:
            failures.append(f"{key} differs by {value:.2e}")
    for failure in failures:
        print("FAIL:", failure)
    print("bridge parity: " + ("FAILED" if failures else "passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
