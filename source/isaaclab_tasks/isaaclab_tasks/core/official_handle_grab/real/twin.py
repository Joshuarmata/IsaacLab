# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mirror the limp real arm in the simulator, and overlay the real D415 on the simulated camera.

Move the real arm by hand and watch the simulated arm follow. A joint that turns the wrong way needs its ``sign``
flipped in ``joint_map.json``; one that is offset needs ``calibrate.py`` again. With ``--camera`` a second window shows
the real and simulated overhead images blended and side by side, with the same crop the policy gets; move the mount
until the tray edges line up. When both look right, set ``"verified": true`` in ``joint_map.json``. Ctrl-C to stop::

    uv run python -m isaaclab_tasks.core.official_handle_grab.real.twin --camera --viz kit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import add_launcher_args

from .joint_map import DEFAULT_PATH, JointMap

TASK = "Isaac-HandleGrab-Camera"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--camera", action="store_true", help="Also overlay the real D415 on the simulated camera.")
    parser.add_argument("--camera_serial", default=None)
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

    from .bridge import to_student_frames
    from .feetech import FeetechBus

    joint_map = JointMap.load(args.map)
    if not joint_map.calibrated:
        raise SystemExit(f"{args.map} is not calibrated: run calibrate.py.")
    env_cfg, _ = resolve_task_config(TASK, None, play_mode=True)
    env_cfg.scene.num_envs = 1
    bus = FeetechBus(joint_map.port, joint_map.motor_ids)
    bus.set_torque(False)
    camera = None
    if args.camera:
        from .realsense import D415

        camera = D415(serial=args.camera_serial)

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        robot, sim_camera = env.scene["robot"], env.scene["camera"]
        env.reset()
        figure = None
        if camera is not None:
            import matplotlib.pyplot as plt

            plt.ion()
            figure, axes = plt.subplots(1, 3, figsize=(15, 3.5))
        last_print = 0.0
        try:
            while True:
                pos = torch.tensor([joint_map.to_rad(bus.read_positions())], device=env.device)
                with torch.inference_mode():
                    robot.write_joint_position_to_sim_index(position=pos)
                    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos))
                    robot.set_joint_position_target_index(target=pos)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=True)
                    env.scene.update(env.physics_dt)
                if time.monotonic() - last_print > 1.0:
                    last_print = time.monotonic()
                    print("real arm [rad]:", np.round(pos[0].cpu().numpy(), 3).tolist(), flush=True)
                    if figure is not None:
                        rgb, depth = camera.read()
                        real = to_student_frames(rgb, depth, camera.intrinsics)[0][0].numpy()
                        sim = sim_camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                        blend = (0.5 * real.astype(np.float32) + 0.5 * sim.astype(np.float32)).astype(np.uint8)
                        for ax, image, title in zip(axes, (real, sim, blend), ("real D415", "simulated", "blend")):
                            ax.clear()
                            ax.imshow(image)
                            ax.set_title(title)
                            ax.axis("off")
                        plt.pause(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            bus.close()
            if camera is not None:
                camera.close()
            env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
