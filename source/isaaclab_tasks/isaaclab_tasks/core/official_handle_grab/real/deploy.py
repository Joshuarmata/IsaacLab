# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the camera student on the real arm: D415 frames and joint readings in, joint targets out, at 30 Hz.

Without ``--execute`` it is a dry run: it reads the robot and camera and runs the policy every step, prints what it
would command, and never writes to the motors. With ``--execute`` it needs a verified joint map and the arm at home
(``home.py``). It stops after ``--seconds`` (an episode is 20 s), on Ctrl-C, or when a safety check trips, and then
holds the arm where it is::

    uv run python -m isaaclab_tasks.core.official_handle_grab.real.deploy --policy <run>/exported/policy.onnx
    uv run python -m isaaclab_tasks.core.official_handle_grab.real.deploy --policy <run>/exported/policy.onnx --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .bridge import (
    DECIMATION,
    POLICY_HZ,
    ObservationBuilder,
    OnnxStudent,
    VelocityEstimator,
    joint_targets,
    to_student_frames,
)
from .joint_map import DEFAULT_PATH, SIM_HOME_RAD, JointMap

START_TOLERANCE = 0.15
"""Largest distance from home [rad] any joint may start at; every training episode in play mode starts at home."""
TRACKING_LIMIT = 0.35
"""Arm joint lag behind its target [rad] that counts as blocked (the gripper is meant to stall on the strap)."""
TRACKING_STEPS = 5
"""Consecutive blocked steps before stopping."""
OVERRUN_STEPS = 10
"""Consecutive steps over twice the policy period before stopping: the policy would be acting on stale frames."""
MAX_TEMPERATURE_C = 65


class Bus(Protocol):
    def read_positions(self) -> list[int]: ...
    def write_goals(self, counts: list[int]) -> None: ...


class Camera(Protocol):
    intrinsics: object

    def read(self) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class EpisodeLog:
    """Every step of a run, for replay and comparison with the simulator."""

    time: list[float] = field(default_factory=list)
    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    target: list[np.ndarray] = field(default_factory=list)
    stop_reason: str = ""

    def save(self, path: Path) -> None:
        np.savez(
            path,
            time=np.array(self.time),
            joint_pos=np.array(self.joint_pos),
            joint_vel=np.array(self.joint_vel),
            action=np.array(self.action),
            target=np.array(self.target),
            stop_reason=self.stop_reason,
        )


def run_episode(
    bus: Bus,
    camera: Camera,
    joint_map: JointMap,
    policy,
    execute: bool,
    seconds: float,
    clock=time.monotonic,
    sleep=time.sleep,
    verbose: bool = True,
) -> EpisodeLog:
    """Run the policy for up to ``seconds``; write goals only when ``execute``. Returns the step log.

    Each step waits for a camera frame, reads the joints, runs the policy, and sends ``measured + scaled action``; then,
    half a period later, reads the joints and sends it again from there (:data:`~.bridge.DECIMATION`).
    """
    if execute and not (joint_map.calibrated and joint_map.verified):
        raise SystemExit("--execute needs a calibrated, verified joint map (calibrate.py, then twin.py).")
    lower, upper = (np.array(v) for v in joint_map.limits())
    builder, velocity = ObservationBuilder(), VelocityEstimator()
    log = EpisodeLog()

    pos = np.array(joint_map.to_rad(bus.read_positions()))
    off_home = np.abs(pos - np.array(SIM_HOME_RAD))
    if execute and off_home.max() > START_TOLERANCE:
        raise SystemExit(f"The arm is {off_home.max():.2f} rad from home (limit {START_TOLERANCE}); run home.py first.")
    # As after a simulator reset: targets at the measured pose, no previous action.
    last_target, last_action = pos.copy(), np.zeros_like(pos)
    blocked = overruns = 0
    start = clock()
    try:
        while (now := clock()) - start < seconds:
            rgb, depth = camera.read()
            pos = np.array(joint_map.to_rad(bus.read_positions()))
            vel = velocity(pos, now)
            frames = to_student_frames(rgb, depth, camera.intrinsics)
            inputs = builder(pos, vel, last_target, last_action, *frames)
            action, target = joint_targets(pos, policy(inputs), lower, upper)
            lag = np.abs(pos - last_target)[:6]
            if execute:
                bus.write_goals(joint_map.to_counts(list(target)))
                for k in range(1, DECIMATION):
                    sleep(max(0.0, now + k / (POLICY_HZ * DECIMATION) - clock()))
                    substep_pos = np.array(joint_map.to_rad(bus.read_positions()))
                    _, target = joint_targets(substep_pos, action, lower, upper)
                    bus.write_goals(joint_map.to_counts(list(target)))
                blocked = blocked + 1 if lag.max() > TRACKING_LIMIT else 0
                if blocked >= TRACKING_STEPS:
                    log.stop_reason = f"arm blocked: joint lag {np.round(lag, 2).tolist()} rad"
                    break
            last_target, last_action = target, action
            log.time.append(now - start)
            log.joint_pos.append(pos)
            log.joint_vel.append(vel)
            log.action.append(action)
            log.target.append(target)
            if verbose and len(log.time) % int(POLICY_HZ) == 0:
                print(
                    f"t={now - start:5.1f}s gripper {pos[6]:+.2f} rad, action {np.round(action, 2).tolist()}",
                    flush=True,
                )
            overruns = overruns + 1 if clock() - now > 2.0 / POLICY_HZ else 0
            if overruns >= OVERRUN_STEPS:
                log.stop_reason = "loop too slow: more than twice the policy period for 10 steps"
                break
        else:
            log.stop_reason = "time limit"
    except KeyboardInterrupt:
        log.stop_reason = "stopped by user"
    finally:
        if execute:
            # Hold where the arm is now, rather than at the last (possibly blocked) target.
            bus.write_goals(bus.read_positions())
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, required=True, help="exported/policy.onnx of a distillation run.")
    parser.add_argument("--map", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--camera_serial", default=None, help="D415 serial number, if more than one is connected.")
    parser.add_argument("--execute", action="store_true", help="Move the robot. Without it, only print.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Run length; training episodes are 20 s.")
    parser.add_argument("--arm_torque", type=float, default=0.6, help="Arm torque limit, fraction of maximum.")
    parser.add_argument(
        "--gripper_torque",
        type=float,
        default=0.3,
        help="Gripper torque limit, fraction of maximum. The simulated jaw squeezes with 0.8 N m, about 0.3 of an "
        "STS3215's stall torque.",
    )
    parser.add_argument("--log", type=Path, default=None, help="Save the step log (.npz).")
    args = parser.parse_args(argv)

    from .feetech import FeetechBus
    from .realsense import D415

    joint_map = JointMap.load(args.map)
    policy = OnnxStudent(args.policy)
    camera = D415(serial=args.camera_serial)
    bus = FeetechBus(joint_map.port, joint_map.motor_ids)
    try:
        if args.execute:
            bus.set_torque_limits([args.arm_torque] * 6 + [args.gripper_torque])
            bus.write_goals(bus.read_positions())
            bus.set_torque(True)
            temps = bus.read_temperatures()
            if max(temps) > MAX_TEMPERATURE_C:
                raise SystemExit(f"Motors too hot to start: {temps} deg C.")
        else:
            print("Dry run: nothing is written to the motors. Add --execute to move the robot.")
        log = run_episode(bus, camera, joint_map, policy, args.execute, args.seconds)
    finally:
        camera.close()
        bus.close()
    print(f"Stopped after {log.time[-1] if log.time else 0.0:.1f} s: {log.stop_reason}")
    if args.log:
        log.save(args.log)
        print(f"Saved {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
