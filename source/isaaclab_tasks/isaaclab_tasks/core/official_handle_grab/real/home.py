# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move the real arm slowly to the simulation's home pose, where every played episode starts::

uv run python -m isaaclab_tasks.core.official_handle_grab.real.home
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .bridge import POLICY_HZ
from .feetech import FeetechBus
from .joint_map import DEFAULT_PATH, SIM_HOME_RAD, JointMap

HOME_SPEED = 0.3
"""Speed of the slowest-finishing joint on the way home [rad/s]."""


def require_verified(joint_map: JointMap, path: Path) -> None:
    """Refuse to move a robot whose map has not been calibrated and checked in ``twin.py``."""
    if not joint_map.calibrated:
        raise SystemExit(f"{path} is not calibrated: run calibrate.py.")
    if not joint_map.verified:
        raise SystemExit(f'{path} is unverified: check it with twin.py, then set "verified": true in the file.')


def move_to(bus: FeetechBus, joint_map: JointMap, goal: np.ndarray, speed: float = HOME_SPEED) -> None:
    """Interpolate every joint linearly from where it is to ``goal`` [rad], arriving together."""
    start = np.array(joint_map.to_rad(bus.read_positions()))
    steps = max(int(np.abs(goal - start).max() / speed * POLICY_HZ), 1)
    for k in range(1, steps + 1):
        bus.write_goals(joint_map.to_counts(list(start + (goal - start) * k / steps)))
        time.sleep(1.0 / POLICY_HZ)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--torque", type=float, default=0.6, help="Torque limit while moving, fraction of maximum.")
    args = parser.parse_args(argv)

    joint_map = JointMap.load(args.map)
    require_verified(joint_map, args.map)
    bus = FeetechBus(joint_map.port, joint_map.motor_ids)
    try:
        bus.set_torque_limits([args.torque] * len(joint_map.joints))
        bus.write_goals(bus.read_positions())
        bus.set_torque(True)
        move_to(bus, joint_map, np.array(SIM_HOME_RAD))
        time.sleep(0.5)
        error = np.array(joint_map.to_rad(bus.read_positions())) - np.array(SIM_HOME_RAD)
        print("At home. Error per joint [rad]:", np.round(error, 3).tolist())
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
