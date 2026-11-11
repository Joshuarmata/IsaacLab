# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record each motor's encoder count with the real arm posed as the simulation's home, into ``real/joint_map.json``.

The motors go limp; pose the arm by hand to match the home pose (``reset.viewer`` shows it as the first row of the home
phase), hold it, and press Enter. Signs stay as they are in the file (+1 for a new one); ``twin.py`` shows which ones to
flip. The map is saved unverified, so ``home.py`` and ``deploy.py --execute`` refuse it until it is checked::

    uv run python -m isaaclab_tasks.core.official_handle_grab.real.calibrate --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feetech import FeetechBus
from .joint_map import DEFAULT_PATH, JointMap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=None, help="Servo bus serial port (default: the map's, or /dev/ttyACM0).")
    parser.add_argument("--ids", type=int, nargs=7, default=None, help="Motor ids of revolute_1 .. revolute_7.")
    parser.add_argument("--map", type=Path, default=DEFAULT_PATH, help="Joint map file to update or create.")
    args = parser.parse_args(argv)

    joint_map = JointMap.load(args.map) if args.map.exists() else JointMap.template()
    if args.ids:
        for joint, mid in zip(joint_map.joints, args.ids):
            joint.motor_id = mid
    if args.port:
        joint_map.port = args.port

    bus = FeetechBus(joint_map.port, joint_map.motor_ids)
    try:
        bus.set_torque(False)
        print("The arm is limp. Pose it as the simulation's home pose:")
        for joint in joint_map.joints:
            print(f"  {joint.name}: {joint.home_rad:+.2f} rad  (motor {joint.motor_id})")
        input("Hold it there and press Enter... ")
        counts = bus.read_positions()
    finally:
        bus.close()

    for joint, count in zip(joint_map.joints, counts):
        joint.home_count = count
    joint_map.verified = False
    joint_map.save(args.map)
    print(f"Saved {args.map} (unverified). Home counts: {counts}")

    seams = [j.name for j in joint_map.joints if j.crosses_seam()]
    if seams:
        print(
            f"WARNING: the range of {', '.join(seams)} runs through the encoder's 0/4095 seam, which a position-mode"
            " servo cannot cross. Re-centre those motors' homing offsets (LeRobot's calibration does) and calibrate"
            " again. The signs in the file affect this check; fix them in twin.py first."
        )
    print("Next: twin.py, to check the signs and the home pose against the simulator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
