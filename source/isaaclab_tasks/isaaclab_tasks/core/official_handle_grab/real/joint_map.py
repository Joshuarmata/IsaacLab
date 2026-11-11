# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-motor map between Feetech encoder counts and the simulation's joint angles.

Each joint is calibrated at the home pose: ``home_count`` is what its motor reads with the real arm posed as the
simulation's home (:data:`~..scene.handle_grab_scene.ARM_HOME_JOINT_POS`). An angle is then
``home_rad + sign * (count - home_count) * 2 pi / 4096``, unwrapped into the joint's range. ``sign`` is +1 when the
motor counts up as the simulated joint angle grows; ``twin.py`` shows whether it does.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

COUNTS_PER_TURN = 4096
"""STS3215 encoder counts per revolution."""
RAD_PER_COUNT = 2.0 * math.pi / COUNTS_PER_TURN

SIM_JOINT_NAMES = tuple(f"revolute_{i}" for i in range(1, 8))
"""Simulation joint order: the policy's 7 inputs per proprioception term and its 7 actions, in this order."""
SIM_HOME_RAD = (0.11, 0.94, -2.17, 0.71, -0.09, -0.37, 0.6)
"""``ARM_HOME_JOINT_POS`` in joint order; every training episode under ``play_mode`` starts here [rad]."""
SIM_LIMITS_RAD = (
    (-2.032, 1.8077),
    (-0.1461, 3.3445),
    (-2.975, 0.1666),
    (-2.2274, 0.9141),
    (-1.6389, 1.5027),
    (-4.2091, 0.6778),
    (-0.3787, 1.2793),
)
"""Joint limits of ``SO_101_sim.usd`` [rad]. ``check_bridge.py`` compares them with the simulator."""
DEFAULT_PATH = Path(__file__).with_name("joint_map.json")
"""Where ``calibrate.py`` writes the map and the other scripts read it."""


@dataclass
class JointCalibration:
    """One motor: its bus id, direction, and the count it reads at the home pose."""

    name: str
    motor_id: int
    home_rad: float
    lower_rad: float
    upper_rad: float
    sign: int = 1
    home_count: int | None = None

    def to_rad(self, count: int) -> float:
        """Joint angle [rad] for an encoder count, unwrapped to ``[lower_rad - margin, lower_rad - margin + 2 pi)``."""
        if self.home_count is None:
            raise ValueError(f"{self.name} is not calibrated; run calibrate.py.")
        raw = self.home_rad + self.sign * (count - self.home_count) * RAD_PER_COUNT
        low = self.lower_rad - _UNWRAP_MARGIN
        return low + (raw - low) % (2.0 * math.pi)

    def to_count(self, rad: float) -> int:
        """Encoder count [0, 4096) for a joint angle; the angle is clamped to the joint's limits first."""
        if self.home_count is None:
            raise ValueError(f"{self.name} is not calibrated; run calibrate.py.")
        rad = min(max(rad, self.lower_rad), self.upper_rad)
        count = self.home_count + self.sign * round((rad - self.home_rad) / RAD_PER_COUNT)
        return int(count) % COUNTS_PER_TURN

    def crosses_seam(self) -> bool:
        """Whether the joint's range runs through count 0/4095.

        A position-mode servo cannot cross that seam: told to go from 4090 to 10, it turns almost a full revolution
        the other way. The motor's homing offset has to put the seam outside the range (LeRobot's calibration does).
        """
        ends = [
            self.home_count + self.sign * (r - self.home_rad) / RAD_PER_COUNT for r in (self.lower_rad, self.upper_rad)
        ]
        return min(ends) < 0 or max(ends) >= COUNTS_PER_TURN


_UNWRAP_MARGIN = 0.25
"""How far below its lower limit [rad] a joint can read before it wraps to the top of its turn."""


@dataclass
class JointMap:
    """The seven motors in simulation joint order, plus whether a person has checked the map in ``twin.py``."""

    joints: list[JointCalibration] = field(default_factory=list)
    verified: bool = False
    port: str = "/dev/ttyACM0"

    @classmethod
    def template(cls, motor_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)) -> JointMap:
        """An uncalibrated map: simulation order, home pose and limits, motor ids 1-7, every sign +1."""
        joints = [
            JointCalibration(name=name, motor_id=mid, home_rad=home, lower_rad=lo, upper_rad=hi)
            for name, mid, home, (lo, hi) in zip(SIM_JOINT_NAMES, motor_ids, SIM_HOME_RAD, SIM_LIMITS_RAD)
        ]
        return cls(joints=joints)

    @property
    def motor_ids(self) -> list[int]:
        return [j.motor_id for j in self.joints]

    @property
    def calibrated(self) -> bool:
        return len(self.joints) == len(SIM_JOINT_NAMES) and all(j.home_count is not None for j in self.joints)

    def to_rad(self, counts: list[int]) -> list[float]:
        return [j.to_rad(c) for j, c in zip(self.joints, counts)]

    def to_counts(self, rads: list[float]) -> list[int]:
        return [j.to_count(r) for j, r in zip(self.joints, rads)]

    def limits(self) -> tuple[list[float], list[float]]:
        return [j.lower_rad for j in self.joints], [j.upper_rad for j in self.joints]

    def save(self, path: Path = DEFAULT_PATH) -> None:
        data = {"verified": self.verified, "port": self.port, "joints": [asdict(j) for j in self.joints]}
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> JointMap:
        data = json.loads(Path(path).read_text())
        joints = [JointCalibration(**j) for j in data["joints"]]
        names = tuple(j.name for j in joints)
        if names != SIM_JOINT_NAMES:
            raise ValueError(f"{path}: joints must be {SIM_JOINT_NAMES} in that order, got {names}.")
        for j in joints:
            if j.sign not in (1, -1):
                raise ValueError(f"{path}: {j.name} sign must be 1 or -1, got {j.sign}.")
        return cls(joints=joints, verified=bool(data.get("verified", False)), port=data.get("port", "/dev/ttyACM0"))
