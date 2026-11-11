# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The arm's Feetech STS3215 bus: read positions, write goal positions, torque on and off, torque limits.

Uses ``feetech-servo-sdk`` (``scservo_sdk``), the driver LeRobot's SO-101 uses; install it with
``uv pip install feetech-servo-sdk``. Register addresses are the STS series control table.
"""

from __future__ import annotations

ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_TORQUE_LIMIT = 48
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_TEMPERATURE = 63
BAUD_RATE = 1_000_000


def _signed(value: int, sign_bit: int = 15) -> int:
    """STS registers store negatives as sign and magnitude, with the sign in ``sign_bit``."""
    return -(value & ~(1 << sign_bit)) if value & (1 << sign_bit) else value


class FeetechBus:
    """Synchronous reads and writes to a set of STS3215 motors on one serial port."""

    def __init__(self, port: str, motor_ids: list[int], baud_rate: int = BAUD_RATE):
        try:
            import scservo_sdk as scs
        except ImportError as exc:
            raise ImportError("The servo driver is missing: uv pip install feetech-servo-sdk") from exc
        self._scs = scs
        self.ids = list(motor_ids)
        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)
        if not self._port.openPort():
            raise OSError(f"Cannot open {port}. Is the arm plugged in, and is your user in the dialout group?")
        if not self._port.setBaudRate(baud_rate):
            raise OSError(f"Cannot set {port} to {baud_rate} baud.")
        self._reader = scs.GroupSyncRead(self._port, self._packet, ADDR_PRESENT_POSITION, 2)
        for mid in self.ids:
            self._reader.addParam(mid)
        self._writer = scs.GroupSyncWrite(self._port, self._packet, ADDR_GOAL_POSITION, 2)

    def close(self) -> None:
        self._port.closePort()

    def read_positions(self) -> list[int]:
        """Present position of every motor [counts], in ``motor_ids`` order."""
        result = self._reader.txRxPacket()
        if result != self._scs.COMM_SUCCESS:
            raise OSError(f"Position read failed: {self._packet.getTxRxResult(result)}")
        counts = []
        for mid in self.ids:
            if not self._reader.isAvailable(mid, ADDR_PRESENT_POSITION, 2):
                raise OSError(f"Motor {mid} did not answer the position read.")
            counts.append(_signed(self._reader.getData(mid, ADDR_PRESENT_POSITION, 2)))
        return counts

    def write_goals(self, counts: list[int]) -> None:
        """Goal position of every motor [counts], in ``motor_ids`` order, in one packet."""
        self._writer.clearParam()
        for mid, count in zip(self.ids, counts):
            count = int(count)
            self._writer.addParam(mid, [count & 0xFF, (count >> 8) & 0xFF])
        result = self._writer.txPacket()
        if result != self._scs.COMM_SUCCESS:
            raise OSError(f"Goal write failed: {self._packet.getTxRxResult(result)}")

    def _write_byte(self, mid: int, address: int, value: int) -> None:
        result, error = self._packet.write1ByteTxRx(self._port, mid, address, value)
        if result != self._scs.COMM_SUCCESS or error:
            raise OSError(f"Motor {mid} register {address} write failed ({result}, {error}).")

    def _write_word(self, mid: int, address: int, value: int) -> None:
        result, error = self._packet.write2ByteTxRx(self._port, mid, address, value)
        if result != self._scs.COMM_SUCCESS or error:
            raise OSError(f"Motor {mid} register {address} write failed ({result}, {error}).")

    def set_torque(self, enabled: bool) -> None:
        """Hold position (True) or go limp so the arm can be moved by hand (False)."""
        for mid in self.ids:
            self._write_byte(mid, ADDR_TORQUE_ENABLE, 1 if enabled else 0)

    def set_torque_limits(self, fractions: list[float]) -> None:
        """Cap each motor's torque at a fraction of its maximum (register unit 0.1 %)."""
        for mid, fraction in zip(self.ids, fractions):
            self._write_word(mid, ADDR_TORQUE_LIMIT, round(1000 * min(max(fraction, 0.0), 1.0)))

    def set_acceleration(self, value: int) -> None:
        """Acceleration ramp of every motor (0 is the fastest; 254 the gentlest)."""
        for mid in self.ids:
            self._write_byte(mid, ADDR_ACCELERATION, value)

    def read_temperatures(self) -> list[int]:
        """Motor temperatures [deg C]."""
        temps = []
        for mid in self.ids:
            value, result, error = self._packet.read1ByteTxRx(self._port, mid, ADDR_PRESENT_TEMPERATURE)
            if result != self._scs.COMM_SUCCESS or error:
                raise OSError(f"Motor {mid} temperature read failed.")
            temps.append(value)
        return temps
