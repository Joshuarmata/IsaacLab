# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Running the distilled camera student on the real 7-motor arm and overhead D415, without ROS.

- :mod:`.joint_map`: Feetech encoder counts to and from simulation joint angles, per motor.
- :mod:`.bridge`: the policy's inputs and outputs, built with the simulation's own image code.
- :mod:`.feetech`, :mod:`.realsense`: thin drivers for the servo bus and the camera.
- Scripts: ``calibrate``, ``twin`` (mirror the real arm in the simulator and overlay the cameras), ``home``,
  ``deploy``, and ``check_bridge`` (bridge and ONNX parity against the simulator).

See "Sim to real" in the task README for the order to run them in.
"""
