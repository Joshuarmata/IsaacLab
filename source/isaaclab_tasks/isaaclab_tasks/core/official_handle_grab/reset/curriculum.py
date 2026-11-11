# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset phases and the phase mixes the task trains and evaluates on.

Each row of the reset dataset is a physics-validated state. Later phases are continued from rows of the phase before
them that already survived physics, the vial generator's seed banks. The phases a policy has to be able to continue
from:

* ``home``: the arm settled at its home pose, then the bag placed in the spawn box and settled, gripper open,
* ``pregrasp``: the jaw tip a few centimetres off the strap along the approach axis, gripper open,
* ``grasp``: the jaw tip on the recorded pinch point, gripper still open,
* ``grasp_closing``: the jaw slid onto the strap and closed most of the way around it, short of
  :data:`~..mdp.task.GRIPPER_HOLD_POS`, so not yet :func:`~..mdp.task.holding`. Every row was checked to hold once
  the jaw closes the rest of the way.

``grasp_closing`` shortens the 0.9 rad close the policy has to discover to about 0.2 rad, without handing it a grasp:
the policy still has to finish the close to earn the grasp and success rewards. The ``reset_fade`` curriculum term in
``env_cfg.py`` phases it out of the training mix. The vial's lift phases are not generated (``LIFT_GRAVEYARD.md``),
and there is no reorient, transport, insert, or release phase either: the scene has no rack.
"""

HOME = "home"
PREGRASP = "pregrasp"
GRASP = "grasp"
GRASP_CLOSING = "grasp_closing"

PHASES = (HOME, PREGRASP, GRASP, GRASP_CLOSING)
"""Phase names in the order of their ids in the dataset."""

ALL_PHASES: dict[str, float] = {phase: 1.0 for phase in PHASES}
"""Training mix: every phase equally likely (before the curriculum fades ``grasp_closing``)."""

CANONICAL_START: dict[str, float] = {HOME: 1.0}
"""Evaluation mix: every episode starts at home."""
