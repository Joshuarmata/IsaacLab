# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset phases and the phase mixes the task trains and evaluates on.

Each row of the reset dataset is a physics-validated state along one successful scripted grab. The four phases are
the states a policy has to be able to continue from:

* ``home``: the arm at its home pose, the gripper open, the bag at rest somewhere in the spawn box,
* ``pregrasp``: the jaw tip a few centimetres off the strap along the approach axis, gripper open,
* ``grasp``: the jaw tip on the recorded pinch point, gripper still open,
* ``grasp_closed``: the jaws closed on the strap, the bag still resting on the tray.

Lifting is deliberately not a phase: the first lift has to be discovered from ``grasp_closed`` starts.
"""

HOME = "home"
PREGRASP = "pregrasp"
GRASP = "grasp"
GRASP_CLOSED = "grasp_closed"

PHASES = (HOME, PREGRASP, GRASP, GRASP_CLOSED)
"""Phase names in the order of their ids in the dataset."""

ALL_PHASES: dict[str, float] = {phase: 1.0 for phase in PHASES}
"""Training mix: every phase equally likely."""

CANONICAL_START: dict[str, float] = {HOME: 1.0}
"""Evaluation mix: every episode starts at home."""

APPROACH_PHASES: dict[str, float] = {HOME: 1.0, PREGRASP: 1.0, GRASP: 1.0}
"""Mix for the ``pinch_only`` fallback. A closed-grasp start is already a held strap, nearly a success."""
