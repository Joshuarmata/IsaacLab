# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The physics-validated reset dataset: format, validation, and phase-balanced sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .curriculum import PHASES

DATASET_VERSION = 1
DEFAULT_DATASET_PATH = Path(__file__).with_name("handle_grab_reset_dataset.pt")
"""Written by :mod:`isaaclab_tasks.core.official_handle_grab.reset.generator`."""

NUM_JOINTS = 7


@dataclass
class ResetDataset:
    """One row per (attempt, phase). Poses are relative to the environment origin; quaternions are ``(x, y, z, w)``."""

    phase: torch.Tensor
    """Phase id of every row, index into :data:`~.curriculum.PHASES`. Shape ``(rows,)``."""
    attempt: torch.Tensor
    """Which successful scripted attempt produced the row. Shape ``(rows,)``."""
    joint_pos: torch.Tensor
    """Arm joint positions [rad] in the robot's joint order. Shape ``(rows, 7)``."""
    bag_pose: torch.Tensor
    """Bag root pose ``(x, y, z, qx, qy, qz, qw)`` in the environment frame. Shape ``(rows, 7)``."""
    meta: dict = field(default_factory=dict)
    """Free-form generator settings, kept for traceability."""
    _rows: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        rows = self.phase.shape[0]
        if self.attempt.shape != (rows,):
            raise ValueError(f"attempt must have shape ({rows},), got {tuple(self.attempt.shape)}.")
        if self.joint_pos.shape != (rows, NUM_JOINTS):
            raise ValueError(f"joint_pos must have shape ({rows}, {NUM_JOINTS}), got {tuple(self.joint_pos.shape)}.")
        if self.bag_pose.shape != (rows, 7):
            raise ValueError(f"bag_pose must have shape ({rows}, 7), got {tuple(self.bag_pose.shape)}.")
        if rows and (int(self.phase.min()) < 0 or int(self.phase.max()) >= len(PHASES)):
            raise ValueError(f"phase ids must lie in [0, {len(PHASES)}).")
        if not torch.isfinite(self.joint_pos).all() or not torch.isfinite(self.bag_pose).all():
            raise ValueError("The reset dataset contains non-finite poses.")

    def to(self, device: str | torch.device) -> ResetDataset:
        return ResetDataset(
            self.phase.to(device),
            self.attempt.to(device),
            self.joint_pos.to(device),
            self.bag_pose.to(device),
            self.meta,
        )

    def rows(self, phase: str) -> torch.Tensor:
        """Indices of the rows of ``phase``, in attempt order."""
        if phase not in self._rows:
            rows = torch.nonzero(self.phase == PHASES.index(phase), as_tuple=False).squeeze(-1)
            self._rows[phase] = rows[torch.argsort(self.attempt[rows])]
        return self._rows[phase]

    def save(self, path: str | Path = DEFAULT_DATASET_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": DATASET_VERSION,
                "phases": list(PHASES),
                "phase": self.phase.cpu(),
                "attempt": self.attempt.cpu(),
                "joint_pos": self.joint_pos.cpu(),
                "bag_pose": self.bag_pose.cpu(),
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATASET_PATH, device: str | torch.device = "cpu") -> ResetDataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No handle-grab reset dataset at '{path}'. Generate it on a GPU machine with"
                " `uv run python -m isaaclab_tasks.core.official_handle_grab.reset.generator`."
            )
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("version") != DATASET_VERSION or data.get("phases") != list(PHASES):
            raise ValueError(
                f"'{path}' has version {data.get('version')} and phases {data.get('phases')}; expected version"
                f" {DATASET_VERSION} and phases {list(PHASES)}. Regenerate it."
            )
        dataset = cls(data["phase"], data["attempt"], data["joint_pos"], data["bag_pose"], data.get("meta", {}))
        return dataset.to(device)


def validate_phase_weights(dataset: ResetDataset, phase_weights: dict[str, float]) -> list[str]:
    """Return the phases with positive weight, raising when one of them is unknown or has no rows."""
    chosen = []
    for phase, weight in phase_weights.items():
        if phase not in PHASES:
            raise ValueError(f"Unknown reset phase '{phase}'. Known phases: {PHASES}.")
        if weight < 0.0:
            raise ValueError(f"Phase weight of '{phase}' must be non-negative, got {weight}.")
        if weight > 0.0:
            if dataset.rows(phase).numel() == 0:
                raise ValueError(f"The reset dataset has no rows for phase '{phase}'.")
            chosen.append(phase)
    if not chosen:
        raise ValueError("phase_weights has no phase with a positive weight.")
    return chosen


def sample_rows(dataset: ResetDataset, phase_weights: dict[str, float], count: int) -> torch.Tensor:
    """Draw ``count`` row indices, choosing the phase by ``phase_weights`` and then a row of that phase uniformly."""
    phases = validate_phase_weights(dataset, phase_weights)
    device = dataset.phase.device
    probabilities = torch.tensor([phase_weights[p] for p in phases], dtype=torch.float32, device=device)
    chosen = torch.multinomial(probabilities, count, replacement=True)
    out = torch.empty(count, dtype=torch.long, device=device)
    for k, phase in enumerate(phases):
        mask = chosen == k
        rows = dataset.rows(phase)
        out[mask] = rows[torch.randint(rows.numel(), (int(mask.sum()),), device=device)]
    return out
