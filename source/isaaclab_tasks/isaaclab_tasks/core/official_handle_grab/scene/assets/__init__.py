# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paths for the handle-grab scene assets.

The same role as the vial example's ``assets/__init__.py``: the scene loads these files, and
:func:`validate_assets` reports any that are missing without opening a USD stage.
"""

from pathlib import Path

ASSET_ROOT = Path(__file__).parent
SANDBOX_USD = ASSET_ROOT / "sandbox_tray.usda"
ARM_USD = ASSET_ROOT / "so101_physics.usda"
BAG_USD = ASSET_ROOT / "bag_single_body.usda"
BAG_BODY_USD = ASSET_ROOT / "bag_body_only.usda"
HANDLE_USD = ASSET_ROOT / "bag_handle_ribbon.usda"
HANDLE_INFO = ASSET_ROOT / "bag_handle_info.json"
JAW_POINTS = ASSET_ROOT / "jaw_points.json"


def validate_assets() -> list[str]:
    """Return missing local asset files, without requiring a USD runtime."""
    required = (SANDBOX_USD, ARM_USD, BAG_USD, BAG_BODY_USD, HANDLE_USD, HANDLE_INFO, JAW_POINTS)
    return [str(path) for path in required if not path.is_file()]
