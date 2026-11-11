# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL algorithm settings for Isaac-HandleGrab-Direct (rl_games PPO)."""

# Registers the handle_grab_camera network when rl_games is installed. The real-arm scripts import this
# package and never use that network, so a checkout without the rl-games extra still imports.
try:
    from . import camera_net as _camera_net
except ModuleNotFoundError as exc:
    if exc.name != "rl_games" and not (exc.name or "").startswith("rl_games."):
        raise
else:
    del _camera_net
