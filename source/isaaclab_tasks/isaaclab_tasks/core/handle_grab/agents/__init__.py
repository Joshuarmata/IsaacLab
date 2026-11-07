# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL algorithm settings for Isaac-HandleGrab-Direct (rl_games PPO)."""

# Registers the handle_grab_camera network with rl_games at import time.
from . import camera_net as _camera_net

del _camera_net
