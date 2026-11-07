# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers Isaac-HandleGrab-Direct: find the strap and close on it.

Train the camera policy (logs to logs/rl_games/handle_grab_camera, separate from handle_grab):
    uv run --extra rl-games isaaclab train --rl_library rl_games --task=Isaac-HandleGrab-Direct --num_envs 128
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HandleGrab-Direct",
    entry_point=f"{__name__}.handle_grab_env:HandleGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handle_grab_env_cfg:HandleGrabEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
