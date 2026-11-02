# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers Isaac-BagGraspHandle-Direct.

Train with:  isaaclab train --rl_library rl_games --task=Isaac-BagGraspHandle-Direct
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BagGraspHandle-Direct",
    entry_point="isaaclab_tasks.core.bag_lift.bag_lift_env:BagLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_handle_env_cfg:GraspHandleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
