# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registers Isaac-BagScoopLift-Direct.

Test the teacher (zero actions):  see ../../README.md
Train with:  isaaclab train --rl_library rl_games --task=Isaac-BagScoopLift-Direct
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BagScoopLift-Direct",
    entry_point="isaaclab_tasks.core.bag_lift.bag_lift_env:BagLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scoop_lift_env_cfg:ScoopLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
