# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based handle-grab task: an SO-101 closes its jaws on a bag's strap and holds the bag up.

The recipe is the NVIDIA SO-101 vial task's: relative joint steps, dense shaping with latched milestones, resets from a
physics-validated dataset, an RSL-RL PPO state teacher, then a DAgger-style distilled camera student. The scene is the
one the direct ``Isaac-HandleGrab-Direct`` task uses.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-HandleGrab",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HandleGrabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandleGrabStatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-HandleGrab-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:HandleGrabCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandleGrabCameraPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:HandleGrabCameraDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
