# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-teacher to overhead-camera-student distillation configuration."""

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


class BoundedTeacherDistillation(Distillation):
    """RSL-RL distillation whose labels are the teacher's actions clipped to [-1, 1], as the env applies them."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        return actions


@configclass
class HandleGrabCameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """DAgger-style distillation: the student acts, the loaded state teacher labels every visited state."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 1600
    save_interval = 100
    experiment_name = "handle_grab_camera_distillation"
    run_name = ""
    obs_groups = {"student": ["overhead_rgb", "proprioception"], "teacher": ["teacher_state"]}
    clip_actions = 1.0
    student = RslRlCNNModelCfg(
        # Compact encoder for the 160 x 90 overhead image.
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32], kernel_size=[5, 3, 3], stride=[2, 2, 2], activation="elu"
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.05, std_type="log"),
    )
    # Must mirror the state actor so its checkpoint loads; the teacher always acts deterministically.
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=(
            "isaaclab_tasks.core.official_handle_grab.agents.rsl_rl_distillation_cfg:BoundedTeacherDistillation"
        ),
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
