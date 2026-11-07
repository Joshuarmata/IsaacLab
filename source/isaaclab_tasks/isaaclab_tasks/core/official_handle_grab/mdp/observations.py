# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms: task-specific state features and the encoder-coordinate proprioception of the camera student."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .progress import grab_progress
from .task import BAG, ROBOT, gripper_closed_fraction, lower_body_force, strap_contacts, strap_target_w, tip_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint-position targets currently sent to the drives [rad]."""
    return env.scene[asset_cfg.name].data.joint_pos_target.torch[:, asset_cfg.joint_ids]


def encoder_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions as the robot's encoders report them, including the sampled calibration bias [rad]."""
    bias = getattr(env, "_hg_encoder_bias", None)
    pos = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return pos if bias is None else pos + bias[:, asset_cfg.joint_ids]


def encoder_joint_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint targets in the same (biased) encoder coordinates as :func:`encoder_joint_pos` [rad]."""
    bias = getattr(env, "_hg_encoder_bias", None)
    target = joint_target(env, asset_cfg)
    return target if bias is None else target + bias[:, asset_cfg.joint_ids]


def strap_in_base_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The recorded pinch point on the strap in the robot's base frame [m]."""
    robot = env.scene[ROBOT]
    target = strap_target_w(env)
    return quat_apply_inverse(robot.data.root_link_quat_w.torch, target - robot.data.root_link_pos_w.torch)


def grab_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from the jaw tip to the strap, the bag's rise, the bag's speed, and how far the jaw has closed."""
    progress = grab_progress(env)
    distance = torch.linalg.vector_norm(tip_pos_w(env) - strap_target_w(env), dim=-1)
    speed = torch.linalg.vector_norm(env.scene[BAG].data.root_lin_vel_w.torch, dim=-1)
    return torch.stack((distance, progress.lift, speed, gripper_closed_fraction(env)), dim=-1)


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Latched grasp and lift milestones, which make the once-per-episode rewards Markov."""
    progress = grab_progress(env)
    return torch.stack((progress.grasped, progress.lifted), dim=-1).float()


def jaw_contact_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged rigid-jaw, moving-jaw and bilateral strap contact, plus the force on the lower body [N]."""
    rigid_on, moving_on = strap_contacts(env)
    force = lower_body_force(env)
    return torch.stack((rigid_on.float(), moving_on.float(), (rigid_on & moving_on).float(), force), dim=-1)
