# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events: phase-balanced resets from the physics-validated dataset, and encoder calibration bias."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from ..reset.curriculum import ALL_PHASES, GRASP_CLOSED, HOME, PHASES
from ..reset.dataset import DEFAULT_DATASET_PATH, ResetDataset, sample_rows, validate_phase_weights
from .task import BAG, ROBOT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetFromDataset(ManagerTermBase):
    """Reset the arm and the bag to rows of the physics-validated reset dataset.

    Rows are drawn by ``phase_weights``, so one training run sees the approach, the pinch and the closed grasp. Every
    row is a state a scripted attempt reached in physics, so none starts interpenetrating or off the strap.

    With ``sequential`` the term walks the rows of the single positive-weight phase in attempt order, which is how play
    and evaluation start every episode from the same home placements. Home rows can get a little bag position and
    heading noise, since at home the arm and the bag are not coupled.

    The phase of each environment's start is kept on ``env._hg_reset_phase``; ``env._hg_reset_grasped`` marks closed
    grasps, so the success termination restores the grasp milestone instead of making the policy earn it again.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.dataset = ResetDataset.load(params.get("dataset_path", DEFAULT_DATASET_PATH), env.device)
        validate_phase_weights(self.dataset, params.get("phase_weights", ALL_PHASES))
        self.robot = env.scene[ROBOT]
        self.bag = env.scene[BAG]
        joint_names = self.dataset.meta.get("joint_names")
        if joint_names is not None and list(joint_names) != list(self.robot.joint_names):
            raise ValueError(
                f"The reset dataset was generated for joints {list(joint_names)}, but the robot has"
                f" {list(self.robot.joint_names)}. Regenerate it."
            )
        self.cursor = 0
        env._hg_reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._hg_reset_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        dataset_path: str = str(DEFAULT_DATASET_PATH),
        phase_weights: dict[str, float] = ALL_PHASES,
        sequential: bool = False,
        home_position_noise: float = 0.0,
        home_heading_noise: float = 0.0,
    ) -> None:
        all_ids = torch.arange(env.num_envs, device=env.device)
        ids = all_ids if env_ids is None else all_ids[env_ids]
        n = ids.numel()

        if sequential:
            phase = validate_phase_weights(self.dataset, phase_weights)
            if len(phase) != 1:
                raise ValueError("Sequential resets need exactly one phase with a positive weight.")
            pool = self.dataset.rows(phase[0])
            rows = pool[(self.cursor + torch.arange(n, device=env.device)) % pool.numel()]
            self.cursor = (self.cursor + n) % pool.numel()
        else:
            rows = sample_rows(self.dataset, phase_weights, n)

        phase_id = self.dataset.phase[rows]
        joint_pos = self.dataset.joint_pos[rows]
        bag_pose = self.dataset.bag_pose[rows].clone()

        home = (phase_id == PHASES.index(HOME)).float()
        if home_position_noise > 0.0:
            bag_pose[:, :2] += home.unsqueeze(-1) * sample_uniform(
                -home_position_noise, home_position_noise, (n, 2), env.device
            )
        if home_heading_noise > 0.0:
            yaw = home * sample_uniform(-home_heading_noise, home_heading_noise, (n,), env.device)
            zeros = torch.zeros_like(yaw)
            bag_pose[:, 3:7] = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), bag_pose[:, 3:7])
        bag_pose[:, :3] += env.scene.env_origins[ids]

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=ids)
        # The drive holds the dataset pose until the first action, instead of pulling toward the last episode's target.
        self.robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=None, env_ids=ids)
        self.bag.write_root_pose_to_sim_index(root_pose=bag_pose, env_ids=ids)
        self.bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=ids)

        env._hg_reset_phase[ids] = phase_id
        env._hg_reset_grasped[ids] = phase_id == PHASES.index(GRASP_CLOSED)


class randomize_encoder_bias(ManagerTermBase):
    """Sample a per-episode calibration bias [rad] for every joint encoder.

    The bias is added to the joint positions and targets the camera student observes
    (:func:`~.observations.encoder_joint_pos`), not to the physics, as a mis-calibrated real robot would report.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        env._hg_encoder_bias = torch.zeros(env.num_envs, env.scene[ROBOT].num_joints, device=env.device)

    def __call__(self, env: ManagerBasedEnv, env_ids: Sequence[int] | torch.Tensor | None, bound: float = 0.01) -> None:
        ids = slice(None) if env_ids is None else env_ids
        bias = env._hg_encoder_bias
        bias[ids] = sample_uniform(-bound, bound, bias[ids].shape, env.device)
