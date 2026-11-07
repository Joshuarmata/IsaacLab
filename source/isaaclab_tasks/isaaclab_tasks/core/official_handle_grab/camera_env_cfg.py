# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overhead-camera variant of the handle-grab task, distilled from the state teacher.

It adds the fixed RealSense D415 over the tray to :class:`~.env_cfg.HandleGrabEnvCfg`. The student sees two stacked
RGB frames and its own noisy proprioception; the teacher's state, and the critic's privileged state, ride along in
extra observation groups.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import D415_CFG

from . import mdp
from .env_cfg import CriticStateGroupCfg, HandleGrabEnvCfg, HandleGrabSceneCfg, PolicyStateGroupCfg


@configclass
class HandleGrabCameraSceneCfg(HandleGrabSceneCfg):
    """The handle-grab scene plus the overhead D415, RGB only (the student never sees depth)."""

    camera = D415_CFG.replace(data_types=["rgb"])


@configclass
class OverheadImageCfg(ObsGroup):
    """Two stacked overhead RGB frames, channel first, per-image normalized, with pixel noise during training."""

    image = ObsTerm(
        func=mdp.image_rgb,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "rgb",
            "normalize": True,
            "channel_first": True,
            "frame_stack": 2,
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ProprioceptionCfg(ObsGroup):
    """Proprioception available on the real robot, in the encoders' (possibly biased) coordinates."""

    joint_pos = ObsTerm(
        func=mdp.encoder_joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    joint_target = ObsTerm(
        func=mdp.encoder_joint_target,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005),
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    """Deployable student inputs, the state teacher's inputs (the distillation labels), and the privileged critic."""

    overhead_rgb: OverheadImageCfg = OverheadImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class HandleGrabCameraEnvCfg(HandleGrabEnvCfg):
    """Handle grab from overhead RGB and proprioception, distilled from the state teacher."""

    scene: HandleGrabCameraSceneCfg = HandleGrabCameraSceneCfg(num_envs=128, env_spacing=2.5)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
