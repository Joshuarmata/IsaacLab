# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overhead-camera variant of the handle-grab task, distilled from the state teacher.

It adds a fixed RealSense D415 above the tray to :class:`~.env_cfg.HandleGrabEnvCfg`. The student sees two stacked
RGB frames, two stacked depth frames, and its own noisy proprioception; the teacher's state, and the critic's
privileged state, ride along in extra observation groups.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.core.official_handle_grab.scene.handle_grab_scene import (
    BAG_REST_Z,
    BAG_SPAWN_XY,
    D415_CFG,
)
from isaaclab_tasks.utils import preset

from . import mdp
from .env_cfg import (
    CriticStateGroupCfg,
    EventsCfg,
    HandleGrabEnvCfg,
    HandleGrabSceneCfg,
    PolicyStateGroupCfg,
    Sim2RealEventsCfg,
)

STUDENT_CAMERA_WIDTH = 256
STUDENT_CAMERA_HEIGHT_PX = 144
"""Student image size, 16:9 like the D415. At 160 x 90 the strap was only two or three pixels wide."""


def _look_at_ros(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
    right: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> tuple[float, ...]:
    """Quaternion (x, y, z, w) of a ROS optical frame (+Z forward, +X right, +Y down) at ``pos`` looking at ``target``.

    Image right is ``right`` made perpendicular to the view, so it must not be parallel to it. The default, world -x,
    matches the straight-down camera of the base scene: arm at the top of the image, tray long side horizontal.
    """
    z = [t - p for t, p in zip(target, pos)]
    z = [v / math.hypot(*z) for v in z]
    along = sum(r * v for r, v in zip(right, z))
    x = [r - along * v for r, v in zip(right, z)]
    x = [v / math.hypot(*x) for v in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    # Columns x, y, z form the rotation matrix; convert with Shepperd's method.
    m00, m11, m22 = x[0], y[1], z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 2.0 * math.sqrt(1.0 + trace)
        return ((y[2] - z[1]) / s, (z[0] - x[2]) / s, (x[1] - y[0]) / s, 0.25 * s)
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return (0.25 * s, (y[0] + x[1]) / s, (z[0] + x[2]) / s, (y[2] - z[1]) / s)
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return ((y[0] + x[1]) / s, 0.25 * s, (z[1] + y[2]) / s, (z[0] - x[2]) / s)
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return ((z[0] + x[2]) / s, (z[1] + y[2]) / s, 0.25 * s, (x[1] - y[0]) / s)


# High enough to see the whole tray with the arm at home, and nearly straight down so the bag never hides a strap
# on its far side (the bag spawns at any yaw). The arm reaches in along +y, so a camera exactly above it sees only
# the top of the wrist at the grasp; a small tilt in from +x looks past the wrist at the side of the jaw.
STUDENT_CAMERA_POS = (0.12, 0.09, 0.70)
STUDENT_CAMERA_TARGET = (BAG_SPAWN_XY[0], BAG_SPAWN_XY[1] + 0.05, BAG_REST_Z)
STUDENT_CAMERA_ROT = _look_at_ros(STUDENT_CAMERA_POS, STUDENT_CAMERA_TARGET)

KEY_LIGHT_TILT = math.radians(35.0)
"""Tilt of the key light from straight down; it shines from +y toward -y."""

STUDENT_DEPTH_MAX_M = 1.0
"""Depth is clipped at this range and divided by it, so the student sees depth in [0, 1]. The tray floor is under it."""


@configclass
class HandleGrabCameraSceneCfg(HandleGrabSceneCfg):
    """The handle-grab scene plus the overhead D415 (RGB and depth) under soft, even studio lighting."""

    camera = D415_CFG.replace(
        offset=D415_CFG.offset.replace(pos=STUDENT_CAMERA_POS, rot=STUDENT_CAMERA_ROT),
        spawn=D415_CFG.spawn.replace(focus_distance=math.dist(STUDENT_CAMERA_POS, STUDENT_CAMERA_TARGET)),
        width=STUDENT_CAMERA_WIDTH,
        height=STUDENT_CAMERA_HEIGHT_PX,
        data_types=["rgb", "distance_to_image_plane"],
    )
    # Bright, shadow-free fill so no part of the bag or strap falls into darkness.
    ambient = AssetBaseCfg(
        prim_path="/World/Ambient",
        spawn=sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 1.0, 1.0)),
    )
    # Key light from above the far (+y) end of the tray, onto the front of the jaw as it reaches in. Off the camera
    # axis, its reflection on the floor misses the lens; the wide angle keeps shadow edges soft, and the intensity
    # stays below where the white tray floor clips.
    overhead = AssetBaseCfg(
        prim_path="/World/OverheadLight",
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0), angle=25.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(-math.sin(0.5 * KEY_LIGHT_TILT), 0.0, 0.0, math.cos(0.5 * KEY_LIGHT_TILT))
        ),
    )


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
class OverheadDepthCfg(ObsGroup):
    """Two stacked overhead depth frames, channel first, scaled to [0, 1], with a few millimetres of noise.

    Its own group, so the student gets a separate CNN for depth. Pixels with no return read 0, as on the real D415.
    """

    image = ObsTerm(
        func=mdp.image_depth,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "distance_to_image_plane",
            "normalize": True,
            "invalid_value": 0.0,
            "max_depth": STUDENT_DEPTH_MAX_M,
            "channel_first": True,
            "frame_stack": 2,
        },
        noise=UniformNoiseCfg(n_min=-0.004, n_max=0.004),
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
    overhead_depth: OverheadDepthCfg = OverheadDepthCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class AugmentedOverheadImageCfg(ObsGroup):
    """:class:`OverheadImageCfg` seen through a per-episode random camera setting (:class:`~.mdp.augmented_rgb`)."""

    image = ObsTerm(
        func=mdp.augmented_rgb,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb", "channel_first": True, "frame_stack": 2},
    )


@configclass
class AugmentedOverheadDepthCfg(ObsGroup):
    """:class:`OverheadDepthCfg` with the D415's calibration error, noise and holes (:class:`~.mdp.augmented_depth`)."""

    image = ObsTerm(
        func=mdp.augmented_depth,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "distance_to_image_plane",
            "max_depth": STUDENT_DEPTH_MAX_M,
            "channel_first": True,
            "frame_stack": 2,
        },
    )


@configclass
class RealProprioceptionCfg(ProprioceptionCfg):
    """:class:`ProprioceptionCfg` with the joint-velocity noise of a 30 Hz difference of 4096-count encoders."""

    def __post_init__(self):
        super().__post_init__()
        # One encoder count (2 pi / 4096 rad) over one policy step (1/30 s) is 0.046 rad/s.
        self.joint_vel.noise = UniformNoiseCfg(n_min=-0.06, n_max=0.06)


@configclass
class CameraSim2RealObservationsCfg(CameraObservationsCfg):
    """The same groups, with real-camera image variation and real-encoder velocity noise."""

    overhead_rgb: AugmentedOverheadImageCfg = AugmentedOverheadImageCfg()
    overhead_depth: AugmentedOverheadDepthCfg = AugmentedOverheadDepthCfg()
    proprioception: RealProprioceptionCfg = RealProprioceptionCfg()


SCENE_RECOLOR_INTERVAL_S = (20.0, 40.0)
"""Seconds between scene-wide colour and light changes; about one change per episode, never one per reset."""


def _recolor(asset: str, colors: dict[str, tuple[float, float]], mesh_name: str = "") -> EventTerm:
    return EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=SCENE_RECOLOR_INTERVAL_S,
        is_global_time=True,
        params={
            "event_name": f"recolor_{asset}{mesh_name}",
            "asset_cfg": SceneEntityCfg(asset),
            "colors": colors,
            "mesh_name": mesh_name,
        },
    )


@configclass
class CameraSim2RealEventsCfg(Sim2RealEventsCfg):
    """:class:`~.env_cfg.Sim2RealEventsCfg` plus what a real overhead camera sees differently.

    Per episode: the camera's mount, within 2 cm and 3 degrees. Scene-wide, every 20 to 40 s: light brightness, tint
    and key-light direction, and the colours of the bag, strap, tray floor and tray walls. Each environment gets its
    own colours.
    """

    camera_pose = EventTerm(func=mdp.randomize_camera_pose, mode="reset")
    lights = EventTerm(func=mdp.randomize_lights, mode="interval", interval_range_s=SCENE_RECOLOR_INTERVAL_S)
    bag_color = _recolor("bag", {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)})
    strap_color = _recolor("handle", {"r": (0.0, 0.6), "g": (0.0, 0.6), "b": (0.0, 0.6)})
    floor_color = _recolor("sandbox", {"r": (0.45, 0.95), "g": (0.4, 0.95), "b": (0.3, 0.95)}, "TrayFloor")
    wall_color = _recolor("sandbox", {"r": (0.2, 0.9), "g": (0.2, 0.9), "b": (0.2, 0.9)}, "TrayWall.*")


@configclass
class HandleGrabCameraEnvCfg(HandleGrabEnvCfg):
    """Handle grab from overhead RGB, depth and proprioception, distilled from the state teacher.

    ``presets=sim2real`` adds the state task's physics variation and, for the camera, :class:`CameraSim2RealEventsCfg`
    and :class:`CameraSim2RealObservationsCfg`.
    """

    scene: HandleGrabCameraSceneCfg = HandleGrabCameraSceneCfg(num_envs=128, env_spacing=2.5)
    events: EventsCfg = preset(default=EventsCfg(), sim2real=CameraSim2RealEventsCfg())
    observations: CameraObservationsCfg = preset(
        default=CameraObservationsCfg(), sim2real=CameraSim2RealObservationsCfg()
    )

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 8)
