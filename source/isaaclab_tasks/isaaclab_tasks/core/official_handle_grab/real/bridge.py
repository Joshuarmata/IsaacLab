# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The camera student's inputs and outputs on the real robot, built exactly as the simulator builds them.

The images go through the same :class:`~isaaclab.utils.images.CameraFrameStack`,
:func:`~isaaclab.utils.images.normalize_rgb` and :func:`~isaaclab.utils.images.normalize_depth` as the camera task's
observation terms, after the real frame is cropped and resized to the simulated D415's field of view. Actions are
turned into joint targets as
:class:`~isaaclab.envs.mdp.actions.RelativeJointPositionAction` does: the measured position plus the scaled action,
renewed :data:`DECIMATION` times per policy step.
Nothing here touches hardware, so it is tested without a robot.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from isaaclab.utils.images import CameraFrameStack, normalize_depth, normalize_rgb

from ..camera_env_cfg import STUDENT_CAMERA_HEIGHT_PX, STUDENT_CAMERA_WIDTH, STUDENT_DEPTH_MAX_M
from ..env_cfg import ARM_ACTION_SCALE, GRIPPER_ACTION_SCALE
from ..scene.handle_grab_scene import D415_HFOV_DEG

FRAME_STACK = 2
"""Frames per image group, oldest first, as in ``OverheadImageCfg`` and ``OverheadDepthCfg``."""
POLICY_HZ = 30.0
"""Policy rate: physics at 60 Hz with decimation 2."""
DECIMATION = 2
"""Joint-target updates per policy step. The simulator re-applies the relative action at every physics step, each
time from the joint's position at that step, so the real loop sends ``measured + scaled action`` twice per step too."""
ACTION_SCALE = (ARM_ACTION_SCALE,) * 6 + (GRIPPER_ACTION_SCALE,)
"""Largest joint step per policy step [rad], per joint, in simulation joint order."""


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics [px] of an image."""

    fx: float
    fy: float
    cx: float
    cy: float


def sim_intrinsics() -> Intrinsics:
    """The simulated D415 at the student's resolution: square pixels, principal point at the centre."""
    f = 0.5 * STUDENT_CAMERA_WIDTH / math.tan(math.radians(0.5 * D415_HFOV_DEG))
    return Intrinsics(f, f, 0.5 * STUDENT_CAMERA_WIDTH, 0.5 * STUDENT_CAMERA_HEIGHT_PX)


CROP_OVERFLOW = 0.03
"""Share of the crop that may fall outside the real frame (filled with the edge pixels). A D415's 1280 x 720 color view
is about 69.4 degrees wide, only just wider than the simulated 69, so a unit with a slightly long lens needs this."""


def crop_window(real: Intrinsics, width: int, height: int) -> tuple[float, float, float, float]:
    """The ``(left, top, crop_width, crop_height)`` [px] of a real frame that spans the simulated camera's view.

    Raises:
        ValueError: If more than :data:`CROP_OVERFLOW` of the crop falls outside the real frame: the real camera's view
            is narrower than the simulated one.
    """
    sim = sim_intrinsics()
    crop_w = STUDENT_CAMERA_WIDTH * real.fx / sim.fx
    crop_h = STUDENT_CAMERA_HEIGHT_PX * real.fy / sim.fy
    # RealSense reports the principal point with pixel centres at whole numbers (OpenCV); the window is in edge
    # coordinates, where pixel i spans [i, i + 1], as is the simulated camera's centred principal point.
    left = real.cx + 0.5 - sim.cx * real.fx / sim.fx
    top = real.cy + 0.5 - sim.cy * real.fy / sim.fy
    slack_w, slack_h = CROP_OVERFLOW * crop_w, CROP_OVERFLOW * crop_h
    if left < -slack_w or top < -slack_h or left + crop_w > width + slack_w or top + crop_h > height + slack_h:
        raise ValueError(
            f"The real camera's view ({width} x {height}, fx {real.fx:.1f}) does not cover the simulated "
            f"{D415_HFOV_DEG} degree view; crop ({left:.1f}, {top:.1f}, {crop_w:.1f}, {crop_h:.1f})."
        )
    return left, top, crop_w, crop_h


def _resample(image: torch.Tensor, window: tuple[float, float, float, float], mode: str) -> torch.Tensor:
    """Sample ``window`` of a ``(C, H, W)`` image onto the student's grid; ``nearest`` keeps depth holes intact."""
    _, h, w = image.shape
    left, top, crop_w, crop_h = window
    # Pixel centres of the output grid, in the input's normalized [-1, 1] coordinates (align_corners=False).
    xs = left + (torch.arange(STUDENT_CAMERA_WIDTH, dtype=torch.float32) + 0.5) * crop_w / STUDENT_CAMERA_WIDTH
    ys = top + (torch.arange(STUDENT_CAMERA_HEIGHT_PX, dtype=torch.float32) + 0.5) * crop_h / STUDENT_CAMERA_HEIGHT_PX
    grid = torch.stack(torch.meshgrid(2.0 * ys / h - 1.0, 2.0 * xs / w - 1.0, indexing="ij")[::-1], dim=-1)
    if mode == "area":
        # Low-pass first so a 5x downsample does not alias, then sample.
        factor = max(1, int(crop_w / STUDENT_CAMERA_WIDTH))
        if factor > 1:
            image = F.avg_pool2d(image[None], factor, stride=1, padding=factor // 2, count_include_pad=False)[0]
            image = image[:, :h, :w]
        mode = "bilinear"
    return F.grid_sample(image[None], grid[None], mode=mode, padding_mode="border", align_corners=False)[0]


def to_student_frames(
    rgb: np.ndarray, depth_m: np.ndarray, intrinsics: Intrinsics
) -> tuple[torch.Tensor, torch.Tensor]:
    """A real color frame ``(H, W, 3)`` uint8 and the depth aligned to it ``(H, W)`` [m, 0 for no return] as the
    simulated camera's ``(1, 144, 256, 3)`` uint8 RGB and ``(1, 144, 256, 1)`` float depth."""
    height, width = depth_m.shape
    window = crop_window(intrinsics, width, height)
    color = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    color = _resample(color, window, "area").round().clamp(0, 255).to(torch.uint8)
    depth = torch.from_numpy(np.ascontiguousarray(depth_m, dtype=np.float32))[None]
    depth = _resample(depth, window, "nearest")
    return color.permute(1, 2, 0)[None], depth.permute(1, 2, 0)[None]


class ObservationBuilder:
    """The student's three inputs, ``obs`` (proprioception), ``overhead_rgb`` and ``overhead_depth``, batch of one."""

    def __init__(self):
        # The normalizers run Warp kernels; outside Isaac Sim nothing else initializes Warp.
        wp.init()
        self._rgb = CameraFrameStack(1, "cpu", frame_stack=FRAME_STACK, channel_first=True)
        self._depth = CameraFrameStack(1, "cpu", frame_stack=FRAME_STACK, channel_first=True)
        self._depth_normalizer = functools.partial(normalize_depth, invalid_value=0.0, max_depth=STUDENT_DEPTH_MAX_M)

    def reset(self) -> None:
        """Start a new episode: the next frame fills the whole stack, as after a simulator reset."""
        self._rgb.reset()
        self._depth.reset()

    def __call__(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_target: np.ndarray,
        last_action: np.ndarray,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Inputs for one policy step.

        Args:
            joint_pos, joint_vel, joint_target, last_action: Seven values each, simulation joint order [rad, rad/s].
            rgb: ``(1, 144, 256, 3)`` uint8 from :func:`to_student_frames`.
            depth: ``(1, 144, 256, 1)`` [m] from :func:`to_student_frames`.
        """
        proprio = np.concatenate([joint_pos, joint_vel, joint_target, last_action]).astype(np.float32)[None]
        return {
            "obs": proprio,
            "overhead_rgb": self._rgb(rgb, normalize_rgb).numpy().astype(np.float32),
            "overhead_depth": self._depth(depth, self._depth_normalizer).numpy().astype(np.float32),
        }


def joint_targets(
    measured: np.ndarray, action: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The policy's action clipped to [-1, 1] (``clip_actions``), and the joint targets it commands [rad].

    A target is the measured position plus the scaled action, kept inside the joint limits.
    """
    action = np.clip(action, -1.0, 1.0)
    return action, np.clip(measured + action * np.asarray(ACTION_SCALE), lower, upper)


class VelocityEstimator:
    """Joint velocity [rad/s] from successive encoder readings, as the simulator's ``joint_vel`` would read."""

    def __init__(self, smoothing: float = 0.5):
        self._smoothing = smoothing
        self._last_pos: np.ndarray | None = None
        self._last_time = 0.0
        self._vel: np.ndarray | None = None

    def reset(self) -> None:
        self._last_pos = None

    def __call__(self, pos: np.ndarray, now: float) -> np.ndarray:
        if self._last_pos is None or now <= self._last_time:
            vel = np.zeros_like(pos)
        else:
            raw = (pos - self._last_pos) / (now - self._last_time)
            vel = raw if self._vel is None else self._smoothing * self._vel + (1.0 - self._smoothing) * raw
        self._last_pos, self._last_time, self._vel = pos.copy(), now, vel
        return vel


class OnnxStudent:
    """The exported student (``exported/policy.onnx``, observation normalizer included) on ONNX Runtime."""

    INPUTS = ("obs", "overhead_rgb", "overhead_depth")

    def __init__(self, path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=providers or ort.get_available_providers())
        names = tuple(i.name for i in self.session.get_inputs())
        if set(names) != set(self.INPUTS):
            raise ValueError(f"{path} takes {names}; the camera student takes {self.INPUTS}.")

    def __call__(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        return self.session.run(None, {name: inputs[name] for name in self.INPUTS})[0][0]
