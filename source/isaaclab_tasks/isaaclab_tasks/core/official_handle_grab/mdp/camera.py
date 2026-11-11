# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overhead D415 images with a per-episode real-camera look, for training a student that transfers to the robot.

Both terms change the raw frame and then hand it to the stock pipeline (:class:`~isaaclab.utils.images.CameraFrameStack`
with :func:`~isaaclab.utils.images.normalize_rgb` or :func:`~isaaclab.utils.images.normalize_depth`), the same
pipeline ``real/bridge.py`` runs on real frames. The RGB term rounds back to ``uint8`` first, as a real frame is.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.envs.mdp.observations import CameraImageBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.images import is_depth_like, is_rgb_like, normalize_depth, normalize_rgb
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


def _gaussian_blur(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Blur each ``(B, C, H, W)`` image with its own Gaussian ``sigma`` [px]; ``sigma`` 0 leaves it unchanged."""
    radius = 3
    offsets = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-0.5 * (offsets[None] / sigma.clamp(min=1e-3)[:, None]) ** 2)
    kernel = torch.where(sigma[:, None] > 0, kernel, (offsets[None] == 0).to(images.dtype))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    b, c, h, w = images.shape
    # One grouped convolution per axis, one group per (image, channel).
    flat = images.reshape(1, b * c, h, w)
    k = kernel.repeat_interleave(c, dim=0)
    flat = F.conv2d(F.pad(flat, (radius, radius, 0, 0), mode="replicate"), k[:, None, None, :], groups=b * c)
    flat = F.conv2d(F.pad(flat, (0, 0, radius, radius), mode="replicate"), k[:, None, :, None], groups=b * c)
    return flat.reshape(b, c, h, w)


class augmented_rgb(CameraImageBase):
    """Color frames with a per-episode exposure, white balance, contrast, gamma and focus, plus per-frame sensor noise.

    Sampled at every reset, so each episode looks like a different camera setting under different light:

    - ``exposure``: one gain for all channels.
    - ``white_balance``: a gain per channel, ``1 +- white_balance``.
    - ``contrast``: scales the image about its mean.
    - ``gamma``: the tone curve.
    - ``blur_probability`` / ``blur_sigma``: a share of episodes are slightly out of focus [px].
    - ``noise_std``: per-pixel Gaussian sensor noise, fresh every frame (fraction of full scale).

    The result is rounded to ``uint8`` and then normalized and stacked exactly as :class:`~isaaclab.envs.mdp.image_rgb`
    does. Returns ``(num_envs, 3 * frame_stack, H, W)`` when ``channel_first``.
    """

    @staticmethod
    def _accepts(data_type: str) -> bool:
        return is_rgb_like(data_type)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self._exposure = torch.ones(n, 1, 1, 1, device=env.device)
        self._balance = torch.ones(n, 3, 1, 1, device=env.device)
        self._contrast = torch.ones(n, 1, 1, 1, device=env.device)
        self._gamma = torch.ones(n, 1, 1, 1, device=env.device)
        self._sigma = torch.zeros(n, device=env.device)
        self._sample(slice(None), cfg.params)

    def _sample(self, ids, params: dict) -> None:
        n = self._exposure[ids].shape[0]
        device = self._exposure.device
        self._exposure[ids] = sample_uniform(*params["exposure"], (n, 1, 1, 1), device)
        wb = params["white_balance"]
        self._balance[ids] = sample_uniform(1.0 - wb, 1.0 + wb, (n, 3, 1, 1), device)
        self._contrast[ids] = sample_uniform(*params["contrast"], (n, 1, 1, 1), device)
        # Log-uniform, so a gamma of 1/g is as likely as g.
        lo, hi = params["gamma"]
        self._gamma[ids] = torch.exp(sample_uniform(math.log(lo), math.log(hi), (n, 1, 1, 1), device))
        blurred = torch.rand(n, device=device) < params["blur_probability"]
        sigma = sample_uniform(*params["blur_sigma"], (n,), device)
        self._sigma[ids] = torch.where(blurred, sigma, torch.zeros_like(sigma))

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        super().reset(env_ids)
        self._sample(slice(None) if env_ids is None else env_ids, self.cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        data_type: str = "rgb",
        channel_first: bool = False,
        frame_stack: int = 1,
        exposure: tuple[float, float] = (0.6, 1.4),
        white_balance: float = 0.12,
        contrast: tuple[float, float] = (0.75, 1.25),
        gamma: tuple[float, float] = (0.75, 1.33),
        blur_probability: float = 0.3,
        blur_sigma: tuple[float, float] = (0.3, 1.2),
        noise_std: float = 0.01,
    ) -> torch.Tensor:
        raw = env.scene.sensors[sensor_cfg.name].data.output[data_type].torch[..., :3]
        x = raw.permute(0, 3, 1, 2).float() / 255.0
        x = x * self._exposure * self._balance
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * self._contrast + mean
        x = x.clamp(0.0, 1.0) ** self._gamma
        if bool((self._sigma > 0).any()):
            x = _gaussian_blur(x, self._sigma)
        x = x + noise_std * torch.randn_like(x)
        frame = (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
        return self._frames(frame, normalize_rgb)


class augmented_depth(CameraImageBase):
    """Depth frames with the D415's errors: calibration scale and offset, range-dependent noise, and missing pixels.

    - ``scale`` / ``offset`` [m]: a per-episode calibration error, ``depth * scale + offset``.
    - ``noise_at_1m`` [m]: per-pixel Gaussian noise, growing with the square of the range as stereo depth does.
    - ``dropout``: a per-episode share of pixels with no return.
    - ``edge_dropout``: chance that a pixel on a depth edge (a jump over ``edge_jump`` [m] to a neighbour) has no
      return; stereo cameras lose depth along object outlines.

    Missing pixels read 0, as on the real D415, and then go through :func:`~isaaclab.utils.images.normalize_depth`
    with ``max_depth``, exactly as :class:`~isaaclab.envs.mdp.image_depth` does.
    """

    @staticmethod
    def _accepts(data_type: str) -> bool:
        return is_depth_like(data_type)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self._scale = torch.ones(n, 1, 1, 1, device=env.device)
        self._offset = torch.zeros(n, 1, 1, 1, device=env.device)
        self._dropout = torch.zeros(n, 1, 1, 1, device=env.device)
        self._sample(slice(None), cfg.params)

    def _sample(self, ids, params: dict) -> None:
        n = self._scale[ids].shape[0]
        device = self._scale.device
        self._scale[ids] = sample_uniform(*params["scale"], (n, 1, 1, 1), device)
        self._offset[ids] = sample_uniform(*params["offset"], (n, 1, 1, 1), device)
        self._dropout[ids] = sample_uniform(*params["dropout"], (n, 1, 1, 1), device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        super().reset(env_ids)
        self._sample(slice(None) if env_ids is None else env_ids, self.cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        data_type: str = "distance_to_image_plane",
        max_depth: float = 1.0,
        channel_first: bool = False,
        frame_stack: int = 1,
        scale: tuple[float, float] = (0.99, 1.01),
        offset: tuple[float, float] = (-0.005, 0.005),
        noise_at_1m: float = 0.003,
        dropout: tuple[float, float] = (0.0, 0.03),
        edge_dropout: float = 0.5,
        edge_jump: float = 0.02,
    ) -> torch.Tensor:
        raw = env.scene.sensors[sensor_cfg.name].data.output[data_type].torch
        d = raw.permute(0, 3, 1, 2).float()
        valid = torch.isfinite(d) & (d > 0)
        d = torch.where(valid, d, torch.zeros_like(d))
        d = d * self._scale + self._offset + noise_at_1m * d.square() * torch.randn_like(d)
        # A depth edge: any 4-neighbour more than edge_jump away.
        padded = F.pad(d, (1, 1, 1, 1), mode="replicate")
        jump = torch.stack(
            [
                (padded[..., 1:-1, :-2] - d).abs(),
                (padded[..., 1:-1, 2:] - d).abs(),
                (padded[..., :-2, 1:-1] - d).abs(),
                (padded[..., 2:, 1:-1] - d).abs(),
            ]
        ).amax(dim=0)
        lost = torch.rand_like(d) < self._dropout
        lost |= (jump > edge_jump) & (torch.rand_like(d) < edge_dropout)
        d = torch.where(valid & ~lost, d.clamp(min=0.0), torch.zeros_like(d))
        frame = d.permute(0, 2, 3, 1)
        return self._frames(frame, functools.partial(normalize_depth, invalid_value=0.0, max_depth=max_depth))
