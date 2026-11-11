# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The overhead D415: color and depth at 30 Hz, depth aligned to the color image and in metres.

Uses ``pyrealsense2`` (``uv pip install pyrealsense2``). The color stream is 1280 x 720, whose 69 degree view covers the
simulated camera's; :func:`~.bridge.to_student_frames` crops and resizes it. No depth hole filling: the student was
trained on holes, which read 0.
"""

from __future__ import annotations

import numpy as np

from .bridge import Intrinsics

WIDTH, HEIGHT, FPS = 1280, 720, 30


class D415:
    """A running D415 pipeline; one :meth:`read` per policy step."""

    def __init__(self, serial: str | None = None, auto_exposure: bool = True):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("The camera driver is missing: uv pip install pyrealsense2") from exc
        self._rs = rs
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        device = profile.get_device()
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
        self._align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy)
        # Let auto exposure settle before the first frame is used.
        for _ in range(30):
            self._pipeline.wait_for_frames()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """The newest ``(720, 1280, 3)`` uint8 RGB frame and the depth aligned to it, ``(720, 1280)`` [m], 0 = none."""
        frames = self._align.process(self._pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("The D415 returned an incomplete frame set.")
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return rgb, depth_m

    def close(self) -> None:
        self._pipeline.stop()
