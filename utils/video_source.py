"""
Video Source
============
Abstracts webcam and video file inputs under a unified interface.
Handles resolution setting, FPS matching, and reconnect logic.
"""

import time
from typing import Tuple, Union

import cv2
import numpy as np

from config.settings import AppConfig
from utils.logger import setup_logger


class VideoSource:
    """
    Unified video source (webcam or file).

    Usage:
        src = VideoSource(0, config)       # webcam
        src = VideoSource("video.mp4", config)
        while True:
            ret, frame = src.read()
            if not ret: break
        src.release()
    """

    def __init__(self, source: Union[int, str], config: AppConfig):
        self.source = source
        self.config = config
        self.logger = setup_logger("VideoSource")
        self._cap: cv2.VideoCapture = None
        self._is_file = isinstance(source, str)
        self._open()

    def _open(self):
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self.logger.error(f"Cannot open source: {self.source}")
            return

        # Configure resolution for webcam
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)
            # Some cameras need a buffer size hint
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(
            f"Source opened: {self.source} | "
            f"{actual_w}x{actual_h} @ {actual_fps:.1f} fps"
        )

    def read(self) -> Tuple[bool, np.ndarray]:
        """Read next frame. Returns (success, frame)."""
        if not self.is_open():
            return False, None

        ret, frame = self._cap.read()

        if not ret and self._is_file:
            # Video file ended
            return False, None

        if not ret and not self._is_file:
            # Webcam hiccup — retry once
            self.logger.warning("Frame read failed, retrying...")
            time.sleep(0.05)
            ret, frame = self._cap.read()

        return ret, frame

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self):
        if self._cap:
            self._cap.release()
            self.logger.info("Video source released.")

    @property
    def total_frames(self) -> int:
        """Total frames (for video files), -1 for webcam."""
        if self._is_file:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
