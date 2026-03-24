"""
AR Pipeline
===========
Orchestrates detection → tracking → AR rendering per frame.
Handles FPS calculation, statistics, and video saving.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import AppConfig, Domain
from core.detector import ObjectDetector
from core.tracker import TrailTracker
from ar.renderer import ARRenderer
from ar.hud import HUDRenderer
from utils.logger import setup_logger


class ARPipeline:
    """
    Full AR processing pipeline.

    Per-frame flow:
        1. Detect objects (YOLO)
        2. Update trail tracker
        3. Render AR overlays on bounding boxes
        4. Render HUD (stats, domain info)
        5. Optionally write to video file
    """

    def __init__(self, config: AppConfig, detector: ObjectDetector):
        self.config = config
        self.detector = detector
        self.logger = setup_logger("Pipeline")

        self.ar_renderer = ARRenderer(config)
        self.hud_renderer = HUDRenderer(config)
        self.trail_tracker = TrailTracker(config)

        self.video_writer: Optional[cv2.VideoWriter] = None
        self._fps_buffer = []
        self._last_time = time.perf_counter()
        self._frame_count = 0
        self._stats: Dict = {
            "fps": 0.0,
            "inference_ms": 0.0,
            "detections": 0,
            "counts": {},
        }

        if config.save_output:
            self._init_writer()

    def _init_writer(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"{self.config.output_dir}/ar_{self.config.domain.value}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            out_path,
            fourcc,
            self.config.fps_target,
            (self.config.frame_width, self.config.frame_height),
        )
        self.logger.info(f"Saving output to: {out_path}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through the full AR pipeline.

        Returns:
            (rendered_frame, stats_dict)
        """
        # Resize to configured resolution
        frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

        # 1. Detect
        detections, inference_ms = self.detector.detect(frame)

        # 2. Update trails
        self.trail_tracker.update(detections)

        # 3. Render AR overlays
        rendered = self.ar_renderer.render(frame.copy(), detections, self.trail_tracker)

        # 4. Render HUD
        self._update_fps()
        counts = self._count_by_class(detections)
        rendered = self.hud_renderer.render(
            rendered,
            fps=self._stats["fps"],
            inference_ms=inference_ms,
            counts=counts,
            frame_count=self._frame_count,
        )

        # 5. Write output
        if self.video_writer:
            self.video_writer.write(rendered)

        self._frame_count += 1
        self._stats.update(
            {
                "inference_ms": inference_ms,
                "detections": len(detections),
                "counts": counts,
            }
        )

        return rendered, self._stats.copy()

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            fps = 1.0 / dt
            self._fps_buffer.append(fps)
            if len(self._fps_buffer) > 30:
                self._fps_buffer.pop(0)
            self._stats["fps"] = sum(self._fps_buffer) / len(self._fps_buffer)

    def _count_by_class(self, detections) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in detections:
            counts[det.domain_label] = counts.get(det.domain_label, 0) + 1
        return counts

    def finalize(self):
        """Release resources."""
        if self.video_writer:
            self.video_writer.release()
            self.logger.info("Video writer closed.")
