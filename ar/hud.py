"""
HUD Renderer
============
Renders a heads-up display overlay with:
- Domain title & subtitle
- FPS & inference time
- Object count summary
- Frame counter
- Domain-specific status bar
"""

import time
from typing import Dict

import cv2
import numpy as np

from config.settings import AppConfig, Domain


class HUDRenderer:
    """Renders a persistent HUD on top of AR frames."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._start_time = time.time()

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        inference_ms: float,
        counts: Dict[str, int],
        frame_count: int,
    ) -> np.ndarray:
        """Render all HUD elements onto the frame."""
        h, w = frame.shape[:2]

        self._draw_top_bar(frame, w, fps, inference_ms)
        self._draw_bottom_bar(frame, h, w, counts, frame_count)
        self._draw_domain_badge(frame, w)

        return frame

    # ------------------------------------------------------------------ #
    #  Top Bar                                                             #
    # ------------------------------------------------------------------ #
    def _draw_top_bar(self, frame, w, fps, inference_ms):
        bar_h = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.config.get_color("hud_bg"), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Accent line at bottom of bar
        cv2.line(frame, (0, bar_h), (w, bar_h), self.config.get_color("primary"), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = self.config.get_color("hud_text")
        white = (220, 220, 220)

        # Title (left)
        title = self.config.domain_meta.get("title", "AR Detection System")
        cv2.putText(frame, title, (12, 26), font, 0.58, text_color, 1, cv2.LINE_AA)

        # FPS (right side)
        if self.config.show_fps:
            fps_str = f"FPS: {fps:.1f}"
            (tw, _), _ = cv2.getTextSize(fps_str, font, 0.55, 1)
            fps_color = (
                (0, 220, 80)
                if fps >= 20
                else (0, 165, 255) if fps >= 10 else (0, 0, 220)
            )
            cv2.putText(
                frame,
                fps_str,
                (w - tw - 120, 26),
                font,
                0.55,
                fps_color,
                1,
                cv2.LINE_AA,
            )

        # Inference time
        inf_str = f"INF: {inference_ms:.1f}ms"
        (tw2, _), _ = cv2.getTextSize(inf_str, font, 0.48, 1)
        cv2.putText(
            frame, inf_str, (w - tw2 - 14, 26), font, 0.48, white, 1, cv2.LINE_AA
        )

    # ------------------------------------------------------------------ #
    #  Bottom Bar                                                          #
    # ------------------------------------------------------------------ #
    def _draw_bottom_bar(self, frame, h, w, counts, frame_count):
        bar_h = 36
        bar_y = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), self.config.get_color("hud_bg"), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.line(frame, (0, bar_y), (w, bar_y), self.config.get_color("primary"), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = self.config.get_color("hud_text")

        # Object count chips
        x_cursor = 12
        total = sum(counts.values())

        cv2.putText(
            frame,
            f"DETECTED: {total}",
            (x_cursor, h - 12),
            font,
            0.50,
            text_color,
            1,
            cv2.LINE_AA,
        )
        x_cursor += 130

        for label, count in sorted(counts.items()):
            chip_text = f"{label.upper()}: {count}"
            (tw, _), _ = cv2.getTextSize(chip_text, font, 0.42, 1)
            if x_cursor + tw + 20 > w - 150:
                break
            # Chip background
            cv2.rectangle(
                frame,
                (x_cursor - 2, h - 28),
                (x_cursor + tw + 8, h - 8),
                self.config.get_color("primary"),
                1,
            )
            cv2.putText(
                frame,
                chip_text,
                (x_cursor + 4, h - 12),
                font,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            x_cursor += tw + 18

        # Frame counter + elapsed time (right)
        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)
        right_text = f"FRAME: {frame_count:05d}  {mins:02d}:{secs:02d}"
        (tw, _), _ = cv2.getTextSize(right_text, font, 0.44, 1)
        cv2.putText(
            frame,
            right_text,
            (w - tw - 12, h - 12),
            font,
            0.44,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    #  Domain Badge (top-right corner)                                    #
    # ------------------------------------------------------------------ #
    def _draw_domain_badge(self, frame, w):
        icon = self.config.domain_meta.get("icon", "AR")
        color = self.config.get_color("primary")

        # Badge box
        bx1 = w // 2 - 100
        bx2 = w // 2 + 100
        by1 = 4
        by2 = 35

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

        subtitle = self.config.domain_meta.get("subtitle", "")
        cv2.putText(
            frame,
            subtitle,
            (bx1 + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
