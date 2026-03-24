"""
AR Renderer
===========
Draws augmented reality overlays on detected objects.
Supports domain-specific visual styles:
- Fancy corner brackets instead of plain boxes
- Animated-style label panels
- Motion trails
- Domain-specific contextual info cards
- Alert pulsing for high-risk detections
"""

import math
from typing import List, Tuple

import cv2
import numpy as np

from config.settings import AppConfig, Domain
from core.detector import Detection
from core.tracker import TrailTracker


class ARRenderer:
    """Renders per-object AR overlays onto frames."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._frame_counter = 0

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        trail_tracker: TrailTracker,
    ) -> np.ndarray:
        """Apply all AR overlays to the frame."""
        self._frame_counter += 1

        for det in detections:
            # 1. Motion trails
            if self.config.show_trails and det.track_id is not None:
                trail = trail_tracker.get_trail(det.track_id)
                self._draw_trail(frame, trail, det)

            # 2. Bounding box with corner brackets
            self._draw_bbox(frame, det)

            # 3. Label panel
            self._draw_label(frame, det)

            # 4. Domain-specific info card
            self._draw_domain_info(frame, det)

            # 5. Alert glow for high-risk
            if det.is_alert or det.extra.get("hazard"):
                self._draw_alert_overlay(frame, det)

        return frame

    # ------------------------------------------------------------------ #
    #  Bounding Box                                                        #
    # ------------------------------------------------------------------ #
    def _draw_bbox(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        color = self._get_detection_color(det)
        thickness = self.config.box_thickness

        # Semi-transparent fill
        overlay = frame.copy()
        alpha = 0.07 if not det.is_alert else 0.15
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Outer box (thin)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Corner brackets
        self._draw_corners(frame, x1, y1, x2, y2, color, thickness + 1)

    def _draw_corners(self, frame, x1, y1, x2, y2, color, thickness):
        """Draw corner bracket decorations."""
        length = max(12, min(30, (x2 - x1) // 5))

        corners = [
            # top-left
            [(x1, y1 + length), (x1, y1), (x1 + length, y1)],
            # top-right
            [(x2 - length, y1), (x2, y1), (x2, y1 + length)],
            # bottom-left
            [(x1, y2 - length), (x1, y2), (x1 + length, y2)],
            # bottom-right
            [(x2 - length, y2), (x2, y2), (x2, y2 - length)],
        ]
        for pts in corners:
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Label Panel                                                         #
    # ------------------------------------------------------------------ #
    def _draw_label(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        color = self._get_detection_color(det)

        label = det.domain_label.replace("_", " ").upper()
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        if self.config.show_confidence:
            label += f"  {det.confidence:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.config.font_scale
        ft = self.config.font_thickness

        (tw, th), baseline = cv2.getTextSize(label, font, fs, ft)
        pad_x, pad_y = 8, 5

        # Panel background
        panel_x1 = x1
        panel_y1 = max(0, y1 - th - pad_y * 2 - 2)
        panel_x2 = x1 + tw + pad_x * 2
        panel_y2 = y1

        # Clamp to frame
        h, w = frame.shape[:2]
        panel_x2 = min(panel_x2, w)

        # Dark background with colored left accent bar
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (10, 10, 20), -1
        )
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Accent left bar
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x1 + 3, panel_y2), color, -1)

        # Text
        cv2.putText(
            frame,
            label,
            (panel_x1 + pad_x + 2, panel_y2 - pad_y),
            font,
            fs,
            color,
            ft,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    #  Domain-Specific Info Card                                           #
    # ------------------------------------------------------------------ #
    def _draw_domain_info(self, frame: np.ndarray, det: Detection):
        domain = self.config.domain
        extra = det.extra

        if domain == Domain.TRAFFIC:
            self._draw_traffic_card(frame, det, extra)
        elif domain == Domain.RETAIL:
            self._draw_retail_card(frame, det, extra)
        elif domain == Domain.SAFETY:
            self._draw_safety_card(frame, det, extra)
        elif domain == Domain.SPORTS:
            self._draw_sports_card(frame, det, extra)

    def _draw_traffic_card(self, frame, det, extra):
        if det.domain_label not in ["car", "bus", "truck", "motorcycle"]:
            return
        lines = []
        if extra.get("proximity"):
            lines.append(f"DIST: {extra['proximity']}")
        if extra.get("est_speed"):
            lines.append(f"SPD : {extra['est_speed']}")
        self._draw_info_card(frame, det.bbox, lines, self.config.get_color("accent"))

    def _draw_retail_card(self, frame, det, extra):
        lines = []
        if extra.get("price"):
            lines.append(f"PRICE: {extra['price']}")
        if extra.get("category"):
            lines.append(extra["category"])
        if lines:
            self._draw_info_card(
                frame, det.bbox, lines, self.config.get_color("accent")
            )

    def _draw_safety_card(self, frame, det, extra):
        risk = extra.get("risk", "")
        if not risk:
            return
        risk_colors = {
            "HIGH": (0, 0, 220),
            "MEDIUM": (0, 165, 255),
            "LOW": (50, 200, 50),
        }
        color = risk_colors.get(risk, (200, 200, 200))
        lines = [f"RISK: {risk}"]
        if extra.get("hazard"):
            lines.append("!! HAZARD DETECTED")
        self._draw_info_card(frame, det.bbox, lines, color)

    def _draw_sports_card(self, frame, det, extra):
        role = extra.get("role", "")
        if not role:
            return
        lines = [f"ROLE: {role}"]
        self._draw_info_card(frame, det.bbox, lines, self.config.get_color("accent"))

    def _draw_info_card(self, frame, bbox, lines, color):
        if not lines:
            return
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.config.font_scale * 0.75
        ft = 1
        pad = 5
        line_h = 18
        card_w = 160
        card_h = len(lines) * line_h + pad * 2

        # Position: bottom-right of box, clamped to frame
        cx = min(x2 + 8, w - card_w - 4)
        cy = min(y2, h - card_h - 4)
        cx = max(cx, x1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), color, 1)

        for i, line in enumerate(lines):
            ty = cy + pad + (i + 1) * line_h - 4
            cv2.putText(frame, line, (cx + pad, ty), font, fs, color, ft, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Motion Trail                                                        #
    # ------------------------------------------------------------------ #
    def _draw_trail(self, frame: np.ndarray, trail, det: Detection):
        if len(trail) < 2:
            return
        color = self._get_detection_color(det)

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = max(1, int(alpha * 3))
            faded = tuple(int(c * alpha * 0.7) for c in color)
            cv2.line(frame, trail[i - 1], trail[i], faded, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Alert Overlay                                                       #
    # ------------------------------------------------------------------ #
    def _draw_alert_overlay(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        # Pulsing effect based on frame counter
        pulse = abs(math.sin(self._frame_counter * 0.15))
        intensity = int(pulse * 180)
        alert_color = (0, 0, intensity + 75)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), alert_color, 4)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # "ALERT" tag
        cv2.putText(
            frame,
            "! ALERT",
            (x1 + 4, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _get_detection_color(self, det: Detection) -> Tuple[int, int, int]:
        domain = self.config.domain

        if det.is_alert or det.extra.get("hazard"):
            return self.config.get_color("alert")

        from config.settings import Domain

        if domain == Domain.TRAFFIC:
            if det.domain_label == "person":
                return self.config.get_color("safe")
            return self.config.get_color("primary")
        elif domain == Domain.RETAIL:
            if det.domain_label == "person":
                return self.config.get_color("safe")
            return self.config.get_color("primary")
        elif domain == Domain.SAFETY:
            risk = det.extra.get("risk", "LOW")
            if risk == "HIGH":
                return self.config.get_color("alert")
            elif risk == "MEDIUM":
                return self.config.get_color("accent")
            return self.config.get_color("safe")
        elif domain == Domain.SPORTS:
            role = det.extra.get("role", "")
            if role == "BALL":
                return self.config.get_color("alert")
            elif role == "PLAYER":
                return self.config.get_color("primary")
            return self.config.get_color("neutral")

        return self.config.get_color("primary")
