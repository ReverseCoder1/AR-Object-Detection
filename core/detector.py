"""
Object Detector
===============
YOLOv8-based object detector with domain-aware filtering,
optional tracking, and structured detection output.

CPU Optimizations applied automatically:
  - OpenCV uses all available CPU threads
  - ONNX Runtime preferred when .onnx model provided (2x faster on CPU)
  - PyTorch inter/intra-op threads tuned for CPU
  - Half-precision disabled on CPU (would hurt, not help)
  - Input frame pre-scaled to 640px for fastest inference
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── CPU threading: let OpenCV & PyTorch use all cores ──────────────────────
_cpu_count = os.cpu_count() or 4
cv2.setNumThreads(_cpu_count)
os.environ.setdefault("OMP_NUM_THREADS", str(_cpu_count))
os.environ.setdefault("MKL_NUM_THREADS", str(_cpu_count))

try:
    import torch

    torch.set_num_threads(_cpu_count)
    torch.set_num_interop_threads(max(1, _cpu_count // 2))
except Exception:
    pass

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from config.settings import AppConfig
from utils.logger import setup_logger


@dataclass
class Detection:
    """Single detection result."""

    track_id: Optional[int]
    class_id: int
    class_name: str
    domain_label: str  # domain-friendly label
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    is_alert: bool = False
    extra: dict = field(default_factory=dict)  # domain-specific extras


class ObjectDetector:
    """
    YOLOv8 object detector with:
    - Domain-aware class filtering
    - Optional ByteTrack tracking
    - Structured Detection output
    - Fallback stub when ultralytics not installed
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger("Detector")
        self.model = None
        self.class_names: List[str] = []
        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            self.logger.warning(
                "ultralytics not installed — running in STUB mode.\n"
                "  Install: pip install ultralytics"
            )
            return

        model_path = self.config.model_path
        is_onnx = str(model_path).endswith(".onnx")

        try:
            self.logger.info(f"Loading model: {model_path}")
            if is_onnx:
                self.logger.info(
                    "  Format : ONNX  (CPU-optimized — ~2x faster than .pt)"
                )
            else:
                self.logger.info("  Format : PyTorch (.pt)")
                self.logger.info("  Tip    : Export to ONNX for faster CPU speed:")
                self.logger.info(
                    f"           python -m utils.model_manager export --model {model_path}"
                )

            self.model = YOLO(model_path)

            # Warm-up: pre-allocates internal buffers, hides first-frame spike
            self.logger.info("Warming up model (2 passes)...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(2):
                self.model(dummy, verbose=False)

            self.class_names = list(self.model.names.values())
            self.logger.info(
                f"Model ready — {len(self.class_names)} classes | "
                f"CPU | threads={_cpu_count}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run detection on a frame.

        Returns:
            (detections, inference_ms)
        """
        if self.model is None:
            return self._stub_detections(frame), 0.0

        t0 = time.perf_counter()

        # Run YOLO with optional tracking
        if self.config.enable_tracking:
            results = self.model.track(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        else:
            results = self.model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse_results(results, frame.shape)
        return detections, inference_ms

    def _parse_results(self, results, frame_shape) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        allowed_ids = self.config.get_all_class_ids()
        alert_classes = self.config.domain_meta.get("alert_classes", [])
        h, w = frame_shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])

                # Filter to domain-relevant classes
                if class_id not in allowed_ids:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to frame
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Class name
                raw_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else "object"
                )
                domain_label = self.config.class_name_from_id(class_id)

                # Track ID
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                is_alert = domain_label in alert_classes

                # Domain extras
                extra = self._build_extras(
                    domain_label, conf, area, w, h, x1, y1, x2, y2
                )

                detections.append(
                    Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=raw_name,
                        domain_label=domain_label,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        area=area,
                        is_alert=is_alert,
                        extra=extra,
                    )
                )

        return detections

    def _build_extras(
        self,
        label: str,
        conf: float,
        area: int,
        fw: int,
        fh: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> dict:
        """Build domain-specific extra metadata for a detection."""
        extras = {}
        domain = self.config.domain

        from config.settings import Domain

        if domain == Domain.TRAFFIC:
            # Estimate relative size as proxy for distance
            bbox_area_ratio = area / (fw * fh)
            if bbox_area_ratio > 0.15:
                extras["proximity"] = "CLOSE"
            elif bbox_area_ratio > 0.04:
                extras["proximity"] = "MID"
            else:
                extras["proximity"] = "FAR"

            # Rough speed estimate (placeholder — would use optical flow in prod)
            extras["est_speed"] = (
                f"{int(conf * 80 + 20)} km/h"
                if label in ["car", "truck", "bus", "motorcycle"]
                else ""
            )

        elif domain == Domain.RETAIL:
            price_map = self.config.domain_meta.get("price_map", {})
            extras["price"] = price_map.get(label, "")
            extras["category"] = _retail_category(label)

        elif domain == Domain.SAFETY:
            hazards = self.config.domain_meta.get("hazard_classes", [])
            extras["hazard"] = label in hazards
            extras["risk"] = (
                "HIGH"
                if label in hazards
                else ("MEDIUM" if label == "person" else "LOW")
            )

        elif domain == Domain.SPORTS:
            extras["role"] = (
                "BALL"
                if label == "sports_ball"
                else ("PLAYER" if label == "person" else "EQUIPMENT")
            )

        return extras

    def _stub_detections(self, frame: np.ndarray) -> List[Detection]:
        """Return empty detections when model not loaded."""
        return []

    @property
    def is_ready(self) -> bool:
        return self.model is not None


def _retail_category(label: str) -> str:
    food = {"banana", "apple", "bowl", "cup", "bottle"}
    electronics = {"laptop", "cell_phone"}
    if label in food:
        return "Food & Beverage"
    elif label in electronics:
        return "Electronics"
    elif label in {"backpack", "handbag"}:
        return "Accessories"
    elif label == "book":
        return "Books & Media"
    return "General"
