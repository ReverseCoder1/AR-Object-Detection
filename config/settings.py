"""
Application Configuration
==========================
Central configuration management for domain settings,
model parameters, AR visual styles, and runtime options.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class Domain(Enum):
    TRAFFIC = "traffic"
    RETAIL = "retail"
    SAFETY = "safety"
    SPORTS = "sports"


# COCO class IDs relevant to each domain
DOMAIN_CLASSES: Dict[Domain, Dict[str, List[int]]] = {
    Domain.TRAFFIC: {
        "car": [2],
        "motorcycle": [3],
        "bus": [5],
        "truck": [7],
        "person": [0],
        "traffic_light": [9],
        "stop_sign": [11],
        "bicycle": [1],
    },
    Domain.RETAIL: {
        "person": [0],
        "bottle": [39],
        "cup": [41],
        "bowl": [45],
        "banana": [46],
        "apple": [47],
        "backpack": [24],
        "handbag": [26],
        "cell_phone": [67],
        "laptop": [63],
        "book": [73],
        "chair": [56],
        "potted_plant": [58],
    },
    Domain.SAFETY: {
        "person": [0],
        "car": [2],
        "motorcycle": [3],
        "bus": [5],
        "truck": [7],
        "fire_hydrant": [10],
        "scissors": [76],
        "knife": [43],
        "bicycle": [1],
    },
    Domain.SPORTS: {
        "person": [0],
        "sports_ball": [32],
        "frisbee": [29],
        "skis": [30],
        "snowboard": [31],
        "baseball_bat": [34],
        "baseball_glove": [35],
        "tennis_racket": [38],
        "bottle": [39],
        "umbrella": [25],
    },
}

# AR color themes per domain (BGR format for OpenCV)
DOMAIN_COLORS: Dict[Domain, Dict[str, Tuple[int, int, int]]] = {
    Domain.TRAFFIC: {
        "primary": (0, 220, 255),  # cyan
        "alert": (0, 0, 255),  # red
        "safe": (0, 255, 100),  # green
        "neutral": (200, 200, 200),
        "accent": (0, 165, 255),  # orange
        "hud_bg": (20, 20, 40),
        "hud_text": (0, 220, 255),
    },
    Domain.RETAIL: {
        "primary": (30, 180, 255),  # gold
        "alert": (0, 100, 255),
        "safe": (0, 220, 150),
        "neutral": (200, 200, 200),
        "accent": (200, 100, 255),  # purple
        "hud_bg": (20, 10, 40),
        "hud_text": (30, 180, 255),
    },
    Domain.SAFETY: {
        "primary": (0, 200, 255),  # amber
        "alert": (0, 0, 220),  # red alert
        "safe": (50, 220, 50),
        "neutral": (200, 200, 200),
        "accent": (0, 165, 255),
        "hud_bg": (20, 10, 10),
        "hud_text": (0, 200, 255),
    },
    Domain.SPORTS: {
        "primary": (80, 255, 80),  # lime green
        "alert": (80, 80, 255),
        "safe": (100, 200, 255),
        "neutral": (200, 200, 200),
        "accent": (50, 200, 255),
        "hud_bg": (10, 20, 10),
        "hud_text": (80, 255, 80),
    },
}

# Domain-specific metadata
DOMAIN_META: Dict[Domain, Dict] = {
    Domain.TRAFFIC: {
        "title": "Traffic Monitoring System",
        "subtitle": "Real-time vehicle & pedestrian detection",
        "icon": "TF",
        "alert_classes": ["person"],
        "count_classes": ["car", "bus", "truck", "motorcycle", "bicycle", "person"],
    },
    Domain.RETAIL: {
        "title": "Smart Retail Analytics",
        "subtitle": "Shopper & product detection",
        "icon": "RT",
        "alert_classes": [],
        "count_classes": ["person", "bottle", "cup", "laptop", "cell_phone"],
        "price_map": {
            "bottle": "$2.99",
            "cup": "$4.99",
            "bowl": "$12.99",
            "banana": "$0.59/lb",
            "apple": "$1.29/lb",
            "laptop": "$899.99",
            "cell_phone": "$699.99",
            "book": "$19.99",
            "backpack": "$49.99",
            "handbag": "$89.99",
        },
    },
    Domain.SAFETY: {
        "title": "Industrial Safety Monitor",
        "subtitle": "Hazard & compliance detection",
        "icon": "SF",
        "alert_classes": ["person", "knife", "scissors"],
        "count_classes": ["person"],
        "hazard_classes": ["knife", "scissors"],
        "zone_warnings": True,
    },
    Domain.SPORTS: {
        "title": "Sports Analytics System",
        "subtitle": "Player & equipment tracking",
        "icon": "SP",
        "alert_classes": ["sports_ball"],
        "count_classes": ["person", "sports_ball"],
        "track_trajectories": True,
    },
}


@dataclass
class AppConfig:
    # Domain
    domain: Domain = Domain.TRAFFIC

    # Model
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    device: str = "cpu"

    # Video
    frame_width: int = 1280
    frame_height: int = 720
    fps_target: int = 30

    # AR Visual
    box_thickness: int = 2
    font_scale: float = 0.52
    font_thickness: int = 1
    show_confidence: bool = True
    show_fps: bool = True
    show_count: bool = True
    show_trails: bool = True
    trail_length: int = 40
    overlay_alpha: float = 0.70

    # Output
    save_output: bool = False
    output_dir: str = "output"
    display: bool = True

    # Tracking
    enable_tracking: bool = True
    max_track_age: int = 30

    # Computed (auto-filled)
    domain_classes: Dict = field(default_factory=dict)
    domain_colors: Dict = field(default_factory=dict)
    domain_meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.domain_classes = DOMAIN_CLASSES.get(self.domain, {})
        self.domain_colors = DOMAIN_COLORS.get(
            self.domain, DOMAIN_COLORS[Domain.TRAFFIC]
        )
        self.domain_meta = DOMAIN_META.get(self.domain, {})

    def get_all_class_ids(self) -> List[int]:
        ids = []
        for id_list in self.domain_classes.values():
            ids.extend(id_list)
        return list(set(ids))

    def get_color(self, key: str) -> Tuple[int, int, int]:
        return self.domain_colors.get(key, (200, 200, 200))

    def class_name_from_id(self, class_id: int) -> str:
        for name, ids in self.domain_classes.items():
            if class_id in ids:
                return name
        return "object"
