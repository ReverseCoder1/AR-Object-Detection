"""
Trail Tracker
=============
Maintains movement trails (history of center points) per tracked object.
Used by the AR renderer to draw motion trails.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

from config.settings import AppConfig
from core.detector import Detection


class TrailTracker:
    """
    Stores center-point history for each tracked object (by track_id).
    Falls back to a simple nearest-center matcher when track IDs are absent.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.max_len = config.trail_length
        # track_id -> deque of (cx, cy) points
        self._trails: Dict[int, deque] = {}
        self._next_id = 1000  # fallback IDs when tracker not active
        self._prev_centers: List[Tuple[int, int]] = []

    def update(self, detections: List[Detection]):
        """Update trails with new detections."""
        if not self.config.show_trails:
            return

        current_ids = set()

        for det in detections:
            tid = det.track_id

            # Fallback: assign pseudo-IDs by nearest previous center
            if tid is None:
                tid = self._match_or_new(det.center)

            current_ids.add(tid)

            if tid not in self._trails:
                self._trails[tid] = deque(maxlen=self.max_len)

            self._trails[tid].append(det.center)

        # Remove stale tracks (not seen this frame — age them out)
        stale = [k for k in list(self._trails.keys()) if k not in current_ids]
        for k in stale:
            # Gradually shrink instead of immediate delete
            if len(self._trails[k]) > 0:
                self._trails[k].popleft()
            if len(self._trails[k]) == 0:
                del self._trails[k]

    def get_trail(self, track_id: int) -> List[Tuple[int, int]]:
        """Return list of (cx, cy) points for a track."""
        if track_id in self._trails:
            return list(self._trails[track_id])
        return []

    def _match_or_new(self, center: Tuple[int, int]) -> int:
        """Match center to nearest previous center or assign new ID."""
        cx, cy = center
        best_id = None
        best_dist = float("inf")

        for tid, trail in self._trails.items():
            if len(trail) == 0:
                continue
            px, py = trail[-1]
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist and dist < 80:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1

        return best_id
