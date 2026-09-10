"""Obstacle map — time-decaying spatial obstacle registry.

Each obstacle has a confidence that decays over time (TTL-based).  Expired
obstacles are pruned lazily on insert or query.  This prevents the world model
from retaining stale detections indefinitely.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Obstacle:
    """A single detected obstacle in the vehicle frame."""

    distance_m: float              # range from vehicle
    bearing_deg: float = 0.0       # relative bearing (0 = dead ahead)
    depth_m: Optional[float] = None
    confidence: float = 1.0
    source: str = "sonar"          # sensor that detected it
    detected_at: float = field(default_factory=time.monotonic)
    ttl_s: float = 30.0            # confidence valid for this many seconds

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.detected_at

    @property
    def effective_confidence(self) -> float:
        """Confidence decayed linearly over TTL."""
        fraction_remaining = max(0.0, 1.0 - self.age_s / max(self.ttl_s, 1e-6))
        return self.confidence * fraction_remaining

    @property
    def expired(self) -> bool:
        return self.age_s >= self.ttl_s


class ObstacleMap:
    """Maintains a set of known obstacles with temporal decay.

    Usage::

        om = ObstacleMap(max_range_m=50.0)
        om.update(Obstacle(distance_m=3.5, bearing_deg=10.0))
        nearest = om.nearest_obstacle()
    """

    def __init__(self, max_range_m: float = 50.0) -> None:
        self._obstacles: list[Obstacle] = []
        self.max_range_m = max_range_m

    def update(self, obstacle: Obstacle) -> None:
        """Add or refresh an obstacle reading."""
        self._prune()
        self._obstacles.append(obstacle)

    def nearest_obstacle(self) -> Optional[Obstacle]:
        """Return the closest non-expired obstacle, or None."""
        self._prune()
        if not self._obstacles:
            return None
        return min(self._obstacles, key=lambda o: o.distance_m)

    def all_obstacles(self) -> list[Obstacle]:
        self._prune()
        return list(self._obstacles)

    def clear(self) -> None:
        self._obstacles.clear()

    def _prune(self) -> None:
        self._obstacles = [o for o in self._obstacles if not o.expired]
