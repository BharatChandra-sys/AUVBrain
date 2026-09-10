"""Underwater current and environment model.

The ``CurrentModel`` tracks the estimated water current vector and its
confidence.  It compares the planned trajectory against actual trajectory
to detect and estimate drift, feeding corrections back into planning.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CurrentVector:
    """3-D current velocity in m/s (body or NED frame)."""
    north_ms: float = 0.0
    east_ms: float = 0.0
    down_ms: float = 0.0
    confidence: float = 0.5

    @property
    def magnitude_ms(self) -> float:
        return math.sqrt(self.north_ms ** 2 + self.east_ms ** 2 + self.down_ms ** 2)

    @property
    def bearing_deg(self) -> float:
        """Horizontal bearing the current is flowing toward (0 = North)."""
        return math.degrees(math.atan2(self.east_ms, self.north_ms)) % 360.0


@dataclass
class CurrentModel:
    """Maintains a running estimate of the water current.

    On each update, the model computes the drift (planned vs actual position
    delta) and uses an exponential moving average to track current magnitude
    and direction.

    Args:
        alpha: EMA smoothing factor (0 = never update, 1 = instant update).
        staleness_s: Seconds after which confidence degrades.
    """

    alpha: float = 0.15
    staleness_s: float = 60.0

    _current: CurrentVector = field(default_factory=CurrentVector)
    _last_update_ts: float = field(default_factory=time.monotonic)
    _drift_history: list[tuple[float, float]] = field(default_factory=list)  # (north, east) deltas
    _MAX_HISTORY: int = 20

    @property
    def current(self) -> CurrentVector:
        self._decay_confidence()
        return self._current

    def update_from_drift(
        self,
        planned_north_m: float,
        planned_east_m: float,
        actual_north_m: float,
        actual_east_m: float,
        dt: float,
    ) -> None:
        """Update current estimate from planned vs actual position delta.

        Args:
            planned_north_m / planned_east_m: Where the vehicle expected to be.
            actual_north_m / actual_east_m:   Where it actually is.
            dt: Time interval in seconds.
        """
        if dt <= 0:
            return
        drift_north = (actual_north_m - planned_north_m) / dt
        drift_east  = (actual_east_m  - planned_east_m)  / dt

        self._drift_history.append((drift_north, drift_east))
        if len(self._drift_history) > self._MAX_HISTORY:
            self._drift_history.pop(0)

        # EMA of the current estimate
        self._current.north_ms = (
            (1 - self.alpha) * self._current.north_ms + self.alpha * drift_north
        )
        self._current.east_ms = (
            (1 - self.alpha) * self._current.east_ms + self.alpha * drift_east
        )
        # Build confidence from consistency of recent drift measurements
        self._current.confidence = min(1.0, len(self._drift_history) / 10.0)
        self._last_update_ts = time.monotonic()

    def reset(self) -> None:
        self._current = CurrentVector()
        self._drift_history.clear()

    def _decay_confidence(self) -> None:
        age_s = time.monotonic() - self._last_update_ts
        if age_s > self.staleness_s:
            decay = max(0.0, 1.0 - (age_s - self.staleness_s) / self.staleness_s)
            self._current.confidence = self._current.confidence * decay
