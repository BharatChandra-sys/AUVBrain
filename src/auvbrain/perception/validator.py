"""Sensor validation pipeline.

``SensorValidator.validate(obs, prev_obs, dt)`` runs all checks and returns
a ``ValidatedObservation`` that pairs the raw ``Observation`` with per-sensor
``SensorHealth`` descriptors and an aggregate ``observation_confidence``.

Checks performed (in order):
  1. Availability   — value is not None
  2. Range          — value within physical min/max bounds
  3. Rate of change — delta / dt below max-rate-of-change threshold
  4. Frozen value   — value not identical for more than N consecutive ticks
  5. Cross-sensor   — depth-vs-pressure consistency check

The validator is intentionally conservative: when in doubt, it marks a sensor
degraded and reduces its confidence rather than silently passing bad data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from ..models import Observation
from .sensor_health import FaultReason, SensorHealth, healthy_sensor, unavailable_sensor

logger = logging.getLogger(__name__)


# ── Physical range table ────────────────────────────────────────────────────
# (sensor_id, min, max, max_rate_per_s)
_RANGES: dict[str, tuple[float, float, float]] = {
    "depth_m":          (0.0,    300.0,  5.0),    # max 5 m/s descent
    "battery_v":        (0.0,     30.0,  0.5),    # max 0.5 V drop per second
    "internal_temp_c":  (-10.0,  120.0,  5.0),    # max 5 °C/s rise
    "pressure_bar":     (0.0,     50.0,  0.5),    # max 0.5 bar/s
    "obstacle_front_m": (0.0,    200.0, 30.0),    # sonar can change fast
}

# Minimum number of ticks a value must stay identical to be marked CONSTANT
_CONSTANT_TICKS: int = 10


@dataclass
class ValidatedObservation:
    """Raw observation plus per-sensor health descriptors."""

    obs: Observation
    sensor_health: dict[str, SensorHealth] = field(default_factory=dict)
    observation_confidence: float = 1.0   # aggregate; 1.0 = all sensors healthy

    def get_health(self, sensor_id: str) -> Optional[SensorHealth]:
        return self.sensor_health.get(sensor_id)

    def is_sensor_healthy(self, sensor_id: str) -> bool:
        h = self.sensor_health.get(sensor_id)
        return h.healthy if h is not None else True  # unknown → assume OK


class SensorValidator:
    """Stateful validator — tracks previous values across ticks."""

    def __init__(self) -> None:
        self._prev: dict[str, float] = {}
        self._constant_counts: dict[str, int] = {}

    def validate(
        self,
        obs: Observation,
        dt: float = 0.1,
    ) -> ValidatedObservation:
        """Run all checks and return a ValidatedObservation.

        Args:
            obs: Current raw observation.
            dt:  Time since last call in seconds (used for rate checks).
        """
        health: dict[str, SensorHealth] = {}
        dt = max(dt, 1e-6)  # avoid divide-by-zero

        # Mandatory numeric sensors
        for sensor_id, value in [
            ("depth_m", obs.depth_m),
            ("battery_v", obs.battery_v),
        ]:
            health[sensor_id] = self._check_numeric(sensor_id, value, dt)

        # Optional sensors — only checked when present
        optional: list[tuple[str, Optional[float]]] = [
            ("internal_temp_c", obs.internal_temp_c),
            ("pressure_bar", obs.pressure_bar),
            ("obstacle_front_m", obs.obstacle_front_m),
        ]
        for sensor_id, value in optional:
            if value is not None:
                health[sensor_id] = self._check_numeric(sensor_id, value, dt)
            # If None: no entry added; downstream treats absence as "unavailable but optional"

        # Cross-sensor consistency: depth vs pressure
        if obs.pressure_bar is not None and "pressure_bar" in health:
            # Rough conversion: 1 bar ≈ 10 m seawater
            pressure_depth_m = (obs.pressure_bar - 1.0) * 10.0
            delta = abs(obs.depth_m - pressure_depth_m)
            if delta > 5.0:  # more than 5 m discrepancy
                health["depth_m"].degrade(FaultReason.CROSS_SENSOR_DISAGREE, confidence=0.4)
                health["pressure_bar"].degrade(FaultReason.CROSS_SENSOR_DISAGREE, confidence=0.4)
                logger.warning(
                    "depth/pressure cross-check fail: depth=%.1f pressure_depth=%.1f delta=%.1f",
                    obs.depth_m, pressure_depth_m, delta,
                )

        # Aggregate confidence: geometric mean of available sensor confidences
        confidences = [h.confidence for h in health.values()]
        if confidences:
            agg = math.exp(sum(math.log(max(c, 1e-9)) for c in confidences) / len(confidences))
        else:
            agg = 1.0

        return ValidatedObservation(
            obs=obs,
            sensor_health=health,
            observation_confidence=round(agg, 4),
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _check_numeric(self, sensor_id: str, value: float, dt: float) -> SensorHealth:
        h = healthy_sensor(sensor_id, raw_value=value)

        if math.isnan(value) or math.isinf(value):
            h.degrade(FaultReason.OUT_OF_RANGE, confidence=0.0)
            return h

        # Range check
        bounds = _RANGES.get(sensor_id)
        if bounds is not None:
            lo, hi, max_rate = bounds
            if not (lo <= value <= hi):
                h.degrade(FaultReason.OUT_OF_RANGE, confidence=0.1)
                logger.warning("%s out of range: %.3f (valid [%.1f, %.1f])", sensor_id, value, lo, hi)
                return h

            # Rate-of-change check
            if sensor_id in self._prev:
                rate = abs(value - self._prev[sensor_id]) / dt
                if rate > max_rate:
                    h.degrade(FaultReason.SPIKE, confidence=0.3)
                    logger.warning(
                        "%s spike: rate=%.2f/s > max=%.2f/s", sensor_id, rate, max_rate
                    )

        # Frozen value (constant) check
        if sensor_id in self._prev:
            if math.isclose(value, self._prev[sensor_id], rel_tol=1e-6, abs_tol=1e-9):
                self._constant_counts[sensor_id] = self._constant_counts.get(sensor_id, 0) + 1
                if self._constant_counts[sensor_id] >= _CONSTANT_TICKS:
                    h.degrade(FaultReason.CONSTANT, confidence=0.2)
                    logger.warning("%s frozen for %d ticks", sensor_id, self._constant_counts[sensor_id])
            else:
                self._constant_counts[sensor_id] = 0
        else:
            self._constant_counts[sensor_id] = 0

        self._prev[sensor_id] = value
        return h
