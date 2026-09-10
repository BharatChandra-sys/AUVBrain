"""Per-sensor health model.

``SensorHealth`` captures the runtime quality of a single sensor channel.
The validator populates one instance per sensor channel each tick and attaches
them to the ``ValidatedObservation`` so every downstream consumer can decide
how much trust to place in a reading without re-doing the checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FaultReason(str, Enum):
    NONE = "none"
    UNAVAILABLE = "unavailable"        # sensor returned None unexpectedly
    STALE = "stale"                     # no update within freshness window
    OUT_OF_RANGE = "out_of_range"       # value outside physical bounds
    SPIKE = "spike"                     # delta too large relative to dt
    CONSTANT = "constant"               # value not changing (frozen sensor)
    CROSS_SENSOR_DISAGREE = "cross_sensor_disagree"  # conflicts with another channel
    IMPOSSIBLE_JUMP = "impossible_jump"  # physically impossible rate of change


@dataclass(slots=True)
class SensorHealth:
    """Quality descriptor for a single sensor channel.

    All boolean flags are True when the check passes (sensor is healthy).
    ``confidence`` is a [0, 1] aggregate; 1.0 means fully trusted.
    """

    sensor_id: str                          # e.g. "depth", "battery", "imu"
    available: bool = True                  # value present (not None)
    fresh: bool = True                      # updated within freshness window
    range_valid: bool = True                # within physical bounds
    rate_valid: bool = True                 # delta reasonable for dt
    consistency: bool = True                # agrees with cross-checks
    confidence: float = 1.0                 # [0, 1] aggregate score
    fault_reason: FaultReason = FaultReason.NONE
    raw_value: Optional[float] = None       # last seen value for logging

    @property
    def healthy(self) -> bool:
        return (
            self.available
            and self.fresh
            and self.range_valid
            and self.rate_valid
            and self.consistency
            and self.confidence >= 0.5
        )

    def degrade(self, reason: FaultReason, confidence: float = 0.0) -> None:
        """Mark the sensor as degraded with a specific fault reason."""
        self.fault_reason = reason
        self.confidence = max(0.0, min(1.0, confidence))
        if reason == FaultReason.UNAVAILABLE:
            self.available = False
        elif reason == FaultReason.STALE:
            self.fresh = False
        elif reason == FaultReason.OUT_OF_RANGE:
            self.range_valid = False
        elif reason in (FaultReason.SPIKE, FaultReason.IMPOSSIBLE_JUMP):
            self.rate_valid = False
        elif reason == FaultReason.CONSTANT:
            self.rate_valid = False
        elif reason == FaultReason.CROSS_SENSOR_DISAGREE:
            self.consistency = False


def healthy_sensor(sensor_id: str, raw_value: Optional[float] = None) -> SensorHealth:
    return SensorHealth(sensor_id=sensor_id, raw_value=raw_value)


def unavailable_sensor(sensor_id: str) -> SensorHealth:
    h = SensorHealth(sensor_id=sensor_id, available=False)
    h.degrade(FaultReason.UNAVAILABLE, confidence=0.0)
    return h
