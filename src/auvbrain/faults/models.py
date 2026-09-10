"""Fault data models.

A ``Fault`` represents a detected anomaly with enough context for the diagnosis
and recovery pipeline to act on it without looking at raw sensor data again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class FaultSeverity(str, Enum):
    LOW      = "low"       # note-worthy; no impact on autonomy
    MEDIUM   = "medium"    # degraded capability; mission may continue
    HIGH     = "high"      # significant; mission should adapt
    CRITICAL = "critical"  # immediate action required


class FaultType(str, Enum):
    SENSOR_DROPOUT        = "sensor_dropout"
    SENSOR_SPIKE          = "sensor_spike"
    SENSOR_BIAS           = "sensor_bias"
    SENSOR_DRIFT          = "sensor_drift"
    SENSOR_FROZEN         = "sensor_frozen"
    SENSOR_DISAGREEMENT   = "sensor_disagreement"
    THRUSTER_DEGRADED     = "thruster_degraded"
    THRUSTER_UNRESPONSIVE = "thruster_unresponsive"
    NAVIGATION_DIVERGED   = "navigation_diverged"
    ENERGY_CRITICAL       = "energy_critical"
    COMMUNICATION_LOSS    = "communication_loss"
    WATER_INGRESS         = "water_ingress"
    UNKNOWN               = "unknown"


@dataclass(slots=True)
class Fault:
    """A detected and classified fault.

    Attributes:
        subsystem:   Top-level subsystem (e.g. "navigation", "propulsion").
        component:   Specific component (e.g. "dvl", "motor_0").
        fault_type:  Classification from ``FaultType``.
        severity:    Impact level from ``FaultSeverity``.
        confidence:  How confident we are in this detection [0, 1].
        description: Human-readable summary.
        detected_at: UTC timestamp of first detection.
    """

    subsystem: str
    component: str
    fault_type: FaultType
    severity: FaultSeverity
    confidence: float = 0.9
    description: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "subsystem": self.subsystem,
            "component": self.component,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
        }
