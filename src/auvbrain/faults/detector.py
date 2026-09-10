"""Fault detector — transforms sensor health + estimated state into Fault objects.

Detection is a separate step from diagnosis.  The detector answers:
    "What is wrong right now?"

The diagnoser answers:
    "What caused it, how severe is it, and what should we do?"

Detectors are stateless functions over the current tick's data.  Adding a new
detector is as simple as adding a function to ``DETECTORS`` at the bottom.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..models import Observation
from ..navigation.estimated_state import EstimatedState
from ..perception.sensor_health import FaultReason, SensorHealth
from ..perception.validator import ValidatedObservation
from .models import Fault, FaultSeverity, FaultType

logger = logging.getLogger(__name__)

# Type alias for a detector function
DetectorFn = Callable[
    [ValidatedObservation, EstimatedState],
    list[Fault],
]


def _detect_sensor_faults(
    validated: ValidatedObservation,
    _state: EstimatedState,
) -> list[Fault]:
    """Raise a fault for each degraded sensor channel."""
    faults: list[Fault] = []
    for sid, health in validated.sensor_health.items():
        if health.healthy:
            continue

        reason = health.fault_reason
        if reason == FaultReason.UNAVAILABLE:
            ftype, sev = FaultType.SENSOR_DROPOUT, FaultSeverity.HIGH
        elif reason == FaultReason.SPIKE:
            ftype, sev = FaultType.SENSOR_SPIKE, FaultSeverity.MEDIUM
        elif reason == FaultReason.CONSTANT:
            ftype, sev = FaultType.SENSOR_FROZEN, FaultSeverity.MEDIUM
        elif reason == FaultReason.OUT_OF_RANGE:
            ftype, sev = FaultType.SENSOR_BIAS, FaultSeverity.HIGH
        elif reason == FaultReason.CROSS_SENSOR_DISAGREE:
            ftype, sev = FaultType.SENSOR_DISAGREEMENT, FaultSeverity.MEDIUM
        else:
            ftype, sev = FaultType.UNKNOWN, FaultSeverity.LOW

        faults.append(Fault(
            subsystem="perception",
            component=sid,
            fault_type=ftype,
            severity=sev,
            confidence=max(0.5, 1.0 - health.confidence),
            description=f"{sid} degraded: {reason.value}  confidence={health.confidence:.2f}",
        ))
        logger.warning("fault detected: subsystem=perception component=%s type=%s", sid, ftype.value)

    return faults


def _detect_navigation_divergence(
    validated: ValidatedObservation,
    state: EstimatedState,
) -> list[Fault]:
    """Flag when navigation confidence drops below a usable threshold."""
    faults: list[Fault] = []
    if state.confidence < 0.35:
        faults.append(Fault(
            subsystem="navigation",
            component="estimator",
            fault_type=FaultType.NAVIGATION_DIVERGED,
            severity=FaultSeverity.HIGH,
            confidence=0.9,
            description=f"navigation confidence={state.confidence:.2f} below 0.35 threshold",
        ))
        logger.warning("navigation diverged: confidence=%.2f", state.confidence)
    return faults


def _detect_water_ingress(
    validated: ValidatedObservation,
    _state: EstimatedState,
) -> list[Fault]:
    faults: list[Fault] = []
    if validated.obs.water_ingress is True:
        faults.append(Fault(
            subsystem="hull",
            component="watertight_integrity",
            fault_type=FaultType.WATER_INGRESS,
            severity=FaultSeverity.CRITICAL,
            confidence=1.0,
            description="water ingress detected",
        ))
    return faults


# ── Registry ─────────────────────────────────────────────────────────────────

DETECTORS: list[DetectorFn] = [
    _detect_sensor_faults,
    _detect_navigation_divergence,
    _detect_water_ingress,
]


class FaultDetector:
    """Run all registered detectors and return a consolidated fault list."""

    def detect(
        self,
        validated: ValidatedObservation,
        state: EstimatedState,
    ) -> list[Fault]:
        faults: list[Fault] = []
        for detector in DETECTORS:
            try:
                faults.extend(detector(validated, state))
            except Exception as exc:
                logger.error("fault detector %s raised: %s", detector.__name__, exc, exc_info=True)
        return faults
