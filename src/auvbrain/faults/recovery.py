"""Recovery strategies and the recovery planner.

``RecoveryStrategy`` defines what the vehicle should do when a fault is
diagnosed.  ``RecoveryPlanner.plan()`` takes the current fault list and
returns the highest-priority strategy the mission should adopt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .models import Fault, FaultSeverity, FaultType

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """What the vehicle should do in response to one or more faults."""
    CONTINUE                 = "continue"
    REDUCE_SPEED             = "reduce_speed"
    RECONFIGURE_ESTIMATOR    = "reconfigure_estimator"   # reduce trust in bad sensor
    REPLAN_ROUTE             = "replan_route"            # find a safer path
    HOLD_POSITION            = "hold_position"           # hover and assess
    ASCEND_TO_PERISCOPE      = "ascend_to_periscope"     # get comms/GPS fix
    ABORT_MISSION_AND_RETURN = "abort_mission_and_return"
    EMERGENCY_SURFACE        = "emergency_surface"


@dataclass
class RecoveryStrategy:
    """Chosen recovery response with context.

    Attributes:
        action:      What to do.
        priority:    Lower number = higher urgency (0 = immediate).
        rationale:   Human-readable explanation.
        faults:      The fault(s) that triggered this strategy.
    """
    action: RecoveryAction
    priority: int
    rationale: str
    faults: list[Fault]


# ── Fault → strategy mapping table ───────────────────────────────────────────
# (fault_type, severity) → (action, priority, rationale_template)
_STRATEGY_TABLE: list[tuple[FaultType, FaultSeverity, RecoveryAction, int, str]] = [
    # Water ingress is always emergency
    (FaultType.WATER_INGRESS,         FaultSeverity.CRITICAL, RecoveryAction.EMERGENCY_SURFACE,        0, "water ingress → surface immediately"),

    # Navigation divergence
    (FaultType.NAVIGATION_DIVERGED,   FaultSeverity.HIGH,     RecoveryAction.HOLD_POSITION,            1, "nav diverged → hold and reconfigure"),
    (FaultType.NAVIGATION_DIVERGED,   FaultSeverity.CRITICAL, RecoveryAction.ASCEND_TO_PERISCOPE,      1, "nav diverged critically → get GPS fix"),

    # Sensor faults
    (FaultType.SENSOR_DROPOUT,        FaultSeverity.HIGH,     RecoveryAction.RECONFIGURE_ESTIMATOR,    2, "sensor dropout → reduce sensor trust"),
    (FaultType.SENSOR_DROPOUT,        FaultSeverity.CRITICAL, RecoveryAction.ABORT_MISSION_AND_RETURN, 1, "critical sensor dropout → abort"),
    (FaultType.SENSOR_FROZEN,         FaultSeverity.MEDIUM,   RecoveryAction.RECONFIGURE_ESTIMATOR,    3, "frozen sensor → exclude from fusion"),
    (FaultType.SENSOR_DISAGREEMENT,   FaultSeverity.MEDIUM,   RecoveryAction.RECONFIGURE_ESTIMATOR,    3, "sensor conflict → reduce trust"),
    (FaultType.SENSOR_BIAS,           FaultSeverity.HIGH,     RecoveryAction.RECONFIGURE_ESTIMATOR,    2, "sensor bias → recalibrate / exclude"),

    # Energy
    (FaultType.ENERGY_CRITICAL,       FaultSeverity.CRITICAL, RecoveryAction.EMERGENCY_SURFACE,        0, "energy critical → surface now"),

    # Thruster
    (FaultType.THRUSTER_DEGRADED,     FaultSeverity.MEDIUM,   RecoveryAction.REDUCE_SPEED,             3, "thruster degraded → reduce speed"),
    (FaultType.THRUSTER_DEGRADED,     FaultSeverity.HIGH,     RecoveryAction.REPLAN_ROUTE,             2, "thruster degraded → replan route"),
    (FaultType.THRUSTER_UNRESPONSIVE, FaultSeverity.HIGH,     RecoveryAction.ABORT_MISSION_AND_RETURN, 1, "thruster unresponsive → abort"),
]


class RecoveryPlanner:
    """Select the most appropriate recovery strategy for the current fault set."""

    def plan(self, faults: list[Fault]) -> Optional[RecoveryStrategy]:
        """Return the highest-priority (lowest priority number) strategy, or None.

        Returns None when there are no faults or all are LOW severity with
        no matching strategy.
        """
        if not faults:
            return None

        candidates: list[RecoveryStrategy] = []

        for fault in faults:
            for (ft, sev, action, priority, rationale) in _STRATEGY_TABLE:
                if fault.fault_type == ft and fault.severity == sev:
                    candidates.append(RecoveryStrategy(
                        action=action,
                        priority=priority,
                        rationale=rationale,
                        faults=[fault],
                    ))

        if not candidates:
            # Fallback: if any HIGH/CRITICAL fault has no specific match
            high_or_critical = [
                f for f in faults
                if f.severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL)
            ]
            if high_or_critical:
                return RecoveryStrategy(
                    action=RecoveryAction.HOLD_POSITION,
                    priority=2,
                    rationale="unclassified high/critical fault → hold and assess",
                    faults=high_or_critical,
                )
            return None

        # Return the highest-priority (lowest number) candidate
        best = min(candidates, key=lambda s: s.priority)
        logger.info(
            "recovery strategy selected: action=%s priority=%d rationale=%r",
            best.action.value, best.priority, best.rationale,
        )
        return best
