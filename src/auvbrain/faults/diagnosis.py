"""Fault diagnosis — isolates root cause and assigns severity.

The diagnoser takes the raw fault list from ``FaultDetector`` and applies
domain knowledge to:
  1. Deduplicate / merge related faults.
  2. Escalate severity when multiple related faults appear together.
  3. Produce a single ``RecoveryStrategy`` for the mission to act on.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .models import Fault, FaultSeverity, FaultType
from .recovery import RecoveryPlanner, RecoveryStrategy

logger = logging.getLogger(__name__)


class FaultDiagnosis:
    """Isolates root cause, escalates severity, selects recovery strategy."""

    def __init__(self) -> None:
        self._planner = RecoveryPlanner()

    def diagnose(self, faults: list[Fault]) -> Optional[RecoveryStrategy]:
        """Diagnose a fault list and return a recovery strategy (or None if clean).

        Side-effects: may mutate fault severity in-place for escalation.
        """
        if not faults:
            return None

        faults = self._escalate(faults)
        strategy = self._planner.plan(faults)
        return strategy

    # ── Internal ─────────────────────────────────────────────────────────

    def _escalate(self, faults: list[Fault]) -> list[Fault]:
        """Escalate severity when multiple faults affect the same subsystem."""
        subsystem_faults: dict[str, list[Fault]] = defaultdict(list)
        for f in faults:
            subsystem_faults[f.subsystem].append(f)

        result: list[Fault] = []
        for subsystem, group in subsystem_faults.items():
            high_or_above = sum(
                1 for f in group
                if f.severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL)
            )
            # If 2+ high faults in same subsystem → escalate the most severe
            if high_or_above >= 2:
                for f in group:
                    if f.severity == FaultSeverity.HIGH:
                        # Escalate to CRITICAL via dataclass replace
                        import dataclasses
                        f = dataclasses.replace(
                            f,
                            severity=FaultSeverity.CRITICAL,
                            description=f"{f.description} [escalated: multiple failures in {subsystem}]",
                        )
                        logger.warning(
                            "fault escalated to CRITICAL: subsystem=%s component=%s",
                            subsystem, f.component,
                        )
                result.extend(group)
            else:
                result.extend(group)

        return result
