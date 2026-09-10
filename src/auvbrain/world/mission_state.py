"""Mission state machine.

``MissionPhase`` represents where the AUV is in its dive profile.
``MissionContext`` carries the full context needed by planners and the
agentic layer to make decisions with a mission purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class MissionPhase(str, Enum):
    # Normal dive phases
    SURFACE  = "SURFACE"
    DIVE     = "DIVE"
    TRANSIT  = "TRANSIT"
    SURVEY   = "SURVEY"
    INSPECT  = "INSPECT"
    RETURN   = "RETURN"
    ASCENT   = "ASCENT"

    # Cross-cutting override states
    DEGRADED = "DEGRADED"   # reduced capability but continuing
    RECOVERY = "RECOVERY"   # active fault recovery
    ABORT    = "ABORT"      # mission aborted; heading home
    SAFE     = "SAFE"       # hardware SAFE state


@dataclass(frozen=True, slots=True)
class Waypoint:
    x_m: float = 0.0
    y_m: float = 0.0
    depth_m: float = 0.0
    label: Optional[str] = None


@dataclass
class MissionContext:
    """Active mission context shared by planners and the agentic layer.

    Attributes:
        phase:        Current mission phase.
        goal:         Human-readable mission objective.
        waypoints:    Ordered list of target waypoints.
        start_time:   UTC time mission started.
        max_depth_m:  Mission-specific depth constraint (may be tighter than safety limit).
        min_battery_fraction: Mission abort threshold (0–1).
        notes:        Free-form mission notes / operator instructions.
    """

    phase: MissionPhase = MissionPhase.SURFACE
    goal: str = "no mission loaded"
    waypoints: list[Waypoint] = field(default_factory=list)
    current_waypoint_idx: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_depth_m: float = 100.0
    min_battery_fraction: float = 0.20
    notes: str = ""

    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None

    @property
    def mission_complete(self) -> bool:
        return self.current_waypoint_idx >= len(self.waypoints)

    def advance_waypoint(self) -> None:
        self.current_waypoint_idx += 1

    def transition(self, phase: MissionPhase) -> None:
        self.phase = phase

    def elapsed_s(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
