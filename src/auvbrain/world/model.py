"""WorldState — the coherent representation consumed by planners and agents.

Architecture:
    Sensors → EstimatedState → WorldState → everything else

``WorldState`` is assembled each autonomy cycle and passed to the
tactical planner, action shield, energy planner, and the agentic layer.
It is the single source of truth for all decision-making above the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..faults.models import Fault
from ..navigation.estimated_state import EstimatedState
from ..planning.energy import EnergyState
from .environment import CurrentModel, CurrentVector
from .mission_state import MissionContext, MissionPhase
from .obstacles import Obstacle, ObstacleMap


@dataclass
class WorldState:
    """The complete world state assembled each tick.

    Attributes:
        navigation:      Best current state estimate (position, velocity, heading).
        obstacles:       Known obstacles with confidence/TTL.
        current_model:   Estimated water current.
        energy:          Energy state and mission feasibility.
        faults:          Currently active faults (from FaultDetector).
        mission:         Active mission context (phase, goal, waypoints).
        observation_confidence: Aggregate sensor quality [0, 1].
        tick:            Agent loop tick counter (for tracing).
    """

    navigation: EstimatedState = field(default_factory=EstimatedState)
    obstacles: ObstacleMap = field(default_factory=ObstacleMap)
    current_model: CurrentModel = field(default_factory=CurrentModel)
    energy: EnergyState = field(default_factory=lambda: EnergyState())
    faults: list[Fault] = field(default_factory=list)
    mission: MissionContext = field(default_factory=MissionContext)
    observation_confidence: float = 1.0
    tick: int = 0

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def mission_phase(self) -> MissionPhase:
        return self.mission.phase

    @property
    def nearest_obstacle(self) -> Optional[Obstacle]:
        return self.obstacles.nearest_obstacle()

    @property
    def current(self) -> CurrentVector:
        return self.current_model.current

    @property
    def has_critical_fault(self) -> bool:
        from ..faults.models import FaultSeverity
        return any(f.severity == FaultSeverity.CRITICAL for f in self.faults)

    @property
    def navigation_trusted(self) -> bool:
        return self.navigation.is_trusted

    def summary(self) -> dict:
        """Compact dict for telemetry / agent context."""
        return {
            "depth_m": round(self.navigation.depth_m, 2),
            "nav_confidence": self.navigation.confidence,
            "obs_confidence": self.observation_confidence,
            "energy_soc": round(self.energy.soc, 3),
            "fault_count": len(self.faults),
            "has_critical_fault": self.has_critical_fault,
            "mission_phase": self.mission.phase.value,
            "nearest_obstacle_m": (
                round(self.nearest_obstacle.distance_m, 2)
                if self.nearest_obstacle else None
            ),
            "current_ms": round(self.current.magnitude_ms, 3),
            "tick": self.tick,
        }


class WorldStateBuilder:
    """Assembles a fresh WorldState from the components updated each tick."""

    def __init__(self) -> None:
        self.obstacle_map = ObstacleMap()
        self.current_model = CurrentModel()
        self.mission = MissionContext()

    def build(
        self,
        navigation: EstimatedState,
        energy: "EnergyState",
        faults: list["Fault"],
        observation_confidence: float,
        tick: int,
    ) -> WorldState:
        # Push latest sonar reading into obstacle map if available
        return WorldState(
            navigation=navigation,
            obstacles=self.obstacle_map,
            current_model=self.current_model,
            energy=energy,
            faults=faults,
            mission=self.mission,
            observation_confidence=observation_confidence,
            tick=tick,
        )
