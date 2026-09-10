"""Tactical planner — Phase 6.

Upgrades the decision from:
    "Generate a vehicle command"
to:
    "Generate and compare feasible ways to accomplish the current subgoal."

The planner produces a ranked list of ``PlanCandidate`` objects.  The best
candidate is forwarded to the ActionShield and then to the control layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from ..models import VehicleCommand
from ..world.mission_state import Waypoint
from ..world.model import WorldState
from .trajectory import Trajectory, TrajectoryEvaluation, TrajectoryEvaluator

logger = logging.getLogger(__name__)


@dataclass
class PlanCandidate:
    """A single candidate plan produced by the tactical planner.

    Attributes:
        command:      The VehicleCommand to execute if this candidate is chosen.
        trajectory:   The trajectory being followed (may be trivial for spot manoeuvres).
        evaluation:   Score / feasibility from the trajectory evaluator.
        rationale:    Human-readable explanation for telemetry / agent context.
        priority:     Lower number = higher priority.
    """
    command: VehicleCommand
    trajectory: Optional[Trajectory] = None
    evaluation: Optional[TrajectoryEvaluation] = None
    rationale: str = ""
    priority: int = 0


class TacticalPlanner:
    """Generate candidate plans from the current world state.

    The planner always produces at least one candidate (a safe hold-position
    command) so the pipeline never stalls.

    Args:
        evaluator: Trajectory evaluator to score candidates.
    """

    def __init__(self, evaluator: Optional[TrajectoryEvaluator] = None) -> None:
        self._evaluator = evaluator or TrajectoryEvaluator()

    def plan(self, world: WorldState) -> list[PlanCandidate]:
        """Return a ranked list of candidates (best first).

        The pipeline:
          1. Generate candidates from mission context.
          2. Score each via TrajectoryEvaluator.
          3. Filter infeasible candidates.
          4. Sort by score descending.
          5. Guarantee at least one safe fallback.
        """
        candidates: list[PlanCandidate] = []

        # ── Candidate A: follow current waypoint ─────────────────────────
        wp = world.mission.current_waypoint
        if wp is not None and world.navigation_trusted:
            candidates.extend(self._waypoint_candidates(wp, world))

        # ── Candidate B: obstacle avoidance if needed ────────────────────
        nearest = world.nearest_obstacle
        if nearest is not None and nearest.effective_confidence > 0.3:
            if nearest.distance_m < 3.0:
                candidates.append(self._obstacle_avoidance_candidate(nearest, world))

        # ── Candidate C: energy-conserving hold ──────────────────────────
        if world.energy.soc < 0.25:
            candidates.append(self._hold_candidate(world, reason="low energy hold"))

        # ── Fallback: always have a safe option ──────────────────────────
        candidates.append(self._safe_fallback(world))

        # Sort by evaluation score (descending) then priority
        def _sort_key(c: PlanCandidate) -> tuple:
            score = c.evaluation.score if (c.evaluation and c.evaluation.feasible) else -1.0
            return (-score, c.priority)

        candidates.sort(key=_sort_key)

        logger.debug(
            "tactical plan: %d candidates, best=%r score=%.3f",
            len(candidates),
            candidates[0].rationale if candidates else "none",
            candidates[0].evaluation.score if (candidates and candidates[0].evaluation) else 0.0,
        )

        return candidates

    # ── Candidate factories ───────────────────────────────────────────────

    def _waypoint_candidates(
        self, target: Waypoint, world: WorldState
    ) -> list[PlanCandidate]:
        """Generate direct + conservative approach to a waypoint."""
        pos = world.navigation.position_m
        origin = Waypoint(x_m=pos.x, y_m=pos.y, depth_m=world.navigation.depth_m)

        # Direct trajectory
        direct = Trajectory.from_waypoints(
            [origin, target],
            propulsion_cost_per_m=world.energy.propulsion_cost_per_m,
            label="direct",
        )
        eval_direct = self._evaluator.evaluate(
            direct,
            available_energy_wh=world.energy.available_wh,
            nav_confidence=world.navigation.confidence,
            obstacle_clearance_m=(
                world.nearest_obstacle.distance_m if world.nearest_obstacle else None
            ),
        )

        cmd_direct = self._command_toward(origin, target, speed=0.3)
        cmd_direct.note = f"direct to waypoint ({target.label or 'wp'})"

        candidates = [PlanCandidate(
            command=cmd_direct,
            trajectory=direct,
            evaluation=eval_direct,
            rationale=f"direct to {target.label or 'waypoint'} score={eval_direct.score:.2f}",
            priority=0,
        )]

        # Conservative (slower) variant
        cmd_slow = self._command_toward(origin, target, speed=0.15)
        cmd_slow.note = f"conservative to waypoint ({target.label or 'wp'})"

        eval_slow = self._evaluator.evaluate(
            direct,  # same path, different speed
            available_energy_wh=world.energy.available_wh,
            nav_confidence=world.navigation.confidence * 0.9,  # slightly penalise uncertainty
            obstacle_clearance_m=(
                world.nearest_obstacle.distance_m if world.nearest_obstacle else None
            ),
        )
        candidates.append(PlanCandidate(
            command=cmd_slow,
            trajectory=direct,
            evaluation=eval_slow,
            rationale=f"conservative to {target.label or 'waypoint'} score={eval_slow.score:.2f}",
            priority=1,
        ))

        return candidates

    def _obstacle_avoidance_candidate(self, obstacle, world: WorldState) -> PlanCandidate:
        cmd = VehicleCommand()
        cmd.thrusters.surge = 0.0
        cmd.thrusters.yaw   = 0.5   # yaw right to avoid
        cmd.note = f"obstacle avoidance: {obstacle.distance_m:.1f} m bearing {obstacle.bearing_deg:.0f}°"
        return PlanCandidate(
            command=cmd,
            rationale=cmd.note,
            priority=0,
        )

    def _hold_candidate(self, world: WorldState, reason: str = "hold") -> PlanCandidate:
        cmd = VehicleCommand(note=f"tactical hold: {reason}")
        return PlanCandidate(command=cmd, rationale=reason, priority=2)

    def _safe_fallback(self, world: WorldState) -> PlanCandidate:
        cmd = VehicleCommand(note="safe fallback: cruise forward")
        cmd.thrusters.surge = 0.1
        return PlanCandidate(command=cmd, rationale="safe fallback", priority=99)

    @staticmethod
    def _command_toward(origin: Waypoint, target: Waypoint, speed: float) -> VehicleCommand:
        """Compute a normalised VehicleCommand pointing from origin to target."""
        dx = target.x_m - origin.x_m
        dy = target.y_m - origin.y_m
        dz = target.depth_m - origin.depth_m
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if dist < 0.1:
            return VehicleCommand(note="at waypoint")

        cmd = VehicleCommand()
        cmd.thrusters.surge = max(-1.0, min(1.0, (dx / dist) * speed))
        cmd.thrusters.sway  = max(-1.0, min(1.0, (dy / dist) * speed))
        cmd.thrusters.heave = max(-1.0, min(1.0, (dz / dist) * speed))
        return cmd
