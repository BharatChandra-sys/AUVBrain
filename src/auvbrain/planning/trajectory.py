"""Trajectory types and evaluator.

A ``Trajectory`` is an ordered sequence of waypoints with associated cost
and risk estimates.  ``TrajectoryEvaluator`` scores a trajectory given the
current world state so the planner can choose between candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..world.mission_state import Waypoint


@dataclass(slots=True)
class TrajectorySegment:
    start: Waypoint
    end: Waypoint
    distance_m: float = 0.0
    energy_cost_wh: float = 0.0
    risk_score: float = 0.0      # [0, 1]; 1 = very risky

    @classmethod
    def from_waypoints(
        cls,
        start: Waypoint,
        end: Waypoint,
        propulsion_cost_per_m: float = 0.05,
    ) -> "TrajectorySegment":
        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        dz = end.depth_m - start.depth_m
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return cls(
            start=start,
            end=end,
            distance_m=dist,
            energy_cost_wh=dist * propulsion_cost_per_m,
        )


@dataclass
class Trajectory:
    waypoints: list[Waypoint] = field(default_factory=list)
    segments: list[TrajectorySegment] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_energy_wh: float = 0.0
    total_risk: float = 0.0
    label: str = ""

    @classmethod
    def from_waypoints(
        cls,
        waypoints: list[Waypoint],
        propulsion_cost_per_m: float = 0.05,
        label: str = "",
    ) -> "Trajectory":
        segments: list[TrajectorySegment] = []
        for i in range(len(waypoints) - 1):
            seg = TrajectorySegment.from_waypoints(
                waypoints[i], waypoints[i + 1], propulsion_cost_per_m
            )
            segments.append(seg)

        total_dist  = sum(s.distance_m    for s in segments)
        total_energy = sum(s.energy_cost_wh for s in segments)
        total_risk   = max((s.risk_score    for s in segments), default=0.0)

        return cls(
            waypoints=waypoints,
            segments=segments,
            total_distance_m=total_dist,
            total_energy_wh=total_energy,
            total_risk=total_risk,
            label=label,
        )


@dataclass(slots=True)
class TrajectoryEvaluation:
    trajectory: Trajectory
    feasible: bool
    score: float             # higher = better; composite of energy, risk, uncertainty
    rejection_reason: str = ""


class TrajectoryEvaluator:
    """Score a trajectory given the current world state."""

    def evaluate(
        self,
        traj: Trajectory,
        *,
        available_energy_wh: float,
        nav_confidence: float,
        obstacle_clearance_m: Optional[float] = None,
        w_energy: float = 0.4,
        w_risk: float   = 0.4,
        w_nav: float    = 0.2,
    ) -> TrajectoryEvaluation:
        """
        Returns a TrajectoryEvaluation with a composite score in [0, 1].

        Score = w_energy * energy_score + w_risk * (1-risk) + w_nav * nav_confidence
        """
        # Energy feasibility
        if traj.total_energy_wh > available_energy_wh:
            return TrajectoryEvaluation(
                trajectory=traj,
                feasible=False,
                score=0.0,
                rejection_reason=(
                    f"energy required {traj.total_energy_wh:.1f} Wh > "
                    f"available {available_energy_wh:.1f} Wh"
                ),
            )

        # Obstacle clearance check
        if obstacle_clearance_m is not None and obstacle_clearance_m < 0.6:
            return TrajectoryEvaluation(
                trajectory=traj,
                feasible=False,
                score=0.0,
                rejection_reason=f"obstacle too close: {obstacle_clearance_m:.2f} m",
            )

        # Energy score: fraction of available energy NOT used (efficiency)
        energy_score = max(0.0, 1.0 - traj.total_energy_wh / max(available_energy_wh, 1e-6))

        score = (
            w_energy * energy_score
            + w_risk  * (1.0 - traj.total_risk)
            + w_nav   * nav_confidence
        )

        return TrajectoryEvaluation(
            trajectory=traj,
            feasible=True,
            score=round(score, 4),
        )
