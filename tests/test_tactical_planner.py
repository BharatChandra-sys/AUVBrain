"""Tests for tactical planner — Phase 6."""

import pytest

from auvbrain.faults.models import Fault, FaultSeverity, FaultType
from auvbrain.navigation.estimated_state import EstimatedState, Position3D
from auvbrain.planning.energy import EnergyState
from auvbrain.planning.tactical import PlanCandidate, TacticalPlanner
from auvbrain.world.model import WorldState
from auvbrain.world.obstacles import Obstacle


def test_tactical_planner_normal():
    """Generate plan candidates in normal conditions."""
    world = WorldState(
        navigation=EstimatedState(position_m=Position3D(0, 0, 10)),
        energy=EnergyState(soc=0.8, voltage_v=12.2),
    )
    
    planner = TacticalPlanner()
    candidates = planner.generate_candidates(world, goal_x_m=100, goal_y_m=0)
    
    assert len(candidates) > 0
    best = min(candidates, key=lambda c: c.total_cost)
    assert best.total_cost > 0.0
    assert len(best.trajectory.waypoints) > 0


def test_tactical_planner_obstacle_avoidance():
    """Planner should generate routes that avoid obstacles."""
    from auvbrain.world.obstacles import ObstacleMap
    
    obstacle_map = ObstacleMap()
    obstacle_map.add_or_update(Obstacle(
        x_m=50, y_m=0, z_m=10, distance_m=10, confidence=0.95
    ))
    
    world = WorldState(
        navigation=EstimatedState(position_m=Position3D(0, 0, 10)),
        obstacles=obstacle_map,
        energy=EnergyState(soc=0.8),
    )
    
    planner = TacticalPlanner()
    candidates = planner.generate_candidates(world, goal_x_m=100, goal_y_m=0)
    
    # Best candidate should route around obstacle
    best = min(candidates, key=lambda c: c.total_cost)
    assert len(best.trajectory.waypoints) >= 2


def test_tactical_planner_energy_degraded():
    """Low energy should produce conservative plans."""
    world = WorldState(
        navigation=EstimatedState(position_m=Position3D(0, 0, 10)),
        energy=EnergyState(soc=0.15, voltage_v=10.9),  # low
    )
    
    planner = TacticalPlanner()
    candidates = planner.generate_candidates(world, goal_x_m=1000, goal_y_m=0)
    
    # Should produce at least one candidate, but feasibility may be questionable
    assert len(candidates) > 0
    best = min(candidates, key=lambda c: c.total_cost)
    assert best.risk > 0.3  # should flag as risky


def test_tactical_planner_critical_fault():
    """Critical fault should increase plan risk."""
    world = WorldState(
        navigation=EstimatedState(position_m=Position3D(0, 0, 10), confidence=0.3),
        energy=EnergyState(soc=0.8),
        faults=[
            Fault(
                subsystem="navigation",
                component="dvl",
                fault_type=FaultType.DEGRADED,
                severity=FaultSeverity.CRITICAL,
                confidence=0.92,
            )
        ],
    )
    
    planner = TacticalPlanner()
    candidates = planner.generate_candidates(world, goal_x_m=200, goal_y_m=0)
    
    best = min(candidates, key=lambda c: c.total_cost)
    assert best.risk > 0.5  # critical fault should elevate risk
