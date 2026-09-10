"""Tests for trajectory planning and evaluation — Phase 6."""

import pytest

from auvbrain.navigation.estimated_state import EstimatedState, Position3D, Velocity3D
from auvbrain.planning.trajectory import Trajectory, TrajectoryEvaluator, Waypoint
from auvbrain.world.model import WorldState


def test_waypoint_creation():
    w = Waypoint(x_m=100, y_m=50, depth_m=20, label="Survey Point A")
    assert w.x_m == 100
    assert w.y_m == 50
    assert w.depth_m == 20
    assert w.label == "Survey Point A"


def test_trajectory_length():
    waypoints = [
        Waypoint(0, 0, 0),
        Waypoint(100, 0, 0),
        Waypoint(100, 100, 0),
    ]
    traj = Trajectory(waypoints=waypoints)
    # Simple calculation: 100 + 100 = 200
    assert traj.length_m == pytest.approx(200.0, abs=1.0)


def test_trajectory_evaluator_basic():
    world = WorldState(
        navigation=EstimatedState(
            position_m=Position3D(0, 0, 0),
            velocity_ms=Velocity3D(0.5, 0, 0),
            confidence=0.95,
        ),
    )
    waypoints = [Waypoint(50, 0, 5), Waypoint(100, 0, 10)]
    traj = Trajectory(waypoints=waypoints)
    
    evaluator = TrajectoryEvaluator()
    result = evaluator.evaluate(traj, world)
    
    assert result.risk > 0.0
    assert result.energy_cost_wh > 0.0
    assert result.eta_s > 0.0
    assert result.feasible is True


def test_trajectory_evaluator_high_risk():
    """Low navigation confidence should increase risk score."""
    world = WorldState(
        navigation=EstimatedState(
            position_m=Position3D(0, 0, 0),
            confidence=0.2,  # very low
        ),
    )
    waypoints = [Waypoint(200, 0, 50)]  # deep, far
    traj = Trajectory(waypoints=waypoints)
    
    evaluator = TrajectoryEvaluator()
    result = evaluator.evaluate(traj, world)
    
    assert result.risk > 0.5  # should be flagged as risky


def test_trajectory_evaluator_energy_infeasible():
    """If trajectory energy cost exceeds available, mark as infeasible."""
    from auvbrain.planning.energy import EnergyState
    
    world = WorldState(
        navigation=EstimatedState(position_m=Position3D(0, 0, 0)),
        energy=EnergyState(soc=0.1, voltage_v=10.8, available_wh=10.0),
    )
    waypoints = [Waypoint(1000, 0, 100) for _ in range(10)]  # massive distance
    traj = Trajectory(waypoints=waypoints)
    
    evaluator = TrajectoryEvaluator()
    result = evaluator.evaluate(traj, world)
    
    assert result.feasible is False
    assert result.energy_cost_wh > world.energy.available_wh
