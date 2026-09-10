"""Tests for ActionShield — Phase 5."""

from __future__ import annotations

import pytest

from auvbrain.faults.models import Fault, FaultSeverity, FaultType
from auvbrain.models import VehicleCommand
from auvbrain.navigation.estimated_state import EstimatedState
from auvbrain.planning.energy import EnergyState
from auvbrain.safety.shield import ActionShield, ShieldVerdict
from auvbrain.world.model import WorldState
from auvbrain.world.mission_state import MissionContext
from auvbrain.world.obstacles import Obstacle, ObstacleMap


def _world(
    nav_confidence: float = 0.95,
    depth_m: float = 5.0,
    obstacle_m: float | None = None,
    soc: float = 0.80,
    faults: list[Fault] | None = None,
    max_depth_m: float = 50.0,
) -> WorldState:
    nav = EstimatedState(depth_m=depth_m, confidence=nav_confidence)
    obs_map = ObstacleMap()
    if obstacle_m is not None:
        obs_map.update(Obstacle(distance_m=obstacle_m, confidence=0.9, ttl_s=60.0))
    energy = EnergyState(soc=soc, capacity_wh=100.0, reserve_fraction=0.1)
    mission = MissionContext(max_depth_m=max_depth_m)
    return WorldState(
        navigation=nav,
        obstacles=obs_map,
        energy=energy,
        faults=faults or [],
        mission=mission,
    )


def _cmd(surge: float = 0.3, heave: float = 0.0) -> VehicleCommand:
    cmd = VehicleCommand()
    cmd.thrusters.surge = surge
    cmd.thrusters.heave = heave
    return cmd


shield = ActionShield()


# ── APPROVE ───────────────────────────────────────────────────────────────────

def test_approve_nominal() -> None:
    decision = shield.evaluate(_cmd(), _world())
    assert decision.verdict == ShieldVerdict.APPROVE
    assert decision.allowed is True


# ── REJECT: low nav confidence ────────────────────────────────────────────────

def test_reject_low_nav_confidence() -> None:
    decision = shield.evaluate(_cmd(), _world(nav_confidence=0.10))
    assert decision.verdict == ShieldVerdict.REJECT
    assert "navigation confidence" in decision.reason


# ── REJECT: obstacle too close ────────────────────────────────────────────────

def test_reject_obstacle_within_clearance() -> None:
    decision = shield.evaluate(_cmd(), _world(obstacle_m=0.3))
    assert decision.verdict == ShieldVerdict.REJECT
    assert "obstacle" in decision.reason


def test_approve_obstacle_just_outside_clearance() -> None:
    decision = shield.evaluate(_cmd(), _world(obstacle_m=0.65))
    assert decision.verdict == ShieldVerdict.APPROVE


# ── MODIFY: depth near limit ──────────────────────────────────────────────────

def test_modify_heave_near_depth_limit() -> None:
    # depth=47.5 m, max=50 m → 95% limit
    decision = shield.evaluate(_cmd(heave=0.5), _world(depth_m=47.5, max_depth_m=50.0))
    assert decision.verdict == ShieldVerdict.MODIFY
    assert decision.command.thrusters.heave == pytest.approx(0.0)
    assert "heave clamped" in (decision.command.note or "")


def test_approve_heave_well_below_limit() -> None:
    decision = shield.evaluate(_cmd(heave=0.5), _world(depth_m=10.0, max_depth_m=50.0))
    assert decision.verdict == ShieldVerdict.APPROVE


# ── MODIFY: energy clamp ──────────────────────────────────────────────────────

def test_modify_clamps_thrust_on_low_energy() -> None:
    decision = shield.evaluate(_cmd(surge=0.9), _world(soc=0.15))
    assert decision.verdict == ShieldVerdict.MODIFY
    assert decision.command.thrusters.surge <= 0.40 + 1e-9


def test_approve_moderate_thrust_on_low_energy() -> None:
    decision = shield.evaluate(_cmd(surge=0.3), _world(soc=0.15))
    assert decision.verdict == ShieldVerdict.APPROVE


# ── REJECT: critical fault blocks movement ────────────────────────────────────

def test_reject_movement_during_critical_fault() -> None:
    crit = Fault(
        subsystem="hull", component="seal",
        fault_type=FaultType.WATER_INGRESS,
        severity=FaultSeverity.CRITICAL,
    )
    decision = shield.evaluate(_cmd(surge=0.5), _world(faults=[crit]))
    assert decision.verdict == ShieldVerdict.REJECT
    assert "critical fault" in decision.reason


def test_approve_neutral_command_during_critical_fault() -> None:
    crit = Fault(
        subsystem="hull", component="seal",
        fault_type=FaultType.WATER_INGRESS,
        severity=FaultSeverity.CRITICAL,
    )
    neutral = VehicleCommand()   # all zeros
    decision = shield.evaluate(neutral, _world(faults=[crit]))
    assert decision.verdict == ShieldVerdict.APPROVE
