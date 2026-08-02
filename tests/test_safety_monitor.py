"""Tests for SafetyMonitor — every branch covered."""

from __future__ import annotations

import pytest

from auvbrain.config import Settings
from auvbrain.models import ControlMode, Observation, VehicleCommand
from auvbrain.safety.monitor import SafetyMonitor


@pytest.fixture()
def safety() -> SafetyMonitor:
    return SafetyMonitor(Settings())


# ── SAFE-mode overrides ──────────────────────────────────────────────────────

def test_water_ingress_forces_safe(safety: SafetyMonitor) -> None:
    obs = Observation(water_ingress=True)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode == ControlMode.SAFE
    assert "water ingress" in (cmd.note or "")


def test_max_depth_forces_safe(safety: SafetyMonitor) -> None:
    obs = Observation(depth_m=safety.settings.max_depth_m + 1.0)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode == ControlMode.SAFE
    assert "depth" in (cmd.note or "")


def test_min_battery_forces_safe(safety: SafetyMonitor) -> None:
    obs = Observation(battery_v=safety.settings.min_battery_v - 0.1)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode == ControlMode.SAFE
    assert "battery" in (cmd.note or "")


def test_overtemperature_forces_safe() -> None:
    safety = SafetyMonitor(Settings(max_internal_temp_c=40.0))
    obs = Observation(internal_temp_c=60.0)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode == ControlMode.SAFE
    assert "temp" in (cmd.note or "")


def test_overpressure_forces_safe() -> None:
    safety = SafetyMonitor(Settings(max_pressure_bar=3.0))
    obs = Observation(pressure_bar=4.0)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode == ControlMode.SAFE
    assert "pressure" in (cmd.note or "")


# ── Command-only override (obstacle stop) ────────────────────────────────────

def test_obstacle_stop_does_not_change_mode(safety: SafetyMonitor) -> None:
    """Obstacle stop overrides command but does NOT set SAFE mode."""
    obs = Observation(obstacle_front_m=safety.settings.emergency_obstacle_m - 0.01)
    original_cmd = VehicleCommand()
    original_cmd.thrusters.surge = 0.5

    mode, safe_cmd = safety.enforce(obs, original_cmd)
    assert mode is None  # no mode change
    assert safe_cmd.thrusters.surge == 0.0  # neutral command
    assert "obstacle" in (safe_cmd.note or "")


def test_obstacle_at_exactly_threshold_does_not_trigger(safety: SafetyMonitor) -> None:
    """Obstacle at exactly the threshold distance must NOT trigger."""
    obs = Observation(obstacle_front_m=safety.settings.emergency_obstacle_m)
    cmd = VehicleCommand()
    mode, returned_cmd = safety.enforce(obs, cmd)
    assert mode is None
    assert returned_cmd is cmd  # original command returned unchanged


# ── No-override passthrough ──────────────────────────────────────────────────

def test_all_nominal_passes_through(safety: SafetyMonitor) -> None:
    obs = Observation(
        depth_m=5.0,
        battery_v=12.0,
        internal_temp_c=30.0,
        pressure_bar=1.5,
        water_ingress=False,
        obstacle_front_m=5.0,
    )
    cmd = VehicleCommand()
    mode, returned_cmd = safety.enforce(obs, cmd)
    assert mode is None
    assert returned_cmd is cmd


def test_optional_fields_none_no_trigger(safety: SafetyMonitor) -> None:
    """When optional sensors return None, safety must not trigger."""
    obs = Observation(
        depth_m=0.0,
        battery_v=12.0,
        internal_temp_c=None,
        pressure_bar=None,
        water_ingress=None,
        obstacle_front_m=None,
    )
    cmd = VehicleCommand()
    mode, returned_cmd = safety.enforce(obs, cmd)
    assert mode is None
    assert returned_cmd is cmd
