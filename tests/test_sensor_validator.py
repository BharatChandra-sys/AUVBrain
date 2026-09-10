"""Tests for SensorValidator — Phase 1.

Covers: dropout, stale/frozen, spike, out-of-range, cross-sensor disagreement,
        aggregate confidence, and the healthy passthrough case.
"""

from __future__ import annotations

import pytest

from auvbrain.models import Observation
from auvbrain.perception.sensor_health import FaultReason
from auvbrain.perception.validator import SensorValidator


def _validator() -> SensorValidator:
    return SensorValidator()


# ── Healthy passthrough ───────────────────────────────────────────────────────

def test_healthy_observation_passes_through() -> None:
    v = _validator()
    obs = Observation(depth_m=5.0, battery_v=12.0)
    validated = v.validate(obs, dt=0.1)
    assert validated.observation_confidence > 0.8
    assert validated.is_sensor_healthy("depth_m")
    assert validated.is_sensor_healthy("battery_v")


# ── Out-of-range ──────────────────────────────────────────────────────────────

def test_depth_out_of_range_is_flagged() -> None:
    v = _validator()
    obs = Observation(depth_m=500.0, battery_v=12.0)   # > 300 m limit
    validated = v.validate(obs, dt=0.1)
    h = validated.get_health("depth_m")
    assert h is not None
    assert not h.range_valid
    assert h.fault_reason == FaultReason.OUT_OF_RANGE
    assert h.confidence < 0.5


def test_battery_out_of_range_is_flagged() -> None:
    v = _validator()
    obs = Observation(depth_m=1.0, battery_v=35.0)   # > 30 V limit
    validated = v.validate(obs, dt=0.1)
    h = validated.get_health("battery_v")
    assert h is not None
    assert not h.range_valid


# ── Spike detection ───────────────────────────────────────────────────────────

def test_depth_spike_is_flagged() -> None:
    v = _validator()
    # Establish baseline
    v.validate(Observation(depth_m=5.0, battery_v=12.0), dt=0.1)
    # Jump of 50 m in 0.1 s → rate = 500 m/s >> 5 m/s limit
    obs2 = Observation(depth_m=55.0, battery_v=12.0)
    validated = v.validate(obs2, dt=0.1)
    h = validated.get_health("depth_m")
    assert h is not None
    assert h.fault_reason == FaultReason.SPIKE
    assert h.confidence < 0.5


def test_slow_depth_change_is_not_a_spike() -> None:
    v = _validator()
    v.validate(Observation(depth_m=5.0, battery_v=12.0), dt=0.1)
    obs2 = Observation(depth_m=5.3, battery_v=12.0)   # 0.3 m in 0.1 s = 3 m/s < 5 m/s
    validated = v.validate(obs2, dt=0.1)
    h = validated.get_health("depth_m")
    assert h is None or h.fault_reason == FaultReason.NONE


# ── Frozen sensor ─────────────────────────────────────────────────────────────

def test_frozen_sensor_is_flagged_after_n_ticks() -> None:
    from auvbrain.perception.validator import _CONSTANT_TICKS
    v = _validator()
    for _ in range(_CONSTANT_TICKS + 2):
        validated = v.validate(Observation(depth_m=5.0, battery_v=12.0), dt=0.1)
    h = validated.get_health("depth_m")
    assert h is not None
    assert h.fault_reason == FaultReason.CONSTANT
    assert h.confidence < 0.5


def test_varying_sensor_not_frozen() -> None:
    v = _validator()
    for i in range(15):
        validated = v.validate(
            Observation(depth_m=float(i) * 0.1, battery_v=12.0), dt=0.1
        )
    h = validated.get_health("depth_m")
    assert h is None or h.fault_reason != FaultReason.CONSTANT


# ── Cross-sensor disagreement ─────────────────────────────────────────────────

def test_depth_pressure_disagreement_is_flagged() -> None:
    v = _validator()
    # depth_m=5, but pressure_bar=4.0 → pressure_depth = (4.0-1.0)*10 = 30 m → 25 m discrepancy
    obs = Observation(depth_m=5.0, battery_v=12.0, pressure_bar=4.0)
    validated = v.validate(obs, dt=0.1)
    depth_h = validated.get_health("depth_m")
    pressure_h = validated.get_health("pressure_bar")
    assert depth_h is not None
    assert depth_h.fault_reason == FaultReason.CROSS_SENSOR_DISAGREE
    assert pressure_h is not None
    assert pressure_h.fault_reason == FaultReason.CROSS_SENSOR_DISAGREE


def test_depth_pressure_agreement_passes() -> None:
    v = _validator()
    # depth_m=20, pressure_bar=3.0 → pressure_depth=(3-1)*10=20 m → delta=0
    obs = Observation(depth_m=20.0, battery_v=12.0, pressure_bar=3.0)
    validated = v.validate(obs, dt=0.1)
    depth_h = validated.get_health("depth_m")
    assert depth_h is None or depth_h.fault_reason != FaultReason.CROSS_SENSOR_DISAGREE


# ── Aggregate confidence ──────────────────────────────────────────────────────

def test_aggregate_confidence_drops_with_bad_sensors() -> None:
    v = _validator()
    # Bad depth (out of range) lowers aggregate
    obs = Observation(depth_m=500.0, battery_v=12.0)
    validated = v.validate(obs, dt=0.1)
    assert validated.observation_confidence < 0.5


def test_aggregate_confidence_high_when_all_healthy() -> None:
    v = _validator()
    obs = Observation(depth_m=3.0, battery_v=12.0, obstacle_front_m=5.0)
    validated = v.validate(obs, dt=0.1)
    assert validated.observation_confidence > 0.85


# ── Optional sensor absent ────────────────────────────────────────────────────

def test_optional_sensor_none_does_not_create_entry() -> None:
    v = _validator()
    obs = Observation(depth_m=2.0, battery_v=12.0, pressure_bar=None)
    validated = v.validate(obs, dt=0.1)
    # pressure_bar absent → no health entry for it
    assert validated.get_health("pressure_bar") is None
