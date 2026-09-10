"""Tests for DeadReckoningEstimator — Phase 1."""

from __future__ import annotations

import time

import pytest

from auvbrain.models import Observation
from auvbrain.navigation.dead_reckoning import DeadReckoningEstimator
from auvbrain.perception.validator import SensorValidator


def _validated(depth_m: float = 2.0, battery_v: float = 12.0):
    v = SensorValidator()
    return v.validate(Observation(depth_m=depth_m, battery_v=battery_v), dt=0.1)


# ── Basic behaviour ───────────────────────────────────────────────────────────

def test_initial_state_is_origin() -> None:
    est = DeadReckoningEstimator()
    validated = _validated()
    state = est.update(validated)
    assert state.depth_m == pytest.approx(2.0, abs=0.01)
    assert state.confidence > 0.0


def test_surge_forward_moves_position() -> None:
    est = DeadReckoningEstimator()
    state0 = est.update(_validated())
    # Apply surge for several ticks
    for _ in range(10):
        state = est.update(_validated(), surge=1.0)
    # x position should have advanced
    assert state.position_m.x > state0.position_m.x


def test_depth_fused_from_sensor() -> None:
    est = DeadReckoningEstimator()
    state = est.update(_validated(depth_m=15.0))
    assert state.depth_m == pytest.approx(15.0, abs=0.1)


# ── Confidence ────────────────────────────────────────────────────────────────

def test_confidence_is_between_0_and_1() -> None:
    est = DeadReckoningEstimator()
    for _ in range(20):
        state = est.update(_validated())
    assert 0.0 <= state.confidence <= 1.0


def test_confidence_decays_with_bad_sensor() -> None:
    """When depth sensor confidence is 0, estimator confidence should drop."""
    from auvbrain.perception.validator import SensorValidator
    from auvbrain.perception.sensor_health import FaultReason
    from auvbrain.models import Observation

    est = DeadReckoningEstimator()
    v = SensorValidator()

    # Good baseline
    good = v.validate(Observation(depth_m=5.0, battery_v=12.0), dt=0.1)
    state_good = est.update(good)

    # Now inject bad depth (out of range) to drive observation_confidence down
    bad_v = SensorValidator()
    bad = bad_v.validate(Observation(depth_m=500.0, battery_v=12.0), dt=0.1)
    est2 = DeadReckoningEstimator()
    state_bad = est2.update(bad)

    assert state_bad.confidence < state_good.confidence


def test_reset_clears_position() -> None:
    est = DeadReckoningEstimator()
    for _ in range(10):
        est.update(_validated(), surge=0.5)
    est.reset()
    state = est.update(_validated())
    assert abs(state.position_m.x) < 0.1


# ── Sensor sources ────────────────────────────────────────────────────────────

def test_sensor_sources_populated() -> None:
    est = DeadReckoningEstimator()
    state = est.update(_validated())
    assert len(state.sensor_sources) >= 1


# ── Timestamp ─────────────────────────────────────────────────────────────────

def test_timestamp_is_recent() -> None:
    from datetime import datetime, timezone
    est = DeadReckoningEstimator()
    state = est.update(_validated())
    now = datetime.now(timezone.utc)
    delta = abs((now - state.timestamp).total_seconds())
    assert delta < 2.0
