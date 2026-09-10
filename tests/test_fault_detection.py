"""Tests for fault detection, diagnosis, and recovery — Phase 4."""

from __future__ import annotations

import pytest

from auvbrain.faults.detector import FaultDetector
from auvbrain.faults.diagnosis import FaultDiagnosis
from auvbrain.faults.models import Fault, FaultSeverity, FaultType
from auvbrain.faults.recovery import RecoveryAction, RecoveryPlanner
from auvbrain.models import Observation
from auvbrain.navigation.dead_reckoning import DeadReckoningEstimator
from auvbrain.navigation.estimated_state import EstimatedState
from auvbrain.perception.validator import SensorValidator


def _state(confidence: float = 0.9) -> EstimatedState:
    return EstimatedState(confidence=confidence)


def _validated(obs: Observation):
    v = SensorValidator()
    return v.validate(obs, dt=0.1)


# ── FaultDetector ─────────────────────────────────────────────────────────────

def test_no_faults_on_healthy_observation() -> None:
    det = FaultDetector()
    validated = _validated(Observation(depth_m=5.0, battery_v=12.0))
    faults = det.detect(validated, _state())
    assert faults == []


def test_detects_sensor_fault_on_out_of_range() -> None:
    det = FaultDetector()
    validated = _validated(Observation(depth_m=400.0, battery_v=12.0))
    faults = det.detect(validated, _state())
    assert any(f.subsystem == "perception" for f in faults)


def test_detects_water_ingress_fault() -> None:
    det = FaultDetector()
    validated = _validated(Observation(depth_m=5.0, battery_v=12.0, water_ingress=True))
    faults = det.detect(validated, _state())
    wf = [f for f in faults if f.fault_type == FaultType.WATER_INGRESS]
    assert len(wf) == 1
    assert wf[0].severity == FaultSeverity.CRITICAL


def test_detects_navigation_divergence() -> None:
    det = FaultDetector()
    validated = _validated(Observation(depth_m=5.0, battery_v=12.0))
    low_conf = _state(confidence=0.20)
    faults = det.detect(validated, low_conf)
    nav_faults = [f for f in faults if f.fault_type == FaultType.NAVIGATION_DIVERGED]
    assert len(nav_faults) == 1


def test_no_nav_fault_when_confidence_ok() -> None:
    det = FaultDetector()
    validated = _validated(Observation(depth_m=5.0, battery_v=12.0))
    faults = det.detect(validated, _state(confidence=0.90))
    nav_faults = [f for f in faults if f.fault_type == FaultType.NAVIGATION_DIVERGED]
    assert nav_faults == []


def test_multiple_detectors_run_independently() -> None:
    det = FaultDetector()
    # Both nav and sensor fault
    validated = _validated(Observation(depth_m=400.0, battery_v=12.0))
    faults = det.detect(validated, _state(confidence=0.20))
    types = {f.fault_type for f in faults}
    assert FaultType.NAVIGATION_DIVERGED in types


# ── RecoveryPlanner ───────────────────────────────────────────────────────────

def test_no_strategy_when_no_faults() -> None:
    planner = RecoveryPlanner()
    assert planner.plan([]) is None


def test_water_ingress_maps_to_emergency_surface() -> None:
    planner = RecoveryPlanner()
    fault = Fault(
        subsystem="hull", component="seal",
        fault_type=FaultType.WATER_INGRESS,
        severity=FaultSeverity.CRITICAL,
    )
    strategy = planner.plan([fault])
    assert strategy is not None
    assert strategy.action == RecoveryAction.EMERGENCY_SURFACE
    assert strategy.priority == 0   # highest priority


def test_dvl_dropout_maps_to_reconfigure() -> None:
    planner = RecoveryPlanner()
    fault = Fault(
        subsystem="navigation", component="dvl",
        fault_type=FaultType.SENSOR_DROPOUT,
        severity=FaultSeverity.HIGH,
    )
    strategy = planner.plan([fault])
    assert strategy is not None
    assert strategy.action == RecoveryAction.RECONFIGURE_ESTIMATOR


def test_critical_beats_lower_priority() -> None:
    planner = RecoveryPlanner()
    faults = [
        Fault(subsystem="navigation", component="dvl",
              fault_type=FaultType.SENSOR_DROPOUT, severity=FaultSeverity.HIGH),
        Fault(subsystem="hull", component="seal",
              fault_type=FaultType.WATER_INGRESS, severity=FaultSeverity.CRITICAL),
    ]
    strategy = planner.plan(faults)
    assert strategy is not None
    assert strategy.action == RecoveryAction.EMERGENCY_SURFACE


def test_unmatched_high_fault_returns_hold() -> None:
    planner = RecoveryPlanner()
    fault = Fault(
        subsystem="comms", component="radio",
        fault_type=FaultType.UNKNOWN,
        severity=FaultSeverity.HIGH,
    )
    strategy = planner.plan([fault])
    assert strategy is not None
    assert strategy.action == RecoveryAction.HOLD_POSITION


# ── FaultDiagnosis ────────────────────────────────────────────────────────────

def test_diagnosis_returns_none_for_empty() -> None:
    d = FaultDiagnosis()
    assert d.diagnose([]) is None


def test_diagnosis_escalates_multiple_high_faults_in_same_subsystem() -> None:
    d = FaultDiagnosis()
    faults = [
        Fault(subsystem="navigation", component="dvl",
              fault_type=FaultType.SENSOR_DROPOUT, severity=FaultSeverity.HIGH),
        Fault(subsystem="navigation", component="imu",
              fault_type=FaultType.SENSOR_DROPOUT, severity=FaultSeverity.HIGH),
    ]
    strategy = d.diagnose(faults)
    # Two HIGH faults in navigation → escalated to CRITICAL → abort or surface
    assert strategy is not None
    assert strategy.action in (
        RecoveryAction.ABORT_MISSION_AND_RETURN,
        RecoveryAction.EMERGENCY_SURFACE,
        RecoveryAction.HOLD_POSITION,
    )


def test_diagnosis_single_medium_fault_does_not_escalate() -> None:
    d = FaultDiagnosis()
    faults = [
        Fault(subsystem="navigation", component="dvl",
              fault_type=FaultType.SENSOR_FROZEN, severity=FaultSeverity.MEDIUM),
    ]
    strategy = d.diagnose(faults)
    assert strategy is not None
    assert strategy.action == RecoveryAction.RECONFIGURE_ESTIMATOR
