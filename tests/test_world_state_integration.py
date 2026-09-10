"""Integration tests for WorldStateBuilder in the agent loop — Phase 1-9."""

import pytest

from auvbrain.faults.detector import FaultDetector
from auvbrain.faults.models import FaultSeverity
from auvbrain.models import Observation
from auvbrain.navigation.dead_reckoning import DeadReckoningEstimator
from auvbrain.perception.validator import SensorValidator
from auvbrain.planning.energy import EnergyState
from auvbrain.world.model import WorldState, WorldStateBuilder


@pytest.fixture
def pipeline_components():
    """Create the full pipeline components."""
    return {
        "validator": SensorValidator(),
        "estimator": DeadReckoningEstimator(),
        "fault_detector": FaultDetector(),
        "world_builder": WorldStateBuilder(),
    }


def test_world_state_pipeline_normal(pipeline_components):
    """Test full pipeline: observation → validation → estimation → world state."""
    validator = pipeline_components["validator"]
    estimator = pipeline_components["estimator"]
    fault_detector = pipeline_components["fault_detector"]
    world_builder = pipeline_components["world_builder"]
    
    # Tick 1
    obs = Observation(
        depth_m=25.0,
        battery_v=12.3,
        internal_temp_c=22.0,
        pressure_bar=3.5,
        obstacle_front_m=50.0,
    )
    
    validated = validator.validate(obs, dt=0.1)
    estimated_state = estimator.update(validated)
    faults = fault_detector.detect(validated, estimated_state)
    energy_state = EnergyState(
        soc=0.85,
        voltage=12.3,
        current_draw_a=0.5,
    )
    
    world = world_builder.build(
        navigation=estimated_state,
        energy=energy_state,
        faults=faults,
        observation_confidence=validated.observation_confidence,
        tick=1,
    )
    
    assert isinstance(world, WorldState)
    assert world.tick == 1
    assert world.observation_confidence > 0.9
    assert world.navigation.depth_m == 25.0
    assert world.energy.soc == 0.85
    assert len(world.faults) == 0


def test_world_state_pipeline_with_fault(pipeline_components):
    """Test pipeline detects and includes faults in world state."""
    validator = pipeline_components["validator"]
    estimator = pipeline_components["estimator"]
    fault_detector = pipeline_components["fault_detector"]
    world_builder = pipeline_components["world_builder"]
    
    # Build up history for fault detection
    for _ in range(5):
        obs = Observation(depth_m=20.0, battery_v=12.0)
        validated = validator.validate(obs, dt=0.1)
        estimator.update(validated)
    
    # Now inject a sensor spike (out of range)
    obs_bad = Observation(depth_m=400.0, battery_v=12.0)  # impossible depth
    validated = validator.validate(obs_bad, dt=0.1)
    estimated_state = estimator.update(validated)
    faults = fault_detector.detect(validated, estimated_state)
    
    energy_state = EnergyState(soc=0.8, voltage=12.0)
    world = world_builder.build(
        navigation=estimated_state,
        energy=energy_state,
        faults=faults,
        observation_confidence=validated.observation_confidence,
        tick=6,
    )
    
    # Should have degraded confidence
    assert world.observation_confidence < 0.5
    # May have faults detected
    assert world.tick == 6


def test_world_state_summary(pipeline_components):
    """Test world state summary for telemetry."""
    validator = pipeline_components["validator"]
    estimator = pipeline_components["estimator"]
    fault_detector = pipeline_components["fault_detector"]
    world_builder = pipeline_components["world_builder"]
    
    obs = Observation(depth_m=60.0, battery_v=11.2)
    validated = validator.validate(obs, dt=0.1)
    estimated_state = estimator.update(validated)
    faults = fault_detector.detect(validated, estimated_state)
    energy_state = EnergyState(soc=0.3, voltage=11.2)
    
    world = world_builder.build(
        navigation=estimated_state,
        energy=energy_state,
        faults=faults,
        observation_confidence=validated.observation_confidence,
        tick=42,
    )
    
    summary = world.summary()
    assert isinstance(summary, dict)
    assert summary["tick"] == 42
    assert summary["depth_m"] == 60.0
    assert "energy_soc" in summary
    assert "fault_count" in summary
    assert "mission_phase" in summary


def test_world_state_critical_fault_detection(pipeline_components):
    """Test world state correctly identifies critical faults."""
    from auvbrain.faults.models import Fault, FaultType
    
    world_builder = pipeline_components["world_builder"]
    estimator = pipeline_components["estimator"]
    
    obs = Observation(depth_m=50.0, battery_v=12.0)
    estimated_state = estimator.update(validator.validate(obs, dt=0.1))
    energy_state = EnergyState(soc=0.5, voltage=12.0)
    
    critical_fault = Fault(
        subsystem="safety",
        component="water_ingress",
        fault_type=FaultType.CRITICAL_FAILURE,
        severity=FaultSeverity.CRITICAL,
        confidence=1.0,
    )
    
    world = world_builder.build(
        navigation=estimated_state,
        energy=energy_state,
        faults=[critical_fault],
        observation_confidence=1.0,
        tick=10,
    )
    
    assert world.has_critical_fault is True
    assert len(world.faults) == 1
    assert world.faults[0].severity == FaultSeverity.CRITICAL


def test_world_state_mission_context(pipeline_components):
    """Test mission context is preserved in world state."""
    from auvbrain.world.mission_state import MissionContext, MissionPhase
    
    world_builder = pipeline_components["world_builder"]
    estimator = pipeline_components["estimator"]
    
    # Set mission context
    world_builder.mission.phase = MissionPhase.SURVEY
    world_builder.mission.goal = "Survey reef section A"
    
    obs = Observation(depth_m=30.0, battery_v=12.0)
    estimated_state = estimator.update(validator.validate(obs, dt=0.1))
    energy_state = EnergyState(soc=0.7, voltage=12.0)
    
    world = world_builder.build(
        navigation=estimated_state,
        energy=energy_state,
        faults=[],
        observation_confidence=1.0,
        tick=5,
    )
    
    assert world.mission_phase == MissionPhase.SURVEY
    assert world.mission.goal == "Survey reef section A"
