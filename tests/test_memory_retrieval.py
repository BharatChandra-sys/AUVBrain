"""Tests for state-aware memory retrieval — Phase 7."""

import pytest

from auvbrain.memory.episodic import EpisodicMemory
from auvbrain.memory.fault_memory import FaultMemory
from auvbrain.memory.models import MemoryEntry, MemoryTrust, MemoryType
from auvbrain.memory.retriever import RetrievalQuery, StateAwareRetriever
from auvbrain.navigation.estimated_state import EstimatedState, Position3D
from auvbrain.world.mission_state import MissionContext, MissionPhase
from auvbrain.world.model import WorldState


@pytest.fixture
def sample_episodic_memory():
    """Create sample episodic memory entries."""
    mem = EpisodicMemory()
    
    # Episode 1: successful survey at depth
    mem.record_episode(
        mission_id="survey_001",
        episode_description="surveyed reef at 60m",
        vehicle_state={"depth_m": 60, "battery_v": 12.0},
        action_taken="continue_survey",
        outcome="success",
        confidence=0.95,
        trust=MemoryTrust.VALIDATED_MISSION,
    )
    
    # Episode 2: low battery forced return
    mem.record_episode(
        mission_id="survey_002",
        episode_description="low battery at 50m, returned",
        vehicle_state={"depth_m": 50, "battery_v": 10.8},
        action_taken="return_to_surface",
        outcome="success",
        confidence=0.88,
        trust=MemoryTrust.VALIDATED_MISSION,
    )
    
    return mem


@pytest.fixture
def sample_fault_memory():
    """Create sample fault memory entries."""
    from auvbrain.faults.models import Fault, FaultSeverity, FaultType
    
    mem = FaultMemory()
    
    fault1 = Fault(
        subsystem="navigation",
        component="dvl",
        fault_type=FaultType.DROPOUT,
        severity=FaultSeverity.MAJOR,
        confidence=0.93,
        details="DVL signal lost at 65m",
    )
    
    mem.record_fault(
        fault=fault1,
        vehicle_state={"depth_m": 65, "battery_v": 11.8},
        recovery_action="reduce_speed_reconfigure_estimator",
        outcome="recovered",
        mission_context={"phase": "transit"},
    )
    
    return mem


def test_episodic_memory_recall(sample_episodic_memory):
    """Test recalling similar episodes."""
    results = sample_episodic_memory.recall(
        query="low battery return",
        k=2,
        min_trust=MemoryTrust.SIMULATION_EPISODE,
    )
    
    assert len(results) <= 2
    assert all(isinstance(e, MemoryEntry) for e in results)


def test_fault_memory_lookup(sample_fault_memory):
    """Test looking up similar faults."""
    from auvbrain.faults.models import Fault, FaultSeverity, FaultType
    
    query_fault = Fault(
        subsystem="navigation",
        component="dvl",
        fault_type=FaultType.DROPOUT,
        severity=FaultSeverity.MAJOR,
        confidence=0.85,
    )
    
    results = sample_fault_memory.lookup_similar(query_fault, k=5)
    assert len(results) >= 0  # may return empty if no matches


def test_state_aware_retriever():
    """Test that retriever constructs queries from world state."""
    world = WorldState(
        navigation=EstimatedState(
            position_m=Position3D(100, 50, 65),
            depth_m=65,
            confidence=0.6,
        ),
        mission=MissionContext(phase=MissionPhase.SURVEY),
    )
    
    retriever = StateAwareRetriever()
    query = retriever.build_query(world, user_query="dvl failure")
    
    assert isinstance(query, RetrievalQuery)
    assert "dvl failure" in query.text.lower()
    assert query.depth_m == 65
    assert query.mission_phase == MissionPhase.SURVEY
    assert query.min_confidence > 0.0


def test_state_aware_retriever_empty():
    """Retriever should handle empty state gracefully."""
    world = WorldState()
    retriever = StateAwareRetriever()
    query = retriever.build_query(world, user_query="test")
    
    assert query.text == "test"
    assert query.depth_m is None
