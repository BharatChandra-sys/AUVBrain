"""Memory taxonomy and trust provenance models — Phase 7.

Five memory classes (from the plan):
    PROCEDURAL    — how to do something (dive procedure, recovery steps)
    SEMANTIC      — factual knowledge (vehicle specs, environment facts)
    EPISODIC      — what happened (mission episodes, outcomes)
    FAULT         — fault occurrences and their recoveries
    ENVIRONMENTAL — environment observations (current maps, obstacle history)

Each MemoryEntry carries full provenance so the retriever can trust-filter:
    VERIFIED_PROCEDURE > HUMAN_REVIEWED_EPISODE > VALIDATED_MISSION
    > SIMULATION_EPISODE > AGENT_GENERATED_MEMORY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MemoryType(str, Enum):
    PROCEDURAL    = "procedural"
    SEMANTIC      = "semantic"
    EPISODIC      = "episodic"
    FAULT         = "fault"
    ENVIRONMENTAL = "environmental"


class MemoryTrust(str, Enum):
    """Trust rank — higher ordinal = more trusted."""
    AGENT_GENERATED      = "agent_generated"       # 1 — least trusted
    SIMULATION_EPISODE   = "simulation_episode"    # 2
    VALIDATED_MISSION    = "validated_mission"     # 3
    HUMAN_REVIEWED       = "human_reviewed"        # 4
    VERIFIED_PROCEDURE   = "verified_procedure"    # 5 — most trusted

    @property
    def rank(self) -> int:
        _ranks = {
            "agent_generated":    1,
            "simulation_episode": 2,
            "validated_mission":  3,
            "human_reviewed":     4,
            "verified_procedure": 5,
        }
        return _ranks[self.value]


@dataclass
class VehicleSnapshot:
    """Minimal vehicle state captured at memory creation time."""
    depth_m: float = 0.0
    battery_v: float = 12.0
    mode: str = "AUTONOMOUS"
    nav_confidence: float = 1.0
    mission_phase: str = "UNKNOWN"


@dataclass
class MemoryEntry:
    """A single persistent memory entry with full provenance.

    Attributes:
        memory_type:    Category (PROCEDURAL, EPISODIC, etc.).
        trust:          Trust rank for filtering.
        content:        Natural-language content (what to retrieve).
        source:         Unique key / origin (file path, mission ID, etc.).
        confidence:     How confident we are in this memory [0, 1].
        outcome:        Result of the episode / procedure (e.g. "successful").
        vehicle_state:  Vehicle snapshot when memory was created.
        environment:    Free-form environment context.
        model_version:  LLM / software version that generated this (if AI-created).
        tags:           Optional tags for filtering.
        created_at:     UTC creation timestamp.
        id:             Optional DB row ID (set after persistence).
    """

    memory_type: MemoryType
    trust: MemoryTrust
    content: str
    source: str
    confidence: float = 1.0
    outcome: Optional[str] = None
    vehicle_state: Optional[VehicleSnapshot] = None
    environment: dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type.value,
            "trust": self.trust.value,
            "trust_rank": self.trust.rank,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "model_version": self.model_version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "id": self.id,
        }
