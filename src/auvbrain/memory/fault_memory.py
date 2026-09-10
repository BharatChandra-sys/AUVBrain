"""Fault memory — records fault episodes and recoveries — Phase 7.

Example entry (from plan.md §13):
{
  "type": "fault_episode",
  "mission": "survey_12",
  "state": {"depth": 63, "battery": 0.22, "navigation_confidence": 0.61},
  "fault": "dvl_degraded",
  "action": "return_to_surface",
  "outcome": "successful",
  "verification": "validated_simulation"
}
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Settings
from ..db.engine import get_session
from ..db.repositories import RagDocumentRepository
from ..faults.models import Fault
from .models import MemoryEntry, MemoryTrust, MemoryType, VehicleSnapshot

logger = logging.getLogger(__name__)


class FaultMemory:
    """Persist fault episodes and retrieve similar past faults.

    Used by the agent tool ``retrieve_fault_memory()`` to find how previous
    similar faults were handled and whether the recovery was successful.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def record_fault(
        self,
        *,
        mission_id: str,
        fault: Fault,
        recovery_action: str,
        outcome: str,
        vehicle_state: Optional[VehicleSnapshot] = None,
        trust: MemoryTrust = MemoryTrust.SIMULATION_EPISODE,
    ) -> MemoryEntry:
        """Store a fault episode with its recovery outcome."""
        content = (
            f"Fault episode in mission {mission_id}: "
            f"subsystem={fault.subsystem} component={fault.component} "
            f"type={fault.fault_type.value} severity={fault.severity.value}. "
            f"Recovery action: {recovery_action}. Outcome: {outcome}."
        )

        state_dict: dict = {}
        if vehicle_state:
            state_dict = {
                "depth_m": vehicle_state.depth_m,
                "battery_v": vehicle_state.battery_v,
                "nav_confidence": vehicle_state.nav_confidence,
                "mission_phase": vehicle_state.mission_phase,
            }

        entry = MemoryEntry(
            memory_type=MemoryType.FAULT,
            trust=trust,
            content=content,
            source=f"fault:{mission_id}:{fault.component}",
            confidence=fault.confidence,
            outcome=outcome,
            vehicle_state=vehicle_state,
            environment={"fault": fault.to_dict(), "vehicle_state": state_dict},
            tags=["fault", fault.subsystem, fault.component, fault.fault_type.value],
        )

        await self._persist(entry)
        logger.info(
            "fault memory recorded: mission=%s fault=%s/%s outcome=%s",
            mission_id, fault.subsystem, fault.component, outcome,
        )
        return entry

    async def lookup_similar(
        self,
        fault: Fault,
        top_k: int = 5,
        min_trust: MemoryTrust = MemoryTrust.SIMULATION_EPISODE,
    ) -> list[MemoryEntry]:
        """Find past fault episodes similar to the given fault.

        Searches by component name + fault type string.
        Trust-filters below ``min_trust`` rank.
        """
        query = f"{fault.component} {fault.fault_type.value} {fault.subsystem}"

        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            docs = await repo.search_by_content(query, limit=top_k * 2)

        results: list[MemoryEntry] = []
        for doc in docs:
            meta = doc.metadata_ or {}
            if meta.get("memory_type") != MemoryType.FAULT.value:
                continue
            trust_str = meta.get("trust", MemoryTrust.SIMULATION_EPISODE.value)
            try:
                trust = MemoryTrust(trust_str)
            except ValueError:
                trust = MemoryTrust.SIMULATION_EPISODE

            if trust.rank < min_trust.rank:
                continue

            results.append(MemoryEntry(
                memory_type=MemoryType.FAULT,
                trust=trust,
                content=doc.content,
                source=doc.source,
                outcome=meta.get("outcome"),
                tags=meta.get("tags", []),
                created_at=doc.created_at,
                id=str(doc.id),
            ))

            if len(results) >= top_k:
                break

        return results

    async def _persist(self, entry: MemoryEntry) -> None:
        from ..db.models import RagDocument
        async with get_session(self._settings) as session:
            doc = RagDocument(
                source=entry.source,
                content=entry.content,
                metadata_={
                    "memory_type": entry.memory_type.value,
                    "trust": entry.trust.value,
                    "trust_rank": entry.trust.rank,
                    "outcome": entry.outcome,
                    "tags": entry.tags,
                },
            )
            session.add(doc)
