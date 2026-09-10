"""Episodic memory — records and recalls mission episodes — Phase 7."""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Settings
from ..db.engine import get_session
from .models import MemoryEntry, MemoryTrust, MemoryType, VehicleSnapshot

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Persist and retrieve mission episodes.

    Each episode is stored as a MemoryEntry with type=EPISODIC.
    Recall uses the state-aware retriever (full-text for now, ANN when
    embeddings are available).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def record_episode(
        self,
        *,
        mission_id: str,
        goal: str,
        outcome: str,
        vehicle_state: Optional[VehicleSnapshot] = None,
        environment: Optional[dict] = None,
        notes: str = "",
        trust: MemoryTrust = MemoryTrust.SIMULATION_EPISODE,
    ) -> MemoryEntry:
        """Persist a mission episode to the DB memory store."""
        content = (
            f"Mission {mission_id}: goal='{goal}' outcome='{outcome}'. "
            f"{notes}"
        ).strip()

        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            trust=trust,
            content=content,
            source=f"mission:{mission_id}",
            confidence=1.0,
            outcome=outcome,
            vehicle_state=vehicle_state,
            environment=environment or {},
            tags=["episode", mission_id],
        )

        await self._persist(entry)
        logger.info("episodic memory recorded: mission=%s outcome=%s", mission_id, outcome)
        return entry

    async def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Recall episodes relevant to the query string."""
        return await self._search(MemoryType.EPISODIC, query, top_k)

    # ── DB helpers ────────────────────────────────────────────────────────

    async def _persist(self, entry: MemoryEntry) -> None:
        from ..db.models import RagDocument
        async with get_session(self._settings) as session:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from sqlalchemy import insert as generic_insert

            doc = RagDocument(
                source=entry.source,
                content=entry.content,
                metadata_={
                    "memory_type": entry.memory_type.value,
                    "trust": entry.trust.value,
                    "outcome": entry.outcome,
                    "tags": entry.tags,
                },
            )
            session.add(doc)

    async def _search(
        self, memory_type: MemoryType, query: str, top_k: int
    ) -> list[MemoryEntry]:
        from ..db.engine import get_session
        from ..db.repositories import RagDocumentRepository

        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            docs = await repo.search_by_content(query, limit=top_k)

        results: list[MemoryEntry] = []
        for doc in docs:
            meta = doc.metadata_ or {}
            if meta.get("memory_type") != memory_type.value:
                continue
            trust_str = meta.get("trust", MemoryTrust.SIMULATION_EPISODE.value)
            try:
                trust = MemoryTrust(trust_str)
            except ValueError:
                trust = MemoryTrust.SIMULATION_EPISODE

            results.append(MemoryEntry(
                memory_type=memory_type,
                trust=trust,
                content=doc.content,
                source=doc.source,
                outcome=meta.get("outcome"),
                tags=meta.get("tags", []),
                created_at=doc.created_at,
                id=str(doc.id),
            ))

        return results
