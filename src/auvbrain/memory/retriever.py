"""State-aware RAG retriever — Phase 7.

Instead of a generic semantic similarity query, this retriever constructs
a rich context-aware query from:
    mission goal + estimated state + fault state + energy + vehicle mode

Pipeline:
    construct_query()
        → DB full-text search (ILIKE; pgvector ANN when embeddings available)
        → rerank by trust
        → filter by minimum trust threshold
        → return context list

This replaces the trivial ProtocolRetriever with something the LLM can
actually use to make state-aware decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..config import Settings
from ..db.engine import get_session
from ..db.repositories import RagDocumentRepository
from .models import MemoryEntry, MemoryTrust, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """Structured query built from the current vehicle context."""
    mission_goal: str = ""
    depth_m: float = 0.0
    nav_confidence: float = 1.0
    energy_soc: float = 1.0
    active_faults: list[str] = field(default_factory=list)
    mission_phase: str = "UNKNOWN"
    vehicle_mode: str = "AUTONOMOUS"
    free_text: str = ""

    def to_search_string(self) -> str:
        """Convert to a keyword-rich string for DB full-text search."""
        parts: list[str] = []
        if self.mission_goal:
            parts.append(self.mission_goal)
        if self.active_faults:
            parts.extend(self.active_faults)
        if self.mission_phase and self.mission_phase != "UNKNOWN":
            parts.append(self.mission_phase.lower())
        if self.energy_soc < 0.25:
            parts.append("low energy")
        if self.nav_confidence < 0.5:
            parts.append("navigation degraded")
        if self.free_text:
            parts.append(self.free_text)
        return " ".join(parts) or "general procedure"


@dataclass(slots=True)
class RetrievalResult:
    """A single retrieved memory entry with relevance context."""
    content: str
    source: str
    trust: MemoryTrust
    memory_type: MemoryType
    score: float = 1.0    # relevance score; 1.0 = exact match
    entry_id: Optional[str] = None


class StateAwareRetriever:
    """State-context-aware retriever replacing the trivial ProtocolRetriever.

    Usage::

        retriever = StateAwareRetriever(settings)
        query = RetrievalQuery(
            mission_goal="survey grid A",
            active_faults=["dvl_degraded"],
            energy_soc=0.22,
        )
        results = await retriever.retrieve(query)
    """

    def __init__(
        self,
        settings: Settings,
        *,
        min_trust: MemoryTrust = MemoryTrust.SIMULATION_EPISODE,
        top_k: int = 8,
    ) -> None:
        self._settings = settings
        self._min_trust = min_trust
        self._top_k = top_k

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Run the full retrieval pipeline and return ranked results."""
        k = top_k or self._top_k
        search_str = query.to_search_string()

        logger.debug("state-aware retrieval: query=%r", search_str[:120])

        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            docs = await repo.search_by_content(search_str, limit=k * 2)

        results: list[RetrievalResult] = []
        for doc in docs:
            meta = doc.metadata_ or {}
            trust_str = meta.get("trust", MemoryTrust.SIMULATION_EPISODE.value)
            try:
                trust = MemoryTrust(trust_str)
            except ValueError:
                trust = MemoryTrust.SIMULATION_EPISODE

            # Trust filter
            if trust.rank < self._min_trust.rank:
                continue

            mtype_str = meta.get("memory_type", MemoryType.SEMANTIC.value)
            try:
                mtype = MemoryType(mtype_str)
            except ValueError:
                mtype = MemoryType.SEMANTIC

            results.append(RetrievalResult(
                content=doc.content,
                source=doc.source,
                trust=trust,
                memory_type=mtype,
                score=float(trust.rank) / 5.0,  # rough relevance from trust
                entry_id=str(doc.id),
            ))

        # Rerank: higher trust rank first, then by insertion order
        results.sort(key=lambda r: -r.trust.rank)

        if len(results) > k:
            results = results[:k]

        logger.debug(
            "retrieval complete: %d results (filtered from %d docs)",
            len(results), len(docs),
        )
        return results

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved results as a string for the LLM system prompt."""
        if not results:
            return "[No relevant memory retrieved]"
        lines = ["=== Retrieved Context ==="]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] ({r.memory_type.value}, trust={r.trust.value}) "
                f"source={r.source}\n{r.content}"
            )
        return "\n\n".join(lines)
