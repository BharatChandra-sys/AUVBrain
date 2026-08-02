"""Protocol retrieval (RAG) layer backed by the database.

In development / when pgvector is unavailable, falls back to ILIKE full-text
search.  In production with pgvector enabled, uses ANN cosine similarity.

Usage
-----
Ingest a document::

    retriever = ProtocolRetriever(settings)
    await retriever.ingest("dive_protocol_v2", text, embedding_vector)

Query::

    results = await retriever.retrieve("emergency ascent procedure")

The embedding step (generating the float vector from text) is the caller's
responsibility — this module is embedding-model agnostic so it works with
any provider (Ollama embeddings, OpenAI, llama.cpp, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Settings
from ..db.engine import get_session
from ..db.repositories import RagDocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    text: str
    source: str
    score: float = 1.0  # 1.0 = exact / no ranking; 0..1 for ANN results


class ProtocolRetriever:
    """Retrieve relevant protocol snippets for the LLM context window.

    Falls back to full-text ILIKE search when no embeddings are available.
    Uses pgvector cosine ANN when embeddings are present.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pgvector_available = False
        # Lazy check performed on first ANN query attempt

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return up to ``top_k`` relevant documents for ``query``."""
        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            docs = await repo.search_by_content(query, limit=top_k)

        results = [
            RetrievalResult(text=d.content, source=d.source)
            for d in docs
        ]

        if not results:
            logger.debug("RAG: no results for query=%r", query[:80])

        return results

    async def ingest(
        self,
        source: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Upsert a document into the RAG store.

        Args:
            source:     Unique key (e.g. file path or document ID).
            content:    Raw text to store and search.
            embedding:  Pre-computed embedding vector (optional).
            metadata:   Arbitrary JSON metadata (title, version, etc.).
        """
        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            doc = await repo.upsert(
                source=source,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
            )
        logger.info("RAG: ingested source=%r id=%s", source, doc.id)

    async def list_documents(self) -> list[dict]:
        """Return all ingested document metadata (no content)."""
        async with get_session(self._settings) as session:
            repo = RagDocumentRepository(session)
            docs = await repo.list_all()
        return [
            {
                "id": str(d.id),
                "source": d.source,
                "created_at": d.created_at.isoformat(),
                "metadata": d.metadata_,
            }
            for d in docs
        ]
