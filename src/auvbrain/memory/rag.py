from __future__ import annotations

"""Optional retrieval (RAG) layer.

Keep the interface stable; plug in ChromaDB/pgvector later.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    text: str
    source: str


class ProtocolRetriever:
    async def retrieve(self, query: str) -> list[RetrievalResult]:
        return []
