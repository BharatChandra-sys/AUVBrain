"""Repository layer — thin query wrappers around ORM models.

Each repository is a plain class (no magic) injected with an ``AsyncSession``.
Business logic lives in services, not here.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ApiKey, MissionLog, RagDocument, TelemetryEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _hash_key(raw: str) -> str:
    """Return SHA-256 hex of ``raw``."""
    return hashlib.sha256(raw.encode()).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ApiKeyRepository
# ---------------------------------------------------------------------------

class ApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        scopes: str = "read",
        expires_at: datetime | None = None,
    ) -> tuple[ApiKey, str]:
        """Create a new API key.

        Returns ``(orm_record, raw_key)`` — the raw key is shown **once** and
        never stored.  Callers must surface it to the user immediately.
        """
        raw = secrets.token_urlsafe(32)
        record = ApiKey(
            name=name,
            key_hash=_hash_key(raw),
            scopes=scopes,
            expires_at=expires_at,
        )
        self._session.add(record)
        await self._session.flush()
        logger.info("ApiKey created name=%r id=%s scopes=%r", name, record.id, scopes)
        return record, raw

    async def get_by_hash(self, raw: str) -> ApiKey | None:
        """Look up a key record by raw (plaintext) key."""
        key_hash = _hash_key(raw)
        stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def touch(self, record: ApiKey) -> None:
        """Update last_used_at timestamp (best-effort, never blocks auth)."""
        record.last_used_at = _utcnow()
        await self._session.flush()

    async def revoke(self, key_id: uuid.UUID) -> bool:
        stmt = (
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(is_active=False)
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def list_active(self) -> Sequence[ApiKey]:
        stmt = select(ApiKey).where(ApiKey.is_active == True).order_by(ApiKey.created_at)  # noqa: E712
        result = await self._session.execute(stmt)
        return result.scalars().all()


# ---------------------------------------------------------------------------
# TelemetryRepository
# ---------------------------------------------------------------------------

class TelemetryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, event: dict[str, Any]) -> TelemetryEvent:
        """Persist a raw telemetry event dict."""
        event_type = str(event.get("type", "unknown"))
        record = TelemetryEvent(
            event_type=event_type,
            mode=event.get("mode"),
            source=event.get("source"),
            tick_period_ms=event.get("tick_period_ms"),
            total_ms=event.get("total_ms"),
            read_ms=event.get("read_ms"),
            decide_ms=event.get("decide_ms"),
            apply_ms=event.get("apply_ms"),
            payload=event,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def recent(
        self,
        limit: int = 100,
        event_type: str | None = None,
        mode: str | None = None,
    ) -> Sequence[TelemetryEvent]:
        stmt = select(TelemetryEvent).order_by(TelemetryEvent.ts.desc()).limit(limit)
        if event_type:
            stmt = stmt.where(TelemetryEvent.event_type == event_type)
        if mode:
            stmt = stmt.where(TelemetryEvent.mode == mode)
        result = await self._session.execute(stmt)
        return result.scalars().all()


# ---------------------------------------------------------------------------
# MissionLogRepository
# ---------------------------------------------------------------------------

class MissionLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start(
        self,
        hardware_profile: str,
        use_llm: bool,
        llm_provider: str | None = None,
    ) -> MissionLog:
        record = MissionLog(
            hardware_profile=hardware_profile,
            use_llm=use_llm,
            llm_provider=llm_provider,
        )
        self._session.add(record)
        await self._session.flush()
        logger.info("Mission started id=%s hardware=%r", record.id, hardware_profile)
        return record

    async def finish(
        self,
        mission_id: uuid.UUID,
        *,
        total_ticks: int,
        safe_overrides: int,
        llm_fallbacks: int,
        dropped_telemetry: int,
        notes: str | None = None,
    ) -> bool:
        stmt = (
            update(MissionLog)
            .where(MissionLog.id == mission_id)
            .values(
                ended_at=_utcnow(),
                total_ticks=total_ticks,
                safe_overrides=safe_overrides,
                llm_fallbacks=llm_fallbacks,
                dropped_telemetry=dropped_telemetry,
                notes=notes,
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0

    async def get(self, mission_id: uuid.UUID) -> MissionLog | None:
        stmt = select(MissionLog).where(MissionLog.id == mission_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 20) -> Sequence[MissionLog]:
        stmt = select(MissionLog).order_by(MissionLog.started_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()


# ---------------------------------------------------------------------------
# RagDocumentRepository
# ---------------------------------------------------------------------------

class RagDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        source: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> RagDocument:
        """Insert or replace a document by source key."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(RagDocument).values(
            source=source,
            content=content,
            embedding_json=embedding,
            metadata_=metadata or {},
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_rag_documents_source",
            set_={
                "content": stmt.excluded.content,
                "embedding_json": stmt.excluded.embedding_json,
                "metadata_": stmt.excluded.metadata_,
            },
        )
        await self._session.execute(stmt)
        # Re-fetch to return the ORM object
        return await self._get_by_source(source)  # type: ignore[return-value]

    async def _get_by_source(self, source: str) -> RagDocument | None:
        stmt = select(RagDocument).where(RagDocument.source == source)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def search_by_content(self, query: str, limit: int = 5) -> Sequence[RagDocument]:
        """Simple ILIKE full-text fallback (use pgvector ANN in production)."""
        stmt = (
            select(RagDocument)
            .where(RagDocument.content.ilike(f"%{query}%"))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def list_all(self, limit: int = 200) -> Sequence[RagDocument]:
        stmt = select(RagDocument).order_by(RagDocument.created_at).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()
