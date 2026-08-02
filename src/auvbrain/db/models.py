"""SQLAlchemy ORM models for persistent storage.

Tables:
  - api_keys       — hashed API keys for endpoint auth
  - telemetry_events — structured append-only event store (mirrors JSONL)
  - mission_log    — high-level mission summaries
  - rag_documents  — RAG source documents for pgvector retrieval
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

class ApiKey(Base):
    """Hashed API key records.

    The raw key is never stored.  Only the SHA-256 hex digest is persisted so
    that even a full DB leak cannot be used to authenticate.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="Human label")
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True, comment="SHA-256 hex"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    scopes: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default="read",
        comment="Space-separated scope tokens e.g. 'read write admin'",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_api_keys_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ApiKey id={self.id} name={self.name!r} scopes={self.scopes!r}>"


# ---------------------------------------------------------------------------
# Telemetry events  (mirrored from JSONL for queryable history)
# ---------------------------------------------------------------------------

class TelemetryEvent(Base):
    """Structured telemetry events persisted to Postgres.

    The JSONL file remains the low-latency primary write path.  This table
    is populated asynchronously by the ``DBTelemetrySink`` for queryable
    history, dashboards, and alerting.
    """

    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    mode: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    tick_period_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    read_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    decide_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    apply_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )

    __table_args__ = (
        Index("ix_telemetry_type_ts", "event_type", "ts"),
        Index("ix_telemetry_mode_ts", "mode", "ts"),
    )

    def __repr__(self) -> str:
        return f"<TelemetryEvent id={self.id} type={self.event_type!r} ts={self.ts}>"


# ---------------------------------------------------------------------------
# Mission log
# ---------------------------------------------------------------------------

class MissionLog(Base):
    """High-level mission records.

    One row per mission (agent-loop run), linking to associated telemetry.
    """

    __tablename__ = "mission_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hardware_profile: Mapped[str] = mapped_column(String(64), nullable=False)
    use_llm: Mapped[bool] = mapped_column(Boolean, nullable=False)
    llm_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_ticks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    safe_overrides: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_fallbacks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dropped_telemetry: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<MissionLog id={self.id} started={self.started_at}>"


# ---------------------------------------------------------------------------
# RAG documents  (pgvector embedding column added via migration when available)
# ---------------------------------------------------------------------------

class RagDocument(Base):
    """Source documents for protocol-level retrieval.

    The ``embedding`` column (vector(1536)) is added by a separate migration
    after enabling the pgvector extension.  We leave it as JSONB here so the
    table can be created before the extension is available (e.g. SQLite tests).
    """

    __tablename__ = "rag_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # stored as JSON array of floats; replaced by native vector column in prod
    embedding_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("source", name="uq_rag_documents_source"),
        Index("ix_rag_documents_source", "source"),
    )

    def __repr__(self) -> str:
        return f"<RagDocument id={self.id} source={self.source!r}>"
