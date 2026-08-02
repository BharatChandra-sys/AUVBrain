"""Async SQLAlchemy engine + session factory.

A single async engine is created on first import.  Call ``init_db()`` at
startup to run DDL (CREATE TABLE IF NOT EXISTS) and seed any required rows.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import Settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine(settings: Settings) -> AsyncEngine:
    """Create an async engine from Settings.

    Uses asyncpg driver.  Falls back to aiosqlite when the URL is SQLite
    (useful for tests without a running Postgres instance).
    """
    url = settings.db_url
    engine_kwargs: dict = {
        "echo": settings.db_echo,
        "pool_pre_ping": True,
    }

    if url.startswith("sqlite"):
        # aiosqlite driver for unit tests
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # asyncpg pool sizing
        engine_kwargs["pool_size"] = settings.db_pool_size
        engine_kwargs["max_overflow"] = settings.db_max_overflow
        engine_kwargs["pool_recycle"] = 1800  # seconds

    return create_async_engine(url, **engine_kwargs)


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the module-level engine, creating it on first call."""
    global _engine
    if _engine is None:
        if settings is None:
            from ..config import load_settings
            settings = load_settings()
        _engine = _build_engine(settings)
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_session(
    settings: Settings | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """Context manager that yields a session and handles commit / rollback."""
    factory = get_session_factory(settings)
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(settings: Settings | None = None) -> None:
    """Create all tables (idempotent). Run once at application startup."""
    from .models import Base  # local import to avoid circular

    engine = get_engine(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialised.")


async def close_db() -> None:
    """Dispose the engine connection pool.  Call at application shutdown."""
    global _engine, _async_session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database engine disposed.")
