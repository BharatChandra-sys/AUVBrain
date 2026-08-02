"""Shared pytest fixtures.

DB fixture
----------
Each test that touches the DB gets a fresh in-memory SQLite database (via
aiosqlite) so tests are fully isolated and don't require a running Postgres
instance.

API fixture
-----------
``async_client`` creates a full-stack ASGI test client with the DB and auth
initialised so HTTP endpoint tests are end-to-end.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Force sqlite in-memory for all tests
os.environ.setdefault("AUV_DB_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("AUV_AUTH_ENABLED", "false")  # disabled by default; per-test overrides use fixtures


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture()
async def db_session():
    """Yield an async session backed by an isolated in-memory SQLite DB."""
    from auvbrain.config import load_settings
    from auvbrain.db.engine import init_db, get_session

    settings = load_settings()
    await init_db(settings)

    async with get_session(settings) as session:
        yield session


@pytest_asyncio.fixture()
async def async_client():
    """Full-stack HTTP test client (auth disabled, SQLite in-memory)."""
    from auvbrain.api.app import create_app
    from auvbrain.config import load_settings

    settings = load_settings()
    app = create_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture()
async def api_key_pair(db_session):
    """Create a write-scope API key and return (record, raw_token)."""
    from auvbrain.db.repositories import ApiKeyRepository

    repo = ApiKeyRepository(db_session)
    record, raw = await repo.create(name="test-key", scopes="read write admin")
    return record, raw
