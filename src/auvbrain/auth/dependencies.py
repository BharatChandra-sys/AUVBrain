"""FastAPI dependency providers for authentication + authorization.

Every control endpoint (POST /mode, POST /command, WS /ws/control) must
declare ``Depends(require_scope("write"))`` to enforce bearer-token auth.

Read-only endpoints (GET /state, GET /health) use ``Depends(require_scope("read"))``
— useful when operators want a read-only dashboard key.

The admin scope (``Depends(require_scope("admin"))``) gates key management.

Setup
-----
Bootstrap the first API key from the CLI::

    python -m auvbrain.auth.cli create --name "operator" --scopes "read write"

That prints the raw token once.  Store it in your client .env.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from ..db.engine import get_session
from ..db.models import ApiKey
from ..db.repositories import ApiKeyRepository

logger = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Missing or invalid API key.",
    headers={"WWW-Authenticate": "ApiKey"},
)

_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Insufficient scope.",
)


async def _resolve_key(
    header_key: str | None = Security(_API_KEY_HEADER),
    query_key: str | None = Security(_API_KEY_QUERY),
) -> ApiKey:
    """Validate the API key from header or query string.

    Header takes precedence.  Returns the ``ApiKey`` ORM record.
    Raises ``401`` if missing/invalid, ``403`` if revoked/expired.
    """
    raw = header_key or query_key
    if not raw:
        raise _UNAUTHORIZED

    async with get_session() as session:
        repo = ApiKeyRepository(session)
        record = await repo.get_by_hash(raw)

        if record is None:
            logger.warning("API key lookup miss (unknown key)")
            raise _UNAUTHORIZED

        if not record.is_active:
            logger.warning("API key lookup miss (revoked) id=%s", record.id)
            raise _UNAUTHORIZED

        if record.expires_at is not None:
            if record.expires_at < datetime.now(timezone.utc):
                logger.warning("API key expired id=%s", record.id)
                raise _UNAUTHORIZED

        # Touch last_used_at asynchronously — failure is non-fatal
        try:
            await repo.touch(record)
        except Exception:
            logger.debug("touch last_used_at failed", exc_info=True)

        return record


def require_scope(scope: str):
    """Return a FastAPI dependency that requires ``scope`` on the API key.

    Usage::

        @router.post("/mode")
        async def set_mode(
            mode: ControlMode,
            _: Annotated[ApiKey, Depends(require_scope("write"))],
        ): ...
    """

    async def _check(key: Annotated[ApiKey, Depends(_resolve_key)]) -> ApiKey:
        granted = set(key.scopes.split())
        if scope not in granted:
            logger.warning(
                "Scope denied scope=%r granted=%r key_id=%s", scope, granted, key.id
            )
            raise _FORBIDDEN
        return key

    return _check
