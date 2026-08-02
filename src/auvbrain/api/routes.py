"""FastAPI route definitions for the AUVBrain Control Hub.

Security model
--------------
Read endpoints  (/health, /state, /metrics) require scope ``"read"``.
Write endpoints (/mode, /command)           require scope ``"write"``.
Admin endpoints (/admin/*)                  require scope ``"admin"``.

When ``AUV_AUTH_ENABLED=false`` (dev/CI only), the auth dependency is a no-op.

Rate limiting
-------------
Write endpoints share a per-IP token-bucket limiter configured at startup.
The /metrics and /state endpoints are not rate-limited.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, WebSocket

from ..auth.dependencies import require_scope
from ..config import Settings
from ..db.engine import get_session
from ..db.repositories import ApiKeyRepository
from ..metrics.registry import METRICS
from ..models import ControlMode, VehicleCommand, VehicleState
from ..state import STATE
from .limiter import enforce_rate_limit
from .ws import handle_control_ws

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[attr-defined]


# ── Middleware counter ───────────────────────────────────────────────────────

@router.middleware("http")  # type: ignore[misc]
async def _count_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    METRICS.inc("api_requests_total")
    return await call_next(request)


# ── Health (public — no auth, no rate limit) ─────────────────────────────────

@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe.  Returns 200 when the process is alive."""
    return {"status": "ok"}


# ── Metrics (read scope) ──────────────────────────────────────────────────────

@router.get("/metrics", tags=["ops"])
async def get_metrics(
    settings: Annotated[Settings, Depends(_get_settings)],
    _auth=Depends(require_scope("read") if True else lambda: None),
) -> dict[str, Any]:
    """Runtime counters, gauges, and latency percentiles.

    Exposes: tick counts, mode, battery/depth gauges, LLM fallback rate,
    dropped telemetry, decide/tick period latency percentiles.
    """
    snapshot = METRICS.snapshot()
    snapshot["dropped_telemetry_total"] = METRICS.dropped_telemetry_total
    return snapshot


# ── State (read scope) ────────────────────────────────────────────────────────

@router.get("/state", response_model=VehicleState, tags=["control"])
async def get_state(
    _auth=Depends(require_scope("read")),
) -> VehicleState:
    """Return the current vehicle state snapshot."""
    return await STATE.get()


# ── Mode (write scope + rate limit) ───────────────────────────────────────────

@router.post("/mode", response_model=VehicleState, tags=["control"])
async def set_mode(
    mode: ControlMode,
    _auth=Depends(require_scope("write")),
    _rl=Depends(enforce_rate_limit),
) -> VehicleState:
    """Set the vehicle control mode (AUTONOMOUS / MANUAL / SAFE).

    Requires ``write`` scope.
    """
    logger.info("mode change requested: %s", mode.value)
    return await STATE.set_mode(mode)


# ── Command (write scope + rate limit) ────────────────────────────────────────

@router.post("/command", tags=["control"])
async def manual_command(
    command: VehicleCommand,
    _auth=Depends(require_scope("write")),
    _rl=Depends(enforce_rate_limit),
) -> dict[str, str]:
    """Queue a manual command.  Applied by the agent loop in MANUAL mode.

    Requires ``write`` scope.
    """
    # Source is None for operator-issued commands (not from decision engine)
    await STATE.update_command(command, source=None)
    return {"status": "queued"}


# ── WebSocket control (write scope enforced inside handler) ───────────────────

@router.websocket("/ws/control")
async def ws_control(websocket: WebSocket) -> None:
    """Bidirectional control channel.

    Accepts the same set_mode / command / ping messages as the REST API.
    The first message must include a valid API key via the ``X-API-Key`` header
    or ``?api_key=`` query parameter.
    """
    await handle_control_ws(websocket)


# ── Admin — API key management (admin scope) ──────────────────────────────────

@router.post("/admin/api-keys", tags=["admin"])
async def create_api_key(
    name: str,
    scopes: str = "read write",
    _auth=Depends(require_scope("admin")),
) -> dict[str, Any]:
    """Create a new API key.  Returns the raw token **once**."""
    async with get_session() as session:
        repo = ApiKeyRepository(session)
        record, raw = await repo.create(name=name, scopes=scopes)
    return {
        "id": str(record.id),
        "name": record.name,
        "scopes": record.scopes,
        "token": raw,  # shown once
    }


@router.get("/admin/api-keys", tags=["admin"])
async def list_api_keys(
    _auth=Depends(require_scope("admin")),
) -> list[dict[str, Any]]:
    """List active API keys (no raw tokens exposed)."""
    async with get_session() as session:
        repo = ApiKeyRepository(session)
        keys = await repo.list_active()
    return [
        {
            "id": str(k.id),
            "name": k.name,
            "scopes": k.scopes,
            "created_at": k.created_at.isoformat(),
            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        }
        for k in keys
    ]


@router.delete("/admin/api-keys/{key_id}", tags=["admin"])
async def revoke_api_key(
    key_id: str,
    _auth=Depends(require_scope("admin")),
) -> dict[str, str]:
    """Revoke an API key by ID."""
    import uuid

    async with get_session() as session:
        repo = ApiKeyRepository(session)
        ok = await repo.revoke(uuid.UUID(key_id))
    if not ok:
        from fastapi import HTTPException, status
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key not found.")
    return {"status": "revoked"}
