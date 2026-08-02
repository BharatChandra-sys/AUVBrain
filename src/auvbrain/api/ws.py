"""WebSocket control handler.

Auth
----
The WS auth handshake is performed on the first ``hello`` acknowledgement
using the ``X-API-Key`` header (passed when upgrading the connection).

The write scope is required for ``set_mode`` and ``command`` messages.
Read-scope callers can still connect and receive pong responses, but any
write attempt returns an error frame.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from ..auth.dependencies import _resolve_key, _UNAUTHORIZED
from ..metrics.registry import METRICS
from ..models import ControlMode, ManualControlMessage
from ..state import STATE

logger = logging.getLogger(__name__)


async def handle_control_ws(ws: WebSocket) -> None:
    # ── Authenticate via X-API-Key header or ?api_key= query param ─────────
    raw_key = (
        ws.headers.get("x-api-key")
        or ws.query_params.get("api_key")
    )

    # Attempt authentication before accepting so we can reject with 403
    api_key = None
    if raw_key:
        try:
            from ..db.engine import get_session
            from ..db.repositories import ApiKeyRepository
            import hashlib
            from datetime import timezone

            async with get_session() as session:
                repo = ApiKeyRepository(session)
                api_key = await repo.get_by_hash(raw_key)

            if api_key is None or not api_key.is_active:
                api_key = None
            elif api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
                api_key = None

        except Exception:
            logger.debug("WS auth lookup failed", exc_info=True)
            api_key = None

    if api_key is None:
        METRICS.inc("auth_failures_total")
        await ws.close(code=4001, reason="Unauthorized")
        return

    has_write = "write" in api_key.scopes.split()

    await ws.accept()
    METRICS.inc("ws_connects_total")
    await ws.send_json({
        "type": "hello",
        "msg": "connected",
        "scopes": api_key.scopes,
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = ManualControlMessage.model_validate_json(raw)
            except Exception:
                await ws.send_json({
                    "type": "error",
                    "msg": "invalid message",
                    "raw": raw[:200],
                })
                continue

            if msg.type == "ping":
                await ws.send_json({
                    "type": "pong",
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
                continue

            if msg.type == "set_mode":
                if not has_write:
                    await ws.send_json({"type": "error", "msg": "write scope required"})
                    continue
                if msg.mode is None:
                    await ws.send_json({"type": "error", "msg": "mode required"})
                    continue
                state = await STATE.set_mode(msg.mode)
                await ws.send_json({"type": "mode", "mode": state.mode.value})
                logger.info("WS mode change: %s", msg.mode.value)
                continue

            if msg.type == "command":
                if not has_write:
                    await ws.send_json({"type": "error", "msg": "write scope required"})
                    continue
                if msg.command is None:
                    await ws.send_json({"type": "error", "msg": "command required"})
                    continue
                await STATE.update_command(msg.command, source=None)
                await ws.send_json({"type": "ack"})
                continue

            await ws.send_json({"type": "error", "msg": f"unknown type: {msg.type}"})

    except WebSocketDisconnect:
        logger.info("control websocket disconnected")
    except Exception as exc:
        logger.exception("control websocket error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": "server error"}))
        except Exception:
            pass
    finally:
        METRICS.inc("ws_disconnects_total")
