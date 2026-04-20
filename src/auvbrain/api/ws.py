from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from ..models import ControlMode, ManualControlMessage
from ..state import STATE

logger = logging.getLogger(__name__)


async def handle_control_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({"type": "hello", "msg": "connected"})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = ManualControlMessage.model_validate_json(raw)
            except Exception:
                await ws.send_json({"type": "error", "msg": "invalid message", "raw": raw})
                continue

            if msg.type == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg.type == "set_mode":
                if msg.mode is None:
                    await ws.send_json({"type": "error", "msg": "mode required"})
                    continue
                state = await STATE.set_mode(msg.mode)
                await ws.send_json({"type": "mode", "mode": state.mode})
                continue

            if msg.type == "command":
                if msg.command is None:
                    await ws.send_json({"type": "error", "msg": "command required"})
                    continue
                await STATE.update_command(msg.command, source=None)
                await ws.send_json({"type": "ack"})
                continue

            await ws.send_json({"type": "error", "msg": f"unknown type: {msg.type}"})

    except WebSocketDisconnect:
        logger.info("control websocket disconnected")
    except Exception as e:
        logger.exception("control websocket error: %s", e)
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": "server error"}))
        except Exception:
            pass
