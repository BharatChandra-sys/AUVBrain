from __future__ import annotations

from fastapi import APIRouter, WebSocket

from ..models import ControlMode, VehicleCommand, VehicleState
from ..state import STATE
from .ws import handle_control_ws

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/state", response_model=VehicleState)
async def get_state() -> VehicleState:
    return await STATE.get()


@router.post("/mode", response_model=VehicleState)
async def set_mode(mode: ControlMode) -> VehicleState:
    return await STATE.set_mode(mode)


@router.post("/command")
async def manual_command(command: VehicleCommand) -> dict[str, str]:
    # The agent loop will read STATE and apply in MANUAL mode.
    await STATE.update_command(command, source=None)
    return {"status": "queued"}


@router.websocket("/ws/control")
async def ws_control(websocket: WebSocket) -> None:
    await handle_control_ws(websocket)
