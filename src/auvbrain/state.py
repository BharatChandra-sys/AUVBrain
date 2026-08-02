"""In-process vehicle state store.

A single ``StateStore`` instance (``STATE``) is shared between the agent loop
and the API handlers.  All mutations are serialised through an ``asyncio.Lock``
to guarantee consistency without race conditions.

Dead-man's switch
-----------------
When in MANUAL mode, if no new command is posted within ``manual_deadman_s``
seconds the next ``check_manual_deadman()`` call returns ``True`` and the
caller (agent loop) must transition to SAFE.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from .models import DecisionSource, Observation, ControlMode, VehicleCommand, VehicleState


@dataclass
class StateStore:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _state: VehicleState = field(default_factory=VehicleState)

    # Monotonic timestamp of the last manual command received
    _last_manual_command_ts: float = field(default_factory=time.monotonic)

    # ── Read ─────────────────────────────────────────────────────────────

    async def get(self) -> VehicleState:
        async with self._lock:
            return self._state.model_copy(deep=True)

    async def get_mode_and_last_command(self) -> tuple[ControlMode, VehicleCommand | None]:
        """Low-overhead snapshot for the agent loop.

        Avoids a deep copy of the full VehicleState on every tick.
        """
        async with self._lock:
            mode = self._state.mode
            cmd = self._state.last_command
        return mode, (cmd.model_copy(deep=True) if cmd is not None else None)

    # ── Write ────────────────────────────────────────────────────────────

    async def set_mode(self, mode: ControlMode) -> VehicleState:
        async with self._lock:
            self._state.mode = mode
            return self._state.model_copy(deep=True)

    async def update_observation(self, obs: Observation) -> None:
        async with self._lock:
            self._state.last_observation = obs

    async def update_command(
        self, cmd: VehicleCommand, source: Optional[DecisionSource]
    ) -> None:
        async with self._lock:
            self._state.last_command = cmd
            self._state.last_decision_source = source
            # Track when the last command arrived so the dead-man's switch
            # can compare against the idle time.
            self._last_manual_command_ts = time.monotonic()

    # ── Dead-man's switch ────────────────────────────────────────────────

    async def check_manual_deadman(self, deadman_s: float) -> bool:
        """Return True if MANUAL mode has been idle longer than ``deadman_s``.

        A return value of True means the agent loop should transition to SAFE.
        Disabled (returns False always) when ``deadman_s <= 0``.
        """
        if deadman_s <= 0:
            return False
        async with self._lock:
            mode = self._state.mode
            if mode != ControlMode.MANUAL:
                return False
            idle_s = time.monotonic() - self._last_manual_command_ts
            return idle_s > deadman_s


STATE = StateStore()
