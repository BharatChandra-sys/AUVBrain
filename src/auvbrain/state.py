from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .models import DecisionSource, Observation, ControlMode, VehicleCommand, VehicleState


@dataclass
class StateStore:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _state: VehicleState = field(default_factory=VehicleState)

    async def get(self) -> VehicleState:
        async with self._lock:
            return self._state.model_copy(deep=True)

    async def get_mode_and_last_command(self) -> tuple[ControlMode, VehicleCommand | None]:
        """Low-overhead snapshot for the agent loop.

        The agent loop only needs the current mode and the last (manual) command.
        Avoiding a deep copy of the full VehicleState reduces tail latency.
        """

        async with self._lock:
            mode = self._state.mode
            cmd = self._state.last_command

        # Copy outside the lock.
        return mode, (cmd.model_copy(deep=True) if cmd is not None else None)

    async def set_mode(self, mode: ControlMode) -> VehicleState:
        async with self._lock:
            self._state.mode = mode
            return self._state.model_copy(deep=True)

    async def update_observation(self, obs: Observation) -> None:
        async with self._lock:
            self._state.last_observation = obs

    async def update_command(self, cmd: VehicleCommand, source: Optional[DecisionSource]) -> None:
        async with self._lock:
            self._state.last_command = cmd
            self._state.last_decision_source = source


STATE = StateStore()
