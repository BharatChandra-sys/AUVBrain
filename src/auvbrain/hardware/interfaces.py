from __future__ import annotations

from typing import Protocol

from ..models import Observation, VehicleCommand


class SensorSuite(Protocol):
    async def read(self) -> Observation: ...


class Thrusters(Protocol):
    async def apply(self, command: VehicleCommand) -> None: ...


class ExperimentModule(Protocol):
    async def apply(self, command: VehicleCommand) -> None: ...


class HardwareBundle(Protocol):
    sensors: SensorSuite
    thrusters: Thrusters
    experiments: ExperimentModule
