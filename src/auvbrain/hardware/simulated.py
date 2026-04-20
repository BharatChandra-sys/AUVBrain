from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from random import random

from ..models import Observation, VehicleCommand

logger = logging.getLogger(__name__)


class SimulatedSensors:
    def __init__(self) -> None:
        self._depth = 0.0
        self._battery = 12.4

    async def read(self) -> Observation:
        # very small fake dynamics
        await asyncio.sleep(0)
        self._battery = max(10.0, self._battery - 0.0005)
        obstacle = 1.5 + (random() * 3.0)
        return Observation(
            depth_m=self._depth,
            battery_v=self._battery,
            obstacle_front_m=obstacle,
            sensors={"temp_c": 24.0 + random()},
        )


class SimulatedThrusters:
    async def apply(self, command: VehicleCommand) -> None:
        logger.info("THRUSTERS %s", command.thrusters.model_dump())


class SimulatedExperimentModule:
    async def apply(self, command: VehicleCommand) -> None:
        if command.experiment.enabled:
            logger.info("EXPERIMENT %s", command.experiment.model_dump())


@dataclass
class SimulatedHardware:
    sensors: SimulatedSensors
    thrusters: SimulatedThrusters
    experiments: SimulatedExperimentModule


def make_simulated_hardware() -> SimulatedHardware:
    return SimulatedHardware(
        sensors=SimulatedSensors(),
        thrusters=SimulatedThrusters(),
        experiments=SimulatedExperimentModule(),
    )
