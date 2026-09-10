"""Simulation sensor suite with configurable noise, bias, and dropout — Phase 9.

This replaces the trivial SimulatedSensors with a physics-backed sensor model.
Each channel has independently configurable:
  - Gaussian noise (sigma)
  - Constant bias
  - Dropout probability (channel returns None)
  - Freeze (stuck-at-last-value)

These parameters can be set per-channel, making it trivial to inject realistic
fault scenarios into the agent loop.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..models import Observation
from .dynamics import AUVDynamics


@dataclass
class SensorChannel:
    """Configuration and state for a single simulated sensor channel."""
    noise_sigma: float   = 0.0
    bias: float          = 0.0
    dropout_prob: float  = 0.0   # probability of returning None each tick
    frozen: bool         = False  # True = stuck at last value
    _last_value: Optional[float] = field(default=None, repr=False)

    def sample(self, true_value: float) -> Optional[float]:
        """Apply noise, bias, dropout and freeze to the true value."""
        if self.frozen:
            return self._last_value if self._last_value is not None else true_value

        if random.random() < self.dropout_prob:
            return None  # sensor dropout

        value = true_value + self.bias
        if self.noise_sigma > 0:
            value += random.gauss(0.0, self.noise_sigma)

        self._last_value = value
        return value

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False


class SimSensorSuite:
    """Physics-backed sensor suite for the simulation lab.

    Replaces ``SimulatedSensors`` in ``hardware/simulated.py`` for fault-injection
    scenarios.  The normal SIM hardware adapter still works for benchmarks.

    Usage::
        dynamics = AUVDynamics()
        sensors = SimSensorSuite(dynamics)
        sensors.depth.noise_sigma = 0.05    # add 5cm noise
        sensors.battery.dropout_prob = 0.1  # 10% dropout
        obs = await sensors.read()
    """

    def __init__(
        self,
        dynamics: AUVDynamics,
        *,
        initial_battery_v: float = 12.4,
        battery_drain_rate: float = 0.0002,  # V per tick
    ) -> None:
        self._dyn = dynamics
        self._battery_v = initial_battery_v
        self._battery_drain = battery_drain_rate
        self._tick = 0

        # Per-channel config (all healthy by default)
        self.depth     = SensorChannel(noise_sigma=0.02)
        self.battery   = SensorChannel(noise_sigma=0.01)
        self.temp      = SensorChannel(noise_sigma=0.5, bias=0.0)
        self.pressure  = SensorChannel(noise_sigma=0.01)
        self.obstacle  = SensorChannel(noise_sigma=0.1)

        self._water_ingress: bool = False

    async def read(self) -> Observation:
        await asyncio.sleep(0)  # yield to event loop
        self._tick += 1

        # Drain battery
        self._battery_v = max(9.0, self._battery_v - self._battery_drain)

        # Get true state from dynamics
        true_depth = self._dyn.z
        true_pressure = 1.0 + true_depth / 10.0   # rough seawater conversion

        depth_m    = self.depth.sample(true_depth)
        battery_v  = self.battery.sample(self._battery_v)
        temp_c     = self.temp.sample(25.0)
        pressure   = self.pressure.sample(true_pressure)

        # Simulate obstacle: random between 2–10 m unless overridden
        obstacle_m = self.obstacle.sample(2.0 + random.random() * 8.0)

        obs = Observation(
            depth_m=depth_m if depth_m is not None else true_depth,
            battery_v=battery_v if battery_v is not None else self._battery_v,
            internal_temp_c=temp_c,
            pressure_bar=pressure if pressure is not None else true_pressure,
            obstacle_front_m=obstacle_m,
            water_ingress=self._water_ingress,
            sensors={
                "x_m": round(self._dyn.x, 3),
                "y_m": round(self._dyn.y, 3),
                "heading_deg": round(math.degrees(self._dyn.yaw) % 360.0, 1),
            },
        )
        return obs

    def inject_water_ingress(self) -> None:
        self._water_ingress = True

    def clear_water_ingress(self) -> None:
        self._water_ingress = False

    def drain_battery(self, to_v: float) -> None:
        """Set battery voltage directly for fault injection."""
        self._battery_v = to_v
