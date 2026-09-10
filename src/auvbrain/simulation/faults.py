"""Fault injector for simulation scenarios — Phase 9.

``FaultInjector`` provides named injection methods that configure the
SimSensorSuite channels and AUVDynamics state to reproduce realistic
fault conditions.  Each method is reversible via ``clear()``.
"""

from __future__ import annotations

import logging

from .dynamics import AUVDynamics
from .sensors import SimSensorSuite

logger = logging.getLogger(__name__)


class FaultInjector:
    """Injects and clears fault conditions in the simulation.

    Usage::
        injector = FaultInjector(dynamics, sensors)
        injector.dvl_dropout()          # GPS/DVL loss
        injector.thruster_degraded(0)   # Motor 0 at 40% thrust
        injector.clear_all()
    """

    def __init__(self, dynamics: AUVDynamics, sensors: SimSensorSuite) -> None:
        self._dyn = dynamics
        self._sensors = sensors
        self._thruster_health = [1.0, 1.0, 1.0, 1.0]

    # ── Sensor faults ─────────────────────────────────────────────────────

    def dvl_dropout(self) -> None:
        """Simulate DVL (velocity sensor) dropout → depth dropout for now."""
        self._sensors.depth.dropout_prob = 1.0   # depth as DVL proxy
        logger.info("fault injected: dvl_dropout")

    def imu_bias(self, bias_deg_s: float = 5.0) -> None:
        """Introduce a heading bias simulating IMU drift."""
        self._dyn.vyaw += bias_deg_s * 0.017453  # deg/s to rad/s
        logger.info("fault injected: imu_bias=%.1f deg/s", bias_deg_s)

    def depth_drift(self, bias_m: float = 3.0) -> None:
        """Add a constant bias to depth readings."""
        self._sensors.depth.bias = bias_m
        logger.info("fault injected: depth_drift=%.1f m", bias_m)

    def sensor_freeze(self, sensor: str) -> None:
        """Freeze a specific sensor at its last value."""
        chan = getattr(self._sensors, sensor, None)
        if chan is None:
            logger.warning("unknown sensor for freeze: %s", sensor)
            return
        chan.freeze()
        logger.info("fault injected: sensor_freeze=%s", sensor)

    def battery_critical(self) -> None:
        """Drop battery to critical level."""
        self._sensors.drain_battery(10.5)
        logger.info("fault injected: battery_critical")

    def water_ingress(self) -> None:
        self._sensors.inject_water_ingress()
        logger.info("fault injected: water_ingress")

    def add_current(self, north_ms: float, east_ms: float) -> None:
        """Inject a water current disturbance."""
        self._dyn.set_current(north_ms, east_ms)
        logger.info("fault injected: current north=%.2f east=%.2f m/s", north_ms, east_ms)

    # ── Thruster faults ───────────────────────────────────────────────────

    def thruster_degraded(self, motor_idx: int, health: float = 0.4) -> None:
        """Reduce a specific motor's effective thrust."""
        if 0 <= motor_idx < len(self._thruster_health):
            self._thruster_health[motor_idx] = health
            logger.info("fault injected: thruster_degraded motor=%d health=%.2f", motor_idx, health)

    def thruster_unresponsive(self, motor_idx: int) -> None:
        self.thruster_degraded(motor_idx, health=0.0)
        logger.info("fault injected: thruster_unresponsive motor=%d", motor_idx)

    @property
    def thruster_health(self) -> tuple[float, float, float, float]:
        return tuple(self._thruster_health)  # type: ignore[return-value]

    # ── Reset ─────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Remove all injected faults and restore nominal operation."""
        self._sensors.depth.dropout_prob    = 0.0
        self._sensors.depth.bias            = 0.0
        self._sensors.depth.unfreeze()
        self._sensors.battery.dropout_prob  = 0.0
        self._sensors.clear_water_ingress()
        self._dyn.set_current(0.0, 0.0)
        self._thruster_health               = [1.0, 1.0, 1.0, 1.0]
        logger.info("all faults cleared")
