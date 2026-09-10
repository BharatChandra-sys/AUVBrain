"""Built-in simulation scenarios — Phase 9.

A ``Scenario`` specifies:
  - Initial conditions
  - Fault injection sequence (time → injection fn)
  - Expected behaviours (for assertion in tests)

The full fault-injection matrix from plan.md §20 is represented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .dynamics import AUVDynamics, DynamicsConfig
from .faults import FaultInjector
from .sensors import SimSensorSuite


@dataclass
class ScenarioEvent:
    """A timed fault injection event within a scenario."""
    tick: int
    label: str
    inject: Callable[[FaultInjector], None]


@dataclass
class Scenario:
    """A complete simulation scenario with dynamics, sensors, and fault timeline.

    Usage::
        scenario = Scenarios.dvl_dropout()
        dynamics, sensors, injector = scenario.build()
    """
    name: str
    description: str
    dynamics_config: DynamicsConfig = field(default_factory=DynamicsConfig)
    events: list[ScenarioEvent] = field(default_factory=list)
    initial_battery_v: float = 12.4
    initial_current: tuple[float, float] = (0.0, 0.0)
    expected_behaviors: list[str] = field(default_factory=list)

    def build(self) -> tuple[AUVDynamics, SimSensorSuite, FaultInjector]:
        """Instantiate dynamics, sensors, and injector for this scenario."""
        dyn = AUVDynamics(config=self.dynamics_config)
        dyn.set_current(*self.initial_current)
        sensors = SimSensorSuite(dyn, initial_battery_v=self.initial_battery_v)
        injector = FaultInjector(dyn, sensors)
        return dyn, sensors, injector


class Scenarios:
    """Factory for the fault-injection test matrix from plan.md §20."""

    @staticmethod
    def normal() -> Scenario:
        return Scenario(
            name="normal",
            description="Nominal operation — no faults injected.",
            expected_behaviors=["continue"],
        )

    @staticmethod
    def low_battery() -> Scenario:
        return Scenario(
            name="low_battery",
            description="Battery drains below low threshold.",
            events=[ScenarioEvent(
                tick=20,
                label="drain_to_low",
                inject=lambda inj: inj._sensors.drain_battery(11.0),
            )],
            expected_behaviors=["replan", "reduce_speed"],
        )

    @staticmethod
    def critical_battery() -> Scenario:
        return Scenario(
            name="critical_battery",
            description="Battery critically low — vehicle should return/surface.",
            events=[ScenarioEvent(
                tick=10,
                label="drain_to_critical",
                inject=lambda inj: inj.battery_critical(),
            )],
            expected_behaviors=["return", "surface"],
        )

    @staticmethod
    def dvl_dropout() -> Scenario:
        return Scenario(
            name="dvl_dropout",
            description="DVL drops out — navigation confidence should decrease.",
            events=[ScenarioEvent(
                tick=15,
                label="dvl_out",
                inject=lambda inj: inj.dvl_dropout(),
            )],
            expected_behaviors=["reconfigure_estimator", "reduce_speed"],
        )

    @staticmethod
    def imu_bias() -> Scenario:
        return Scenario(
            name="imu_bias",
            description="IMU develops a heading bias.",
            events=[ScenarioEvent(
                tick=10,
                label="imu_bias",
                inject=lambda inj: inj.imu_bias(bias_deg_s=8.0),
            )],
            expected_behaviors=["reconfigure_estimator"],
        )

    @staticmethod
    def depth_drift() -> Scenario:
        return Scenario(
            name="depth_drift",
            description="Depth sensor drifts with a constant positive bias.",
            events=[ScenarioEvent(
                tick=10,
                label="depth_bias",
                inject=lambda inj: inj.depth_drift(bias_m=4.0),
            )],
            expected_behaviors=["cross_check", "reconfigure_estimator"],
        )

    @staticmethod
    def thruster_degradation() -> Scenario:
        return Scenario(
            name="thruster_degradation",
            description="Motor 0 degrades to 30% thrust.",
            events=[ScenarioEvent(
                tick=5,
                label="motor0_degrade",
                inject=lambda inj: inj.thruster_degraded(0, health=0.3),
            )],
            expected_behaviors=["replan_route", "reduce_speed"],
        )

    @staticmethod
    def obstacle_replan() -> Scenario:
        return Scenario(
            name="obstacle_replan",
            description="Close obstacle requires replanning.",
            events=[ScenarioEvent(
                tick=10,
                label="obstacle_close",
                inject=lambda inj: setattr(inj._sensors.obstacle, "bias", -1.8),
            )],
            expected_behaviors=["replan"],
        )

    @staticmethod
    def strong_current() -> Scenario:
        return Scenario(
            name="strong_current",
            description="Strong water current causes drift.",
            initial_current=(0.8, 0.3),
            expected_behaviors=["replan"],
        )

    @staticmethod
    def communication_loss() -> Scenario:
        return Scenario(
            name="communication_loss",
            description="No operator commands — vehicle continues autonomously.",
            expected_behaviors=["continue"],
        )

    @staticmethod
    def llm_timeout() -> Scenario:
        return Scenario(
            name="llm_timeout",
            description="LLM decision engine times out — rules fallback activates.",
            expected_behaviors=["rules_fallback"],
        )

    @staticmethod
    def water_ingress() -> Scenario:
        return Scenario(
            name="water_ingress",
            description="Water ingress detected — immediate SAFE override.",
            events=[ScenarioEvent(
                tick=5,
                label="ingress",
                inject=lambda inj: inj.water_ingress(),
            )],
            expected_behaviors=["emergency_surface", "safe_mode"],
        )

    @staticmethod
    def sensor_freeze() -> Scenario:
        return Scenario(
            name="sensor_freeze",
            description="Depth sensor freezes at a constant value.",
            events=[ScenarioEvent(
                tick=10,
                label="freeze_depth",
                inject=lambda inj: inj.sensor_freeze("depth"),
            )],
            expected_behaviors=["reconfigure_estimator"],
        )

    @classmethod
    def all_scenarios(cls) -> list[Scenario]:
        return [
            cls.normal(),
            cls.low_battery(),
            cls.critical_battery(),
            cls.dvl_dropout(),
            cls.imu_bias(),
            cls.depth_drift(),
            cls.thruster_degradation(),
            cls.obstacle_replan(),
            cls.strong_current(),
            cls.communication_loss(),
            cls.llm_timeout(),
            cls.water_ingress(),
            cls.sensor_freeze(),
        ]
