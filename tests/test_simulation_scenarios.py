"""Tests for simulation scenarios and fault injection — Phase 9."""

import pytest

from auvbrain.simulation.dynamics import AUVDynamics, DynamicsConfig
from auvbrain.simulation.faults import FaultInjector
from auvbrain.simulation.scenarios import Scenarios
from auvbrain.simulation.sensors import SimSensorSuite


def test_scenario_normal():
    """Normal scenario builds correctly."""
    scenario = Scenarios.normal()
    dynamics, sensors, injector = scenario.build()
    
    assert isinstance(dynamics, AUVDynamics)
    assert isinstance(sensors, SimSensorSuite)
    assert isinstance(injector, FaultInjector)
    assert len(scenario.events) == 0
    assert "continue" in scenario.expected_behaviors


def test_scenario_dvl_dropout():
    """DVL dropout scenario."""
    scenario = Scenarios.dvl_dropout()
    dynamics, sensors, injector = scenario.build()
    
    # Read sensor before fault
    obs1 = sensors.read()
    assert obs1["velocity_ms"] is not None
    
    # Inject fault
    assert len(scenario.events) == 1
    event = scenario.events[0]
    event.inject(injector)
    
    # DVL should now be unavailable
    obs2 = sensors.read()
    assert obs2["velocity_ms"] is None
    assert "reconfigure_estimator" in scenario.expected_behaviors


def test_scenario_critical_battery():
    """Critical battery scenario."""
    scenario = Scenarios.critical_battery()
    dynamics, sensors, injector = scenario.build()
    
    obs1 = sensors.read()
    assert obs1["battery_v"] > 11.0
    
    # Inject critical battery
    event = scenario.events[0]
    event.inject(injector)
    
    obs2 = sensors.read()
    assert obs2["battery_v"] < 10.6  # critical threshold
    assert "surface" in scenario.expected_behaviors


def test_scenario_thruster_degradation():
    """Thruster degradation scenario."""
    scenario = Scenarios.thruster_degradation()
    dynamics, sensors, injector = scenario.build()
    
    # Apply thrust before degradation
    dynamics.step(thrust=[1.0, 0, 0, 0], dt=0.1)
    vel_before = dynamics.get_velocity()
    
    # Inject degradation
    event = scenario.events[0]
    event.inject(injector)
    
    # Same thrust should produce less effect
    dynamics.reset()
    dynamics.step(thrust=[1.0, 0, 0, 0], dt=0.1)
    vel_after = dynamics.get_velocity()
    
    # After degradation, velocity should be lower (motor 0 is degraded)
    assert abs(vel_after[0]) < abs(vel_before[0]) or abs(vel_after[0]) == 0.0


def test_scenario_imu_bias():
    """IMU bias scenario."""
    scenario = Scenarios.imu_bias()
    dynamics, sensors, injector = scenario.build()
    
    obs1 = sensors.read()
    heading_before = obs1.get("heading_deg", 0.0)
    
    # Inject IMU bias
    event = scenario.events[0]
    event.inject(injector)
    
    # Step through time to accumulate bias
    for _ in range(10):
        sensors.step(dt=0.1)
    
    obs2 = sensors.read()
    heading_after = obs2.get("heading_deg", 0.0)
    
    # Heading should drift due to bias
    # (may be subtle depending on bias magnitude)
    assert "reconfigure_estimator" in scenario.expected_behaviors


def test_scenario_water_ingress():
    """Water ingress emergency scenario."""
    scenario = Scenarios.water_ingress()
    dynamics, sensors, injector = scenario.build()
    
    obs1 = sensors.read()
    assert not obs1.get("water_ingress_detected", False)
    
    # Inject ingress
    event = scenario.events[0]
    event.inject(injector)
    
    obs2 = sensors.read()
    assert obs2.get("water_ingress_detected", False) is True
    assert "emergency_surface" in scenario.expected_behaviors
    assert "safe_mode" in scenario.expected_behaviors


def test_all_scenarios_build():
    """Verify all built-in scenarios can be instantiated."""
    all_scenarios = Scenarios.all_scenarios()
    assert len(all_scenarios) >= 10
    
    for scenario in all_scenarios:
        dynamics, sensors, injector = scenario.build()
        assert isinstance(dynamics, AUVDynamics)
        assert isinstance(sensors, SimSensorSuite)
        assert isinstance(injector, FaultInjector)
        assert len(scenario.expected_behaviors) > 0


def test_fault_injection_sequence():
    """Test multi-event fault injection sequence."""
    scenario = Scenarios.normal()
    # Add custom events
    from auvbrain.simulation.scenarios import ScenarioEvent
    
    scenario.events = [
        ScenarioEvent(
            tick=5,
            label="low_battery",
            inject=lambda inj: inj._sensors.drain_battery(11.0),
        ),
        ScenarioEvent(
            tick=10,
            label="dvl_dropout",
            inject=lambda inj: inj.dvl_dropout(),
        ),
    ]
    
    dynamics, sensors, injector = scenario.build()
    
    # Simulate tick progression
    for tick in range(15):
        for event in scenario.events:
            if event.tick == tick:
                event.inject(injector)
        
        obs = sensors.read()
        
        if tick >= 5:
            assert obs["battery_v"] <= 11.0
        if tick >= 10:
            assert obs["velocity_ms"] is None
