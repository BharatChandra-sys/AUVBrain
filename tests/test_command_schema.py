from auvbrain.models import ThrusterCommand, VehicleCommand
from auvbrain.config import Settings
from auvbrain.models import Observation
from auvbrain.safety.monitor import SafetyMonitor


def test_thruster_bounds() -> None:
    cmd = ThrusterCommand(surge=0.0, sway=0.0, heave=0.0, yaw=0.0)
    assert cmd.surge == 0.0


def test_vehicle_command_defaults() -> None:
    cmd = VehicleCommand()
    assert cmd.experiment.enabled is False


def test_safety_forces_safe_on_ingress() -> None:
    settings = Settings()
    safety = SafetyMonitor(settings)

    obs = Observation(water_ingress=True)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode is not None
    assert "water ingress" in (cmd.note or "")


def test_safety_forces_safe_on_overtemp() -> None:
    settings = Settings(max_internal_temp_c=40.0)
    safety = SafetyMonitor(settings)

    obs = Observation(internal_temp_c=60.0)
    mode, cmd = safety.enforce(obs, VehicleCommand())
    assert mode is not None
    assert "overtemperature" in (cmd.note or "")
