from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..models import ControlMode, Observation, VehicleCommand


@dataclass
class SafetyMonitor:
    settings: Settings

    def enforce(self, obs: Observation, cmd: VehicleCommand) -> tuple[ControlMode | None, VehicleCommand]:
        # Hard safety limits that can override any decision engine.
        if obs.water_ingress is True:
            safe_cmd = VehicleCommand(note="SAFE: water ingress")
            return ControlMode.SAFE, safe_cmd

        if obs.depth_m > self.settings.max_depth_m:
            safe_cmd = VehicleCommand(note="SAFE: max depth exceeded")
            return ControlMode.SAFE, safe_cmd

        if obs.battery_v < self.settings.min_battery_v:
            safe_cmd = VehicleCommand(note="SAFE: low battery")
            return ControlMode.SAFE, safe_cmd

        if obs.internal_temp_c is not None and obs.internal_temp_c > self.settings.max_internal_temp_c:
            safe_cmd = VehicleCommand(note="SAFE: overtemperature")
            return ControlMode.SAFE, safe_cmd

        if obs.pressure_bar is not None and obs.pressure_bar > self.settings.max_pressure_bar:
            safe_cmd = VehicleCommand(note="SAFE: overpressure")
            return ControlMode.SAFE, safe_cmd

        # Emergency obstacle stop even if the policy is wrong/noisy.
        if obs.obstacle_front_m is not None and obs.obstacle_front_m < self.settings.emergency_obstacle_m:
            safe_cmd = VehicleCommand(note=f"SAFE: obstacle {obs.obstacle_front_m:.2f}m")
            return None, safe_cmd

        return None, cmd
