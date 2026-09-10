"""AUV dynamics simulation — Phase 9.

Models the physics of an AUV with:
  - Added mass and drag in 6 DOF (simplified diagonal model)
  - Thruster saturation and degradation
  - Water current disturbance

This replaces the trivial SimulatedSensors that just returns constants,
giving us a physics-grounded sim we can inject faults into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DynamicsConfig:
    """Physical parameters for the AUV dynamics model."""
    mass_kg: float          = 12.0    # vehicle mass including added mass
    drag_surge: float       = 3.0     # N·s/m linear drag coefficient
    drag_sway: float        = 5.0     # N·s/m (higher for non-streamlined sway)
    drag_heave: float       = 4.0     # N·s/m
    drag_yaw: float         = 1.5     # N·m·s/rad
    max_thrust_n: float     = 20.0    # max total thrust per axis in Newtons
    gravity_ms2: float      = 9.81
    water_density: float    = 1025.0  # kg/m³ (seawater)
    buoyancy_offset: float  = 0.0     # net buoyancy force (+ve = positively buoyant)


@dataclass
class AUVDynamics:
    """6-DOF simplified AUV dynamics integrator.

    State: [x, y, z(depth), roll, pitch, yaw, vx, vy, vz, vyaw]

    Usage::
        dyn = AUVDynamics()
        obs = dyn.step(surge=0.3, sway=0.0, heave=0.1, yaw=0.0, dt=0.1)
    """

    config: DynamicsConfig = field(default_factory=DynamicsConfig)

    # State
    x: float   = 0.0    # north position (m)
    y: float   = 0.0    # east position (m)
    z: float   = 0.0    # depth (m, positive down)
    yaw: float = 0.0    # heading (rad)
    vx: float  = 0.0    # surge velocity (m/s)
    vy: float  = 0.0    # sway velocity (m/s)
    vz: float  = 0.0    # heave velocity (m/s)
    vyaw: float = 0.0   # yaw rate (rad/s)

    # External disturbance (current)
    current_x: float = 0.0
    current_y: float = 0.0

    def step(
        self,
        surge: float,
        sway: float,
        heave: float,
        yaw_cmd: float,
        dt: float,
        *,
        thruster_health: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> dict:
        """Integrate dynamics one step and return sensor-equivalent readings.

        Args:
            surge/sway/heave/yaw_cmd: Normalised commands [-1, 1].
            dt: Time step in seconds.
            thruster_health: Per-motor health multiplier [0, 1].

        Returns:
            dict with depth_m, vx, vy, vz, heading_deg suitable for Observation.
        """
        cfg = self.config
        avg_health = sum(thruster_health) / len(thruster_health)

        # Convert normalised commands to forces (N) / torques (N·m)
        Fx  = surge    * cfg.max_thrust_n * avg_health
        Fy  = sway     * cfg.max_thrust_n * avg_health
        Fz  = heave    * cfg.max_thrust_n * avg_health
        Mz  = yaw_cmd  * cfg.max_thrust_n * avg_health * 0.3  # torque arm

        # Drag (linear model in body frame)
        Fx -= cfg.drag_surge * self.vx
        Fy -= cfg.drag_sway  * self.vy
        Fz -= cfg.drag_heave * self.vz
        Mz -= cfg.drag_yaw   * self.vyaw

        # Buoyancy
        Fz -= cfg.buoyancy_offset

        # Acceleration
        ax   = Fx / cfg.mass_kg
        ay   = Fy / cfg.mass_kg
        az   = Fz / cfg.mass_kg
        ayaw = Mz / (cfg.mass_kg * 0.1)  # simplified moment of inertia

        # Integrate velocity
        self.vx   += ax   * dt
        self.vy   += ay   * dt
        self.vz   += az   * dt
        self.vyaw += ayaw * dt

        # Integrate position (rotate body velocity to world frame)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        self.x += (self.vx * cos_yaw - self.vy * sin_yaw + self.current_x) * dt
        self.y += (self.vx * sin_yaw + self.vy * cos_yaw + self.current_y) * dt
        self.z += self.vz * dt
        self.yaw += self.vyaw * dt

        # Clamp depth to surface
        self.z = max(0.0, self.z)

        return {
            "x_m": self.x,
            "y_m": self.y,
            "depth_m": self.z,
            "heading_deg": math.degrees(self.yaw) % 360.0,
            "vx_ms": self.vx,
            "vy_ms": self.vy,
            "vz_ms": self.vz,
        }

    def set_current(self, x_ms: float, y_ms: float) -> None:
        self.current_x = x_ms
        self.current_y = y_ms
