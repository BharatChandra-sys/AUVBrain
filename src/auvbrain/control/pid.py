from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PID:
    kp: float
    ki: float
    kd: float
    integrator: float = 0.0
    prev_error: float = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        self.integrator += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integrator + self.kd * derivative
