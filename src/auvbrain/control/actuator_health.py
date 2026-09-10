"""Actuator health tracking.

Goes beyond command validation to actually track whether each motor is
responding as expected.  A health score of 1.0 means full performance;
below 0.5 the motor is considered degraded and the fault system is notified.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MotorHealth:
    """Health descriptor for a single motor/thruster channel.

    Attributes:
        motor_id:          Index (0–3) or name.
        commanded:         Most recent normalised command [-1, 1].
        estimated_response:Estimated actual thrust fraction [-1, 1].
        current_a:         Motor current in Amps (None if unavailable).
        temp_c:            Motor temperature in °C (None if unavailable).
        response_delay_ms: Measured response latency.
        health_score:      Aggregate [0, 1]; 1.0 = fully healthy.
    """

    motor_id: str
    commanded: float = 0.0
    estimated_response: float = 0.0
    current_a: Optional[float] = None
    temp_c: Optional[float] = None
    response_delay_ms: float = 0.0
    health_score: float = 1.0
    _last_update: float = field(default_factory=time.monotonic, repr=False)

    @property
    def degraded(self) -> bool:
        return self.health_score < 0.5

    @property
    def age_s(self) -> float:
        return time.monotonic() - self._last_update


class ActuatorHealth:
    """Tracks health of all motor channels.

    Usage::

        ah = ActuatorHealth(n_motors=4)
        ah.update(0, commanded=0.5, measured_current_a=2.1, temp_c=35.0)
        score = ah.motor(0).health_score
    """

    def __init__(self, n_motors: int = 4) -> None:
        self._motors: dict[str, MotorHealth] = {
            str(i): MotorHealth(motor_id=str(i)) for i in range(n_motors)
        }

        # Limits for health scoring
        self._max_current_a: float = 15.0
        self._max_temp_c: float = 80.0
        self._max_delay_ms: float = 100.0
        self._max_response_error: float = 0.3  # |commanded − response|

    def update(
        self,
        motor_id: int | str,
        *,
        commanded: float = 0.0,
        estimated_response: Optional[float] = None,
        measured_current_a: Optional[float] = None,
        temp_c: Optional[float] = None,
        response_delay_ms: float = 0.0,
    ) -> MotorHealth:
        """Update motor state and recompute health score."""
        key = str(motor_id)
        if key not in self._motors:
            self._motors[key] = MotorHealth(motor_id=key)

        m = self._motors[key]
        m.commanded = commanded
        m.estimated_response = estimated_response if estimated_response is not None else commanded
        m.current_a = measured_current_a
        m.temp_c = temp_c
        m.response_delay_ms = response_delay_ms
        m._last_update = time.monotonic()

        # Compute health score as minimum of sub-scores
        scores: list[float] = [1.0]

        # Response tracking: if estimated response deviates much from command
        if estimated_response is not None:
            err = abs(commanded - estimated_response)
            scores.append(max(0.0, 1.0 - err / max(self._max_response_error, 1e-6)))

        # Current: high current → motor strain
        if measured_current_a is not None:
            scores.append(max(0.0, 1.0 - measured_current_a / self._max_current_a))

        # Temperature: thermal throttling risk
        if temp_c is not None:
            scores.append(max(0.0, 1.0 - temp_c / self._max_temp_c))

        # Latency
        if response_delay_ms > 0:
            scores.append(max(0.0, 1.0 - response_delay_ms / self._max_delay_ms))

        m.health_score = round(min(scores), 3)

        if m.degraded:
            logger.warning(
                "motor %s degraded: health=%.2f current=%.1fA temp=%.1f°C delay=%.1fms",
                key, m.health_score,
                measured_current_a or 0.0, temp_c or 0.0, response_delay_ms,
            )
        return m

    def motor(self, motor_id: int | str) -> Optional[MotorHealth]:
        return self._motors.get(str(motor_id))

    def all_motors(self) -> list[MotorHealth]:
        return list(self._motors.values())

    @property
    def min_health(self) -> float:
        if not self._motors:
            return 1.0
        return min(m.health_score for m in self._motors.values())

    @property
    def any_degraded(self) -> bool:
        return any(m.degraded for m in self._motors.values())
