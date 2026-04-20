from __future__ import annotations

"""Raspberry Pi GPIO adapters.

Keep this file as the *only* place that imports GPIO libraries.
That makes Windows development + CI work without Raspberry Pi deps.

Wire your actual pins/ESCs/relays here.
"""

import logging

from ..config import Settings
from .mixers_4motor import apply_invert, mix_2_horizontal_2_vertical, mix_vectored_4_horizontal

from ..models import VehicleCommand

logger = logging.getLogger(__name__)


class RaspiThrusters:
    def __init__(self, settings: Settings) -> None:
        try:
            from gpiozero import Servo
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "gpiozero not available. Install with: pip install -e .[raspi]"
            ) from e

        self.settings = settings

        self._enabled = bool(settings.pwm_enabled)
        self._servos = None

        pins = [settings.motor_gpio_0, settings.motor_gpio_1, settings.motor_gpio_2, settings.motor_gpio_3]
        if self._enabled and any(p is None for p in pins):
            raise ValueError(
                "PWM enabled but one or more motor GPIO pins are not set (AUV_MOTOR_GPIO_0..3)."
            )

        if self._enabled:
            # Note: gpiozero will use software PWM unless you configure pigpio.
            self._servos = [
                Servo(
                    int(p),
                    min_pulse_width=settings.esc_min_pulse_s,
                    max_pulse_width=settings.esc_max_pulse_s,
                    frame_width=settings.esc_frame_s,
                )
                for p in pins
            ]

        # TODO: initialize your PWM outputs / ESCs

    async def apply(self, command: VehicleCommand) -> None:
        layout = self.settings.thruster_layout.upper().strip()

        if layout == "VECTORED_4_HORIZONTAL":
            mix = mix_vectored_4_horizontal(command.thrusters)
        elif layout in {"H2_V2", "2H2V", "2_HORIZONTAL_2_VERTICAL"}:
            mix = mix_2_horizontal_2_vertical(command.thrusters)
        else:
            mix = mix_vectored_4_horizontal(command.thrusters)

        mix = apply_invert(
            mix,
            (
                self.settings.motor_invert_0,
                self.settings.motor_invert_1,
                self.settings.motor_invert_2,
                self.settings.motor_invert_3,
            ),
        )

        # TODO: map -1..1 to your PWM/ESC command per motor (m0..m3)
        outputs = mix.as_tuple()
        if not self._enabled or self._servos is None:
            logger.info("(GPIO) MOTORS m0..m3=%s (pwm disabled)", outputs)
            return

        for servo, value in zip(self._servos, outputs, strict=True):
            # gpiozero.Servo expects -1..1
            servo.value = float(value)


class RaspiExperimentModule:
    def __init__(self) -> None:
        try:
            from gpiozero import OutputDevice  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "gpiozero not available. Install with: pip install -e .[raspi]"
            ) from e

        # TODO: initialize relay / actuator pins

    async def apply(self, command: VehicleCommand) -> None:
        if not command.experiment.enabled:
            return
        logger.info("(GPIO) EXPERIMENT %s", command.experiment.model_dump())
