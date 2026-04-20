from __future__ import annotations

from ..config import Settings
from .simulated import make_simulated_hardware


def make_hardware(settings: Settings):
    """Create the hardware bundle based on settings.

    Keep SIM as the default so the project runs everywhere.
    """

    hw = settings.hardware.upper().strip()

    if hw == "SIM":
        return make_simulated_hardware()

    if hw in {"RASPI_STUB", "RASPI_GPIO"}:
        # Safe default: keeps sensors simulated; thrusters/experiments are Raspberry Pi stubs.
        from .raspi_gpio import RaspiExperimentModule, RaspiThrusters

        sim = make_simulated_hardware()
        return type(sim)(
            sensors=sim.sensors,
            thrusters=RaspiThrusters(settings),
            experiments=RaspiExperimentModule(),
        )

    raise ValueError(f"Unknown AUV_HARDWARE={settings.hardware!r}")
