from __future__ import annotations

from ..models import ThrusterCommand


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def mix_4dof(surge: float, sway: float, heave: float, yaw: float) -> ThrusterCommand:
    # Placeholder mixer. Keep the contract stable; replace with your vehicle-specific mapping.
    return ThrusterCommand(
        surge=clamp(surge),
        sway=clamp(sway),
        heave=clamp(heave),
        yaw=clamp(yaw),
    )
