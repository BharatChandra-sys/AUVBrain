from __future__ import annotations

from dataclasses import dataclass

from ..models import ThrusterCommand


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True, slots=True)
class MotorMix4:
    """Normalized motor outputs for 4 thrusters (m0..m3 in -1..1)."""

    m0: float
    m1: float
    m2: float
    m3: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.m0, self.m1, self.m2, self.m3)

    def as_list(self) -> list[float]:
        # Compatibility helper; prefer as_tuple() on hot paths.
        return list(self.as_tuple())


def apply_invert(mix: MotorMix4, invert: tuple[bool, bool, bool, bool]) -> MotorMix4:
    i0, i1, i2, i3 = invert
    m0 = -mix.m0 if i0 else mix.m0
    m1 = -mix.m1 if i1 else mix.m1
    m2 = -mix.m2 if i2 else mix.m2
    m3 = -mix.m3 if i3 else mix.m3
    return MotorMix4(m0, m1, m2, m3)


def mix_vectored_4_horizontal(cmd: ThrusterCommand) -> MotorMix4:
    """Mixer for 4 horizontal thrusters allowing surge/sway/yaw.

    Assumption: thrusters are arranged roughly as:
      m0=front-left, m1=front-right, m2=rear-left, m3=rear-right

    Heave is ignored because horizontal thrusters can't move vertically.
    """

    surge = cmd.surge
    sway = cmd.sway
    yaw = cmd.yaw

    # Basic linear mix.
    m0 = surge + sway + yaw
    m1 = surge - sway - yaw
    m2 = surge - sway + yaw
    m3 = surge + sway - yaw

    return MotorMix4(clamp(m0), clamp(m1), clamp(m2), clamp(m3))


def mix_2_horizontal_2_vertical(cmd: ThrusterCommand) -> MotorMix4:
    """Mixer for 2 horizontal + 2 vertical thrusters.

    Assumption:
      m0=horizontal-left, m1=horizontal-right
      m2=vertical-left,   m3=vertical-right

    Sway is ignored.
    """

    surge = cmd.surge
    heave = cmd.heave
    yaw = cmd.yaw

    m0 = surge + yaw
    m1 = surge - yaw
    m2 = heave
    m3 = heave

    return MotorMix4(clamp(m0), clamp(m1), clamp(m2), clamp(m3))
