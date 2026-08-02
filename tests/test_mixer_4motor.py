"""Tests for the 4-motor mixer (both layouts + invert)."""

from __future__ import annotations

import pytest

from auvbrain.hardware.mixers_4motor import (
    MotorMix4,
    apply_invert,
    mix_2_horizontal_2_vertical,
    mix_vectored_4_horizontal,
)
from auvbrain.models import ThrusterCommand


def test_vectored_mix_clamps_all_axes() -> None:
    """Full saturation on all axes must still clamp to [-1, 1]."""
    cmd = ThrusterCommand(surge=1.0, sway=1.0, yaw=1.0)
    mix = mix_vectored_4_horizontal(cmd)
    assert all(-1.0 <= v <= 1.0 for v in mix.as_list())


def test_vectored_mix_pure_surge() -> None:
    """Pure surge should drive all four motors equally forward."""
    cmd = ThrusterCommand(surge=0.5)
    mix = mix_vectored_4_horizontal(cmd)
    # All motors get the same surge contribution when sway/yaw = 0
    assert mix.m0 == mix.m1 == mix.m2 == mix.m3 == 0.5


def test_vectored_mix_pure_yaw() -> None:
    """Pure yaw should produce equal-magnitude, opposite-sign pairs."""
    cmd = ThrusterCommand(yaw=0.5)
    mix = mix_vectored_4_horizontal(cmd)
    # m0 = +yaw, m1 = -yaw, m2 = +yaw, m3 = -yaw
    assert mix.m0 == pytest.approx(0.5)
    assert mix.m1 == pytest.approx(-0.5)
    assert mix.m2 == pytest.approx(0.5)
    assert mix.m3 == pytest.approx(-0.5)


def test_invert_flips_sign_m0_only() -> None:
    cmd = ThrusterCommand(surge=0.2)
    mix = mix_vectored_4_horizontal(cmd)
    inv = apply_invert(mix, (True, False, False, False))
    assert inv.m0 == pytest.approx(-mix.m0)
    assert inv.m1 == pytest.approx(mix.m1)
    assert inv.m2 == pytest.approx(mix.m2)
    assert inv.m3 == pytest.approx(mix.m3)


def test_invert_all_motors() -> None:
    cmd = ThrusterCommand(surge=0.3, yaw=0.2)
    mix = mix_vectored_4_horizontal(cmd)
    inv = apply_invert(mix, (True, True, True, True))
    for original, inverted in zip(mix.as_list(), inv.as_list()):
        assert inverted == pytest.approx(-original)


def test_h2v2_heave_both_vertical() -> None:
    """Heave should only affect m2 and m3 (vertical motors)."""
    cmd = ThrusterCommand(heave=0.6)
    mix = mix_2_horizontal_2_vertical(cmd)
    assert mix.m0 == pytest.approx(0.0)
    assert mix.m1 == pytest.approx(0.0)
    assert mix.m2 == pytest.approx(0.6)
    assert mix.m3 == pytest.approx(0.6)


def test_h2v2_surge_horizontal_only() -> None:
    """Surge should only affect m0 and m1 (horizontal motors)."""
    cmd = ThrusterCommand(surge=0.4)
    mix = mix_2_horizontal_2_vertical(cmd)
    assert mix.m0 == pytest.approx(0.4)
    assert mix.m1 == pytest.approx(0.4)
    assert mix.m2 == pytest.approx(0.0)
    assert mix.m3 == pytest.approx(0.0)


def test_motor_mix4_as_tuple_and_list_equivalent() -> None:
    m = MotorMix4(0.1, 0.2, 0.3, 0.4)
    assert list(m.as_tuple()) == m.as_list()
