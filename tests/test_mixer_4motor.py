from auvbrain.hardware.mixers_4motor import apply_invert, mix_vectored_4_horizontal
from auvbrain.models import ThrusterCommand


def test_vectored_mix_clamps() -> None:
    cmd = ThrusterCommand(surge=1.0, sway=1.0, yaw=1.0)
    mix = mix_vectored_4_horizontal(cmd)
    assert all(-1.0 <= v <= 1.0 for v in mix.as_list())


def test_invert_flips_sign() -> None:
    cmd = ThrusterCommand(surge=0.2, sway=0.0, yaw=0.0)
    mix = mix_vectored_4_horizontal(cmd)
    inv = apply_invert(mix, (True, False, False, False))
    assert inv.m0 == -mix.m0
    assert inv.m1 == mix.m1
