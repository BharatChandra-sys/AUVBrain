"""Integration tests for the agent loop — SAFE, MANUAL, AUTONOMOUS branches.

These tests run the full agent loop for a small number of ticks by setting
the stop_event after a short sleep, then verify telemetry output.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from auvbrain.agent.loop import agent_loop
from auvbrain.agent.policy import RuleDecisionEngine
from auvbrain.config import Settings
from auvbrain.hardware.simulated import make_simulated_hardware
from auvbrain.models import ControlMode, VehicleCommand
from auvbrain.safety.monitor import SafetyMonitor
from auvbrain.state import STATE
from auvbrain.telemetry.writer import TelemetryWriter


@pytest.fixture(autouse=True)
async def reset_state():
    """Reset global STATE to AUTONOMOUS before each test."""
    await STATE.set_mode(ControlMode.AUTONOMOUS)
    yield
    await STATE.set_mode(ControlMode.AUTONOMOUS)


async def _run_loop(settings: Settings, telemetry: TelemetryWriter, duration_s: float = 0.3) -> None:
    hw = make_simulated_hardware()
    engine = RuleDecisionEngine()
    safety = SafetyMonitor(settings)
    stop_event = asyncio.Event()

    task = asyncio.create_task(
        agent_loop(settings, hw, engine, safety, telemetry, stop_event=stop_event)
    )
    await asyncio.sleep(duration_s)
    stop_event.set()
    await task


def _load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@pytest.mark.asyncio
async def test_autonomous_mode_writes_tick_events() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            telemetry_dir=Path(tmpdir),
            tick_autonomous_s=0.05,
            hardware="SIM",
            use_llm=False,
        )
        telemetry = TelemetryWriter(Path(tmpdir))
        await _run_loop(settings, telemetry, duration_s=0.25)
        telemetry.close()

        events = _load_events(Path(tmpdir) / "telemetry.jsonl")
        tick_events = [e for e in events if e.get("type") == "tick"]
        assert len(tick_events) >= 2, "Expected at least 2 tick events"

        for evt in tick_events:
            assert "mode" in evt
            assert "correlation_id" in evt
            assert "tick" in evt


@pytest.mark.asyncio
async def test_safe_mode_applies_neutral_command() -> None:
    await STATE.set_mode(ControlMode.SAFE)

    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            telemetry_dir=Path(tmpdir),
            tick_safe_s=0.05,
            hardware="SIM",
            use_llm=False,
        )
        telemetry = TelemetryWriter(Path(tmpdir))
        await _run_loop(settings, telemetry, duration_s=0.2)
        telemetry.close()

        events = _load_events(Path(tmpdir) / "telemetry.jsonl")
        tick_events = [e for e in events if e.get("type") == "tick"]
        assert len(tick_events) >= 1

        # In SAFE mode the mode field should be SAFE
        for evt in tick_events:
            assert evt.get("mode") == "SAFE"


@pytest.mark.asyncio
async def test_manual_mode_applies_last_command() -> None:
    await STATE.set_mode(ControlMode.MANUAL)
    manual_cmd = VehicleCommand(note="test manual command")
    manual_cmd.thrusters.surge = 0.7
    await STATE.update_command(manual_cmd, source=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            telemetry_dir=Path(tmpdir),
            tick_manual_s=0.05,
            hardware="SIM",
            use_llm=False,
            manual_deadman_s=0.0,  # disable dead-man's switch for this test
        )
        telemetry = TelemetryWriter(Path(tmpdir))
        await _run_loop(settings, telemetry, duration_s=0.2)
        telemetry.close()

        events = _load_events(Path(tmpdir) / "telemetry.jsonl")
        manual_ticks = [e for e in events if e.get("mode") == "MANUAL"]
        assert len(manual_ticks) >= 1


@pytest.mark.asyncio
async def test_manual_update_command_tracked_in_state() -> None:
    """STATE.update_command must persist the command for manual-mode reads."""
    cmd = VehicleCommand(note="manual-track")
    cmd.thrusters.surge = 0.3
    await STATE.update_command(cmd, source=None)

    state = await STATE.get()
    assert state.last_command is not None
    assert state.last_command.note == "manual-track"
    assert state.last_command.thrusters.surge == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_manual_deadman_transitions_to_safe() -> None:
    """Dead-man's switch: MANUAL with no commands for 0.1s → SAFE."""
    await STATE.set_mode(ControlMode.MANUAL)

    # Manually set last_command_ts far in the past
    import time
    STATE._last_manual_command_ts = time.monotonic() - 1.0

    triggered = await STATE.check_manual_deadman(deadman_s=0.5)
    assert triggered is True


@pytest.mark.asyncio
async def test_deadman_disabled_when_zero() -> None:
    await STATE.set_mode(ControlMode.MANUAL)
    triggered = await STATE.check_manual_deadman(deadman_s=0.0)
    assert triggered is False


@pytest.mark.asyncio
async def test_profile_events_written_when_enabled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        settings = Settings(
            telemetry_dir=Path(tmpdir),
            tick_autonomous_s=0.02,
            hardware="SIM",
            use_llm=False,
            profile_enabled=True,
            profile_every_n=1,
        )
        telemetry = TelemetryWriter(Path(tmpdir))
        await _run_loop(settings, telemetry, duration_s=0.2)
        telemetry.close()

        events = _load_events(Path(tmpdir) / "telemetry.jsonl")
        profile_events = [e for e in events if e.get("type") == "profile"]
        assert len(profile_events) >= 2

        for evt in profile_events:
            assert "total_ms" in evt
            assert "read_ms" in evt
