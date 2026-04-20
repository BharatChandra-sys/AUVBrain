from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, TypeVar

from ..config import Settings
from ..hardware.interfaces import HardwareBundle
from ..models import ControlMode, VehicleCommand
from ..state import STATE
from ..telemetry.writer import TelemetryWriter
from ..safety.monitor import SafetyMonitor
from .policy import DecisionEngine

logger = logging.getLogger(__name__)


def _safe_command(reason: str) -> VehicleCommand:
    return VehicleCommand(note=f"SAFE: {reason}")


T = TypeVar("T")


async def _with_timeout(awaitable: Awaitable[T], timeout_s: float | None) -> T:
    if timeout_s is None or timeout_s <= 0:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=float(timeout_s))


async def agent_loop(
    settings: Settings,
    hw: HardwareBundle,
    engine: DecisionEngine,
    safety: SafetyMonitor,
    telemetry: TelemetryWriter,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    logger.info("agent loop started (use_llm=%s)", settings.use_llm)

    loop = asyncio.get_running_loop()

    profile = bool(settings.profile_enabled)
    profile_every_n = max(1, int(settings.profile_every_n))
    tick_count = 0
    prev_tick_start: float | None = None
    prev_tick_start_ns: int | None = None

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        tick_count += 1
        tick_start = loop.time()

        tick_start_ns = time.perf_counter_ns() if profile else 0

        tick_period_ms: float | None = None
        if prev_tick_start is not None:
            tick_period_ms = (tick_start - prev_tick_start) * 1000.0
        prev_tick_start = tick_start

        # High-resolution tick period (preferred for profiling/plots)
        tick_period_ms_hr: float | None = None
        if profile:
            if prev_tick_start_ns is not None:
                tick_period_ms_hr = (tick_start_ns - prev_tick_start_ns) / 1_000_000.0
            prev_tick_start_ns = tick_start_ns

        if profile:
            t0_ns = tick_start_ns

        # Overlap independent awaits to reduce tail latency.
        try:
            (mode, last_command), obs = await asyncio.gather(
                STATE.get_mode_and_last_command(),
                _with_timeout(hw.sensors.read(), settings.sensor_read_timeout_s),
            )
        except asyncio.TimeoutError:
            # Hard-cap latency: if sensors stall, force SAFE and retry next tick.
            await STATE.set_mode(ControlMode.SAFE)
            cmd = _safe_command("sensor read timeout")
            try:
                await asyncio.gather(
                    _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
                    _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
                    STATE.update_command(cmd, source=None),
                )
            except Exception:
                pass
            telemetry.write({"type": "alert", "kind": "sensor_timeout"})
            await asyncio.sleep(settings.tick_safe_s)
            continue

        if profile:
            t1_ns = time.perf_counter_ns()

        if mode == ControlMode.SAFE:
            cmd = _safe_command("forced")
            try:
                await asyncio.gather(
                STATE.update_observation(obs),
                _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
                _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
                STATE.update_command(cmd, source=None),
                )
            except asyncio.TimeoutError:
                await STATE.set_mode(ControlMode.SAFE)
                telemetry.write({"type": "alert", "kind": "apply_timeout", "mode": mode})
            telemetry.write({"type": "tick", "mode": mode, "obs": obs})

            if profile and (tick_count % profile_every_n == 0):
                t2_ns = time.perf_counter_ns()
                telemetry.write({
                    "type": "profile",
                    "mode": mode,
                    "tick_period_ms": tick_period_ms_hr if tick_period_ms_hr is not None else tick_period_ms,
                    "read_ms": (t1_ns - t0_ns) / 1_000_000.0,
                    "apply_ms": (t2_ns - t1_ns) / 1_000_000.0,
                    "total_ms": (t2_ns - t0_ns) / 1_000_000.0,
                })
            period_s = settings.tick_safe_s
            await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))
            continue

        if mode == ControlMode.MANUAL:
            # If a manual command was posted, apply it; otherwise neutral.
            cmd = last_command or VehicleCommand(note="MANUAL: idle")
            try:
                await asyncio.gather(
                STATE.update_observation(obs),
                _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
                _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
            )
            except asyncio.TimeoutError:
                await STATE.set_mode(ControlMode.SAFE)
                telemetry.write({"type": "alert", "kind": "apply_timeout", "mode": mode})
            telemetry.write({
                "type": "tick",
                "mode": mode,
                "obs": obs,
                "cmd": cmd,
                "source": "manual",
            })

            if profile and (tick_count % profile_every_n == 0):
                t2_ns = time.perf_counter_ns()
                telemetry.write({
                    "type": "profile",
                    "mode": mode,
                    "tick_period_ms": tick_period_ms_hr if tick_period_ms_hr is not None else tick_period_ms,
                    "read_ms": (t1_ns - t0_ns) / 1_000_000.0,
                    "apply_ms": (t2_ns - t1_ns) / 1_000_000.0,
                    "total_ms": (t2_ns - t0_ns) / 1_000_000.0,
                })
            period_s = settings.tick_manual_s
            await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))
            continue

        # AUTONOMOUS
        if profile:
            t_decide0_ns = time.perf_counter_ns()

        timeout_s = settings.decision_timeout_s
        if timeout_s is not None and timeout_s > 0:
            try:
                cmd = await asyncio.wait_for(engine.decide(obs), timeout=float(timeout_s))
            except asyncio.TimeoutError:
                cmd = _safe_command(f"decision timeout {timeout_s}s")
        else:
            cmd = await engine.decide(obs)

        if profile:
            t_decide1_ns = time.perf_counter_ns()

        override_mode, safe_cmd = safety.enforce(obs, cmd)
        set_mode_awaitable = None
        if override_mode is not None:
            set_mode_awaitable = STATE.set_mode(override_mode)
            cmd = safe_cmd

        mode_for_tick = override_mode or mode

        awaitables = [
            STATE.update_observation(obs),
            _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
            _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
            STATE.update_command(cmd, source=engine.source),
        ]
        if set_mode_awaitable is not None:
            awaitables.append(set_mode_awaitable)
        try:
            await asyncio.gather(*awaitables)
        except asyncio.TimeoutError:
            await STATE.set_mode(ControlMode.SAFE)
            telemetry.write({"type": "alert", "kind": "apply_timeout", "mode": mode_for_tick})

        telemetry.write({
            "type": "tick",
            "mode": mode_for_tick,
            "obs": obs,
            "cmd": cmd,
            "source": engine.source.value,
        })

        if profile and (tick_count % profile_every_n == 0):
            t2_ns = time.perf_counter_ns()
            telemetry.write({
                "type": "profile",
                "mode": mode_for_tick,
                "tick_period_ms": tick_period_ms_hr if tick_period_ms_hr is not None else tick_period_ms,
                "read_ms": (t1_ns - t0_ns) / 1_000_000.0,
                "decide_ms": (t_decide1_ns - t_decide0_ns) / 1_000_000.0,
                "apply_ms": (t2_ns - t_decide1_ns) / 1_000_000.0,
                "total_ms": (t2_ns - t0_ns) / 1_000_000.0,
            })

        period_s = settings.tick_autonomous_s
        await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))
