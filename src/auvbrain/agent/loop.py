"""Main agent control loop.

Each tick follows the pattern:
  read sensors  →  decide command  →  safety check  →  apply actuators  →  telemetry

Key guarantees:
  - Every I/O operation is timeout-bounded.
  - A safety override or sensor failure forces SAFE immediately.
  - LLM decisions fall back to Rules on timeout / consecutive failures.
  - A dead-man's switch transitions MANUAL → SAFE when the operator goes silent.
  - Every tick gets a unique correlation_id for end-to-end tracing.
  - Metrics counters are incremented in-process for the /metrics endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Awaitable, TypeVar

from ..config import Settings
from ..hardware.interfaces import HardwareBundle
from ..logging_config import correlation_id
from ..metrics.registry import METRICS
from ..models import ControlMode, DecisionSource, VehicleCommand
from ..safety.monitor import SafetyMonitor
from ..state import STATE
from ..telemetry.writer import TelemetryWriter
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
    logger.info(
        "agent loop started",
        extra={"use_llm": settings.use_llm, "hardware": settings.hardware},
    )

    loop = asyncio.get_running_loop()

    profile = bool(settings.profile_enabled)
    profile_every_n = max(1, int(settings.profile_every_n))
    tick_count = 0
    prev_tick_start: float | None = None
    prev_tick_start_ns: int | None = None

    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("agent loop stop_event received, exiting")
            break

        # ── Assign per-tick correlation ID ───────────────────────────────
        tick_id = uuid.uuid4().hex[:12]
        correlation_id.set(tick_id)

        tick_count += 1
        tick_start = loop.time()
        tick_start_ns = time.perf_counter_ns() if profile else 0

        tick_period_ms: float | None = None
        if prev_tick_start is not None:
            tick_period_ms = (tick_start - prev_tick_start) * 1000.0
        prev_tick_start = tick_start

        tick_period_ms_hr: float | None = None
        if profile:
            if prev_tick_start_ns is not None:
                tick_period_ms_hr = (tick_start_ns - prev_tick_start_ns) / 1_000_000.0
            prev_tick_start_ns = tick_start_ns

        if profile:
            t0_ns = tick_start_ns

        # ── Read sensors + current mode concurrently ────────────────────
        try:
            (mode, last_command), obs = await asyncio.gather(
                STATE.get_mode_and_last_command(),
                _with_timeout(hw.sensors.read(), settings.sensor_read_timeout_s),
            )
        except asyncio.TimeoutError:
            logger.error("sensor read timeout — forcing SAFE")
            METRICS.inc("sensor_timeouts_total")
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
            telemetry.write({
                "type": "alert",
                "kind": "sensor_timeout",
                "tick": tick_count,
                "correlation_id": tick_id,
            })
            await asyncio.sleep(settings.tick_safe_s)
            continue

        if profile:
            t1_ns = time.perf_counter_ns()

        # ── Update gauges ────────────────────────────────────────────────
        METRICS.inc("ticks_total")
        METRICS.set_gauge("current_mode", mode.value)
        METRICS.set_gauge("current_battery_v", obs.battery_v)
        METRICS.set_gauge("current_depth_m", obs.depth_m)
        if tick_period_ms_hr is not None:
            METRICS.record_tick_period(tick_period_ms_hr)
        elif tick_period_ms is not None:
            METRICS.record_tick_period(tick_period_ms)

        # ── SAFE mode ────────────────────────────────────────────────────
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
                METRICS.inc("apply_timeouts_total")
                await STATE.set_mode(ControlMode.SAFE)
                telemetry.write({
                    "type": "alert",
                    "kind": "apply_timeout",
                    "mode": mode.value,
                    "tick": tick_count,
                    "correlation_id": tick_id,
                })

            telemetry.write({
                "type": "tick",
                "mode": mode.value,
                "obs": obs,
                "tick": tick_count,
                "correlation_id": tick_id,
            })

            if profile and (tick_count % profile_every_n == 0):
                t2_ns = time.perf_counter_ns()
                telemetry.write(_profile_event(
                    mode=mode.value,
                    tick_count=tick_count,
                    tick_id=tick_id,
                    tick_period_ms=tick_period_ms_hr or tick_period_ms,
                    t0_ns=t0_ns,
                    t1_ns=t1_ns,
                    t2_ns=t2_ns,
                ))

            period_s = settings.tick_safe_s
            await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))
            continue

        # ── Dead-man's switch check ──────────────────────────────────────
        if mode == ControlMode.MANUAL and settings.manual_deadman_s > 0:
            if await STATE.check_manual_deadman(settings.manual_deadman_s):
                logger.warning(
                    "MANUAL dead-man's switch triggered after %.1fs idle — forcing SAFE",
                    settings.manual_deadman_s,
                    extra={"tick": tick_count},
                )
                await STATE.set_mode(ControlMode.SAFE)
                cmd = _safe_command(f"manual deadman {settings.manual_deadman_s:.0f}s")
                try:
                    await asyncio.gather(
                        _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
                        _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
                        STATE.update_command(cmd, source=None),
                    )
                except Exception:
                    pass
                telemetry.write({
                    "type": "alert",
                    "kind": "manual_deadman",
                    "idle_s": settings.manual_deadman_s,
                    "tick": tick_count,
                    "correlation_id": tick_id,
                })
                await asyncio.sleep(settings.tick_safe_s)
                continue

        # ── MANUAL mode ──────────────────────────────────────────────────
        if mode == ControlMode.MANUAL:
            cmd = last_command or VehicleCommand(note="MANUAL: idle")
            try:
                await asyncio.gather(
                    STATE.update_observation(obs),
                    _with_timeout(hw.thrusters.apply(cmd), settings.thruster_apply_timeout_s),
                    _with_timeout(hw.experiments.apply(cmd), settings.experiment_apply_timeout_s),
                    STATE.update_command(cmd, source=None),
                )
            except asyncio.TimeoutError:
                METRICS.inc("apply_timeouts_total")
                await STATE.set_mode(ControlMode.SAFE)
                telemetry.write({
                    "type": "alert",
                    "kind": "apply_timeout",
                    "mode": mode.value,
                    "tick": tick_count,
                    "correlation_id": tick_id,
                })

            telemetry.write({
                "type": "tick",
                "mode": mode.value,
                "obs": obs,
                "cmd": cmd,
                "source": "manual",
                "tick": tick_count,
                "correlation_id": tick_id,
            })

            if profile and (tick_count % profile_every_n == 0):
                t2_ns = time.perf_counter_ns()
                telemetry.write(_profile_event(
                    mode=mode.value,
                    tick_count=tick_count,
                    tick_id=tick_id,
                    tick_period_ms=tick_period_ms_hr or tick_period_ms,
                    t0_ns=t0_ns,
                    t1_ns=t1_ns,
                    t2_ns=t2_ns,
                ))

            period_s = settings.tick_manual_s
            await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))
            continue

        # ── AUTONOMOUS mode ──────────────────────────────────────────────
        if profile:
            t_decide0_ns = time.perf_counter_ns()

        timeout_s = settings.decision_timeout_s
        try:
            if timeout_s is not None and timeout_s > 0:
                cmd = await asyncio.wait_for(engine.decide(obs), timeout=float(timeout_s))
            else:
                cmd = await engine.decide(obs)
        except asyncio.TimeoutError:
            cmd = _safe_command(f"decision timeout {timeout_s}s")

        if profile:
            t_decide1_ns = time.perf_counter_ns()
            decide_ms = (t_decide1_ns - t_decide0_ns) / 1_000_000.0
            METRICS.record_decide_latency(decide_ms)

        # Track decision source metrics
        if engine.source == DecisionSource.LLM:
            METRICS.inc("llm_decisions_total")
        else:
            METRICS.inc("rule_decisions_total")

        override_mode, safe_cmd = safety.enforce(obs, cmd)
        set_mode_awaitable = None
        if override_mode is not None:
            METRICS.inc("safe_overrides_total")
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
            METRICS.inc("apply_timeouts_total")
            await STATE.set_mode(ControlMode.SAFE)
            telemetry.write({
                "type": "alert",
                "kind": "apply_timeout",
                "mode": mode_for_tick.value,
                "tick": tick_count,
                "correlation_id": tick_id,
            })

        telemetry.write({
            "type": "tick",
            "mode": mode_for_tick.value,
            "obs": obs,
            "cmd": cmd,
            "source": engine.source.value,
            "tick": tick_count,
            "correlation_id": tick_id,
        })

        if profile and (tick_count % profile_every_n == 0):
            t2_ns = time.perf_counter_ns()
            telemetry.write(_profile_event(
                mode=mode_for_tick.value,
                tick_count=tick_count,
                tick_id=tick_id,
                tick_period_ms=tick_period_ms_hr or tick_period_ms,
                t0_ns=t0_ns,
                t1_ns=t1_ns,
                t2_ns=t2_ns,
                t_decide0_ns=t_decide0_ns,
                t_decide1_ns=t_decide1_ns,
            ))

        period_s = settings.tick_autonomous_s
        await asyncio.sleep(max(0.0, period_s - (loop.time() - tick_start)))


def _profile_event(
    *,
    mode: str,
    tick_count: int,
    tick_id: str,
    tick_period_ms: float | None,
    t0_ns: int,
    t1_ns: int,
    t2_ns: int,
    t_decide0_ns: int | None = None,
    t_decide1_ns: int | None = None,
) -> dict:
    event: dict = {
        "type": "profile",
        "mode": mode,
        "tick": tick_count,
        "correlation_id": tick_id,
        "tick_period_ms": tick_period_ms,
        "read_ms": (t1_ns - t0_ns) / 1_000_000.0,
        "apply_ms": (t2_ns - (t_decide1_ns or t1_ns)) / 1_000_000.0,
        "total_ms": (t2_ns - t0_ns) / 1_000_000.0,
    }
    if t_decide0_ns is not None and t_decide1_ns is not None:
        event["decide_ms"] = (t_decide1_ns - t_decide0_ns) / 1_000_000.0
    return event
