"""Agent loop entrypoint.

Handles graceful shutdown on SIGTERM / SIGINT:
  1. Sets the stop_event so the agent loop exits after the current tick.
  2. Forces SAFE mode.
  3. Flushes telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from ..config import load_settings
from ..hardware.factory import make_hardware
from ..logging_config import configure_logging
from ..metrics.registry import METRICS
from ..models import ControlMode, VehicleCommand
from ..safety.monitor import SafetyMonitor
from ..state import STATE
from ..telemetry.writer import TelemetryWriter
from .loop import agent_loop
from .policy import FallbackDecisionEngine, LLMDecisionEngine, RuleDecisionEngine

logger = logging.getLogger(__name__)


async def _run_async() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)

    hardware = make_hardware(settings)
    telemetry = TelemetryWriter(
        settings.telemetry_dir,
        flush_interval_s=settings.telemetry_flush_interval_s,
        max_queue=settings.telemetry_max_queue,
    )
    safety = SafetyMonitor(settings)
    stop_event = asyncio.Event()

    if settings.use_llm:
        primary = LLMDecisionEngine(settings)
        fallback = RuleDecisionEngine()
        engine = FallbackDecisionEngine(
            primary,
            fallback,
            timeout_s=settings.decision_timeout_s,
            enabled=settings.llm_fallback_enabled,
            max_consecutive_failures=settings.llm_max_consecutive_failures,
            failure_cooldown_s=settings.llm_failure_cooldown_s,
        )
    else:
        engine = RuleDecisionEngine()

    # ── Graceful shutdown handler ─────────────────────────────────────────
    def _handle_shutdown(sig: int, _frame) -> None:  # type: ignore[type-arg]
        sig_name = signal.Signals(sig).name
        logger.info("received %s — initiating graceful shutdown", sig_name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle_shutdown)
        except (OSError, ValueError):
            # On Windows SIGTERM is limited; best-effort
            pass

    try:
        await agent_loop(settings, hardware, engine, safety, telemetry, stop_event=stop_event)
    finally:
        logger.info("agent loop exited — forcing SAFE and flushing telemetry")

        # Force SAFE before shutting down
        safe_cmd = VehicleCommand(note="SAFE: shutdown")
        try:
            await STATE.set_mode(ControlMode.SAFE)
            await hardware.thrusters.apply(safe_cmd)
        except Exception:
            pass

        # Write final mission summary to telemetry
        telemetry.write({
            "type": "mission_end",
            "total_ticks": METRICS.ticks_total,
            "safe_overrides": METRICS.safe_overrides_total,
            "llm_fallbacks": METRICS.llm_fallbacks_total,
            "dropped_telemetry": METRICS.dropped_telemetry_total,
        })

        telemetry.close()

        aclose = getattr(engine, "aclose", None)
        if callable(aclose):
            await aclose()

        logger.info("shutdown complete")


def run() -> None:
    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        # Suppress noisy traceback on Ctrl-C
        sys.exit(0)
