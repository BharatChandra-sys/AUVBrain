from __future__ import annotations

import asyncio

from ..config import load_settings
from ..hardware.factory import make_hardware
from ..logging_config import configure_logging
from ..safety.monitor import SafetyMonitor
from ..telemetry.writer import TelemetryWriter
from .loop import agent_loop
from .policy import FallbackDecisionEngine, LLMDecisionEngine, RuleDecisionEngine


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

    try:
        await agent_loop(settings, hardware, engine, safety, telemetry)
    finally:
        telemetry.close()
        aclose = getattr(engine, "aclose", None)
        if callable(aclose):
            await aclose()


def run() -> None:
    asyncio.run(_run_async())
