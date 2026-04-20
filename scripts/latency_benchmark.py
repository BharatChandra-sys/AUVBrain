from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auvbrain.agent.loop import agent_loop
from auvbrain.agent.policy import RuleDecisionEngine
from auvbrain.config import Settings
from auvbrain.hardware.factory import make_hardware
from auvbrain.safety.monitor import SafetyMonitor
from auvbrain.telemetry.writer import TelemetryWriter


@dataclass(frozen=True)
class BenchmarkResult:
    telemetry_path: Path
    duration_s: float


async def _run_for(settings: Settings, duration_s: float) -> BenchmarkResult:
    hw = make_hardware(settings)
    telemetry = TelemetryWriter(
        settings.telemetry_dir,
        flush_interval_s=settings.telemetry_flush_interval_s,
        max_queue=settings.telemetry_max_queue,
    )
    safety = SafetyMonitor(settings)
    engine = RuleDecisionEngine()
    stop_event = asyncio.Event()

    try:
        task = asyncio.create_task(
            agent_loop(settings, hw, engine, safety, telemetry, stop_event=stop_event)
        )
        await asyncio.sleep(duration_s)
        stop_event.set()
        await task
    finally:
        telemetry.close()

    return BenchmarkResult(telemetry_path=telemetry.path, duration_s=duration_s)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def summarize_profiles(telemetry_path: Path) -> dict[str, Any]:
    totals: list[float] = []
    reads: list[float] = []
    decides: list[float] = []
    applies: list[float] = []
    tick_periods: list[float] = []

    with telemetry_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") != "profile":
                continue
            if "total_ms" in event:
                totals.append(float(event["total_ms"]))
            if "read_ms" in event:
                reads.append(float(event["read_ms"]))
            if "decide_ms" in event:
                decides.append(float(event["decide_ms"]))
            if "apply_ms" in event:
                applies.append(float(event["apply_ms"]))
            if event.get("tick_period_ms") is not None:
                tick_periods.append(float(event["tick_period_ms"]))

    def stats(xs: list[float]) -> dict[str, float]:
        return {
            "n": float(len(xs)),
            "p50": _percentile(xs, 50),
            "p95": _percentile(xs, 95),
            "p99": _percentile(xs, 99),
            "max": max(xs) if xs else float("nan"),
        }

    return {
        "tick_period_ms": stats(tick_periods),
        "total_ms": stats(totals),
        "read_ms": stats(reads),
        "decide_ms": stats(decides),
        "apply_ms": stats(applies),
    }


def main() -> None:
    # Force deterministic, fully-local benchmark:
    # - SIM hardware
    # - Rules engine (no network)
    # - profile every tick
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_dir = Path(".telemetry_benchmark")
    telemetry_path = telemetry_dir / "telemetry.jsonl"
    if telemetry_path.exists():
        telemetry_path.unlink()

    settings = Settings(
        hardware="SIM",
        use_llm=False,
        profile_enabled=True,
        profile_every_n=1,
        telemetry_dir=telemetry_dir,
        tick_autonomous_s=0.02,  # 50Hz request; actual depends on runtime
    )

    duration_s = 5.0
    result = asyncio.run(_run_for(settings, duration_s=duration_s))
    summary = summarize_profiles(result.telemetry_path)

    summary_path = out_dir / "latency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"telemetry={result.telemetry_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
