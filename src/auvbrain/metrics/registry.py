"""In-process metrics registry.

A lightweight, lock-free counter/gauge store.  No external dependency.
The ``/metrics`` endpoint reads from this registry.  The agent loop and
safety monitor update it in-process via the module-level ``METRICS`` singleton.

Why not Prometheus client?  The Pi doesn't run a scrape endpoint, and adding
prometheus_client pulls in a heavy dependency.  Expose raw JSON at ``/metrics``
instead — easy to scrape with any client or turn into a Prometheus exporter
later.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsRegistry:
    """Thread-safe atomic counters + gauges for runtime observability."""

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ── counters (monotonically increasing) ──────────────────────────────
    ticks_total: int = 0
    safe_overrides_total: int = 0
    llm_fallbacks_total: int = 0
    llm_failures_total: int = 0
    rule_decisions_total: int = 0
    llm_decisions_total: int = 0
    dropped_telemetry_total: int = 0
    sensor_timeouts_total: int = 0
    apply_timeouts_total: int = 0
    api_requests_total: int = 0
    ws_connects_total: int = 0
    ws_disconnects_total: int = 0
    auth_failures_total: int = 0
    rate_limit_hits_total: int = 0

    # ── gauges (current value) ────────────────────────────────────────────
    current_mode: str = "UNKNOWN"
    current_battery_v: float = 0.0
    current_depth_m: float = 0.0

    # ── latency samples (ring-buffer for p-tile calc) ─────────────────────
    _decide_latency_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _tick_period_ms: list[float] = field(default_factory=list, init=False, repr=False)
    _MAX_SAMPLES: int = field(default=2048, init=False, repr=False)

    # ── startup timestamp ─────────────────────────────────────────────────
    started_at: float = field(default_factory=time.monotonic, init=False)

    # ─────────────────────────────────────────────────────────────────────
    # Counter helpers
    # ─────────────────────────────────────────────────────────────────────

    def inc(self, name: str, delta: int = 1) -> None:
        with self._lock:
            current = getattr(self, name, 0)
            setattr(self, name, current + delta)

    def set_gauge(self, name: str, value: Any) -> None:
        with self._lock:
            setattr(self, name, value)

    # ─────────────────────────────────────────────────────────────────────
    # Latency ring-buffer
    # ─────────────────────────────────────────────────────────────────────

    def record_decide_latency(self, ms: float) -> None:
        with self._lock:
            self._decide_latency_ms.append(ms)
            if len(self._decide_latency_ms) > self._MAX_SAMPLES:
                self._decide_latency_ms = self._decide_latency_ms[-self._MAX_SAMPLES :]

    def record_tick_period(self, ms: float) -> None:
        with self._lock:
            self._tick_period_ms.append(ms)
            if len(self._tick_period_ms) > self._MAX_SAMPLES:
                self._tick_period_ms = self._tick_period_ms[-self._MAX_SAMPLES :]

    # ─────────────────────────────────────────────────────────────────────
    # Snapshot
    # ─────────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all metrics."""
        with self._lock:
            decide_p = _percentiles(list(self._decide_latency_ms))
            tick_p = _percentiles(list(self._tick_period_ms))
            uptime_s = time.monotonic() - self.started_at

            return {
                "uptime_s": round(uptime_s, 2),
                "current_mode": self.current_mode,
                "current_battery_v": self.current_battery_v,
                "current_depth_m": self.current_depth_m,
                "counters": {
                    "ticks_total": self.ticks_total,
                    "safe_overrides_total": self.safe_overrides_total,
                    "llm_fallbacks_total": self.llm_fallbacks_total,
                    "llm_failures_total": self.llm_failures_total,
                    "rule_decisions_total": self.rule_decisions_total,
                    "llm_decisions_total": self.llm_decisions_total,
                    "dropped_telemetry_total": self.dropped_telemetry_total,
                    "sensor_timeouts_total": self.sensor_timeouts_total,
                    "apply_timeouts_total": self.apply_timeouts_total,
                    "api_requests_total": self.api_requests_total,
                    "ws_connects_total": self.ws_connects_total,
                    "ws_disconnects_total": self.ws_disconnects_total,
                    "auth_failures_total": self.auth_failures_total,
                    "rate_limit_hits_total": self.rate_limit_hits_total,
                },
                "decide_latency_ms": decide_p,
                "tick_period_ms": tick_p,
            }


def _percentiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    values.sort()
    n = len(values)
    return {
        "n": n,
        "p50": _pct(values, 50),
        "p95": _pct(values, 95),
        "p99": _pct(values, 99),
        "max": values[-1],
    }


def _pct(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# Module-level singleton shared across agent loop + API handlers
METRICS = MetricsRegistry()
