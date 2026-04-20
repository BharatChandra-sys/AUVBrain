from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary_path = Path("docs") / "latency_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run scripts/latency_benchmark.py first.")

    s = _load_summary(summary_path)

    total = s["total_ms"]
    read = s["read_ms"]
    decide = s["decide_ms"]
    apply = s["apply_ms"]
    tick = s.get("tick_period_ms")

    fig = plt.figure(figsize=(12, 5), dpi=160)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Left: tick period jitter (ms)
    if tick and tick["n"] > 0:
        labels1 = ["tick p50", "tick p95", "tick p99", "tick max"]
        values1 = [tick["p50"], tick["p95"], tick["p99"], tick["max"]]
        ax1.bar(labels1, values1)
        ax1.set_ylabel("milliseconds")
        ax1.set_title("Tick period (ms)")
        ax1.tick_params(axis="x", rotation=20)
    else:
        ax1.set_title("Tick period (ms)")
        ax1.text(0.5, 0.5, "no tick_period_ms data", ha="center", va="center")
        ax1.set_axis_off()

    # Right: work time in microseconds (so very fast loops are still visible)
    to_us = lambda ms: ms * 1000.0
    labels2 = ["read p95", "decide p95", "apply p95", "work p95", "work p99", "work max"]
    values2 = [
        to_us(read["p95"]),
        to_us(decide["p95"]),
        to_us(apply["p95"]),
        to_us(total["p95"]),
        to_us(total["p99"]),
        to_us(total["max"]),
    ]
    ax2.bar(labels2, values2)
    ax2.set_ylabel("microseconds")
    ax2.set_title("Work time (µs, excludes sleep)")
    ax2.tick_params(axis="x", rotation=20)

    title = "AUVBrain latency proof (SIM + Rules, profile every tick)"
    fig.suptitle(title)

    txt = (
        f"work n={int(total['n'])}\n"
        f"work p50={to_us(total['p50']):.1f}µs  p95={to_us(total['p95']):.1f}µs  "
        f"p99={to_us(total['p99']):.1f}µs  max={to_us(total['max']):.1f}µs"
    )
    fig.text(0.02, 0.92, txt, va="top", ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_path = Path("docs") / "latency_proof.png"
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
