"""Visualize autonomy pipeline benchmark results.

Creates comprehensive charts showing:
  - Component latency breakdown
  - Latency distribution over time
  - Confidence scores over time
  - Fault detection timeline
  - Throughput comparison across scenarios
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_data(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all benchmark results from directory."""
    results = {}
    
    for summary_file in benchmark_dir.glob("*_summary.json"):
        scenario_name = summary_file.stem.replace("_summary", "")
        with open(summary_file) as f:
            summary = json.load(f)
        
        ticks_file = benchmark_dir / f"{scenario_name}_ticks.json"
        if ticks_file.exists():
            with open(ticks_file) as f:
                ticks = json.load(f)
        else:
            ticks = []
        
        results[scenario_name] = {"summary": summary, "ticks": ticks}
    
    return results


def plot_latency_breakdown(data: dict[str, dict], output_path: Path):
    """Create stacked bar chart of component latencies."""
    scenarios = list(data.keys())
    n = len(scenarios)
    
    components = ["validation", "estimation", "fault_detection", "world_building", "decision"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n)
    width = 0.6
    
    bottoms = np.zeros(n)
    for component, color in zip(components, colors):
        values = [data[s]["summary"][f"{component}_p50"] for s in scenarios]
        ax.bar(x, values, width, label=component.replace("_", " ").title(),
               bottom=bottoms, color=color, alpha=0.9)
        bottoms += values
    
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (µs)", fontsize=12, fontweight="bold")
    ax.set_title("Autonomy Pipeline Latency Breakdown (P50)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
    ax.legend(loc="upper left", frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_latency_percentiles(data: dict[str, dict], output_path: Path):
    """Create grouped bar chart of P50/P95/P99 latencies."""
    scenarios = list(data.keys())
    n = len(scenarios)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n)
    width = 0.25
    
    p50 = [data[s]["summary"]["total_p50"] for s in scenarios]
    p95 = [data[s]["summary"]["total_p95"] for s in scenarios]
    p99 = [data[s]["summary"]["total_p99"] for s in scenarios]
    
    ax.bar(x - width, p50, width, label="P50", color="#3498db", alpha=0.9)
    ax.bar(x, p95, width, label="P95", color="#e74c3c", alpha=0.9)
    ax.bar(x + width, p99, width, label="P99", color="#9b59b6", alpha=0.9)
    
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Latency (µs)", fontsize=12, fontweight="bold")
    ax.set_title("End-to-End Latency Percentiles", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_over_time(data: dict[str, dict], scenario: str, output_path: Path):
    """Plot observation and navigation confidence over time."""
    if scenario not in data or not data[scenario]["ticks"]:
        print(f"⚠ No tick data for scenario: {scenario}")
        return
    
    ticks = data[scenario]["ticks"]
    
    tick_nums = [t["tick"] for t in ticks]
    obs_conf = [t["observation_confidence"] for t in ticks]
    nav_conf = [t["navigation_confidence"] for t in ticks]
    faults = [t["fault_count"] for t in ticks]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Confidence plot
    ax1.plot(tick_nums, obs_conf, label="Observation Confidence", color="#3498db", linewidth=1.5, alpha=0.8)
    ax1.plot(tick_nums, nav_conf, label="Navigation Confidence", color="#2ecc71", linewidth=1.5, alpha=0.8)
    ax1.fill_between(tick_nums, obs_conf, alpha=0.2, color="#3498db")
    ax1.fill_between(tick_nums, nav_conf, alpha=0.2, color="#2ecc71")
    ax1.set_ylabel("Confidence", fontsize=12, fontweight="bold")
    ax1.set_title(f"Confidence & Faults Over Time — {scenario.replace('_', ' ').title()}", 
                  fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_ylim([0, 1.05])
    
    # Fault count plot
    ax2.fill_between(tick_nums, faults, color="#e74c3c", alpha=0.6, label="Faults Detected")
    ax2.plot(tick_nums, faults, color="#c0392b", linewidth=1, alpha=0.8)
    ax2.set_xlabel("Tick", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Fault Count", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, shadow=True)
    ax2.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_latency_timeline(data: dict[str, dict], scenario: str, output_path: Path):
    """Plot component latencies over time."""
    if scenario not in data or not data[scenario]["ticks"]:
        print(f"⚠ No tick data for scenario: {scenario}")
        return
    
    ticks = data[scenario]["ticks"]
    
    tick_nums = [t["tick"] for t in ticks]
    validation = [t["validation_us"] for t in ticks]
    estimation = [t["estimation_us"] for t in ticks]
    fault_detection = [t["fault_detection_us"] for t in ticks]
    decision = [t["decision_us"] for t in ticks]
    total = [t["total_us"] for t in ticks]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(tick_nums, total, label="Total", color="#2c3e50", linewidth=2, alpha=0.9)
    ax.plot(tick_nums, validation, label="Validation", color="#3498db", linewidth=1, alpha=0.7)
    ax.plot(tick_nums, estimation, label="Estimation", color="#2ecc71", linewidth=1, alpha=0.7)
    ax.plot(tick_nums, fault_detection, label="Fault Detection", color="#e74c3c", linewidth=1, alpha=0.7)
    ax.plot(tick_nums, decision, label="Decision", color="#9b59b6", linewidth=1, alpha=0.7)
    
    ax.set_xlabel("Tick", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (µs)", fontsize=12, fontweight="bold")
    ax.set_title(f"Component Latency Timeline — {scenario.replace('_', ' ').title()}", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, shadow=True, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_throughput_comparison(data: dict[str, dict], output_path: Path):
    """Create bar chart comparing throughput across scenarios."""
    scenarios = list(data.keys())
    throughput = [data[s]["summary"]["throughput_tps"] for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#3498db" if t > 50 else "#e74c3c" if t < 20 else "#f39c12" for t in throughput]
    bars = ax.bar(range(len(scenarios)), throughput, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (ticks/s)", fontsize=12, fontweight="bold")
    ax.set_title("Pipeline Throughput by Scenario", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label="10 Hz target")
    ax.legend(frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all benchmark visualizations."""
    benchmark_dir = Path("docs/benchmarks")
    
    if not benchmark_dir.exists():
        print(f"✗ Benchmark directory not found: {benchmark_dir}")
        print("  Run autonomy_benchmark.py first!")
        return
    
    print("\nLoading benchmark data...")
    data = load_benchmark_data(benchmark_dir)
    
    if not data:
        print("✗ No benchmark data found!")
        return
    
    print(f"✓ Loaded {len(data)} scenarios: {', '.join(data.keys())}\n")
    
    print("Generating visualizations...")
    
    # Overall comparison charts
    plot_latency_breakdown(data, benchmark_dir / "latency_breakdown.png")
    plot_latency_percentiles(data, benchmark_dir / "latency_percentiles.png")
    plot_throughput_comparison(data, benchmark_dir / "throughput_comparison.png")
    
    # Per-scenario detailed charts
    for scenario in data.keys():
        plot_confidence_over_time(data, scenario, benchmark_dir / f"{scenario}_confidence.png")
        plot_latency_timeline(data, scenario, benchmark_dir / f"{scenario}_latency.png")
    
    print(f"\n{'='*60}")
    print("✓ All visualizations complete!")
    print(f"  Output directory: {benchmark_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
