"""Convenience script to run all benchmarks and generate visualizations."""

import asyncio
import subprocess
import sys
from pathlib import Path


async def main():
    """Run benchmark suite and generate plots."""
    scripts_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("AUVBrain Autonomy Pipeline Benchmark Suite")
    print("="*60 + "\n")
    
    # Step 1: Run benchmarks
    print("Step 1/2: Running benchmarks...")
    print("-" * 60)
    
    benchmark_script = scripts_dir / "autonomy_benchmark.py"
    result = subprocess.run([sys.executable, str(benchmark_script)])
    
    if result.returncode != 0:
        print("\n✗ Benchmark failed!")
        return 1
    
    print("\n" + "="*60)
    
    # Step 2: Generate plots
    print("\nStep 2/2: Generating visualizations...")
    print("-" * 60)
    
    plot_script = scripts_dir / "plot_autonomy_benchmark.py"
    result = subprocess.run([sys.executable, str(plot_script)])
    
    if result.returncode != 0:
        print("\n✗ Visualization failed!")
        return 1
    
    print("\n" + "="*60)
    print("✓ Benchmark suite complete!")
    print("  View results in docs/benchmarks/")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
