"""Comprehensive autonomy pipeline benchmark — Phase 1-9 performance evaluation.

This benchmark measures the end-to-end latency of the complete autonomy stack:
  observation → validation → estimation → fault detection → world building →
  planning → decision → shield → safety → actuation

Metrics tracked:
  - Per-component latency (validation, estimation, fault detection, etc.)
  - End-to-end tick latency
  - Throughput (ticks/second)
  - Memory usage
  - Confidence scores over time
  - Fault detection rate
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from auvbrain.agent.policy import FallbackDecisionEngine, LLMDecisionEngine, RuleDecisionEngine
from auvbrain.config import Settings
from auvbrain.faults.detector import FaultDetector
from auvbrain.models import Observation
from auvbrain.navigation.dead_reckoning import DeadReckoningEstimator
from auvbrain.perception.validator import SensorValidator
from auvbrain.planning.energy import EnergyPlanner, EnergyState
from auvbrain.safety.monitor import SafetyMonitor
from auvbrain.simulation.scenarios import Scenarios
from auvbrain.world.model import WorldStateBuilder


@dataclass
class TickMetrics:
    """Per-tick performance measurements."""
    tick: int
    validation_us: float
    estimation_us: float
    fault_detection_us: float
    world_building_us: float
    decision_us: float
    shield_us: float
    safety_us: float
    total_us: float
    observation_confidence: float
    navigation_confidence: float
    energy_soc: float
    fault_count: int
    decision_source: str


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    scenario_name: str
    total_ticks: int
    duration_s: float
    throughput_tps: float
    
    # Latency statistics (microseconds)
    validation_p50: float
    validation_p95: float
    validation_p99: float
    
    estimation_p50: float
    estimation_p95: float
    estimation_p99: float
    
    fault_detection_p50: float
    fault_detection_p95: float
    fault_detection_p99: float
    
    world_building_p50: float
    world_building_p95: float
    world_building_p99: float
    
    decision_p50: float
    decision_p95: float
    decision_p99: float
    
    total_p50: float
    total_p95: float
    total_p99: float
    total_max: float
    
    # Quality metrics
    avg_observation_confidence: float
    avg_navigation_confidence: float
    fault_detection_count: int
    rule_fallback_count: int
    llm_decision_count: int


class AutonomyBenchmark:
    """Benchmarks the full autonomy pipeline."""
    
    def __init__(self, use_llm: bool = False, enable_shield: bool = True):
        self.use_llm = use_llm
        self.enable_shield = enable_shield
        self.tick_metrics: list[TickMetrics] = []
    
    async def run_scenario(self, scenario_name: str, num_ticks: int = 1000) -> BenchmarkResults:
        """Run a complete scenario and collect metrics."""
        print(f"\n{'='*60}")
        print(f"Benchmark: {scenario_name}")
        print(f"Ticks: {num_ticks}, LLM: {self.use_llm}, Shield: {self.enable_shield}")
        print(f"{'='*60}\n")
        
        # Load scenario
        scenario = getattr(Scenarios, scenario_name.lower())()
        dynamics, sensors, injector = scenario.build()
        
        # Initialize pipeline components
        validator = SensorValidator()
        estimator = DeadReckoningEstimator()
        fault_detector = FaultDetector()
        world_builder = WorldStateBuilder()
        energy_planner = EnergyPlanner()
        
        # Decision engine
        rules_engine = RuleDecisionEngine()
        if self.use_llm:
            settings = Settings()
            settings.use_llm = True
            primary_engine = LLMDecisionEngine(settings)
            engine = FallbackDecisionEngine(
                primary=primary_engine,
                fallback=rules_engine,
                timeout_s=0.5,
                enabled=True,
            )
        else:
            engine = rules_engine
        
        safety = SafetyMonitor()
        
        # Setup scenario event timeline
        event_map = {event.tick: event for event in scenario.events}
        
        # Benchmark loop
        start_time = time.perf_counter()
        self.tick_metrics.clear()
        
        for tick in range(num_ticks):
            # Inject faults if scheduled
            if tick in event_map:
                event = event_map[tick]
                event.inject(injector)
            
            # Read observation
            obs_dict = sensors.read()
            obs = Observation(
                depth_m=obs_dict["depth_m"],
                battery_v=obs_dict["battery_v"],
                internal_temp_c=obs_dict.get("internal_temp_c"),
                pressure_bar=obs_dict.get("pressure_bar"),
                obstacle_front_m=obs_dict.get("obstacle_front_m"),
                water_ingress=obs_dict.get("water_ingress_detected", False),
            )
            
            # Measure each pipeline stage
            t0 = time.perf_counter_ns()
            
            # 1. Validation
            validated = validator.validate(obs, dt=0.1)
            t1 = time.perf_counter_ns()
            
            # 2. Estimation
            estimated_state = estimator.update(validated, surge=0.2, sway=0.0, heave=0.0)
            t2 = time.perf_counter_ns()
            
            # 3. Fault detection
            faults = fault_detector.detect(validated, estimated_state)
            t3 = time.perf_counter_ns()
            
            # 4. Energy state
            energy_state = EnergyState.from_observation(obs.battery_v)
            
            # 5. World building
            world = world_builder.build(
                navigation=estimated_state,
                energy=energy_state,
                faults=faults,
                observation_confidence=validated.observation_confidence,
                tick=tick,
            )
            t4 = time.perf_counter_ns()
            
            # 6. Decision
            cmd = await engine.decide(obs)
            t5 = time.perf_counter_ns()
            
            # 7. Shield (if enabled)
            if self.enable_shield:
                from auvbrain.safety.shield import ActionShield
                shield = ActionShield()
                shield_decision = shield.evaluate(cmd, world)
                if shield_decision.approved:
                    final_cmd = cmd
                else:
                    final_cmd = shield_decision.modified_command or cmd
            else:
                final_cmd = cmd
            t6 = time.perf_counter_ns()
            
            # 8. Safety monitor
            override_mode, safe_cmd = safety.enforce(obs, final_cmd)
            if override_mode:
                final_cmd = safe_cmd
            t7 = time.perf_counter_ns()
            
            # Record metrics
            metrics = TickMetrics(
                tick=tick,
                validation_us=(t1 - t0) / 1000,
                estimation_us=(t2 - t1) / 1000,
                fault_detection_us=(t3 - t2) / 1000,
                world_building_us=(t4 - t3) / 1000,
                decision_us=(t5 - t4) / 1000,
                shield_us=(t6 - t5) / 1000,
                safety_us=(t7 - t6) / 1000,
                total_us=(t7 - t0) / 1000,
                observation_confidence=validated.observation_confidence,
                navigation_confidence=estimated_state.confidence,
                energy_soc=energy_state.soc,
                fault_count=len(faults),
                decision_source=engine.source.value,
            )
            self.tick_metrics.append(metrics)
            
            # Progress report every 100 ticks
            if (tick + 1) % 100 == 0:
                avg_total = sum(m.total_us for m in self.tick_metrics[-100:]) / 100
                print(f"  Tick {tick + 1:4d}: avg latency {avg_total:6.1f} µs, "
                      f"faults: {metrics.fault_count}, conf: {metrics.navigation_confidence:.2f}")
            
            # Step simulation
            dynamics.step(
                thrust=[cmd.thrusters.surge, cmd.thrusters.sway, cmd.thrusters.heave, cmd.thrusters.yaw],
                dt=0.1,
            )
            sensors.step(dt=0.1)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Compute statistics
        results = self._compute_statistics(scenario_name, duration)
        
        # Cleanup
        if self.use_llm and hasattr(engine, '_primary'):
            await engine._primary.aclose()
        
        return results
    
    def _compute_statistics(self, scenario_name: str, duration_s: float) -> BenchmarkResults:
        """Compute aggregate statistics from tick metrics."""
        n = len(self.tick_metrics)
        
        def percentile(values: list[float], p: float) -> float:
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]
        
        validation = [m.validation_us for m in self.tick_metrics]
        estimation = [m.estimation_us for m in self.tick_metrics]
        fault_detection = [m.fault_detection_us for m in self.tick_metrics]
        world_building = [m.world_building_us for m in self.tick_metrics]
        decision = [m.decision_us for m in self.tick_metrics]
        total = [m.total_us for m in self.tick_metrics]
        
        fault_count = sum(m.fault_count for m in self.tick_metrics)
        rule_count = sum(1 for m in self.tick_metrics if m.decision_source == "rules")
        llm_count = sum(1 for m in self.tick_metrics if m.decision_source == "llm")
        
        return BenchmarkResults(
            scenario_name=scenario_name,
            total_ticks=n,
            duration_s=duration_s,
            throughput_tps=n / duration_s if duration_s > 0 else 0,
            
            validation_p50=percentile(validation, 50),
            validation_p95=percentile(validation, 95),
            validation_p99=percentile(validation, 99),
            
            estimation_p50=percentile(estimation, 50),
            estimation_p95=percentile(estimation, 95),
            estimation_p99=percentile(estimation, 99),
            
            fault_detection_p50=percentile(fault_detection, 50),
            fault_detection_p95=percentile(fault_detection, 95),
            fault_detection_p99=percentile(fault_detection, 99),
            
            world_building_p50=percentile(world_building, 50),
            world_building_p95=percentile(world_building, 95),
            world_building_p99=percentile(world_building, 99),
            
            decision_p50=percentile(decision, 50),
            decision_p95=percentile(decision, 95),
            decision_p99=percentile(decision, 99),
            
            total_p50=percentile(total, 50),
            total_p95=percentile(total, 95),
            total_p99=percentile(total, 99),
            total_max=max(total),
            
            avg_observation_confidence=sum(m.observation_confidence for m in self.tick_metrics) / n,
            avg_navigation_confidence=sum(m.navigation_confidence for m in self.tick_metrics) / n,
            fault_detection_count=fault_count,
            rule_fallback_count=rule_count,
            llm_decision_count=llm_count,
        )
    
    def save_results(self, results: BenchmarkResults, output_dir: Path):
        """Save benchmark results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / f"{results.scenario_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        
        # Save detailed tick data
        detail_path = output_dir / f"{results.scenario_name}_ticks.json"
        with open(detail_path, "w") as f:
            json.dump([asdict(m) for m in self.tick_metrics], f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir}")
    
    def print_results(self, results: BenchmarkResults):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {results.scenario_name}")
        print(f"{'='*60}\n")
        
        print(f"Duration:    {results.duration_s:.2f} s")
        print(f"Throughput:  {results.throughput_tps:.1f} ticks/s")
        print(f"Total ticks: {results.total_ticks}")
        print()
        
        print("Latency Breakdown (µs):")
        print(f"  Component         P50      P95      P99")
        print(f"  {'─'*45}")
        print(f"  Validation    {results.validation_p50:7.1f}  {results.validation_p95:7.1f}  {results.validation_p99:7.1f}")
        print(f"  Estimation    {results.estimation_p50:7.1f}  {results.estimation_p95:7.1f}  {results.estimation_p99:7.1f}")
        print(f"  Fault Detect  {results.fault_detection_p50:7.1f}  {results.fault_detection_p95:7.1f}  {results.fault_detection_p99:7.1f}")
        print(f"  World Build   {results.world_building_p50:7.1f}  {results.world_building_p95:7.1f}  {results.world_building_p99:7.1f}")
        print(f"  Decision      {results.decision_p50:7.1f}  {results.decision_p95:7.1f}  {results.decision_p99:7.1f}")
        print(f"  {'─'*45}")
        print(f"  TOTAL         {results.total_p50:7.1f}  {results.total_p95:7.1f}  {results.total_p99:7.1f}  (max: {results.total_max:.1f})")
        print()
        
        print("Quality Metrics:")
        print(f"  Avg obs confidence:  {results.avg_observation_confidence:.3f}")
        print(f"  Avg nav confidence:  {results.avg_navigation_confidence:.3f}")
        print(f"  Faults detected:     {results.fault_detection_count}")
        print(f"  Rule decisions:      {results.rule_fallback_count}")
        print(f"  LLM decisions:       {results.llm_decision_count}")
        print()


async def main():
    """Run benchmark suite."""
    output_dir = Path("docs/benchmarks")
    
    # Scenarios to benchmark
    scenarios = [
        "normal",
        "dvl_dropout",
        "critical_battery",
        "thruster_degradation",
        "imu_bias",
    ]
    
    # Run benchmarks
    benchmark = AutonomyBenchmark(use_llm=False, enable_shield=True)
    
    all_results = []
    for scenario in scenarios:
        results = await benchmark.run_scenario(scenario, num_ticks=500)
        benchmark.print_results(results)
        benchmark.save_results(results, output_dir)
        all_results.append(results)
    
    # Save combined summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ All benchmarks complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
