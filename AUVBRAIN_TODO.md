# AUVBrain — Master Development Roadmap

Cross-referenced against `plan.md` and the current codebase.
Items marked ✅ are committed and tested.  Items marked 🔨 are in-progress.
Items marked ☐ are not yet started.

---

## FOUNDATION (already done)

✅ FastAPI control hub + WebSocket  
✅ observe → decide → act agent loop  
✅ LLM decision engine (Ollama / OpenAI-compat / llama.cpp)  
✅ Rules fallback + FallbackDecisionEngine (timeout + cooldown)  
✅ Hard SafetyMonitor (depth / battery / temp / pressure / ingress / obstacle)  
✅ SAFE / MANUAL / AUTONOMOUS modes  
✅ Dead-man's switch (MANUAL → SAFE on idle)  
✅ API key auth (read / write / admin scopes)  
✅ Per-IP token-bucket rate limiter  
✅ Structured JSON logging + correlation IDs per tick  
✅ `/metrics` endpoint (counters, gauges, latency percentiles)  
✅ Graceful SIGTERM shutdown (force SAFE + flush telemetry)  
✅ Schema versioning on VehicleCommand  
✅ Input validation bounds (note, params)  
✅ Telemetry (append-only JSONL + async DB sink)  
✅ PostgreSQL + SQLite support (SQLAlchemy async)  
✅ API key DB management (create / list / revoke)  
✅ RAG backed by DB (full-text search; pgvector ANN is next)  
✅ Full test suite (SafetyMonitor, FallbackDecisionEngine, agent loop, mixers)  
✅ GitHub Actions CI (pytest on every push/PR, Python 3.10–3.12, ruff, coverage)  
✅ SIM + Raspberry Pi hardware adapters  
✅ 4-motor mixer (VECTORED_4_HORIZONTAL + H2_V2)  
✅ PID controller  

---

## PHASE 1 — EstimatedState + sensor models  ✅ COMPLETE

### 1.1 Sensor health model
- ✅ `src/auvbrain/perception/sensor_health.py` — `SensorHealth` dataclass per sensor  
  Fields: `available`, `fresh`, `range_valid`, `rate_valid`, `consistency`, `confidence`, `fault_reason`  
- ✅ `src/auvbrain/perception/validator.py` — `SensorValidator.validate(obs) → ValidatedObservation`  
  Checks: dropout, stale data, constant value, spike, impossible jump, out-of-range  
- ✅ Tests: `tests/test_sensor_validator.py` (dropout, stale, spike, range)  

### 1.2 EstimatedState model
- ✅ `src/auvbrain/navigation/estimated_state.py`  
  Fields: `position_m`, `velocity_ms`, `heading_deg`, `depth_m`, `orientation`, `covariance`, `confidence`, `sensor_sources`, `timestamp`  
- ✅ Integrate into agent loop: `Observation → validate → EstimatedState`  

### 1.3 Simple dead-reckoning estimator (placeholder for EKF)
- ✅ `src/auvbrain/navigation/dead_reckoning.py` — `DeadReckoningEstimator`  
  Integrates velocity over dt, decays confidence without sensor updates  
- ✅ Tests: `tests/test_dead_reckoning.py`  

---

## PHASE 2 — World Model  ✅ COMPLETE

### 2.1 WorldState
- ✅ `src/auvbrain/world/model.py` — `WorldState`  
  Fields: `vehicle`, `navigation`, `obstacles`, `currents`, `hazards`, `energy`, `faults`, `mission_state`  
- ✅ `src/auvbrain/world/obstacles.py` — `ObstacleMap` (obstacle list with confidence + TTL)  
- ✅ `src/auvbrain/world/environment.py` — `CurrentModel` (vector, confidence, trend)  

### 2.2 Mission state machine
- ✅ `src/auvbrain/world/mission_state.py` — `MissionPhase` enum  
  Phases: `SURFACE`, `DIVE`, `TRANSIT`, `SURVEY`, `INSPECT`, `RETURN`, `ASCENT`  
  Cross-cutting: `DEGRADED`, `RECOVERY`, `ABORT`, `SAFE`  
  `MissionContext` dataclass: phase, goal, waypoints, start_time, constraints  

---

## PHASE 3 — Energy Model  ✅ COMPLETE

### 3.1 EnergyState
- ✅ `src/auvbrain/planning/energy.py` — `EnergyState`  
  Fields: `soc`, `voltage`, `current_draw_a`, `propulsion_cost_per_m`, `mission_cost_estimate`, `return_cost_estimate`, `reserve_fraction`  
- ✅ `EnergyPlanner.feasibility(world) → EnergyDecision` (continue / replan / abort / surface)  
- ✅ Tests: `tests/test_energy_planner.py`  

---

## PHASE 4 — Fault Detection → Diagnosis → Recovery  ✅ COMPLETE

### 4.1 Fault model
- ✅ `src/auvbrain/faults/models.py` — `Fault`, `FaultSeverity`, `FaultType`  
- ✅ `src/auvbrain/faults/detector.py` — `FaultDetector.detect(obs, estimated_state) → list[Fault]`  
  Detects: DVL residual spike, IMU freeze, depth drift, thruster degradation  
- ✅ `src/auvbrain/faults/diagnosis.py` — `FaultDiagnosis.diagnose(faults) → RecoveryStrategy`  
- ✅ `src/auvbrain/faults/recovery.py` — `RecoveryStrategy` enum + `RecoveryPlanner`  
- ✅ Tests: `tests/test_fault_detection.py`  

### 4.2 Actuator health
- ✅ `src/auvbrain/control/actuator_health.py` — `ActuatorHealth`  
  Tracks: commanded vs response, motor current, temperature, response delay, `health_score: float`  

---

## PHASE 5 — Action Safety Shield  ✅ COMPLETE

### 5.1 ActionShield
- ✅ `src/auvbrain/safety/shield.py` — `ActionShield.evaluate(candidate, world) → ShieldDecision`  
  Checks: depth, velocity, energy feasibility, navigation confidence, sensor validity, actuator limits, mission constraints  
  Decision: `APPROVE` / `REJECT` / `MODIFY`  
- ✅ Wire into agent loop: `engine.decide → ActionShield → SafetyMonitor → hw.apply`  
- ✅ Tests: `tests/test_action_shield.py`  

---

## PHASE 6 — Tactical Planner  ✅ COMPLETE

### 6.1 Planner
- ✅ `src/auvbrain/planning/tactical.py` — `TacticalPlanner`  
  Inputs: pose, velocity, obstacles, currents, energy, faults, uncertainty, mission priority  
  Output: `PlanCandidate` (trajectory, action, cost, risk, energy_cost)  
- ✅ `src/auvbrain/planning/trajectory.py` — `Waypoint`, `Trajectory`, `TrajectoryEvaluator`  
- ✅ Tests: `tests/test_trajectory_planner.py`, `tests/test_tactical_planner.py`  

---

## PHASE 7 — Persistent Memory (5-class taxonomy)  ✅ COMPLETE

### 7.1 Memory models
- ✅ `src/auvbrain/memory/models.py` — `MemoryType` enum (PROCEDURAL, SEMANTIC, EPISODIC, FAULT, ENVIRONMENTAL)  
  `MemoryEntry`: type, source, timestamp, verification, confidence, model_version, vehicle_state, environment, outcome  

### 7.2 Episodic + fault memory
- ✅ `src/auvbrain/memory/episodic.py` — `EpisodicMemory.record_episode(...)` + `recall(query)`  
- ✅ `src/auvbrain/memory/fault_memory.py` — `FaultMemory.record_fault(...)` + `lookup_similar(fault)`  
- ✅ DB schema updated: `memory_entries` table with trust provenance columns  

### 7.3 State-aware RAG
- ✅ `src/auvbrain/memory/retriever.py` — `StateAwareRetriever`  
  Query built from: mission goal + estimated state + fault state + energy + vehicle mode  
  Pipeline: construct query → search → rerank → trust-filter → return context  
- ✅ Tests: `tests/test_memory_retrieval.py`  

---

## PHASE 8 — Agent Tools  ✅ COMPLETE

### 8.1 Tool definitions
- ✅ `src/auvbrain/agent/tools/` package  
  - `get_vehicle_state()` → VehicleState snapshot  
  - `get_sensor_health()` → per-sensor SensorHealth  
  - `get_energy_state()` → EnergyState  
  - `get_navigation_estimate()` → EstimatedState  
  - `retrieve_fault_memory(query)` → list[MemoryEntry]  
  - `retrieve_procedure(query)` → list[MemoryEntry]  
  - `evaluate_trajectory(waypoints)` → TrajectoryEvaluation  
  - `check_action_safety(candidate)` → ShieldDecision  
  - `request_replan(reason)` → PlanCandidate  
  - `execute_mission_action(action)` → ExecutionResult  
- ✅ `src/auvbrain/agent/tool_registry.py` — `ToolRegistry` dispatcher  
- ✅ Wire into `LLMDecisionEngine` so model can invoke tools before emitting final command  

---

## PHASE 9 — Simulation Lab  ✅ COMPLETE

### 9.1 Scenario-based simulation
- ✅ `src/auvbrain/simulation/dynamics.py` — `AUVDynamics` (mass, drag, thruster model)  
- ✅ `src/auvbrain/simulation/sensors.py` — `SimSensorSuite` with configurable noise, bias, dropout  
- ✅ `src/auvbrain/simulation/faults.py` — `FaultInjector` (DVL dropout, IMU bias, thruster degradation)  
- ✅ `src/auvbrain/simulation/scenarios.py` — `Scenario` dataclass + built-in scenarios  

### 9.2 Fault-injection test matrix
- ✅ `tests/test_simulation_scenarios.py` — all 13 scenarios from `plan.md` §20  
- ✅ Integration test: `tests/test_world_state_integration.py` — full pipeline validation  

---

## PHASE 10 — LangGraph Mission Workflows  ← FUTURE (after core is stable)

- [ ] Add `langgraph` to optional deps in `pyproject.toml`  
- [ ] `src/auvbrain/agent/graph.py` — mission graph  
  Nodes: load_mission → get_world_state → check_confidence → retrieve_memory  
        → plan_subgoal → generate_candidates → risk_eval → shield → execute  
        → observe_outcome → update_state → replan/recover/continue  
- [ ] `src/auvbrain/agent/nodes/` — one file per graph node  
- [ ] Integration test: full autonomous recovery scenario end-to-end  

---

## PHASE 11 — EKF / ESKF Navigation  ← FUTURE (needs sensor hardware data)

- [ ] `src/auvbrain/navigation/ekf.py` — `ExtendedKalmanFilter`  
  Fuses: IMU, DVL, depth, heading  
- [ ] `src/auvbrain/navigation/eskf.py` — `ErrorStateKalmanFilter` (better for IMU bias)  
- [ ] Sensor driver interfaces: IMU, DVL, depth, heading  
- [ ] Tests: normal, DVL dropout, IMU bias, depth drift, multi-degraded  
- [ ] Evaluation: position RMSE, velocity RMSE, heading RMSE, drift, confidence behavior  

---

## PHASE 12 — Evaluation + Benchmarking

- [ ] Architecture ablation benchmark (A–H from `plan.md` §23)  
  A: deterministic baseline → B: + state estimation → C: + world model →  
  D: + fault diagnosis → E: + energy planning → F: + memory/RAG →  
  G: + LangGraph → H: full system  
- [ ] Metrics: mission success, navigation error, safety violations, energy consumed,  
  fault recovery, latency  
- [ ] `scripts/ablation_benchmark.py`  

---

## PHASE 13 — ROS2 Integration  ← FUTURE

- [ ] Stable simulation milestone required first  
- [ ] Message adapters (sensor publishers, VehicleCommand subscriber)  
- [ ] Navigation node, planner node, control node  

---

## PHASE 14 — Raspberry Pi Hardware Validation  ← FUTURE

- [ ] Verify `raspi_gpio.py` on real hardware + log evidence  
- [ ] Hardware-level watchdog (relay / separate MCU)  
- [ ] Real sensor driver wiring  

---

## OPEN (from previous hardening pass)

- [ ] Test: `hw.thrusters.apply()` throws mid-tick (not timeout — explicit exception)  
- [ ] README: separate "implemented+tested" vs "hardware-unverified" vs "planned"  
- [ ] README: known-limitations section  

---

**Last updated:** 2026-09-11  
**Phase 1–9 status:** ✅ COMPLETE — Core autonomy pipeline implemented with full test coverage  
  - Sensor validation → state estimation → world model → fault detection → planning  
  - Memory systems (episodic, fault, procedural) with trust provenance  
  - Agent tool registry for LLM-callable functions  
  - Simulation lab with 13 fault-injection scenarios  
  - Integration tests verify end-to-end pipeline  
**Phase 10–14:** future milestones — LangGraph workflows, EKF/ESKF, hardware validation
