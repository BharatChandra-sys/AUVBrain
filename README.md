# AUVBrain

AUVBrain is an **uncertainty-aware, fault-tolerant autonomy stack** for an experimental AUV (Raspberry Pi friendly).
It provides the complete wiring from sensors → state estimation → world modeling → planning → decisions → actuators,
with multiple safety layers, persistent memory, telemetry logging, and an operator control hub (FastAPI + WebSocket).

The "brain" is not the base model weights. The defensible engineering is:
- **Sensor validation** and health monitoring with cross-sensor consistency checks
- **State estimation** (dead-reckoning baseline; EKF-ready architecture)
- **World model** with obstacles, currents, energy, faults, and mission context
- **Fault detection, diagnosis, and recovery** strategies
- **Energy-aware planning** with mission feasibility prediction
- **Action safety shield** between AI decisions and physical actuation
- **Persistent memory** (episodic, fault, procedural) with trust provenance
- **Agent tools** for LLM-callable functions (state queries, trajectory evaluation, safety checks)
- **Simulation lab** with 13 fault-injection scenarios for validation
- A deterministic **observe → validate → estimate → decide → shield → act** agent loop
- A safety monitor that can **force SAFE** regardless of the decision engine
- Hardware adapters that let the same code run in simulation or on the vehicle
- An API/control hub for monitoring and manual override

## What this repo includes

### Core Autonomy Pipeline (Phase 1-9 ✅ Complete)

- **Sensor validation** (`src/auvbrain/perception/`) — per-sensor health, range checks, spike detection, frozen value detection, cross-sensor consistency
- **State estimation** (`src/auvbrain/navigation/`) — EstimatedState with position, velocity, heading, confidence, covariance; dead-reckoning baseline (EKF/ESKF ready)
- **World model** (`src/auvbrain/world/`) — unified WorldState with navigation, obstacles, currents, energy, faults, mission context
- **Energy planning** (`src/auvbrain/planning/energy.py`) — forward-looking energy feasibility (can_complete_mission, can_return, energy_margin)
- **Fault detection** (`src/auvbrain/faults/`) — detects DVL dropout, IMU freeze, depth drift, thruster degradation, water ingress; diagnosis → recovery strategies
- **Tactical planner** (`src/auvbrain/planning/`) — generates trajectory candidates with cost, risk, and energy evaluation
- **Action safety shield** (`src/auvbrain/safety/shield.py`) — validates candidate actions for depth, velocity, energy feasibility, navigation confidence, sensor validity
- **Persistent memory** (`src/auvbrain/memory/`) — 5-class taxonomy (PROCEDURAL, SEMANTIC, EPISODIC, FAULT, ENVIRONMENTAL) with trust provenance (VERIFIED_PROCEDURE → AGENT_GENERATED)
- **Agent tools** (`src/auvbrain/agent/tools/`) — LLM-callable functions: get_vehicle_state, get_energy_state, get_navigation_estimate, get_sensor_health, evaluate_trajectory, check_action_safety, request_replan
- **Simulation lab** (`src/auvbrain/simulation/`) — physics-based dynamics, sensor suite with noise/bias/dropout, fault injector, 13 built-in scenarios (normal, DVL dropout, battery critical, thruster degradation, water ingress, etc.)

### Foundation (Already Complete)

- **Agent loop** that produces VehicleCommand outputs at a fixed cadence (now with full pipeline: validate → estimate → detect faults → build world → plan → decide → shield → act)
- **Safety monitor** (depth/battery/temp/ingress/obstacle limits) that overrides unsafe actions
- **Telemetry** (append-only JSONL + async DB sink) for replay/debugging and mission evidence
- **Observability** — structured JSON logging, correlation IDs, metrics endpoint (/metrics), latency percentiles, fallback counters, safety override counters
- **Database** — PostgreSQL + SQLite support (SQLAlchemy async), API key management, RAG backend, schema versioning
- **Hardware bridge** for thrusters + sensors + experiment module (SIM + Raspberry Pi adapters)
- **LLM providers** (optional):
	- Rules engine (always offline)
	- Ollama
	- OpenAI-compatible local server (vLLM/TGI/llama.cpp server/etc.)
	- In-process GGUF via llama.cpp (fully offline; no server)
- **Test suite** — 80+ tests covering sensor validation, dead reckoning, energy planning, fault detection, action shield, trajectory planning, memory retrieval, simulation scenarios, world state integration
- **CI/CD** — GitHub Actions (pytest on Python 3.10–3.12, ruff, coverage ≥70%)
- **Latency benchmark + PNG proof** you can regenerate offline

## Architecture

```mermaid
flowchart TD
	Sensors["Sensors\n(depth, IMU, DVL, sonar, battery, temp)"] -->|Observation| Validator["Sensor Validator\nhealth checks"]
	Validator -->|ValidatedObservation| Estimator["State Estimator\n(dead-reckoning/EKF)"]
	Estimator -->|EstimatedState| FaultDetector["Fault Detector"]
	FaultDetector -->|list[Fault]| WorldBuilder["World State Builder"]
	Validator -->|observation_confidence| WorldBuilder
	EnergyPlanner["Energy Planner"] -->|EnergyState| WorldBuilder
	WorldBuilder -->|WorldState| TacticalPlanner["Tactical Planner"]
	WorldBuilder -->|WorldState| Memory["Persistent Memory\n(episodic/fault/procedural)"]
	TacticalPlanner -->|PlanCandidate| DecisionEngine["Decision Engine\n(LLM + Tools / Rules)"]
	Memory -.->|context| DecisionEngine
	DecisionEngine -->|VehicleCommand| ActionShield["Action Safety Shield"]
	ActionShield -->|validated| SafetyMonitor["Safety Monitor\n(hard limits)"]
	SafetyMonitor -->|final command| Thrusters["Thrusters\n(PWM/ESC)"]
	SafetyMonitor -->|final command| Experiments["Experimental Module\n(pump/camera/relay)"]

	SafetyMonitor -->|telemetry + world state| Telemetry[(Telemetry Log + DB)]

	UI["Operator / Dashboard"] <--> |WebSocket /ws/control| API["FastAPI Control Hub"]
	API -->|mode + manual command| SafetyMonitor
```

## Modes

- **AUTONOMOUS**: agent loop drives the vehicle
- **MANUAL**: vehicle applies operator commands (REST/WS)
- **SAFE**: neutral thrusters + experiments disabled

## Quick start (Windows dev / SIM)

```powershell
cd C:\Users\bc833\llm
py -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# Terminal 1: API (docs + manual override)
auv-api

# Terminal 2: agent loop (simulation adapters)
auv-agent
```

Open API docs: http://127.0.0.1:8000/docs

## Fully offline autonomy (no internet)

You can run **fully offline** in two layers:

1) Always-on deterministic autonomy (Rules)
- Set `AUV_USE_LLM=false` to run purely offline with Rules.

2) Offline local model (optional)
- Keep safety + Rules fallback enabled so the vehicle stays autonomous even if the model stalls.

### Option A: In-process GGUF via llama.cpp (no server)

```powershell
pip install -e .[local-llm]

# .env
AUV_USE_LLM=true
AUV_LLM_PROVIDER=LLAMA_CPP
AUV_LLAMA_CPP_MODEL_PATH=C:\path\to\model.gguf

# Bound worst-case decision latency; on timeout it falls back to Rules
AUV_DECISION_TIMEOUT_S=0.4
AUV_LLM_FALLBACK_ENABLED=true
```

### Option B: Your own local model server (OpenAI-compatible)

Works with many offline servers that expose `POST /v1/chat/completions`.

```powershell
# .env
AUV_USE_LLM=true
AUV_LLM_PROVIDER=OPENAI_COMPAT
AUV_OPENAI_COMPAT_URL=http://127.0.0.1:8001
AUV_OPENAI_COMPAT_MODEL=local-model

AUV_DECISION_TIMEOUT_S=0.4
AUV_LLM_FALLBACK_ENABLED=true
```

## Hardware + experiments

### Thrusters (4 motors)

The control contract is 4-DOF thrusters:
- surge (forward/back)
- sway (left/right)
- heave (up/down)
- yaw (turn)

Adapters mix those into 4 motor outputs (m0..m3) using one of these layouts:
- `AUV_THRUSTER_LAYOUT=VECTORED_4_HORIZONTAL` → surge+sway+yaw
- `AUV_THRUSTER_LAYOUT=H2_V2` → surge+heave+yaw

### Sensors (typical 4-sensor set)

Recommended core signals:
- Depth/pressure → depth_m (and optionally pressure_bar)
- Battery voltage → battery_v
- Obstacle distance → obstacle_front_m
- Leak or internal temperature → water_ingress / internal_temp_c

### Experiments (adaptable)

Experiments are intentionally open-ended:
- enabled / action
- params (free-form JSON)

That allows pumps/cameras/relays/samplers without changing the command schema each time.

## Control hub API

- GET /health
- GET /state
- POST /mode
- POST /command
- WS /ws/control

Run: `auv-api` then open http://127.0.0.1:8000/docs

## Telemetry

Telemetry is append-only JSONL written off the event loop for low latency, plus async DB sink.
Each tick records: mode, observation, command, decision source, world state summary (depth, nav confidence, energy SOC, fault count, mission phase, nearest obstacle), correlation ID.
The benchmark writes under docs/ and is safe to regenerate.

## Testing

Run the full test suite:

```powershell
pip install -e .[dev]
pytest
```

Key test coverage:
- **Sensor validation** — dropout, stale data, spike, range, cross-sensor consistency
- **Dead reckoning** — position integration, confidence decay, depth fusion
- **Energy planning** — SOC calculation, mission feasibility, return feasibility, energy margin
- **Fault detection** — DVL dropout, IMU freeze, depth drift, thruster degradation
- **Action shield** — depth limits, energy feasibility, navigation confidence, velocity checks
- **Trajectory planning** — waypoint evaluation, obstacle avoidance, energy-constrained planning
- **Tactical planner** — candidate generation, risk scoring, fault-aware planning
- **Memory retrieval** — episodic recall, fault memory lookup, state-aware RAG
- **Simulation scenarios** — 13 fault-injection scenarios (normal, DVL dropout, battery critical, IMU bias, thruster degradation, water ingress, etc.)
- **World state integration** — full pipeline from observation → validated → estimated → world state

## Latency proof (PNG)

This repo includes a fully offline benchmark that runs the agent in **SIM + Rules** mode and
renders a PNG chart with tick jitter + work-time percentiles.

![Latency proof](docs/latency_proof.png)

Current benchmark snapshot (Windows SIM, Rules, profile every tick):
- Work time p95 ≈ 0.839 ms, p99 ≈ 1.332 ms, max ≈ 2.55 ms
- Tick period p95 ≈ 32.01 ms (Windows timer resolution dominates cadence)

Regenerate (fully offline):

```powershell
pip install -e .[dev]
python scripts/latency_benchmark.py
python scripts/plot_latency.py
```

## GitHub Pages (public flow + outputs)

This repo ships a tiny static site in `docs/` that shows the system flow diagrams and the benchmark outputs.

- https://chandu1234678.github.io/AUVBrain/

## Configuration

See the complete list of environment variables in .env.example.

## Raspberry Pi notes

- Run headless to save RAM.
- Install Pi adapters: `pip install -e .[raspi]`
- For real thrusters/experiments, wire GPIO logic in src/auvbrain/hardware/raspi_gpio.py

## Future Work (Phase 10-14)

- **LangGraph mission workflows** — structured mission decomposition, tool coordination, replanning, human escalation
- **EKF/ESKF navigation** — sensor fusion with IMU, DVL, depth, heading
- **ROS2 integration** — sensor publishers, command subscribers, navigation/planner/control nodes
- **Raspberry Pi hardware validation** — real sensor drivers, hardware watchdog
- **Architecture ablation benchmarks** — measure impact of each autonomy layer (baseline → +state estimation → +world model → +fault diagnosis → +energy planning → +memory → +LangGraph)
