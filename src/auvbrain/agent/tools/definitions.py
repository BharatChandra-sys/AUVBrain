"""Concrete agent tool implementations — Phase 8.

Every tool is an async function with a typed signature.  The ToolRegistry
wraps these so the LLM agent can call them by name via JSON.

Architecture:
    LLM → tool_call JSON → ToolRegistry.dispatch() → tool fn → result dict
    → back into LLM context → final VehicleCommand
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...faults.models import Fault
from ...memory.models import MemoryTrust
from ...memory.retriever import RetrievalQuery, StateAwareRetriever
from ...metrics.registry import METRICS
from ...models import VehicleState
from ...navigation.estimated_state import EstimatedState
from ...planning.energy import EnergyState
from ...planning.tactical import TacticalPlanner
from ...planning.trajectory import TrajectoryEvaluator
from ...safety.shield import ActionShield
from ...state import STATE
from ...world.model import WorldState

logger = logging.getLogger(__name__)


# ── Tool: get_vehicle_state ───────────────────────────────────────────────────

async def get_vehicle_state() -> dict[str, Any]:
    """Return the current operational vehicle state snapshot."""
    state: VehicleState = await STATE.get()
    return {
        "mode": state.mode.value,
        "last_decision_source": (
            state.last_decision_source.value if state.last_decision_source else None
        ),
        "depth_m": state.last_observation.depth_m if state.last_observation else None,
        "battery_v": state.last_observation.battery_v if state.last_observation else None,
    }


# ── Tool: get_energy_state ────────────────────────────────────────────────────

async def get_energy_state(world: WorldState) -> dict[str, Any]:
    """Return energy state and forward-looking feasibility."""
    e = world.energy
    return {
        "soc": round(e.soc, 3),
        "voltage": e.voltage,
        "available_wh": round(e.available_wh, 2),
        "mission_cost_estimate_wh": round(e.mission_cost_estimate, 2),
        "return_cost_estimate_wh": round(e.return_cost_estimate, 2),
        "energy_margin_wh": round(e.energy_margin, 2),
        "can_complete_mission": e.can_complete_mission,
        "can_return": e.can_return,
    }


# ── Tool: get_navigation_estimate ─────────────────────────────────────────────

async def get_navigation_estimate(world: WorldState) -> dict[str, Any]:
    """Return the current navigation state estimate."""
    nav = world.navigation
    return {
        "depth_m": round(nav.depth_m, 3),
        "heading_deg": round(nav.heading_deg, 1),
        "confidence": nav.confidence,
        "is_trusted": nav.is_trusted,
        "position": {
            "x_m": round(nav.position_m.x, 3),
            "y_m": round(nav.position_m.y, 3),
            "z_m": round(nav.position_m.z, 3),
        },
        "velocity": {
            "surge_ms": round(nav.velocity_ms.x, 3),
            "sway_ms":  round(nav.velocity_ms.y, 3),
            "heave_ms": round(nav.velocity_ms.z, 3),
        },
        "sensor_sources": list(nav.sensor_sources),
    }


# ── Tool: get_sensor_health ───────────────────────────────────────────────────

async def get_sensor_health(world: WorldState) -> dict[str, Any]:
    """Return per-sensor health summary from the last validated observation."""
    # The world state summary includes fault count; detailed health comes from
    # the last ValidatedObservation (stored in world context by the loop).
    faults_by_sensor: dict[str, str] = {}
    for fault in world.faults:
        faults_by_sensor[fault.component] = fault.fault_type.value

    return {
        "observation_confidence": world.observation_confidence,
        "active_faults": [f.to_dict() for f in world.faults],
        "degraded_sensors": list(faults_by_sensor.keys()),
    }


# ── Tool: retrieve_fault_memory ───────────────────────────────────────────────

async def retrieve_fault_memory(
    fault: Fault,
    retriever: StateAwareRetriever,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve past fault episodes similar to the current fault."""
    from ...memory.fault_memory import FaultMemory
    # Use retriever directly for consistent pipeline
    query = RetrievalQuery(
        free_text=f"{fault.component} {fault.fault_type.value} recovery",
        active_faults=[fault.fault_type.value],
    )
    results = await retriever.retrieve(query, top_k=top_k)
    return [
        {
            "content": r.content,
            "source": r.source,
            "trust": r.trust.value,
            "outcome": None,  # would be in content
        }
        for r in results
    ]


# ── Tool: retrieve_procedure ──────────────────────────────────────────────────

async def retrieve_procedure(
    query_text: str,
    retriever: StateAwareRetriever,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve relevant procedures (PROCEDURAL memory type)."""
    from ...memory.models import MemoryType
    query = RetrievalQuery(
        mission_goal=query_text,
        free_text=query_text,
    )
    results = await retriever.retrieve(query, top_k=top_k)
    return [
        {"content": r.content, "source": r.source, "trust": r.trust.value}
        for r in results
        if r.memory_type == MemoryType.PROCEDURAL
    ] or [{"content": r.content, "source": r.source, "trust": r.trust.value} for r in results]


# ── Tool: evaluate_trajectory ─────────────────────────────────────────────────

async def evaluate_trajectory(
    waypoints_raw: list[dict],
    world: WorldState,
) -> dict[str, Any]:
    """Evaluate a proposed trajectory against current world state."""
    from ...world.mission_state import Waypoint
    from ...planning.trajectory import Trajectory

    waypoints = [
        Waypoint(
            x_m=float(wp.get("x_m", 0)),
            y_m=float(wp.get("y_m", 0)),
            depth_m=float(wp.get("depth_m", 0)),
            label=wp.get("label"),
        )
        for wp in waypoints_raw
    ]

    traj = Trajectory.from_waypoints(
        waypoints,
        propulsion_cost_per_m=world.energy.propulsion_cost_per_m,
    )
    evaluator = TrajectoryEvaluator()
    nearest = world.nearest_obstacle
    result = evaluator.evaluate(
        traj,
        available_energy_wh=world.energy.available_wh,
        nav_confidence=world.navigation.confidence,
        obstacle_clearance_m=nearest.distance_m if nearest else None,
    )
    return {
        "feasible": result.feasible,
        "score": result.score,
        "total_distance_m": round(traj.total_distance_m, 2),
        "total_energy_wh": round(traj.total_energy_wh, 2),
        "rejection_reason": result.rejection_reason,
    }


# ── Tool: check_action_safety ─────────────────────────────────────────────────

async def check_action_safety(
    command_dict: dict,
    world: WorldState,
) -> dict[str, Any]:
    """Run the ActionShield on a candidate command."""
    from ...models import VehicleCommand

    try:
        candidate = VehicleCommand.model_validate(command_dict)
    except Exception as exc:
        return {"verdict": "reject", "reason": f"invalid command schema: {exc}"}

    shield = ActionShield()
    decision = shield.evaluate(candidate, world)
    return {
        "verdict": decision.verdict.value,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "command": decision.command.model_dump(mode="json"),
    }


# ── Tool: request_replan ──────────────────────────────────────────────────────

async def request_replan(
    reason: str,
    world: WorldState,
) -> dict[str, Any]:
    """Request a tactical replan and return the best candidate."""
    planner = TacticalPlanner()
    candidates = planner.plan(world)
    if not candidates:
        return {"status": "no_candidates", "reason": reason}

    best = candidates[0]
    return {
        "status": "replanned",
        "reason": reason,
        "candidate_count": len(candidates),
        "best_rationale": best.rationale,
        "best_command": best.command.model_dump(mode="json"),
    }


# ── Tool: execute_mission_action ──────────────────────────────────────────────

async def execute_mission_action(
    action: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Execute a named mission-level action.

    Supported actions: advance_waypoint, set_phase, hold_position
    """
    # This tool is wired to the WorldStateBuilder's mission context
    # via the agent loop in a full integration.  Here we return a
    # structured intent that the loop applies.
    return {
        "status": "queued",
        "action": action,
        "params": params,
    }
