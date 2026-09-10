"""Tool registry and dispatcher — Phase 8.

The ToolRegistry:
  1. Maps tool names (strings) to async callable functions.
  2. Dispatches a tool call from LLM JSON output.
  3. Returns structured results for injection into the next LLM context.

This is the pattern from FactCheckAI adapted for AUVBrain:
    LLM JSON → ToolRegistry.dispatch() → fn(**args) → result dict

Adding a new tool: call ``registry.register(name, fn)``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

from ..world.model import WorldState

logger = logging.getLogger(__name__)

# Type alias for tool functions
ToolFn = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


class ToolRegistry:
    """Dispatches LLM tool calls to Python functions.

    Usage::

        registry = ToolRegistry(world)
        result = await registry.dispatch("get_energy_state", {})
    """

    def __init__(self, world: WorldState) -> None:
        self._world = world
        self._tools: dict[str, ToolFn] = {}
        self._register_defaults()

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn
        logger.debug("tool registered: %s", name)

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call the named tool with ``args`` and return its result.

        Returns an error dict if the tool is unknown or raises.
        """
        fn = self._tools.get(name)
        if fn is None:
            logger.warning("unknown tool: %s", name)
            return {"error": f"unknown tool: {name}"}

        try:
            result = await fn(**args)
            logger.debug("tool %s → %s", name, str(result)[:120])
            return result
        except Exception as exc:
            logger.error("tool %s raised: %s", name, exc, exc_info=True)
            return {"error": str(exc), "tool": name}

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    # ── Default tool wiring ───────────────────────────────────────────────

    def _register_defaults(self) -> None:
        from .tools.definitions import (
            check_action_safety,
            evaluate_trajectory,
            get_energy_state,
            get_navigation_estimate,
            get_sensor_health,
            get_vehicle_state,
            request_replan,
        )

        world = self._world

        # Stateless tools (no world dependency injected)
        self.register("get_vehicle_state", get_vehicle_state)

        # World-bound tools (partial application)
        async def _energy() -> dict:           return await get_energy_state(world)
        async def _nav() -> dict:              return await get_navigation_estimate(world)
        async def _health() -> dict:           return await get_sensor_health(world)

        async def _eval_traj(waypoints: list) -> dict:
            return await evaluate_trajectory(waypoints, world)

        async def _safety(command: dict) -> dict:
            return await check_action_safety(command, world)

        async def _replan(reason: str = "") -> dict:
            return await request_replan(reason, world)

        self.register("get_energy_state",          _energy)
        self.register("get_navigation_estimate",   _nav)
        self.register("get_sensor_health",         _health)
        self.register("evaluate_trajectory",       _eval_traj)
        self.register("check_action_safety",       _safety)
        self.register("request_replan",            _replan)


def build_tool_schema() -> list[dict]:
    """Return the OpenAI-style tools schema for injection into the system prompt."""
    return [
        {
            "name": "get_vehicle_state",
            "description": "Get current operational mode, depth, and battery.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_energy_state",
            "description": "Get SOC, available Wh, mission cost estimate, and energy margin.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_navigation_estimate",
            "description": "Get estimated position, velocity, heading, depth, and nav confidence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_sensor_health",
            "description": "Get per-sensor health, active faults, and observation confidence.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "evaluate_trajectory",
            "description": "Score a proposed waypoint list for energy feasibility and risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x_m": {"type": "number"},
                                "y_m": {"type": "number"},
                                "depth_m": {"type": "number"},
                                "label": {"type": "string"},
                            },
                        },
                    }
                },
                "required": ["waypoints"],
            },
        },
        {
            "name": "check_action_safety",
            "description": "Run the ActionShield on a candidate VehicleCommand dict.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "object"}},
                "required": ["command"],
            },
        },
        {
            "name": "request_replan",
            "description": "Request a tactical replan and return the best plan candidate.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": [],
            },
        },
    ]
