"""Decision engines for the AUVBrain agent loop.

Hierarchy
---------
  FallbackDecisionEngine
    ├── primary:  LLMDecisionEngine   (optional)
    └── fallback: RuleDecisionEngine  (always-on deterministic)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..config import Settings
from ..llm.ollama_client import OllamaClient
from ..llm.openai_compat_client import OpenAICompatClient
from ..llm.prompts import SYSTEM_PROMPT, user_prompt
from ..metrics.registry import METRICS
from ..models import DecisionSource, Observation, VehicleCommand

if TYPE_CHECKING:
    from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class DecisionEngine:
    source: DecisionSource

    async def decide(self, obs: Observation) -> VehicleCommand:  # pragma: no cover
        raise NotImplementedError


class RuleDecisionEngine(DecisionEngine):
    source = DecisionSource.RULES

    async def decide(self, obs: Observation) -> VehicleCommand:
        cmd = VehicleCommand()

        # Emergency obstacle avoidance: stop surge and yaw away
        if obs.obstacle_front_m is not None and obs.obstacle_front_m < 1.0:
            cmd.thrusters.surge = 0.0
            cmd.thrusters.yaw = 0.6
            cmd.note = f"Obstacle {obs.obstacle_front_m:.2f}m: yaw right"
            return cmd

        # Default: slow forward cruise
        cmd.thrusters.surge = 0.2
        cmd.note = "Cruise"
        return cmd


class LLMDecisionEngine(DecisionEngine):
    source = DecisionSource.LLM

    def __init__(self, settings: Settings, tool_registry: "ToolRegistry | None" = None) -> None:
        self.tool_registry = tool_registry
        provider = settings.llm_provider
        if provider == "OLLAMA":
            self.client = OllamaClient(
                settings.ollama_url,
                settings.ollama_model,
                timeout_s=settings.ollama_timeout_s,
            )
        elif provider in {"OPENAI_COMPAT", "OPENAI", "OPENAI_COMPATIBLE"}:
            self.client = OpenAICompatClient(
                settings.openai_compat_url,
                settings.openai_compat_model,
                api_key=settings.openai_compat_api_key,
                timeout_s=settings.openai_compat_timeout_s,
            )
        elif provider in {"LLAMA_CPP", "LLAMACPP", "LLAMA.CPP"}:
            from ..llm.llama_cpp_client import LlamaCppClient

            if not settings.llama_cpp_model_path:
                raise ValueError("AUV_LLAMA_CPP_MODEL_PATH is required for LLAMA_CPP")
            self.client = LlamaCppClient(
                settings.llama_cpp_model_path,
                n_ctx=settings.llama_cpp_n_ctx,
                n_threads=settings.llama_cpp_n_threads,
                n_gpu_layers=settings.llama_cpp_n_gpu_layers,
                temperature=settings.llama_cpp_temperature,
            )
        else:
            raise ValueError(f"Unknown AUV_LLM_PROVIDER={settings.llm_provider!r}")

    async def aclose(self) -> None:
        aclose = getattr(self.client, "aclose", None)
        if callable(aclose):
            await aclose()

    async def decide(self, obs: Observation) -> VehicleCommand:
        obs_json = obs.model_dump_json()
        
        # If tool registry is available, allow multi-turn tool calls
        if self.tool_registry is not None:
            return await self._decide_with_tools(obs_json)
        
        # Otherwise fallback to single-shot decision (legacy path)
        raw = await self.client.chat_json(system=SYSTEM_PROMPT, user=user_prompt(obs_json))

        try:
            return VehicleCommand.model_validate_json(raw)
        except ValidationError:
            cleaned = _extract_json_object(raw)
            if cleaned is None:
                logger.warning("LLM output not valid JSON: %r", raw[:200])
                METRICS.inc("llm_failures_total")
                return VehicleCommand(note="LLM parse fail; SAFE neutral")
            try:
                return VehicleCommand.model_validate_json(cleaned)
            except ValidationError:
                logger.warning("LLM output JSON failed schema: %r", cleaned[:200])
                METRICS.inc("llm_failures_total")
                return VehicleCommand(note="LLM schema fail; SAFE neutral")

    async def _decide_with_tools(self, obs_json: str) -> VehicleCommand:
        """Multi-turn decision loop with tool calls.
        
        LLM can call tools to gather context before emitting final VehicleCommand.
        Max 3 tool-call rounds to prevent runaway.
        """
        from .tool_registry import build_tool_schema
        
        tools_schema = build_tool_schema()
        system_with_tools = SYSTEM_PROMPT + "\n\n**Available Tools:**\n" + json.dumps(tools_schema, indent=2)
        
        context = user_prompt(obs_json)
        max_rounds = 3
        
        for round_idx in range(max_rounds):
            raw = await self.client.chat_json(system=system_with_tools, user=context)
            
            # Try to parse as VehicleCommand first (final decision)
            try:
                return VehicleCommand.model_validate_json(raw)
            except ValidationError:
                pass
            
            # Otherwise check if it's a tool call request
            try:
                parsed = json.loads(_extract_json_object(raw) or raw)
                if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                    tool_name = parsed["tool"]
                    tool_args = parsed.get("args", {})
                    logger.debug("LLM tool call: %s(%s)", tool_name, tool_args)
                    
                    result = await self.tool_registry.dispatch(tool_name, tool_args)
                    context += f"\n\n**Tool Result ({tool_name}):**\n{json.dumps(result, indent=2)}\n\nNow emit final VehicleCommand."
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
            
            # If we can't parse it as command or tool call, fail gracefully
            logger.warning("LLM output not valid after round %d: %r", round_idx, raw[:200])
            METRICS.inc("llm_failures_total")
            return VehicleCommand(note="LLM output invalid; SAFE neutral")
        
        # Max rounds exceeded
        logger.warning("LLM tool loop exceeded %d rounds, forcing neutral", max_rounds)
        return VehicleCommand(note="LLM tool loop timeout; SAFE neutral")


class FallbackDecisionEngine(DecisionEngine):
    """Bounds decision latency and degrades gracefully from LLM to Rules.

    - Primary (LLM) is tried first.
    - On timeout or exception, the deterministic fallback is used.
    - After ``max_consecutive_failures`` failures, enter a cooldown where
      the fallback is used directly for ``failure_cooldown_s`` seconds.
    """

    def __init__(
        self,
        primary: DecisionEngine,
        fallback: DecisionEngine,
        *,
        timeout_s: float | None,
        enabled: bool = True,
        max_consecutive_failures: int = 3,
        failure_cooldown_s: float = 5.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._timeout_s = timeout_s if (timeout_s is None or timeout_s > 0) else None
        self._enabled = bool(enabled)
        self._max_failures = max(1, int(max_consecutive_failures))
        self._cooldown_s = max(0.0, float(failure_cooldown_s))

        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        self.source = fallback.source

    async def decide(self, obs: Observation) -> VehicleCommand:
        if not self._enabled:
            self.source = self._primary.source
            return await self._primary.decide(obs)

        now = time.monotonic()
        if self._cooldown_s > 0 and now < self._cooldown_until:
            METRICS.inc("llm_fallbacks_total")
            self.source = self._fallback.source
            return await self._fallback.decide(obs)

        try:
            if self._timeout_s is not None:
                import asyncio

                cmd = await asyncio.wait_for(self._primary.decide(obs), timeout=self._timeout_s)
            else:
                cmd = await self._primary.decide(obs)

            note = cmd.note or ""
            if "LLM parse fail" in note or "LLM schema fail" in note:
                raise RuntimeError("LLM invalid output")

            self._consecutive_failures = 0
            self.source = self._primary.source
            return cmd

        except Exception as exc:
            self._consecutive_failures += 1
            METRICS.inc("llm_fallbacks_total")
            logger.warning(
                "Primary engine failed (consecutive=%d): %s",
                self._consecutive_failures,
                exc,
            )
            if self._consecutive_failures >= self._max_failures and self._cooldown_s > 0:
                self._cooldown_until = time.monotonic() + self._cooldown_s
                logger.warning(
                    "LLM cooldown engaged for %.1fs after %d consecutive failures",
                    self._cooldown_s,
                    self._consecutive_failures,
                )
            self.source = self._fallback.source
            return await self._fallback.decide(obs)


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
