"""Tests for FallbackDecisionEngine — timeout, cooldown, failure counting."""

from __future__ import annotations

import asyncio

import pytest

from auvbrain.agent.policy import FallbackDecisionEngine, RuleDecisionEngine
from auvbrain.models import DecisionSource, Observation, VehicleCommand


def _obs() -> Observation:
    return Observation()


class _AlwaysSucceedEngine(RuleDecisionEngine):
    """Immediately returns a known command."""
    source = DecisionSource.LLM

    async def decide(self, obs: Observation) -> VehicleCommand:
        return VehicleCommand(note="llm-success")


class _AlwaysTimeoutEngine(RuleDecisionEngine):
    """Simulates a slow LLM that always times out."""
    source = DecisionSource.LLM

    async def decide(self, obs: Observation) -> VehicleCommand:
        await asyncio.sleep(10)  # will be cancelled by timeout
        return VehicleCommand()


class _AlwaysRaiseEngine(RuleDecisionEngine):
    source = DecisionSource.LLM

    async def decide(self, obs: Observation) -> VehicleCommand:
        raise RuntimeError("simulated LLM error")


class _ParseFailEngine(RuleDecisionEngine):
    """Simulates an LLM that returns a 'parse fail' command."""
    source = DecisionSource.LLM

    async def decide(self, obs: Observation) -> VehicleCommand:
        return VehicleCommand(note="LLM parse fail; SAFE neutral")


# ── Basic fallback on timeout ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_on_timeout() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysTimeoutEngine(),
        RuleDecisionEngine(),
        timeout_s=0.05,
        enabled=True,
    )
    cmd = await engine.decide(_obs())
    assert engine.source == DecisionSource.RULES
    # Rule engine gives cruise command
    assert cmd.thrusters.surge == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_fallback_on_exception() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysRaiseEngine(),
        RuleDecisionEngine(),
        timeout_s=None,
        enabled=True,
    )
    cmd = await engine.decide(_obs())
    assert engine.source == DecisionSource.RULES


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysSucceedEngine(),
        RuleDecisionEngine(),
        timeout_s=5.0,
        enabled=True,
    )
    cmd = await engine.decide(_obs())
    assert cmd.note == "llm-success"
    assert engine.source == DecisionSource.LLM


# ── Parse-fail treated as failure ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_fail_treated_as_failure() -> None:
    engine = FallbackDecisionEngine(
        _ParseFailEngine(),
        RuleDecisionEngine(),
        timeout_s=5.0,
        enabled=True,
    )
    cmd = await engine.decide(_obs())
    assert engine.source == DecisionSource.RULES


# ── Consecutive-failure cooldown ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_consecutive_failures_trigger_cooldown() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysRaiseEngine(),
        RuleDecisionEngine(),
        timeout_s=None,
        enabled=True,
        max_consecutive_failures=3,
        failure_cooldown_s=60.0,
    )
    for _ in range(3):
        await engine.decide(_obs())

    # After 3 failures, cooldown is active
    assert engine._consecutive_failures >= 3
    assert engine._cooldown_until > 0.0


@pytest.mark.asyncio
async def test_cooldown_uses_fallback_directly() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysRaiseEngine(),
        RuleDecisionEngine(),
        timeout_s=None,
        enabled=True,
        max_consecutive_failures=1,
        failure_cooldown_s=60.0,
    )
    # Trigger cooldown
    await engine.decide(_obs())
    # Now cooldown is active — next call should go straight to rules
    engine._consecutive_failures = 999  # ensure we don't recurse into primary
    cmd = await engine.decide(_obs())
    assert engine.source == DecisionSource.RULES


# ── Disabled fallback ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_fallback_uses_primary_directly() -> None:
    engine = FallbackDecisionEngine(
        _AlwaysSucceedEngine(),
        RuleDecisionEngine(),
        timeout_s=5.0,
        enabled=False,
    )
    cmd = await engine.decide(_obs())
    assert cmd.note == "llm-success"
    assert engine.source == DecisionSource.LLM


# ── Consecutive failure reset on success ──────────────────────────────────────

@pytest.mark.asyncio
async def test_success_resets_consecutive_failures() -> None:
    primary_calls = 0

    class _FailThenSucceed(RuleDecisionEngine):
        source = DecisionSource.LLM

        async def decide(self, obs: Observation) -> VehicleCommand:
            nonlocal primary_calls
            primary_calls += 1
            if primary_calls < 3:
                raise RuntimeError("fail")
            return VehicleCommand(note="recovered")

    engine = FallbackDecisionEngine(
        _FailThenSucceed(),
        RuleDecisionEngine(),
        timeout_s=None,
        enabled=True,
        max_consecutive_failures=99,
        failure_cooldown_s=0.0,
    )
    # Two failures
    await engine.decide(_obs())
    await engine.decide(_obs())
    assert engine._consecutive_failures == 2
    # Success resets
    await engine.decide(_obs())
    assert engine._consecutive_failures == 0
