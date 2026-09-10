"""Tests for EnergyState and EnergyPlanner — Phase 3."""

from __future__ import annotations

import pytest

from auvbrain.planning.energy import EnergyDecision, EnergyPlanner, EnergyState


# ── EnergyState ───────────────────────────────────────────────────────────────

def test_from_observation_full_battery() -> None:
    e = EnergyState.from_observation(12.6, nominal_v=12.6, cutoff_v=10.0)
    assert e.soc == pytest.approx(1.0)


def test_from_observation_empty_battery() -> None:
    e = EnergyState.from_observation(10.0, nominal_v=12.6, cutoff_v=10.0)
    assert e.soc == pytest.approx(0.0)


def test_from_observation_midpoint() -> None:
    e = EnergyState.from_observation(11.3, nominal_v=12.6, cutoff_v=10.0)
    assert 0.4 < e.soc < 0.6


def test_available_wh_respects_reserve() -> None:
    e = EnergyState(soc=1.0, capacity_wh=100.0, reserve_fraction=0.10)
    assert e.available_wh == pytest.approx(90.0)


def test_energy_margin_positive_when_feasible() -> None:
    e = EnergyState(
        soc=1.0, capacity_wh=100.0, reserve_fraction=0.1,
        mission_cost_estimate=20.0, return_cost_estimate=10.0
    )
    assert e.energy_margin == pytest.approx(60.0)
    assert e.can_complete_mission is True


def test_energy_margin_negative_when_infeasible() -> None:
    e = EnergyState(
        soc=0.15, capacity_wh=100.0, reserve_fraction=0.1,
        mission_cost_estimate=50.0, return_cost_estimate=10.0
    )
    assert e.energy_margin < 0
    assert e.can_complete_mission is False


def test_cannot_return_when_available_below_return_cost() -> None:
    e = EnergyState(
        soc=0.08, capacity_wh=100.0, reserve_fraction=0.10,
        return_cost_estimate=5.0,
    )
    assert e.can_return is False


# ── EnergyPlanner ─────────────────────────────────────────────────────────────

@pytest.fixture()
def planner() -> EnergyPlanner:
    return EnergyPlanner(critical_soc=0.10, low_soc=0.20, replan_margin_wh=5.0)


def test_continue_when_energy_ample(planner: EnergyPlanner) -> None:
    e = EnergyState(soc=0.80, capacity_wh=100.0, reserve_fraction=0.1,
                    mission_cost_estimate=10.0, return_cost_estimate=5.0)
    assert planner.decide(e) == EnergyDecision.CONTINUE


def test_replan_when_margin_tight(planner: EnergyPlanner) -> None:
    # margin = 90 * 0.4 - 30 - 5 = 1 Wh → tight
    e = EnergyState(soc=0.40, capacity_wh=100.0, reserve_fraction=0.1,
                    mission_cost_estimate=30.0, return_cost_estimate=5.0)
    decision = planner.decide(e)
    assert decision in (EnergyDecision.REPLAN, EnergyDecision.ABORT)


def test_abort_when_mission_infeasible(planner: EnergyPlanner) -> None:
    e = EnergyState(soc=0.30, capacity_wh=100.0, reserve_fraction=0.1,
                    mission_cost_estimate=80.0, return_cost_estimate=5.0)
    assert planner.decide(e) == EnergyDecision.ABORT


def test_return_when_low_soc(planner: EnergyPlanner) -> None:
    e = EnergyState(soc=0.18, capacity_wh=100.0, reserve_fraction=0.1,
                    mission_cost_estimate=5.0, return_cost_estimate=3.0)
    assert planner.decide(e) == EnergyDecision.RETURN


def test_surface_when_critical_soc(planner: EnergyPlanner) -> None:
    e = EnergyState(soc=0.05, capacity_wh=100.0, reserve_fraction=0.1,
                    return_cost_estimate=3.0)
    assert planner.decide(e) == EnergyDecision.SURFACE


def test_surface_when_cannot_return(planner: EnergyPlanner) -> None:
    # available = 13% * 100 - 10% * 100 = 3 Wh, return needs 10 Wh
    e = EnergyState(soc=0.13, capacity_wh=100.0, reserve_fraction=0.10,
                    return_cost_estimate=10.0)
    assert planner.decide(e) == EnergyDecision.SURFACE
