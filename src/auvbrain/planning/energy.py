"""Energy-aware autonomy — beyond a simple low-battery SAFE threshold.

``EnergyState`` tracks state-of-charge, consumption rate, and mission cost
estimates.  ``EnergyPlanner`` turns those numbers into an actionable decision:
continue, replan, abort, return, or surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EnergyDecision(str, Enum):
    CONTINUE = "continue"
    REPLAN   = "replan"    # re-optimize route to reduce energy spend
    ABORT    = "abort"     # abandon current goal, complete minimum mission
    RETURN   = "return"    # head back to surface/dock now
    SURFACE  = "surface"   # emergency ascent


@dataclass
class EnergyState:
    """Current energy status and forward-looking cost estimates.

    All cost fields are in Wh (watt-hours) for dimensional consistency.
    The planner only needs the ratios, so the units can be arbitrary as long
    as they are internally consistent.

    Attributes:
        soc:                    State of charge [0, 1].  1.0 = full.
        voltage:                Pack voltage in Volts.
        current_draw_a:         Instantaneous draw in Amps.
        capacity_wh:            Total pack capacity in Wh.
        propulsion_cost_per_m:  Estimated Wh per metre of travel.
        mission_cost_estimate:  Wh needed to finish current mission goal.
        return_cost_estimate:   Wh needed to return to surface/dock.
        reserve_fraction:       Fraction of capacity kept as reserve [0, 1].
    """

    soc: float = 1.0
    voltage: float = 12.0
    current_draw_a: float = 0.0
    capacity_wh: float = 100.0
    propulsion_cost_per_m: float = 0.05
    mission_cost_estimate: float = 0.0
    return_cost_estimate: float = 5.0
    reserve_fraction: float = 0.10

    @property
    def available_wh(self) -> float:
        """Energy available above the reserve."""
        return max(0.0, self.soc * self.capacity_wh - self.reserve_fraction * self.capacity_wh)

    @property
    def energy_margin(self) -> float:
        """available_wh − mission_cost − return_cost.  Negative = infeasible."""
        return self.available_wh - self.mission_cost_estimate - self.return_cost_estimate

    @property
    def can_complete_mission(self) -> bool:
        return self.energy_margin >= 0.0

    @property
    def can_return(self) -> bool:
        return self.available_wh >= self.return_cost_estimate

    @classmethod
    def from_observation(
        cls,
        battery_v: float,
        *,
        capacity_wh: float = 100.0,
        nominal_v: float = 12.6,
        cutoff_v: float = 10.0,
        current_draw_a: float = 2.0,
    ) -> "EnergyState":
        """Construct from a raw battery voltage reading."""
        span = max(nominal_v - cutoff_v, 0.1)
        soc = max(0.0, min(1.0, (battery_v - cutoff_v) / span))
        return cls(
            soc=soc,
            voltage=battery_v,
            current_draw_a=current_draw_a,
            capacity_wh=capacity_wh,
        )


class EnergyPlanner:
    """Convert EnergyState into an actionable EnergyDecision.

    Thresholds (all as fraction of capacity):
        critical_soc:   Force SURFACE immediately.
        low_soc:        Force RETURN.
        replan_margin:  Trigger route replan when margin drops below this.
    """

    def __init__(
        self,
        *,
        critical_soc: float = 0.10,
        low_soc: float = 0.20,
        replan_margin_wh: float = 5.0,
    ) -> None:
        self._critical_soc = critical_soc
        self._low_soc = low_soc
        self._replan_margin_wh = replan_margin_wh

    def decide(self, energy: EnergyState) -> EnergyDecision:
        """Return the recommended energy action.

        Priority order: SURFACE > RETURN > ABORT > REPLAN > CONTINUE.
        """
        if energy.soc <= self._critical_soc:
            logger.error("ENERGY CRITICAL soc=%.2f → SURFACE", energy.soc)
            return EnergyDecision.SURFACE

        if not energy.can_return:
            logger.error("ENERGY: cannot return  available=%.1f Wh → SURFACE", energy.available_wh)
            return EnergyDecision.SURFACE

        if energy.soc <= self._low_soc:
            logger.warning("ENERGY LOW soc=%.2f → RETURN", energy.soc)
            return EnergyDecision.RETURN

        if not energy.can_complete_mission:
            logger.warning(
                "ENERGY: mission infeasible  margin=%.1f Wh → ABORT", energy.energy_margin
            )
            return EnergyDecision.ABORT

        if energy.energy_margin < self._replan_margin_wh:
            logger.info("ENERGY: margin tight (%.1f Wh) → REPLAN", energy.energy_margin)
            return EnergyDecision.REPLAN

        return EnergyDecision.CONTINUE
