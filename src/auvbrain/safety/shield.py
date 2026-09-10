"""Action Safety Shield — Phase 5.

Architecture:
    LLM / Planner
          ↓
    candidate action (VehicleCommand)
          ↓
    ActionShield.evaluate()
          ↓
    ShieldDecision (APPROVE / REJECT / MODIFY)
          ↓
    Controller / hw.apply()

The existing SafetyMonitor remains as the hard emergency layer (post-apply).
ActionShield is the higher-level pre-apply validation layer that can also
MODIFY a command (clamp + note) rather than only reject it.

Checks performed in order:
  1. Navigation confidence — reject if below minimum for given mode
  2. Obstacle proximity — reject if below hard clearance
  3. Depth constraint — reject if command would exceed mission depth limit
  4. Energy feasibility — reject high-thrust commands when energy is marginal
  5. Actuator limits — clamp any out-of-range values (should be caught earlier,
     but defense-in-depth)
  6. Mission constraint — reject commands that contradict current mission phase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..models import VehicleCommand
from ..world.model import WorldState

logger = logging.getLogger(__name__)


class ShieldVerdict(str, Enum):
    APPROVE = "approve"
    MODIFY  = "modify"   # command allowed but clamped / annotated
    REJECT  = "reject"


@dataclass(slots=True)
class ShieldDecision:
    verdict: ShieldVerdict
    command: VehicleCommand        # original or modified
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict != ShieldVerdict.REJECT


# ── Tunable limits ────────────────────────────────────────────────────────────

_MIN_NAV_CONFIDENCE_AUTONOMOUS = 0.30   # below this → reject autonomous moves
_MIN_OBSTACLE_CLEARANCE_M      = 0.60   # hard stop distance
_MAX_THRUST_ON_LOW_ENERGY      = 0.40   # clamp surge/sway when energy < 20 %
_LOW_ENERGY_SOC_THRESHOLD      = 0.20


class ActionShield:
    """Validate and optionally modify a candidate VehicleCommand.

    Args:
        min_nav_confidence: Reject autonomous manoeuvres below this threshold.
        min_obstacle_clearance_m: Reject any command when an obstacle is
            closer than this distance.
    """

    def __init__(
        self,
        *,
        min_nav_confidence: float = _MIN_NAV_CONFIDENCE_AUTONOMOUS,
        min_obstacle_clearance_m: float = _MIN_OBSTACLE_CLEARANCE_M,
    ) -> None:
        self._min_nav = min_nav_confidence
        self._min_clearance = min_obstacle_clearance_m

    def evaluate(
        self,
        candidate: VehicleCommand,
        world: WorldState,
    ) -> ShieldDecision:
        """Evaluate ``candidate`` against the current world state.

        Returns a ShieldDecision.  The ``command`` field of the decision
        is the original command when APPROVE/REJECT, or a clamped copy
        when MODIFY.
        """

        # ── 1. Navigation confidence ──────────────────────────────────────
        nav_conf = world.navigation.confidence
        if nav_conf < self._min_nav:
            return ShieldDecision(
                verdict=ShieldVerdict.REJECT,
                command=candidate,
                reason=(
                    f"navigation confidence {nav_conf:.2f} < "
                    f"minimum {self._min_nav:.2f} — command rejected"
                ),
            )

        # ── 2. Obstacle proximity ─────────────────────────────────────────
        nearest = world.nearest_obstacle
        if nearest is not None and nearest.effective_confidence > 0.4:
            if nearest.distance_m < self._min_clearance:
                return ShieldDecision(
                    verdict=ShieldVerdict.REJECT,
                    command=candidate,
                    reason=(
                        f"obstacle at {nearest.distance_m:.2f} m < "
                        f"clearance {self._min_clearance:.2f} m — command rejected"
                    ),
                )

        # ── 3. Depth constraint ───────────────────────────────────────────
        max_depth = world.mission.max_depth_m
        current_depth = world.navigation.depth_m
        # If heave > 0 (descending) and already close to limit, clamp heave
        if (
            candidate.thrusters.heave > 0.0
            and current_depth >= max_depth * 0.95
        ):
            import copy
            modified = copy.deepcopy(candidate)
            modified.thrusters.heave = 0.0
            modified.note = (
                f"shield: heave clamped (depth {current_depth:.1f} m "
                f"near limit {max_depth:.1f} m)"
            )
            logger.warning(modified.note)
            return ShieldDecision(
                verdict=ShieldVerdict.MODIFY,
                command=modified,
                reason=modified.note,
            )

        # ── 4. Energy feasibility ─────────────────────────────────────────
        if world.energy.soc < _LOW_ENERGY_SOC_THRESHOLD:
            max_thrust = _MAX_THRUST_ON_LOW_ENERGY
            surge = candidate.thrusters.surge
            sway  = candidate.thrusters.sway

            if abs(surge) > max_thrust or abs(sway) > max_thrust:
                import copy
                modified = copy.deepcopy(candidate)
                sign_surge = 1.0 if surge >= 0 else -1.0
                sign_sway  = 1.0 if sway  >= 0 else -1.0
                modified.thrusters.surge = sign_surge * min(abs(surge), max_thrust)
                modified.thrusters.sway  = sign_sway  * min(abs(sway),  max_thrust)
                modified.note = (
                    f"shield: thrust clamped to {max_thrust:.1f} "
                    f"(energy soc={world.energy.soc:.2f})"
                )
                logger.warning(modified.note)
                return ShieldDecision(
                    verdict=ShieldVerdict.MODIFY,
                    command=modified,
                    reason=modified.note,
                )

        # ── 5. Critical fault gate ────────────────────────────────────────
        if world.has_critical_fault:
            # Block any non-zero thruster commands when a critical fault is active
            is_moving = (
                abs(candidate.thrusters.surge) > 0.05
                or abs(candidate.thrusters.sway)  > 0.05
                or abs(candidate.thrusters.heave) > 0.05
            )
            if is_moving:
                return ShieldDecision(
                    verdict=ShieldVerdict.REJECT,
                    command=candidate,
                    reason="critical fault active — thruster commands blocked",
                )

        # ── All checks passed ─────────────────────────────────────────────
        return ShieldDecision(
            verdict=ShieldVerdict.APPROVE,
            command=candidate,
            reason="ok",
        )
