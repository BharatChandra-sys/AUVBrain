"""Safety monitor — hard-overrides any decision engine.

All conditions are checked on every tick.  The first matching condition wins.
Overrides are logged at ERROR level and fire the optional alert webhook.

Webhook
-------
Set ``AUV_ALERT_WEBHOOK_URL`` to receive a POST with a JSON body whenever
the safety monitor forces a mode change:

    {
      "kind": "safety_override",
      "reason": "SAFE: max depth exceeded",
      "mode": "SAFE"
    }

The webhook call is best-effort (failure is logged, never fatal).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import Settings
from ..metrics.registry import METRICS
from ..models import ControlMode, Observation, VehicleCommand

logger = logging.getLogger(__name__)


@dataclass
class SafetyMonitor:
    settings: Settings

    def enforce(
        self, obs: Observation, cmd: VehicleCommand
    ) -> tuple[ControlMode | None, VehicleCommand]:
        """Apply hard safety limits.

        Returns ``(override_mode, safe_cmd)`` where ``override_mode`` is
        non-None when a full mode change is required (forces SAFE), or
        ``(None, safe_cmd)`` for a command-only override (obstacle stop).

        The caller is responsible for applying both the command and the mode.
        """
        # ── Water ingress (highest priority) ─────────────────────────────
        if obs.water_ingress is True:
            return self._override("water ingress detected", ControlMode.SAFE)

        # ── Depth limit ───────────────────────────────────────────────────
        if obs.depth_m > self.settings.max_depth_m:
            return self._override(
                f"depth {obs.depth_m:.1f}m > max {self.settings.max_depth_m:.1f}m",
                ControlMode.SAFE,
            )

        # ── Battery limit ─────────────────────────────────────────────────
        if obs.battery_v < self.settings.min_battery_v:
            return self._override(
                f"battery {obs.battery_v:.2f}V < min {self.settings.min_battery_v:.2f}V",
                ControlMode.SAFE,
            )

        # ── Internal temperature ──────────────────────────────────────────
        if (
            obs.internal_temp_c is not None
            and obs.internal_temp_c > self.settings.max_internal_temp_c
        ):
            return self._override(
                f"temp {obs.internal_temp_c:.1f}°C > max {self.settings.max_internal_temp_c:.1f}°C",
                ControlMode.SAFE,
            )

        # ── Hull pressure ─────────────────────────────────────────────────
        if (
            obs.pressure_bar is not None
            and obs.pressure_bar > self.settings.max_pressure_bar
        ):
            return self._override(
                f"pressure {obs.pressure_bar:.2f}bar > max {self.settings.max_pressure_bar:.2f}bar",
                ControlMode.SAFE,
            )

        # ── Emergency obstacle stop (command-only override, no mode change) ─
        if (
            obs.obstacle_front_m is not None
            and obs.obstacle_front_m < self.settings.emergency_obstacle_m
        ):
            reason = f"obstacle {obs.obstacle_front_m:.2f}m"
            note = f"SAFE: {reason}"
            logger.warning("safety: obstacle stop — %s", reason)
            safe_cmd = VehicleCommand(note=note)
            return None, safe_cmd

        return None, cmd

    # ── Internals ─────────────────────────────────────────────────────────

    def _override(
        self, reason: str, mode: ControlMode
    ) -> tuple[ControlMode, VehicleCommand]:
        note = f"SAFE: {reason}"
        logger.error("SAFETY OVERRIDE → %s  reason=%s", mode.value, reason)
        METRICS.inc("safe_overrides_total")
        safe_cmd = VehicleCommand(note=note)
        self._fire_alert(reason=note, mode=mode)
        return mode, safe_cmd

    def _fire_alert(self, *, reason: str, mode: ControlMode) -> None:
        """Best-effort alert webhook.  Non-blocking."""
        url = self.settings.alert_webhook_url
        if not url:
            return
        try:
            asyncio.get_running_loop().create_task(
                _post_webhook(url, {"kind": "safety_override", "reason": reason, "mode": mode.value})
            )
        except RuntimeError:
            # No running event loop (e.g. test context) — skip webhook
            pass


async def _post_webhook(url: str, payload: dict) -> None:
    """Fire-and-forget HTTP POST to the alert webhook URL."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("alert webhook OK status=%d", resp.status_code)
    except Exception as exc:
        logger.warning("alert webhook failed: %s", exc)
