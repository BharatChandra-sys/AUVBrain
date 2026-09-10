"""Dead-reckoning state estimator.

Integrates velocity commands over time to produce a pose estimate.
Confidence decays when no sensor updates arrive (expressing growing
positional uncertainty without an absolute reference).

This is a placeholder for a full EKF/ESKF — it uses the same interface
so the EKF can replace it without touching callers.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from ..models import Observation
from ..perception.validator import ValidatedObservation
from .estimated_state import EstimatedState, Vec3

logger = logging.getLogger(__name__)

# Confidence decay rate per second without absolute sensor updates
_CONFIDENCE_DECAY_PER_S: float = 0.02
_MIN_CONFIDENCE: float = 0.10


class DeadReckoningEstimator:
    """Integrate velocity forward and decay confidence over time.

    Args:
        initial_depth_confidence: Confidence assigned when a fresh depth
            reading is available (typically 0.9 — depth sensors are reliable).
        velocity_scale: Scale factor applied to ThrusterCommand surge/sway/heave
            to produce an approximate velocity in m/s.  Highly approximate;
            replace with a proper DVL integration once available.
    """

    def __init__(
        self,
        *,
        initial_depth_confidence: float = 0.90,
        velocity_scale: float = 0.5,
    ) -> None:
        self._pos = Vec3()
        self._vel = Vec3()
        self._heading_deg: float = 0.0
        self._confidence: float = 0.85
        self._last_ts: float = time.monotonic()
        self._initial_depth_confidence = initial_depth_confidence
        self._velocity_scale = velocity_scale

    def update(
        self,
        validated: ValidatedObservation,
        surge: float = 0.0,
        sway: float = 0.0,
        heave: float = 0.0,
    ) -> EstimatedState:
        """Produce a new EstimatedState from the latest validated observation.

        Args:
            validated:  Output of SensorValidator.
            surge/sway/heave: Normalised thruster commands [-1, 1] used to
                         estimate velocity direction.
        """
        now = time.monotonic()
        dt = now - self._last_ts
        self._last_ts = now

        obs = validated.obs
        depth_health = validated.get_health("depth_m")
        depth_confidence = depth_health.confidence if depth_health else 0.5

        # ── Integrate position ───────────────────────────────────────────
        v = self._velocity_scale
        self._vel = Vec3(surge * v, sway * v, heave * v)
        self._pos = Vec3(
            self._pos.x + self._vel.x * dt,
            self._pos.y + self._vel.y * dt,
            self._pos.z + self._vel.z * dt,
        )

        # ── Fuse depth reading ───────────────────────────────────────────
        depth_m = obs.depth_m
        if depth_health and depth_health.healthy:
            # Reset z-position drift using depth sensor
            self._pos = Vec3(self._pos.x, self._pos.y, depth_m)
            # Partial confidence recovery from reliable depth update
            self._confidence = min(
                1.0,
                self._confidence + depth_confidence * 0.05,
            )
        else:
            # Decay confidence without absolute reference
            self._confidence = max(
                _MIN_CONFIDENCE,
                self._confidence - _CONFIDENCE_DECAY_PER_S * dt,
            )

        # ── Aggregate confidence ─────────────────────────────────────────
        agg_confidence = min(self._confidence, validated.observation_confidence)

        sources = tuple(
            sid for sid, h in validated.sensor_health.items() if h.healthy
        )

        if agg_confidence < 0.4:
            logger.warning(
                "navigation confidence low: %.2f  sources=%s", agg_confidence, sources
            )

        return EstimatedState(
            position_m=Vec3(self._pos.x, self._pos.y, self._pos.z),
            velocity_ms=self._vel,
            heading_deg=self._heading_deg,
            depth_m=depth_m,
            orientation=Vec3(0.0, 0.0, self._heading_deg),
            covariance=(
                # Rough diagonal: position variance grows with dt, velocity uncertain
                dt * 0.1, dt * 0.1, 0.01,   # pos x, y, z
                0.1, 0.1, 0.05,              # vel x, y, z
            ),
            confidence=round(agg_confidence, 4),
            sensor_sources=sources,
            timestamp=datetime.now(timezone.utc),
        )

    def reset(self) -> None:
        """Reset position estimate to origin (e.g. on surfacing)."""
        self._pos = Vec3()
        self._vel = Vec3()
        self._confidence = 0.85
        self._last_ts = time.monotonic()
        logger.info("dead-reckoning estimator reset")
