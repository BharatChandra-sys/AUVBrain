"""EstimatedState — the central navigation output consumed by planners and agents.

Every autonomy cycle produces:
    Observation → validate → EstimatedState + uncertainty

``EstimatedState`` is immutable (frozen dataclass) so it can be passed between
components without risk of mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True, slots=True)
class Vec3:
    """Simple 3-vector (x, y, z) for position and velocity."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


@dataclass(frozen=True, slots=True)
class EstimatedState:
    """Full vehicle state estimate with uncertainty.

    position_m:   NED or body-relative position in metres.
    velocity_ms:  Velocity in m/s (surge, sway, heave).
    heading_deg:  Magnetic/true heading in degrees [0, 360).
    depth_m:      Depth in metres (positive down).
    orientation:  Roll, pitch, yaw in degrees.
    covariance:   Diagonal of the 6-DOF state covariance matrix [pos+vel].
                  Lower = more certain.  None = no covariance estimate.
    confidence:   Scalar [0, 1] aggregate confidence (1.0 = fully trusted).
    sensor_sources: Which sensor channels contributed to this estimate.
    timestamp:    UTC timestamp when this estimate was produced.
    """

    position_m: Vec3 = Vec3()
    velocity_ms: Vec3 = Vec3()
    heading_deg: float = 0.0
    depth_m: float = 0.0
    orientation: Vec3 = Vec3()          # roll, pitch, yaw (deg)
    covariance: Optional[tuple[float, ...]] = None  # 6-element diagonal
    confidence: float = 1.0
    sensor_sources: tuple[str, ...] = ()
    timestamp: datetime = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Can't assign in frozen dataclass — use object.__setattr__
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc))

    @property
    def position_uncertainty_m(self) -> Optional[float]:
        """Scalar position uncertainty (RMS of position diagonal)."""
        if self.covariance is None or len(self.covariance) < 3:
            return None
        return (sum(c ** 2 for c in self.covariance[:3]) / 3) ** 0.5

    @property
    def is_trusted(self) -> bool:
        return self.confidence >= 0.6

    def with_confidence(self, confidence: float) -> "EstimatedState":
        """Return a copy with updated confidence."""
        return EstimatedState(
            position_m=self.position_m,
            velocity_ms=self.velocity_ms,
            heading_deg=self.heading_deg,
            depth_m=self.depth_m,
            orientation=self.orientation,
            covariance=self.covariance,
            confidence=max(0.0, min(1.0, confidence)),
            sensor_sources=self.sensor_sources,
            timestamp=self.timestamp,
        )
