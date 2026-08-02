"""Domain models for AUVBrain.

All models use Pydantic v2.  Validators enforce hard limits so that malformed
operator input can never reach the hardware layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ── Current schema version — bump when the command shape changes ──────────
COMMAND_SCHEMA_VERSION = "1"


class DecisionSource(str, Enum):
    RULES = "rules"
    LLM = "llm"


class ThrusterCommand(BaseModel):
    surge: float = Field(0.0, ge=-1.0, le=1.0, description="Forward/back")
    sway: float = Field(0.0, ge=-1.0, le=1.0, description="Left/right")
    heave: float = Field(0.0, ge=-1.0, le=1.0, description="Up/down")
    yaw: float = Field(0.0, ge=-1.0, le=1.0, description="Turn left/right")


class ExperimentCommand(BaseModel):
    enabled: bool = False
    action: Optional[Annotated[str, Field(max_length=64)]] = None
    # Bound the free-form params blob: max 16 keys, values are scalar only.
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _bound_params(cls, v: dict) -> dict:
        if len(v) > 16:
            raise ValueError("experiment.params may have at most 16 keys")
        for key, val in v.items():
            if not isinstance(key, str):
                raise ValueError("experiment.params keys must be strings")
            if len(key) > 64:
                raise ValueError(f"experiment.params key too long: {key!r}")
            if isinstance(val, (dict, list)):
                raise ValueError(
                    f"experiment.params values must be scalars; got {type(val).__name__!r} "
                    f"for key {key!r}"
                )
        return v


class VehicleCommand(BaseModel):
    """Command sent from the decision engine to the hardware layer.

    ``schema_version`` lets the API and hardware detect mismatches without
    silent failures.
    """

    schema_version: str = Field(
        default=COMMAND_SCHEMA_VERSION,
        description="Schema version token — bump when shape changes",
    )
    thrusters: ThrusterCommand = Field(default_factory=ThrusterCommand)
    experiment: ExperimentCommand = Field(default_factory=ExperimentCommand)
    note: Optional[Annotated[str, Field(max_length=256)]] = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v != COMMAND_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: expected {COMMAND_SCHEMA_VERSION!r}, got {v!r}. "
                "Upgrade your client."
            )
        return v


class Observation(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    depth_m: float = 0.0
    battery_v: float = 12.0

    # Health / fault signals (optional, depending on your hardware)
    water_ingress: Optional[bool] = None
    internal_temp_c: Optional[float] = None
    pressure_bar: Optional[float] = None

    imu_ok: Optional[bool] = None
    sonar_ok: Optional[bool] = None
    experiment_ok: Optional[bool] = None

    # distance in metres; None means unavailable
    obstacle_front_m: Optional[float] = None

    # free-form extra sensor readings (pH, salinity, temp, etc.)
    sensors: dict[str, Any] = Field(default_factory=dict)

    # Human-readable alerts produced by sensor validation / fault detection
    alerts: list[str] = Field(default_factory=list)


class ControlMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"
    SAFE = "SAFE"


class VehicleState(BaseModel):
    mode: ControlMode = ControlMode.AUTONOMOUS
    last_observation: Optional[Observation] = None
    last_command: Optional[VehicleCommand] = None
    last_decision_source: Optional[DecisionSource] = None


class ManualControlMessage(BaseModel):
    type: Literal["command", "set_mode", "ping"]
    command: Optional[VehicleCommand] = None
    mode: Optional[ControlMode] = None
    client_ts: Optional[str] = None
