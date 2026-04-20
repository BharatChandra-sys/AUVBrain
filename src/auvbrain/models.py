from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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
    action: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class VehicleCommand(BaseModel):
    thrusters: ThrusterCommand = Field(default_factory=ThrusterCommand)
    experiment: ExperimentCommand = Field(default_factory=ExperimentCommand)
    note: Optional[str] = None


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

    # distance in meters; None means unavailable
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
