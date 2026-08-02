"""Application settings loaded from environment variables / .env file.

All settings are prefixed with ``AUV_``.  Run ``auv-api`` or ``auv-agent``
after setting the values in a ``.env`` file (see .env.example).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VehicleMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"
    SAFE = "SAFE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUV_", env_file=".env", extra="ignore")

    mode: VehicleMode = VehicleMode.AUTONOMOUS
    log_level: str = "INFO"

    # ── Hardware ────────────────────────────────────────────────────────────
    # - SIM: simulated sensors/thrusters
    # - RASPI_STUB / RASPI_GPIO: Raspberry Pi adapters
    hardware: str = "SIM"

    # ── API server ──────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # ── Database ────────────────────────────────────────────────────────────
    # asyncpg URL for production; sqlite+aiosqlite for dev/CI
    db_url: str = "sqlite+aiosqlite:///.auvbrain.db"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # ── Auth ────────────────────────────────────────────────────────────────
    # Set to false ONLY for isolated local dev / CI (no auth on any endpoint).
    auth_enabled: bool = True

    # ── Rate limiting ───────────────────────────────────────────────────────
    # Max write commands per client per second (0 = disabled)
    rate_limit_commands_per_s: int = 20
    rate_limit_window_s: float = 1.0

    # ── LLM ─────────────────────────────────────────────────────────────────
    use_llm: bool = False
    # OLLAMA | OPENAI_COMPAT | LLAMA_CPP
    llm_provider: str = "OLLAMA"

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "phi4-mini"
    ollama_timeout_s: float = 30.0

    openai_compat_url: str = "http://127.0.0.1:8001"
    openai_compat_model: str = "local-model"
    openai_compat_api_key: str = ""
    openai_compat_timeout_s: float = 30.0

    llama_cpp_model_path: str = ""
    llama_cpp_n_ctx: int = 2048
    llama_cpp_n_threads: int | None = None
    llama_cpp_n_gpu_layers: int = 0
    llama_cpp_temperature: float = 0.2

    # ── Safety limits ───────────────────────────────────────────────────────
    max_depth_m: float = 20.0
    min_battery_v: float = 10.8
    max_internal_temp_c: float = 75.0
    max_pressure_bar: float = 5.0
    emergency_obstacle_m: float = 0.4

    # ── Dead-man's switch (MANUAL mode) ─────────────────────────────────────
    # If no new command arrives within this many seconds in MANUAL mode,
    # auto-transition to SAFE.  Set to 0 to disable.
    manual_deadman_s: float = 10.0

    # ── Telemetry ───────────────────────────────────────────────────────────
    telemetry_dir: Path = Field(default=Path(".telemetry"))

    # Also persist telemetry to DB (async, non-blocking to control loop)
    telemetry_db_enabled: bool = True

    # ── Control-loop cadence (seconds) ──────────────────────────────────────
    tick_safe_s: float = 0.2
    tick_manual_s: float = 0.05
    tick_autonomous_s: float = 0.1

    # ── Decision latency caps ───────────────────────────────────────────────
    decision_timeout_s: float | None = None
    sensor_read_timeout_s: float | None = None
    thruster_apply_timeout_s: float | None = None
    experiment_apply_timeout_s: float | None = None

    # ── LLM fallback ────────────────────────────────────────────────────────
    llm_fallback_enabled: bool = True
    llm_max_consecutive_failures: int = 3
    llm_failure_cooldown_s: float = 5.0

    # ── Profiling ───────────────────────────────────────────────────────────
    profile_enabled: bool = False
    profile_every_n: int = 50

    # ── Telemetry writer tuning ─────────────────────────────────────────────
    telemetry_flush_interval_s: float = 0.5
    telemetry_max_queue: int = 10_000

    # ── 4-motor configuration ────────────────────────────────────────────────
    thruster_layout: str = "VECTORED_4_HORIZONTAL"
    motor_invert_0: bool = False
    motor_invert_1: bool = False
    motor_invert_2: bool = False
    motor_invert_3: bool = False

    # ── ESC / GPIO ──────────────────────────────────────────────────────────
    pwm_enabled: bool = False
    motor_gpio_0: int | None = None
    motor_gpio_1: int | None = None
    motor_gpio_2: int | None = None
    motor_gpio_3: int | None = None

    esc_min_pulse_s: float = 0.001
    esc_max_pulse_s: float = 0.002
    esc_frame_s: float = 0.02

    # ── Alerting (safety override webhook) ──────────────────────────────────
    # Optional URL that receives a POST when SafetyMonitor forces SAFE mode.
    # Leave empty to disable.
    alert_webhook_url: str = ""

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("hardware")
    @classmethod
    def _normalise_hardware(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("llm_provider")
    @classmethod
    def _normalise_llm_provider(cls, v: str) -> str:
        return v.upper().strip()


def load_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
