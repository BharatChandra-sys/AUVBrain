from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VehicleMode(str, Enum):
    MANUAL = "MANUAL"
    AUTONOMOUS = "AUTONOMOUS"
    SAFE = "SAFE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUV_", env_file=".env", extra="ignore")

    mode: VehicleMode = VehicleMode.AUTONOMOUS
    log_level: str = "INFO"

    # Hardware profile
    # - SIM: simulated sensors/thrusters
    # - RASPI_STUB: Raspberry Pi adapters that currently log motor outputs (safe until you wire PWM)
    hardware: str = "SIM"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    use_llm: bool = False

    # LLM provider selection
    # - OLLAMA: uses Ollama's /api/chat
    # - OPENAI_COMPAT: uses OpenAI-compatible /v1/chat/completions (works with many local servers)
    # - LLAMA_CPP: uses llama-cpp-python to run a local GGUF model in-process (fully offline)
    llm_provider: str = "OLLAMA"

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "phi4-mini"
    ollama_timeout_s: float = 30.0

    # OpenAI-compatible server (for fully-offline local serving)
    openai_compat_url: str = "http://127.0.0.1:8001"
    openai_compat_model: str = "local-model"
    openai_compat_api_key: str = ""
    openai_compat_timeout_s: float = 30.0

    # llama.cpp in-process (GGUF) configuration
    llama_cpp_model_path: str = ""
    llama_cpp_n_ctx: int = 2048
    llama_cpp_n_threads: int | None = None
    llama_cpp_n_gpu_layers: int = 0
    llama_cpp_temperature: float = 0.2

    max_depth_m: float = 20.0
    min_battery_v: float = 10.8

    max_internal_temp_c: float = 75.0
    max_pressure_bar: float = 5.0

    emergency_obstacle_m: float = 0.4

    telemetry_dir: Path = Field(default=Path(".telemetry"))

    # Control-loop cadence (seconds). Defaults preserve existing behavior.
    tick_safe_s: float = 0.2
    tick_manual_s: float = 0.05
    tick_autonomous_s: float = 0.1

    # Optional cap for decision latency (seconds).
    # When LLM fallback is enabled, timeouts fall back to rules; otherwise the exception
    # will propagate.
    decision_timeout_s: float | None = None

    # Optional caps for I/O latency (seconds). If set, slow I/O will time out.
    sensor_read_timeout_s: float | None = None
    thruster_apply_timeout_s: float | None = None
    experiment_apply_timeout_s: float | None = None

    # Autonomy robustness (LLM fallback)
    llm_fallback_enabled: bool = True
    llm_max_consecutive_failures: int = 3
    llm_failure_cooldown_s: float = 5.0

    # Optional profiling (kept off by default to avoid overhead).
    profile_enabled: bool = False
    profile_every_n: int = 50

    # Telemetry writer tuning
    telemetry_flush_interval_s: float = 0.5
    telemetry_max_queue: int = 10_000

    # 4-motor configuration
    # - VECTORED_4_HORIZONTAL: 4 horizontal thrusters for surge/sway/yaw
    # - H2_V2: 2 horizontal + 2 vertical thrusters for surge/heave/yaw
    thruster_layout: str = "VECTORED_4_HORIZONTAL"
    motor_invert_0: bool = False
    motor_invert_1: bool = False
    motor_invert_2: bool = False
    motor_invert_3: bool = False

    # ESC PWM output (direct GPIO). Disabled by default for safety.
    pwm_enabled: bool = False
    motor_gpio_0: int | None = None
    motor_gpio_1: int | None = None
    motor_gpio_2: int | None = None
    motor_gpio_3: int | None = None

    # Typical ESC pulse widths (seconds). Adjust for your ESC calibration.
    esc_min_pulse_s: float = 0.001
    esc_max_pulse_s: float = 0.002
    esc_frame_s: float = 0.02


def load_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
