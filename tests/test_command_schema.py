"""Tests for VehicleCommand schema, validation, and versioning."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from auvbrain.models import (
    COMMAND_SCHEMA_VERSION,
    ExperimentCommand,
    ThrusterCommand,
    VehicleCommand,
)


def test_thruster_bounds_default() -> None:
    cmd = ThrusterCommand()
    assert cmd.surge == 0.0
    assert cmd.sway == 0.0
    assert cmd.heave == 0.0
    assert cmd.yaw == 0.0


def test_thruster_accepts_valid_range() -> None:
    cmd = ThrusterCommand(surge=1.0, sway=-1.0, heave=0.5, yaw=-0.5)
    assert cmd.surge == 1.0
    assert cmd.sway == -1.0


def test_thruster_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ThrusterCommand(surge=1.1)
    with pytest.raises(ValidationError):
        ThrusterCommand(sway=-1.1)


def test_vehicle_command_defaults() -> None:
    cmd = VehicleCommand()
    assert cmd.experiment.enabled is False
    assert cmd.thrusters.surge == 0.0
    assert cmd.schema_version == COMMAND_SCHEMA_VERSION


def test_schema_version_valid() -> None:
    cmd = VehicleCommand(schema_version=COMMAND_SCHEMA_VERSION)
    assert cmd.schema_version == COMMAND_SCHEMA_VERSION


def test_schema_version_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        VehicleCommand(schema_version="99")


def test_note_max_length_enforced() -> None:
    """Note longer than 256 chars must be rejected."""
    with pytest.raises(ValidationError):
        VehicleCommand(note="x" * 257)


def test_note_max_length_accepted() -> None:
    cmd = VehicleCommand(note="x" * 256)
    assert len(cmd.note) == 256


def test_experiment_action_max_length() -> None:
    with pytest.raises(ValidationError):
        ExperimentCommand(action="a" * 65)


def test_experiment_params_key_count_limit() -> None:
    with pytest.raises(ValidationError):
        ExperimentCommand(params={str(i): i for i in range(17)})


def test_experiment_params_rejects_nested_dict() -> None:
    with pytest.raises(ValidationError):
        ExperimentCommand(params={"nested": {"a": 1}})


def test_experiment_params_rejects_nested_list() -> None:
    with pytest.raises(ValidationError):
        ExperimentCommand(params={"arr": [1, 2, 3]})


def test_experiment_params_accepts_scalars() -> None:
    cmd = ExperimentCommand(
        enabled=True,
        action="sample",
        params={"depth": 5.0, "duration_s": 30, "label": "test"},
    )
    assert cmd.params["depth"] == 5.0
