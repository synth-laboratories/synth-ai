"""CLI launch input boundary. See backend provenance and resource-inventory contracts.

Files select real receipts and resource references. This parser supplies no
receipts, resources, or authority defaults on behalf of a caller.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_ai.sdk.research.public import (
        DeploymentPin,
        ProvenanceMode,
        ResourceLimit,
        RunResourceBindings,
    )

MAX_INPUT_BYTES = 65536


class LaunchInputCode(StrEnum):
    UNREADABLE = "launch_input_unreadable"
    FILE_INVALID = "launch_input_file_invalid"
    TOO_LARGE = "launch_input_too_large"
    JSON_INVALID = "launch_input_json_invalid"
    BUDGET_INVALID = "launch_input_budget_invalid"
    FIELDS_INVALID = "launch_input_fields_invalid"
    VALUE_INVALID = "launch_input_value_invalid"
    DUPLICATE_REPOSITORY = "launch_input_duplicate_repository"
    DUPLICATE_REFERENCE = "launch_input_duplicate_reference"


class LaunchInputRefusalError(ValueError):
    """Safe closed input refusal; causes stay attached without rendering values."""

    def __init__(self, code: str, field: str):
        self.code = LaunchInputCode(code)
        self.field = field
        super().__init__(f"{code}: {field}")


@dataclass(frozen=True)
class LaunchInputs:
    deployment_pins: tuple[DeploymentPin, ...]
    provenance_mode: ProvenanceMode
    resource_bindings: RunResourceBindings
    limit: ResourceLimit


def _closed_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def _read_document(path: str, field: str):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        raise LaunchInputRefusalError("launch_input_unreadable", field) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise LaunchInputRefusalError("launch_input_file_invalid", field)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise LaunchInputRefusalError("launch_input_too_large", field)
        try:
            return json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_closed_object,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")),
            )
        except (ValueError, RecursionError) as error:
            raise LaunchInputRefusalError("launch_input_json_invalid", field) from error
    except OSError as error:
        raise LaunchInputRefusalError("launch_input_unreadable", field) from error
    finally:
        os.close(descriptor)


def _text(value, field, maximum=2048):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise LaunchInputRefusalError("launch_input_value_invalid", field)
    return value


def parse_launch_inputs(
    *, pins_path: str, resources_path: str, provenance_mode: str, budget_limit_usd: float
) -> LaunchInputs:
    """Validate all supplied launch authority before constructing a client.

    # See: backend/app/api/v1/managed_research/projects.py (launch contract)
    """
    from synth_ai.sdk.research.public import (
        DeploymentPin,
        ProvenanceMode,
        ResourceLimit,
        RunResourceBindings,
    )

    if not math.isfinite(budget_limit_usd) or budget_limit_usd < 0:
        raise LaunchInputRefusalError("launch_input_budget_invalid", "budget_limit_usd")
    pins_document = _read_document(pins_path, "deployment_pins")
    if not isinstance(pins_document, dict) or set(pins_document) != {"deployment_pins"}:
        raise LaunchInputRefusalError("launch_input_fields_invalid", "deployment_pins")
    pins = pins_document["deployment_pins"]
    if not isinstance(pins, list) or not 1 <= len(pins) <= 16:
        raise LaunchInputRefusalError("launch_input_value_invalid", "deployment_pins")
    parsed = []
    repositories = set()
    required = {"repository", "commit_sha", "environment", "resolved_at"}
    optional = {"deployment_id", "artifact_digest"}
    for pin in pins:
        if not isinstance(pin, dict) or not required <= set(pin) or set(pin) - required - optional:
            raise LaunchInputRefusalError("launch_input_fields_invalid", "deployment_pins")
        for field in required:
            _text(pin[field], "deployment_pins." + field)
        for field in optional & set(pin):
            if pin[field] is not None:
                _text(pin[field], "deployment_pins." + field)
        if pin["repository"] in repositories:
            raise LaunchInputRefusalError("launch_input_duplicate_repository", "deployment_pins")
        repositories.add(pin["repository"])
        try:
            parsed.append(
                DeploymentPin(**{**pin, "resolved_at": datetime.fromisoformat(pin["resolved_at"])})
            )
        except (ValueError, TypeError) as error:
            raise LaunchInputRefusalError(
                "launch_input_value_invalid", "deployment_pins"
            ) from error
    resources = _read_document(resources_path, "resource_bindings")
    inventory_fields = {"model_file_ids", "external_repository_ids", "credential_ref_ids"}
    if not isinstance(resources, dict) or set(resources) != inventory_fields:
        raise LaunchInputRefusalError("launch_input_fields_invalid", "resource_bindings")
    for field, values in resources.items():
        if not isinstance(values, list) or len(values) > 256:
            raise LaunchInputRefusalError(
                "launch_input_value_invalid", "resource_bindings." + field
            )
        for value in values:
            _text(value, "resource_bindings." + field, maximum=256)
        if len(values) != len(set(values)):
            raise LaunchInputRefusalError(
                "launch_input_duplicate_reference", "resource_bindings." + field
            )
    return LaunchInputs(
        tuple(parsed),
        ProvenanceMode(provenance_mode),
        RunResourceBindings.from_wire(resources),
        ResourceLimit(max_spend_usd=budget_limit_usd),
    )
