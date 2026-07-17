"""One-shot Research Factory stand-up workflow over the public SDK."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import json
import re
import sys
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any, cast

from synth_ai.managed_research.models.factories import FactoryWakeDueResult
from synth_ai.managed_research.models.types import SmrRunnableProjectRequest
from synth_ai.managed_research.sdk.client import ManagedResearchClient

_BUILTIN_PLANS = {
    "rsi-synth-on-synth": "rsi_synth_on_synth.plan.json",
}
_OPERATOR_INPUT_PLAN_SCHEMA_VERSION = "synth.factory_standup_plan.rsi.v1"
_FILE_EVIDENCE_REF_RE = re.compile(r"^file:/[^#\s]+#sha256=(?P<digest>[0-9a-f]{64})$")
_WAIVER_EVIDENCE_REF_RE = re.compile(
    r"^waiver:(?P<expiry>\d{4}-\d{2}-\d{2}):"
    r"(?P<locator>file:/[^#\s]+)#sha256=(?P<digest>[0-9a-f]{64})$"
)
_EVIDENCE_PACKET_SCHEMA_VERSION = "synth.rsi_operator_probe_attestation.v1"
_EVIDENCE_BINDING_SCHEMA_VERSION = "synth.rsi_activation_evidence_binding.v1"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return _jsonable(to_wire())
    return value


def _json_object(raw: str, *, field: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("builtin:"):
        plan_name = value.removeprefix("builtin:").strip()
        resource_name = _BUILTIN_PLANS.get(plan_name)
        if resource_name is None:
            available = ", ".join(sorted(_BUILTIN_PLANS))
            raise ValueError(f"unknown built-in plan {plan_name!r}; available: {available}")
        value = (
            resources.files("synth_ai.managed_research.factory_plans")
            .joinpath(resource_name)
            .read_text(encoding="utf-8")
        )
    elif value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text()
    elif not value.startswith("{"):
        candidate = Path(value).expanduser()
        if candidate.exists():
            value = candidate.read_text()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object or @path to one")
    return dict(parsed)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _required_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def _required_string(parent: Mapping[str, Any], key: str) -> str:
    value = str(parent.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _required_typed_string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _json_contract_equal(actual: object, expected: object) -> bool:
    """Compare JSON claims without Python's bool/integer equality coercion."""

    try:
        actual_json = json.dumps(actual, sort_keys=True, separators=(",", ":"))
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual_json, expected_json)


def _evidence_locator_and_digest(reference: str) -> tuple[str, str]:
    waiver_match = _WAIVER_EVIDENCE_REF_RE.fullmatch(reference)
    if waiver_match is not None:
        return waiver_match.group("locator"), waiver_match.group("digest")
    file_match = _FILE_EVIDENCE_REF_RE.fullmatch(reference)
    if file_match is None:
        raise ValueError(
            "evidence references must be absolute file: URIs ending in "
            "#sha256=<64 lowercase hex characters>; dated waivers wrap that URI "
            "as waiver:YYYY-MM-DD:file:/...#sha256=..."
        )
    locator = reference.rsplit("#sha256=", maxsplit=1)[0]
    return locator, file_match.group("digest")


def _validate_evidence_packet(packet: object) -> dict[str, Any]:
    try:
        normalized_packet = dict(packet) if isinstance(packet, Mapping) else None
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence packet must be a JSON object") from exc
    packet = normalized_packet
    if not isinstance(packet, dict):
        raise ValueError("evidence packet must be a JSON object")
    allowed_packet_fields = {
        "schema_version",
        "subject_id",
        "authority",
        "attestation_type",
        "attested_by",
        "captured_at",
        "valid_until",
        "probe",
        "claims",
    }
    if set(packet) - allowed_packet_fields:
        raise ValueError("evidence packet contains unknown fields")
    if packet.get("schema_version") != _EVIDENCE_PACKET_SCHEMA_VERSION:
        raise ValueError(
            f"evidence packet schema_version must equal {_EVIDENCE_PACKET_SCHEMA_VERSION!r}"
        )
    _required_typed_string(packet, "subject_id")
    _required_typed_string(packet, "authority")
    if packet.get("attestation_type") != "operator_probe_capture":
        raise ValueError("evidence packet attestation_type must be operator_probe_capture")
    _required_typed_string(packet, "attested_by")
    captured_at = _required_typed_string(packet, "captured_at")
    try:
        captured_timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("evidence packet captured_at must be RFC 3339") from exc
    if captured_timestamp.tzinfo is None or captured_timestamp.utcoffset() is None:
        raise ValueError("evidence packet captured_at must include a UTC offset")
    if captured_timestamp > datetime.now().astimezone():
        raise ValueError("evidence packet captured_at cannot be in the future")
    probe = _required_mapping(packet, "probe")
    if set(probe) != {"command", "exit_code", "output", "output_digest"}:
        raise ValueError("evidence packet probe has unknown or missing fields")
    _required_typed_string(probe, "command")
    if type(probe.get("exit_code")) is not int or probe.get("exit_code") != 0:
        raise ValueError("evidence packet probe must record exit_code 0")
    probe_output = probe.get("output")
    if not isinstance(probe_output, str):
        raise ValueError("evidence packet probe output must be a JSON string")
    output_digest = _required_typed_string(probe, "output_digest")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", output_digest) is None:
        raise ValueError("evidence packet probe output_digest must be a SHA-256 digest")
    actual_output_digest = f"sha256:{hashlib.sha256(probe_output.encode('utf-8')).hexdigest()}"
    if not hmac.compare_digest(actual_output_digest, output_digest):
        raise ValueError("evidence packet probe output does not match output_digest")
    claims = _required_mapping(packet, "claims")
    try:
        output_claims = json.loads(probe_output)
    except json.JSONDecodeError as exc:
        raise ValueError("evidence packet probe output must be a JSON object") from exc
    if not isinstance(output_claims, Mapping) or not _json_contract_equal(
        dict(output_claims),
        claims,
    ):
        raise ValueError("evidence packet claims must exactly equal parsed probe output")
    return packet


def _load_evidence_packet(reference: str) -> dict[str, Any]:
    """Resolve one immutable operator attestation with captured probe output."""

    locator, expected_digest = _evidence_locator_and_digest(reference)
    parsed = urllib.parse.urlsplit(locator)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("evidence locator must be an absolute local file URI")
    evidence_path = Path(urllib.parse.unquote(parsed.path))
    if not evidence_path.is_absolute() or not evidence_path.is_file():
        raise ValueError("evidence locator does not resolve to an existing regular file")
    evidence_bytes = evidence_path.read_bytes()
    actual_digest = hashlib.sha256(evidence_bytes).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ValueError("evidence packet SHA-256 does not match its reference")
    try:
        packet = json.loads(evidence_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence packet must be UTF-8 JSON") from exc
    return _validate_evidence_packet(packet)


def _resolve_evidence_binding(reference: str) -> dict[str, Any]:
    packet = _load_evidence_packet(reference)
    _locator, source_digest = _evidence_locator_and_digest(reference)
    canonical_packet = json.dumps(
        packet,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    waiver_match = _WAIVER_EVIDENCE_REF_RE.fullmatch(reference)
    return {
        "schema_version": _EVIDENCE_BINDING_SCHEMA_VERSION,
        "source_digest": f"sha256:{source_digest}",
        "packet_digest": f"sha256:{hashlib.sha256(canonical_packet).hexdigest()}",
        "waiver_expires_on": (waiver_match.group("expiry") if waiver_match is not None else None),
        "packet": packet,
    }


def _load_evidence_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("rendered evidence must be an embedded evidence binding")
    binding = {str(key): item for key, item in value.items()}
    if set(binding) != {
        "schema_version",
        "source_digest",
        "packet_digest",
        "waiver_expires_on",
        "packet",
    }:
        raise ValueError("rendered evidence binding has unknown or missing fields")
    if binding.get("schema_version") != _EVIDENCE_BINDING_SCHEMA_VERSION:
        raise ValueError("rendered evidence binding has an unsupported schema")
    for key in ("source_digest", "packet_digest"):
        digest = _required_typed_string(binding, key)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"evidence binding {key} must be a SHA-256 digest")
    packet_digest = _required_typed_string(binding, "packet_digest")
    waiver_expires_on = binding.get("waiver_expires_on")
    if waiver_expires_on is not None and (
        not isinstance(waiver_expires_on, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", waiver_expires_on) is None
    ):
        raise ValueError("evidence binding waiver_expires_on must be YYYY-MM-DD or null")
    packet = _validate_evidence_packet(binding.get("packet"))
    canonical_packet = json.dumps(
        packet,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    expected_packet_digest = f"sha256:{hashlib.sha256(canonical_packet).hexdigest()}"
    if not hmac.compare_digest(packet_digest, expected_packet_digest):
        raise ValueError("embedded evidence packet does not match packet_digest")
    return binding


def _required_evidence_binding(
    parent: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    try:
        return _load_evidence_binding(parent.get(key))
    except ValueError as exc:
        raise ValueError(f"{key} must be a verified embedded evidence binding: {exc}") from exc


def _verify_evidence_packet(
    binding_value: object,
    *,
    subject_id: str,
    authority: str,
    expected_claims: Mapping[str, object] | None = None,
    require_current: bool = False,
    max_validity_seconds: int | None = None,
) -> None:
    binding = _load_evidence_binding(binding_value)
    packet = _validate_evidence_packet(binding["packet"])
    if packet.get("subject_id") != subject_id:
        raise ValueError(f"evidence packet does not prove subject {subject_id!r}")
    if packet.get("authority") != authority:
        raise ValueError(
            f"operator attestation for {subject_id!r} must name source authority {authority!r}"
        )
    if require_current:
        if max_validity_seconds is None or max_validity_seconds < 1:
            raise ValueError("mutable evidence requires a positive maximum validity")
        valid_until = _required_typed_string(packet, "valid_until")
        captured_at = _required_typed_string(packet, "captured_at")
        try:
            valid_until_timestamp = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
            captured_timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("mutable evidence valid_until must be RFC 3339") from exc
        if valid_until_timestamp.tzinfo is None or valid_until_timestamp.utcoffset() is None:
            raise ValueError("mutable evidence valid_until must include a UTC offset")
        now = datetime.now().astimezone()
        if valid_until_timestamp <= now:
            raise ValueError(f"evidence packet for {subject_id!r} is expired")
        if valid_until_timestamp <= captured_timestamp:
            raise ValueError(
                f"evidence packet for {subject_id!r} has a nonpositive validity window"
            )
        maximum_window = timedelta(seconds=max_validity_seconds)
        if now - captured_timestamp > maximum_window:
            raise ValueError(f"evidence packet for {subject_id!r} is too old")
        if valid_until_timestamp - captured_timestamp > maximum_window:
            raise ValueError(f"evidence packet for {subject_id!r} exceeds its maximum validity")
        waiver_expiry = binding.get("waiver_expires_on")
        if isinstance(waiver_expiry, str) and (
            valid_until_timestamp.date() > datetime.fromisoformat(waiver_expiry).date()
        ):
            raise ValueError(f"evidence packet for {subject_id!r} outlives its dated waiver")
    claims = _required_mapping(packet, "claims")
    for key, expected in dict(expected_claims or {}).items():
        if not _json_contract_equal(claims.get(key), expected):
            raise ValueError(f"evidence packet for {subject_id!r} changed claim {key!r}")


def _parse_plan_inputs(values: list[str]) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for item in values:
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        if not separator or not name or not raw_value.strip():
            raise ValueError("--plan-input must use NAME=VALUE with a non-empty value")
        if name in inputs:
            raise ValueError(f"duplicate --plan-input {name!r}")
        inputs[name] = raw_value.strip()
    return inputs


def _verify_rsi_operator_bindings(plan: Mapping[str, Any]) -> None:
    """Keep authoritative RSI budget values bound to explicit operator inputs."""

    if plan.get("schema_version") != _OPERATOR_INPUT_PLAN_SCHEMA_VERSION:
        return
    input_specs = _required_mapping(plan, "operator_inputs")

    def verify_evidence_references(value: object, *, field: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                item_field = f"{field}.{key}"
                if key.endswith("evidence_ref") or key.endswith("receipt_ref"):
                    if not isinstance(item, Mapping) or set(item) != {"$operator_input"}:
                        raise ValueError(
                            f"{item_field} must remain bound to an evidence operator input"
                        )
                    input_name = _required_string(item, "$operator_input")
                    spec = _required_mapping(input_specs, input_name)
                    if spec.get("type") != "evidence_ref":
                        raise ValueError(
                            f"{item_field} must reference an evidence_ref operator input"
                        )
                    continue
                verify_evidence_references(item, field=item_field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                verify_evidence_references(item, field=f"{field}[{index}]")

    verify_evidence_references(
        {key: value for key, value in plan.items() if key != "operator_inputs"},
        field="plan",
    )
    factory = _required_mapping(plan, "factory")
    if (
        set(factory)
        != {
            "name",
            "kind",
            "description",
            "budget_policy",
            "cap_policy",
            "publication_policy",
            "authorization_policy",
            "metadata",
        }
        or factory.get("name") != "synth-rsi"
        or factory.get("kind") != "internal"
        or _required_mapping(factory, "metadata")
        != {
            "definition_id": "rsi_synth_on_synth.v1",
            "metric_id": "factorybench_score",
            "delivery_posture": "draft_pr_only",
            "human_merge_required": True,
        }
    ):
        raise ValueError("RSI Factory identity or safety metadata changed")
    factory_budget = _required_mapping(factory, "budget_policy")
    efforts = plan.get("efforts")
    if not isinstance(efforts, list) or len(efforts) != 1:
        raise ValueError("RSI standup requires exactly one effort")
    effort = _mapping(efforts[0], field="efforts[0]")
    effort_budget = _required_mapping(effort, "budget_policy")
    project = _required_mapping(plan, "project")
    launch_profile = _required_mapping(project, "default_launch_profile")
    launch_limit = _required_mapping(launch_profile, "limit")
    required_bindings = (
        (
            factory_budget.get("limit"),
            "factory_monthly_limit_usd",
            "factory.budget_policy.limit",
        ),
        (
            factory_budget.get("ordinary_run_limit_usd"),
            "ordinary_run_limit_usd",
            "factory.budget_policy.ordinary_run_limit_usd",
        ),
        (
            factory_budget.get("ordinary_run_target_usd"),
            "ordinary_run_target_usd",
            "factory.budget_policy.ordinary_run_target_usd",
        ),
        (
            effort_budget.get("limit"),
            "factory_monthly_limit_usd",
            "efforts[0].budget_policy.limit",
        ),
        (
            effort_budget.get("ordinary_run_limit_usd"),
            "ordinary_run_limit_usd",
            "efforts[0].budget_policy.ordinary_run_limit_usd",
        ),
        (
            effort_budget.get("ordinary_run_target_usd"),
            "ordinary_run_target_usd",
            "efforts[0].budget_policy.ordinary_run_target_usd",
        ),
        (
            launch_limit.get("max_spend_usd"),
            "ordinary_run_limit_usd",
            "project.default_launch_profile.limit.max_spend_usd",
        ),
    )
    for actual, input_name, field in required_bindings:
        if actual != {"$operator_input": input_name}:
            raise ValueError(f"{field} must remain bound to operator input {input_name!r}")


def _coerce_plan_input(name: str, value: object, spec: Mapping[str, Any]) -> object:
    input_type = _required_string(spec, "type")
    if input_type in {"string", "evidence_ref", "rfc3339"}:
        if not isinstance(value, str):
            raise ValueError(f"plan input {name!r} must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"plan input {name!r} must be a non-empty string")
        evidence_binding: dict[str, Any] | None = None
        if input_type == "evidence_ref":
            try:
                evidence_binding = _resolve_evidence_binding(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"plan input {name!r} is not a verified evidence packet: {exc}"
                ) from exc
        if input_type == "rfc3339":
            try:
                timestamp = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"plan input {name!r} must be RFC 3339") from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"plan input {name!r} must include a UTC offset")
        pattern = _optional_string(spec.get("pattern"))
        if pattern is not None and re.fullmatch(pattern, normalized) is None:
            raise ValueError(f"plan input {name!r} does not match {pattern!r}")
        allowed = spec.get("enum")
        if allowed is not None and (not isinstance(allowed, list) or normalized not in allowed):
            raise ValueError(f"plan input {name!r} must be one of {allowed!r}")
        return evidence_binding if evidence_binding is not None else normalized
    if input_type in {"number", "integer", "money_usd"}:
        if isinstance(value, bool):
            raise ValueError(f"plan input {name!r} must be a number")
        try:
            normalized_number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"plan input {name!r} must be a number") from exc
        if not normalized_number.is_finite():
            raise ValueError(f"plan input {name!r} must be finite")
        exclusive_minimum = spec.get("exclusive_minimum")
        if exclusive_minimum is not None and normalized_number <= Decimal(str(exclusive_minimum)):
            raise ValueError(f"plan input {name!r} must be greater than {exclusive_minimum}")
        maximum = spec.get("maximum")
        if maximum is not None and normalized_number > Decimal(str(maximum)):
            raise ValueError(f"plan input {name!r} must not exceed {maximum}")
        if input_type == "integer":
            integral = normalized_number.to_integral_value()
            if normalized_number != integral:
                raise ValueError(f"plan input {name!r} must be an integer")
            return int(integral)
        if input_type == "money_usd":
            normalized_exponent = cast(
                int,
                normalized_number.normalize().as_tuple().exponent,
            )
            if normalized_exponent < -2:
                raise ValueError(f"plan input {name!r} must use whole cents")
        return format(normalized_number, "f")
    raise ValueError(f"plan input {name!r} has unsupported type {input_type!r}")


def render_factory_standup_plan(
    plan: Mapping[str, Any],
    *,
    inputs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Resolve declared operator inputs, rejecting missing values and defaults."""

    raw_specs = plan.get("operator_inputs")
    supplied = dict(inputs or {})
    if raw_specs is None:
        if plan.get("schema_version") == _OPERATOR_INPUT_PLAN_SCHEMA_VERSION:
            raise ValueError("RSI standup plans must retain their operator_inputs contract")
        if supplied:
            raise ValueError("plan does not declare operator_inputs")
        return dict(plan)
    if plan.get("schema_version") != _OPERATOR_INPUT_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"operator-input plan schema_version must equal {_OPERATOR_INPUT_PLAN_SCHEMA_VERSION!r}"
        )
    _verify_rsi_operator_bindings(plan)
    specs = _mapping(raw_specs, field="operator_inputs")
    unknown = sorted(set(supplied) - set(specs))
    if unknown:
        raise ValueError(f"unknown plan inputs: {', '.join(unknown)}")
    missing = sorted(set(specs) - set(supplied))
    if missing:
        raise ValueError(f"missing required plan inputs: {', '.join(missing)}")

    resolved: dict[str, object] = {}
    for name, raw_spec in specs.items():
        spec = _mapping(raw_spec, field=f"operator_inputs.{name}")
        if "default" in spec:
            raise ValueError(f"operator_inputs.{name} must not declare a default")
        resolved[name] = _coerce_plan_input(name, supplied[name], spec)
    for name, raw_spec in specs.items():
        spec = _mapping(raw_spec, field=f"operator_inputs.{name}")
        maximum_input = _optional_string(spec.get("maximum_input"))
        if maximum_input is None:
            continue
        if maximum_input not in resolved:
            raise ValueError(
                f"operator_inputs.{name}.maximum_input references unknown input {maximum_input!r}"
            )
        if Decimal(str(resolved[name])) > Decimal(str(resolved[maximum_input])):
            raise ValueError(f"plan input {name!r} must not exceed {maximum_input!r}")

    used_inputs: set[str] = set()

    def replace(value: object, *, field: str) -> object:
        if isinstance(value, Mapping):
            if "$operator_input" in value:
                if set(value) != {"$operator_input"}:
                    raise ValueError(
                        f"{field} operator-input reference must contain only $operator_input"
                    )
                input_name = _required_string(value, "$operator_input")
                if input_name not in resolved:
                    raise ValueError(f"{field} references unknown plan input {input_name!r}")
                used_inputs.add(input_name)
                return resolved[input_name]
            return {str(key): replace(item, field=f"{field}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
        return value

    rendered = replace(
        {key: value for key, value in plan.items() if key != "operator_inputs"},
        field="plan",
    )
    if not isinstance(rendered, dict):
        raise ValueError("rendered plan must be a JSON object")
    unused_inputs = sorted(set(resolved) - used_inputs)
    if unused_inputs:
        raise ValueError("RSI plan declares unused operator inputs: " + ", ".join(unused_inputs))
    return rendered


def _verify_rsi_activation_gate(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != _OPERATOR_INPUT_PLAN_SCHEMA_VERSION:
        return
    if "wake_due" in plan:
        raise ValueError("RSI plan cannot embed a wake_due launch override")
    efforts = plan.get("efforts")
    if not isinstance(efforts, list) or len(efforts) != 1:
        raise ValueError("RSI standup requires exactly one effort")
    effort = _mapping(efforts[0], field="efforts[0]")
    if set(effort) != {
        "name",
        "effort_type",
        "status",
        "hypothesis_or_topic",
        "recurrence_policy",
        "next_wake_at",
        "budget_policy",
        "publication_policy",
        "authorization_policy",
        "actor_notes",
        "metadata",
    }:
        raise ValueError("RSI effort surface changed or attempted a project override")
    if effort.get("effort_type") != "research" or effort.get("status") != "active":
        raise ValueError("RSI effort must start as active research")
    metadata = _mapping(effort.get("metadata"), field="efforts[0].metadata")
    dependencies = metadata.get("activation_dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("RSI standup requires activation_dependencies")
    dependency_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_dependency in enumerate(dependencies):
        dependency = _mapping(
            raw_dependency,
            field=f"efforts[0].metadata.activation_dependencies[{index}]",
        )
        dependency_id = _required_string(dependency, "id")
        if dependency_id in dependency_by_id:
            raise ValueError(f"duplicate activation dependency {dependency_id!r}")
        if dependency.get("required_before") != "standup":
            raise ValueError(
                f"activation dependency {dependency_id!r} must be required_before standup"
            )
        dependency_by_id[dependency_id] = dependency

    required_ids = {
        "backend-340-cloud-slot",
        "hosted-coordinator",
        "github-delivery-credential",
        "m2-terminal-and-matrix",
        "backend-delivery-join",
        "decision-task-synthesis",
        "factorybench-synth-repository-adapter",
    }
    if set(dependency_by_id) != required_ids:
        raise ValueError(
            "RSI activation dependencies must exactly match: " + ", ".join(sorted(required_ids))
        )

    for dependency_id, dependency in dependency_by_id.items():
        evidence_values = [
            (key, value)
            for key, value in dependency.items()
            if key.endswith("evidence_ref") or key.endswith("receipt_ref")
        ]
        if not evidence_values:
            raise ValueError(
                f"activation dependency {dependency_id!r} has no durable evidence reference"
            )
        for _key, value in evidence_values:
            try:
                evidence_binding = _load_evidence_binding(value)
            except ValueError as exc:
                raise ValueError(
                    f"activation dependency {dependency_id!r} has invalid evidence: {exc}"
                ) from exc
            if (
                dependency_id != "github-delivery-credential"
                and evidence_binding.get("waiver_expires_on") is not None
            ):
                raise ValueError(
                    f"activation dependency {dependency_id!r} cannot use waiver evidence"
                )

    m2_dependency = dependency_by_id["m2-terminal-and-matrix"]
    if re.fullmatch(r"[0-9a-f]{40}", str(m2_dependency.get("contract_sha") or "")) is None:
        raise ValueError("M2 activation evidence requires an exact 40-character contract SHA")

    coordinator_mode = str(dependency_by_id["hosted-coordinator"].get("mode") or "")
    if coordinator_mode not in {"railway", "supervised_local"}:
        raise ValueError("coordinator mode must be railway or supervised_local")

    credential_dependency = dependency_by_id["github-delivery-credential"]
    credential_mode = str(credential_dependency.get("mode") or "")
    credential_evidence = _required_evidence_binding(
        credential_dependency,
        "evidence_ref",
    )
    waiver_expiry = credential_evidence.get("waiver_expires_on")
    if credential_mode == "dated_waiver":
        if not isinstance(waiver_expiry, str):
            raise ValueError("dated waiver evidence must encode its YYYY-MM-DD expiry")
        try:
            expiry = datetime.fromisoformat(waiver_expiry).date()
        except ValueError as exc:
            raise ValueError("dated waiver evidence has an invalid expiry date") from exc
        if expiry <= datetime.now().astimezone().date():
            raise ValueError("dated waiver evidence is expired")
    elif credential_mode == "pat":
        if waiver_expiry is not None:
            raise ValueError("PAT mode cannot use waiver evidence")
    else:
        raise ValueError("GitHub delivery credential mode must be pat or dated_waiver")

    cloud_dependency = dependency_by_id["backend-340-cloud-slot"]
    _verify_evidence_packet(
        _required_evidence_binding(cloud_dependency, "evidence_ref"),
        subject_id="backend-340-cloud-slot",
        authority="backend",
        expected_claims={
            "satisfied": True,
            "active": True,
            "cloud_deployment_id": _required_string(
                cloud_dependency,
                "cloud_deployment_id",
            ),
            "pool_id": _required_string(cloud_dependency, "pool_id"),
        },
        require_current=True,
        max_validity_seconds=3600,
    )
    _verify_evidence_packet(
        _required_evidence_binding(
            dependency_by_id["hosted-coordinator"],
            "evidence_ref",
        ),
        subject_id="hosted-coordinator",
        authority="internal-code-factory",
        expected_claims={
            "satisfied": True,
            "healthy": True,
            "mode": coordinator_mode,
        },
        require_current=True,
        max_validity_seconds=3600,
    )
    _verify_evidence_packet(
        credential_evidence,
        subject_id="github-delivery-credential",
        authority="internal-code-factory",
        expected_claims={
            "satisfied": True,
            "mode": credential_mode,
            ("authorized" if credential_mode == "pat" else "waiver_approved"): True,
        },
        require_current=True,
        max_validity_seconds=3600,
    )
    _verify_evidence_packet(
        _required_evidence_binding(m2_dependency, "terminal_receipt_ref"),
        subject_id="m2-terminal-receipt",
        authority="internal-code-factory",
        expected_claims={
            "satisfied": True,
            "passed": True,
            "contract_sha": str(m2_dependency["contract_sha"]),
        },
    )
    _verify_evidence_packet(
        _required_evidence_binding(m2_dependency, "combined_matrix_evidence_ref"),
        subject_id="m2-combined-matrix",
        authority="internal-code-factory",
        expected_claims={
            "satisfied": True,
            "passed": True,
            "contract_sha": str(m2_dependency["contract_sha"]),
        },
    )
    _verify_evidence_packet(
        _required_evidence_binding(
            dependency_by_id["backend-delivery-join"],
            "evidence_ref",
        ),
        subject_id="backend-delivery-join",
        authority="backend",
        expected_claims={"satisfied": True, "integrated": True},
    )
    _verify_evidence_packet(
        _required_evidence_binding(
            dependency_by_id["decision-task-synthesis"],
            "evidence_ref",
        ),
        subject_id="decision-task-synthesis",
        authority="backend",
        expected_claims={"satisfied": True, "integrated": True},
    )
    _verify_evidence_packet(
        _required_evidence_binding(
            dependency_by_id["factorybench-synth-repository-adapter"],
            "evidence_ref",
        ),
        subject_id="factorybench-synth-repository-adapter",
        authority="evals",
        expected_claims={
            "satisfied": True,
            "integrated": True,
            "execution_contract_version": "factorybench.synth_repository_delivery.v1",
        },
    )

    candidate_contract = _mapping(
        metadata.get("candidate_contract"),
        field="efforts[0].metadata.candidate_contract",
    )
    if (
        candidate_contract.get("schema_version") != "factory_candidate.v1"
        or candidate_contract.get("execution_contract_version")
        != "factorybench.synth_repository_delivery.v1"
        or candidate_contract.get("adapter_owner") != "evals"
    ):
        raise ValueError(
            "RSI candidate execution contract is not the approved FactoryBench adapter"
        )
    registration_contract = _mapping(
        candidate_contract.get("repository_registration_contract"),
        field="candidate_contract.repository_registration_contract",
    )
    if (
        registration_contract.get("authority") != "internal_code_factory"
        or registration_contract.get("owner_read_required") is not True
    ):
        raise ValueError("RSI repository registrations require Code Factory owner reads")
    required_policy = _mapping(
        registration_contract.get("required_policy"),
        field="candidate_contract.repository_registration_contract.required_policy",
    )
    expected_required_policy = {
        "default_integration_ref": "dev",
        "allowed_base_refs_must_include": "dev",
        "access_mode": "read_write",
        "branch_policy": {
            "nonempty_prefix_required": True,
            "non_force_only": True,
            "remote_delete_allowed": False,
        },
        "required_github_permissions": [
            "metadata:read",
            "contents:write",
            "pull_requests:write",
            "checks:read",
        ],
        "delivery_policy": {
            "draft_required": True,
            "real_product_merge_allowed": False,
            "fixture_ready_allowed": False,
            "fixture_merge_allowed": False,
            "fixture_close_allowed": False,
        },
    }
    if required_policy != expected_required_policy:
        raise ValueError("RSI repository registration policy must match the draft-only contract")
    allowed_repositories = candidate_contract.get("allowed_repositories")
    if not isinstance(allowed_repositories, list) or len(allowed_repositories) != 3:
        raise ValueError("RSI candidate contract requires exactly three repositories")
    repository_keys: set[str] = set()
    expected_git_remotes = {
        "backend": "https://github.com/synth-laboratories/backend.git",
        "synth-ai": "https://github.com/synth-laboratories/synth-ai.git",
        "evals": "https://github.com/synth-laboratories/evals.git",
    }
    for index, raw_repository in enumerate(allowed_repositories):
        repository = _mapping(
            raw_repository,
            field=f"candidate_contract.allowed_repositories[{index}]",
        )
        repository_key = _required_string(repository, "repository_key")
        repository_keys.add(repository_key)
        if (
            repository.get("git_remote") != expected_git_remotes.get(repository_key)
            or repository.get("integration_ref") != "dev"
            or repository.get("code_factory_registration_required") is not True
        ):
            raise ValueError("RSI repository allowlist drifted from its exact dev registration")
        repository_registration_id = _required_string(
            repository,
            "repository_registration_id",
        )
        policy_version = repository.get("policy_version")
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 1
        ):
            raise ValueError("repository policy_version must be a positive integer")
        evidence_binding = _required_evidence_binding(
            repository,
            "registration_evidence_ref",
        )
        if evidence_binding.get("waiver_expires_on") is not None:
            raise ValueError("repository registration owner reads cannot use waiver evidence")
        _verify_evidence_packet(
            evidence_binding,
            subject_id=f"repository-registration:{repository_key}",
            authority="internal-code-factory",
            expected_claims={
                "satisfied": True,
                "active": True,
                "repository_key": repository_key,
                "repository_registration_id": repository_registration_id,
                "git_remote": expected_git_remotes[repository_key],
                "policy_version": policy_version,
                "owner_read": True,
                "policy": expected_required_policy,
            },
            require_current=True,
            max_validity_seconds=3600,
        )
    if repository_keys != {"backend", "synth-ai", "evals"}:
        raise ValueError("RSI repository registrations must be backend, synth-ai, and evals")

    delivery_policy = _mapping(
        metadata.get("delivery_policy"),
        field="efforts[0].metadata.delivery_policy",
    )
    for field, expected in {
        "product_noun": "Delivery",
        "service": "code_factory",
        "draft_required": True,
        "exact_head_required": True,
        "human_merge_required": True,
        "automation_may_merge": False,
    }.items():
        if delivery_policy.get(field) != expected:
            raise ValueError(f"RSI delivery policy changed safety field {field!r}")

    factory = _required_mapping(plan, "factory")
    factory_budget = _required_mapping(factory, "budget_policy")
    effort_budget = _required_mapping(effort, "budget_policy")
    expected_budget_keys = {
        "currency",
        "period",
        "limit",
        "ordinary_run_limit_usd",
        "ordinary_run_target_usd",
    }
    if (
        set(factory_budget) != expected_budget_keys
        or factory_budget != effort_budget
        or factory_budget.get("currency") != "USD"
        or factory_budget.get("period") != "monthly"
    ):
        raise ValueError("RSI Factory and Effort must share the exact approved USD budget")
    try:
        monthly_limit = Decimal(str(factory_budget["limit"]))
        ordinary_limit = Decimal(str(factory_budget["ordinary_run_limit_usd"]))
        ordinary_target = Decimal(str(factory_budget["ordinary_run_target_usd"]))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("RSI budget values must be decimal USD amounts") from exc
    budget_values = (monthly_limit, ordinary_limit, ordinary_target)
    if any(
        not value.is_finite()
        or value <= 0
        or value > Decimal("1000000")
        or cast(int, value.normalize().as_tuple().exponent) < -2
        for value in budget_values
    ):
        raise ValueError("RSI budgets must be positive, bounded whole-cent USD amounts")
    if ordinary_target > ordinary_limit or ordinary_limit > monthly_limit:
        raise ValueError("RSI ordinary target and limit must remain within the monthly cap")
    if _required_mapping(factory, "cap_policy") != {
        "max_active_efforts": 1,
        "max_active_runs": 1,
    }:
        raise ValueError("RSI Factory cap policy must preserve one effort and one run")
    expected_publication = {
        "visibility": "private",
        "publish_reports": True,
        "publish_work_products": True,
    }
    expected_authorization = {
        "enabled": True,
        "requires_audit_trail": True,
    }
    if (
        _required_mapping(factory, "publication_policy") != expected_publication
        or _required_mapping(effort, "publication_policy") != expected_publication
        or _required_mapping(factory, "authorization_policy") != expected_authorization
        or _required_mapping(effort, "authorization_policy") != expected_authorization
    ):
        raise ValueError("RSI publication and authorization policy must remain private and audited")

    recurrence_policy = _mapping(
        effort.get("recurrence_policy"),
        field="efforts[0].recurrence_policy",
    )
    expected_recurrence_policy = {
        "enabled": True,
        "on_run_complete": True,
        "delay_seconds": 3600,
        "max_active_runs": 1,
        "research": {"enabled": True},
        "maintenance": {"enabled": True, "every_n_research_runs": 1},
    }
    if recurrence_policy != expected_recurrence_policy:
        raise ValueError("RSI effort recurrence policy must match the one-lane hourly contract")
    next_wake_at = _required_string(effort, "next_wake_at")
    first_wake = datetime.fromisoformat(next_wake_at.replace("Z", "+00:00"))
    if first_wake.tzinfo is None or first_wake.utcoffset() is None:
        raise ValueError("RSI effort next_wake_at must include a UTC offset")

    project = _mapping(plan.get("project"), field="project")
    create_project = _required_mapping(plan, "create_project")
    agent_profiles = _required_mapping(create_project, "agent_profiles")
    orchestrator_profile_id = _required_string(
        agent_profiles,
        "orchestrator_profile_id",
    )
    worker_profile_id = _required_string(
        agent_profiles,
        "default_worker_profile_id",
    )
    launch_profile = _mapping(
        project.get("default_launch_profile"),
        field="project.default_launch_profile",
    )
    launch_limit = _mapping(
        launch_profile.get("limit"),
        field="project.default_launch_profile.limit",
    )
    project_metadata = _required_mapping(project, "metadata")
    cloud_slot_dependency = _required_mapping(
        project_metadata,
        "cloud_slot_dependency",
    )
    if (
        set(project)
        != {
            "role",
            "status",
            "display_name",
            "description",
            "default_launch_profile",
            "metadata",
        }
        or project.get("role") != "canonical"
        or project.get("status") != "active"
        or set(launch_profile)
        != {
            "runbook_preset",
            "host_kind",
            "timebox_seconds",
            "limit",
            "required_capabilities",
        }
        or set(launch_limit) != {"max_spend_usd"}
        or set(create_project)
        != {
            "name",
            "timezone",
            "pool_id",
            "runtime_kind",
            "environment_kind",
            "agent_profiles",
            "execution_policy",
            "research",
            "notes",
        }
        or set(agent_profiles)
        != {
            "orchestrator_profile_id",
            "default_worker_profile_id",
            "worker_profile_ids",
        }
        or create_project.get("name") != "synth-rsi"
        or create_project.get("timezone") != "UTC"
        or agent_profiles.get("worker_profile_ids") != [worker_profile_id]
        or not orchestrator_profile_id
        or _required_mapping(create_project, "execution_policy")
        != {"task_completion_review": "reviewer", "one_operator_per_lane": True}
        or _required_mapping(create_project, "research")
        != {
            "metric_id": "factorybench_score",
            "metric_name": "FactoryBench score",
            "metric_authority": "evals",
            "evidence_policy": "sealed_held_out_benchmark_owned",
        }
        or launch_profile.get("runbook_preset") != "lite"
        or launch_profile.get("host_kind") != "daytona"
        or launch_profile.get("timebox_seconds") != 3600
        or launch_profile.get("required_capabilities") != ["inference"]
        or "agent_profile" in launch_profile
        or launch_limit.get("max_spend_usd") != effort_budget.get("ordinary_run_limit_usd")
        or create_project.get("runtime_kind") != "sandbox_agent"
        or create_project.get("environment_kind") != "daytona"
        or create_project.get("pool_id") != cloud_dependency.get("pool_id")
        or cloud_slot_dependency.get("pool_id") != cloud_dependency.get("pool_id")
        or cloud_slot_dependency.get("cloud_deployment_id")
        != cloud_dependency.get("cloud_deployment_id")
        or cloud_slot_dependency.get("evidence_ref") != cloud_dependency.get("evidence_ref")
    ):
        raise ValueError(
            "RSI Project must preserve the proved Daytona slot and bounded one-hour launch"
        )


def _project_id_from_response(payload: Mapping[str, Any]) -> str:
    project_id = _optional_string(payload.get("project_id") or payload.get("id"))
    if project_id:
        return project_id
    project = payload.get("project")
    if isinstance(project, Mapping):
        project_id = _optional_string(project.get("project_id") or project.get("id"))
        if project_id:
            return project_id
    raise ValueError("create_project response did not include project_id")


def _factory_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    factory = _required_mapping(plan, "factory")
    return {
        "name": _required_string(factory, "name"),
        "kind": str(factory.get("kind") or "customer"),
        "description": _optional_string(factory.get("description")),
        "budget_policy": _mapping(
            factory.get("budget_policy"),
            field="factory.budget_policy",
        ),
        "cap_policy": _mapping(factory.get("cap_policy"), field="factory.cap_policy"),
        "publication_policy": _mapping(
            factory.get("publication_policy"),
            field="factory.publication_policy",
        ),
        "authorization_policy": _mapping(
            factory.get("authorization_policy"),
            field="factory.authorization_policy",
        ),
        "metadata": _mapping(factory.get("metadata"), field="factory.metadata"),
    }


def _project_link_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    project = _mapping(plan.get("project"), field="project")
    return {
        "role": str(project.get("role") or "canonical"),
        "status": str(project.get("status") or "active"),
        "display_name": _optional_string(project.get("display_name")),
        "description": _optional_string(project.get("description")),
        "workspace_policy": _mapping(
            project.get("workspace_policy"),
            field="project.workspace_policy",
        ),
        "resource_bindings": _mapping(
            project.get("resource_bindings"),
            field="project.resource_bindings",
        ),
        "feed_health": _mapping(project.get("feed_health"), field="project.feed_health"),
        "default_launch_profile": _mapping(
            project.get("default_launch_profile"),
            field="project.default_launch_profile",
        ),
        "metadata": _mapping(project.get("metadata"), field="project.metadata"),
    }


def _effort_plans(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    efforts = plan.get("efforts")
    if not isinstance(efforts, list) or not efforts:
        raise ValueError("efforts must be a non-empty JSON array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(efforts):
        if not isinstance(item, Mapping):
            raise ValueError(f"efforts[{index}] must be a JSON object")
        result.append({str(item_key): item_value for item_key, item_value in item.items()})
    return result


def _effort_kwargs(effort: Mapping[str, Any], *, default_project_id: str) -> dict[str, Any]:
    return {
        "name": _required_string(effort, "name"),
        "project_id": _optional_string(effort.get("project_id")) or default_project_id,
        "hypothesis_or_topic": _optional_string(
            effort.get("hypothesis_or_topic") or effort.get("topic")
        ),
        "effort_type": str(effort.get("effort_type") or effort.get("type") or "research"),
        "status": str(effort.get("status") or "active"),
        "recurrence_policy": _mapping(
            effort.get("recurrence_policy"),
            field="effort.recurrence_policy",
        ),
        "next_wake_at": _optional_string(effort.get("next_wake_at")),
        "latest_run_id": _optional_string(effort.get("latest_run_id")),
        "latest_report_id": _optional_string(effort.get("latest_report_id")),
        "latest_work_product_id": _optional_string(effort.get("latest_work_product_id")),
        "decision_needed": bool(effort.get("decision_needed") or False),
        "decision_note": _optional_string(effort.get("decision_note")),
        "budget_policy": _mapping(
            effort.get("budget_policy"),
            field="effort.budget_policy",
        ),
        "publication_policy": _mapping(
            effort.get("publication_policy"),
            field="effort.publication_policy",
        ),
        "authorization_policy": _mapping(
            effort.get("authorization_policy"),
            field="effort.authorization_policy",
        ),
        "actor_notes": _mapping(effort.get("actor_notes"), field="effort.actor_notes"),
        "metadata": _mapping(effort.get("metadata"), field="effort.metadata"),
    }


def _wake_due_preview_kwargs(plan: Mapping[str, Any]) -> dict[str, Any]:
    wake_due = _mapping(plan.get("wake_due"), field="wake_due")
    return {
        "launch_request": _mapping(
            wake_due.get("launch_request"),
            field="wake_due.launch_request",
        )
        or None,
        "limit": int(wake_due.get("limit") or 10),
        "allow_overlap": bool(wake_due.get("allow_overlap") or False),
        "continue_on_error": bool(wake_due.get("continue_on_error", True)),
        "dry_run": True,
    }


def _confirm_wake_preview(
    *,
    client: ManagedResearchClient,
    factory_id: str,
    preview: FactoryWakeDueResult,
) -> FactoryWakeDueResult:
    if preview.factory_id != factory_id:
        raise RuntimeError("wake preview factory_id does not match the created Factory")
    if not preview.dry_run or not preview.confirmation_required:
        raise RuntimeError("wake preview is not confirmation-ready")
    if preview.preview_id is None or preview.preview_token is None:
        raise RuntimeError("wake preview omitted its preview_id or preview_token")
    contract = preview.request_contract
    if contract is None:
        raise RuntimeError("wake preview omitted its resolved request_contract")
    if contract.confirmed_preview_token is not None:
        raise RuntimeError("wake preview request_contract is not confirmation-ready")
    result = client.factories.wake_due(
        factory_id,
        launch_request=contract.launch_request,
        limit=contract.limit,
        allow_overlap=contract.allow_overlap,
        dry_run=False,
        continue_on_error=contract.continue_on_error,
        confirmed_preview_id=preview.preview_id,
        confirmed_preview_token=preview.preview_token,
    )
    if result.confirmed_preview_id != preview.preview_id or result.receipt_id is None:
        raise RuntimeError("wake receipt is not durably bound to the confirmed preview")
    return result


def _wake_result_proof(result: FactoryWakeDueResult | None) -> Any:
    payload = _jsonable(result)
    if not isinstance(payload, dict):
        return payload
    payload.pop("preview_token", None)
    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw.pop("preview_token", None)
    return payload


def execute_factory_standup(
    plan: Mapping[str, Any],
    *,
    client: ManagedResearchClient,
    plan_inputs: Mapping[str, object] | None = None,
    dry_run: bool = False,
    wake_due: bool = False,
    wake_due_launch: bool = False,
) -> dict[str, Any]:
    plan = render_factory_standup_plan(plan, inputs=plan_inputs)
    _verify_rsi_activation_gate(plan)
    factory_payload = _factory_payload(plan)
    project = _mapping(plan.get("project"), field="project")
    create_project = _mapping(plan.get("create_project"), field="create_project")
    project_id = _optional_string(project.get("project_id"))
    effort_plans = _effort_plans(plan)
    link_payload = _project_link_payload(plan)
    should_wake = wake_due or wake_due_launch or bool(plan.get("wake_due"))

    if project_id is None and not create_project:
        raise ValueError("project.project_id or create_project is required")
    if create_project:
        SmrRunnableProjectRequest.from_wire(create_project)

    if dry_run:
        return {
            "dry_run": True,
            "factory": factory_payload,
            "project_id": project_id,
            "create_project": create_project or None,
            "project_link": link_payload,
            "efforts": effort_plans,
            "wake_due": _wake_due_preview_kwargs(plan) if should_wake else None,
        }

    created_project: dict[str, Any] | None = None
    if project_id is None:
        created_project = client.create_runnable_project(create_project)
        project_id = _project_id_from_response(created_project)
    if project_id is None:
        raise RuntimeError("project_id resolution failed")

    factory = client.factories.create(factory_payload)
    link = client.factories.link_project(
        factory.factory_id,
        project_id,
        **link_payload,
    )
    efforts = [
        client.factories.create_effort(
            factory.factory_id,
            **_effort_kwargs(effort, default_project_id=project_id),
        )
        for effort in effort_plans
    ]
    wake_preview = None
    wake_result = None
    if should_wake:
        wake_preview = client.factories.wake_due(
            factory.factory_id,
            **_wake_due_preview_kwargs(plan),
        )
        if wake_due_launch and wake_preview.ready > 0 and not wake_preview.confirmation_required:
            raise RuntimeError("wake preview has ready work but is not confirmation-ready")
        wake_result = (
            _confirm_wake_preview(
                client=client,
                factory_id=factory.factory_id,
                preview=wake_preview,
            )
            if wake_due_launch and wake_preview.confirmation_required
            else wake_preview
        )

    return {
        "dry_run": False,
        "factory_id": factory.factory_id,
        "project_id": project_id,
        "effort_ids": [effort.effort_id for effort in efforts],
        "factory": _jsonable(factory),
        "created_project": _jsonable(created_project),
        "project_link": _jsonable(link),
        "efforts": _jsonable(efforts),
        "wake_due_preview": (_wake_result_proof(wake_preview) if wake_due_launch else None),
        "wake_due": _wake_result_proof(wake_result),
        "status": _jsonable(client.factories.status(factory.factory_id)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth-ai-research-factory-standup",
        description="Create, link, and seed a Research Factory from one JSON plan.",
    )
    parser.add_argument("--api-key", default=None, help="defaults to $SYNTH_API_KEY")
    parser.add_argument("--backend", default=None, help="defaults to $SYNTH_BACKEND_URL")
    parser.add_argument(
        "--plan",
        required=True,
        help="JSON object, @path, or builtin:rsi-synth-on-synth",
    )
    parser.add_argument(
        "--plan-input",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="required typed input declared by the plan; repeat for each input",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wake-due",
        action="store_true",
        help="call wake-due after effort creation; defaults to wake dry-run",
    )
    parser.add_argument(
        "--wake-due-launch",
        action="store_true",
        help="preview wake-due, then confirm that exact preview and launch its runs",
    )
    parser.add_argument("--output", default=None, help="optional proof JSON path")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = _json_object(args.plan, field="plan")
        plan_inputs = _parse_plan_inputs(args.plan_input)
        client = ManagedResearchClient(api_key=args.api_key, backend_base=args.backend)
        proof = execute_factory_standup(
            plan,
            client=client,
            plan_inputs=plan_inputs,
            dry_run=args.dry_run,
            wake_due=args.wake_due,
            wake_due_launch=args.wake_due_launch,
        )
        rendered = json.dumps(proof, indent=args.indent, sort_keys=True)
        if args.output:
            Path(args.output).expanduser().write_text(f"{rendered}\n")
        print(rendered)
        return 0
    except Exception as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
