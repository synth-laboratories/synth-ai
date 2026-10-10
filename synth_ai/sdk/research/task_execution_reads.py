"""Scoped Orchestra Task facts, without reconstructing Sublinear Task status.

# See: testing/specifications/sdk/owner_reads.md
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import object_value
from synth_ai.sdk.research.execution_reads import ExecutionReadRetention

if TYPE_CHECKING:
    from synth_ai.sdk.research.owner_reads import OwnerReadScope

SCHEMA = "orchestra.task-execution-view.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ATTEMPT_FIELDS = {
    "actor_id", "attempt_id", "actor_class", "slot_key", "allocation", "inputs",
    "allocated_at_ms", "started_at_ms", "finished_at_ms", "session", "liveness",
    "completion", "quiescence", "outputs",
}


def task_identity(value: str) -> str:
    """Validate an owner entity before selecting a resource path."""
    identity = UUID(value)
    if identity.int == 0 or str(identity) != value:
        raise ValueError("Task execution read requires canonical nonzero UUID")
    return value


def _object(value: JsonValue, fields: set[str]) -> JsonObject:
    result = object_value(value, operation_id="Task execution read")
    if set(result) != fields:
        raise ValueError("Task execution owner fields drifted")
    return result


def _integer(value: JsonValue, *, maximum: int = 2**64 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Task execution owner integer outside wire bounds")
    return value


def _reference(value: JsonValue) -> JsonObject:
    result = _object(value, {"owner", "schema", "reference", "sha256"})
    if not all(isinstance(part, str) and part for part in result.values()):
        raise ValueError("Task execution evidence reference invalid")
    digest = result["sha256"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("Task execution evidence digest invalid")
    return result


def _attempt(value: JsonValue, scope: JsonObject, configuration: str) -> JsonObject:
    attempt = _object(value, _ATTEMPT_FIELDS)
    for field in ("actor_id", "attempt_id"):
        identity = attempt[field]
        if not isinstance(identity, str):
            raise ValueError("Task execution attempt identity invalid")
        task_identity(identity)
    _integer(attempt["allocated_at_ms"], maximum=2**63 - 1)
    if attempt["started_at_ms"] is not None or attempt["finished_at_ms"] is not None:
        raise ValueError("Task execution v1 has no authoritative attempt clock")
    inputs = _object(attempt["inputs"], {"scope", "attempt_id", "configuration_digest",
                     "materialized_configuration", "actor_catalogue", "profile",
                     "output_registry", "graph", "task", "assignment",
                     "requirement_revision", "dynamic_sources"})
    if (inputs.get("scope") != scope or inputs.get("attempt_id") != attempt["attempt_id"]
            or inputs.get("configuration_digest") != configuration):
        raise ValueError("Task execution attempt leaves its binding")
    allocation = _reference(attempt["allocation"])
    if (allocation["owner"] != "orchestra"
            or allocation["schema"] != "orchestra.prelaunch-attempt.v1"
            or allocation["reference"] != attempt["attempt_id"]):
        raise ValueError("Task execution allocation differs from attempt")
    session = attempt["session"]
    if session is not None:
        session = _object(session, {"session_id", "carrier", "original_session", "original_admission"})
        identity = session["session_id"]
        if not isinstance(identity, str):
            raise ValueError("Task execution session identity invalid")
        task_identity(identity)
        for field in ("carrier", "original_session", "original_admission"):
            _reference(session[field])
    lease = attempt["liveness"]
    if lease is not None:
        lease = _object(lease, {"scope", "attempt_id", "actor_id", "attempt_generation",
                              "controller_generation", "revision", "observed_at_ms",
                              "lease_until_ms", "execution_deadline_ms", "source", "expired"})
        if (lease["scope"] != scope or lease["actor_id"] != attempt["actor_id"]
                or lease["attempt_id"] != attempt["attempt_id"] or type(lease["expired"]) is not bool):
            raise ValueError("Task execution lease leaves its attempt")
        for field in ("attempt_generation", "controller_generation", "revision"):
            if _integer(lease[field]) == 0:
                raise ValueError("Task execution lease fence invalid")
        for field in ("observed_at_ms", "lease_until_ms", "execution_deadline_ms"):
            _integer(lease[field], maximum=2**63 - 1)
        _reference(lease["source"])
    completion = attempt["completion"]
    if completion is not None:
        completion = _object(completion, {"receipt", "execution_state", "original_completion", "session"})
        if (not isinstance(completion["execution_state"], str)
                or completion["execution_state"] not in {"succeeded", "failed", "cancelled"}
                or session is None or completion["session"] != session["carrier"]):
            raise ValueError("Task execution completion leaves its session")
        for field in ("receipt", "original_completion", "session"):
            _reference(completion[field])
    if attempt["quiescence"] is not None:
        _reference(attempt["quiescence"])
    outputs = attempt["outputs"]
    if not isinstance(outputs, list):
        raise ValueError("Task execution output inventory invalid")
    for value in outputs:
        output = _object(value, {"output_id", "requirement_id", "registry", "inputs",
                                 "kind", "subtype", "operation_id", "sha256",
                                 "published", "validation", "acceptance"})
        if output.get("inputs") != inputs:
            raise ValueError("Task execution output leaves its attempt")
    return attempt


@dataclass(frozen=True, slots=True)
class TaskExecutionRead:
    """One machine position and exact Task view.

    # See: testing/specifications/sdk/owner_reads.md
    """

    scope: OwnerReadScope
    task_id: str
    machine: JsonObject
    sequence: int
    controller_generation: int
    configuration_digest: str
    raw_json: str
    sha256: str
    view: JsonObject
    attempts: tuple[JsonObject, ...]
    retention: ExecutionReadRetention

    @classmethod
    def from_wire(cls, value: JsonValue, *, scope: OwnerReadScope, task_id: str) -> TaskExecutionRead:
        task_identity(task_id)
        body = _object(value, {"scope", "task_id", "machine", "position",
                               "configuration_digest", "view", "retention"})
        wire_scope: JsonObject = {"organization_id": scope.organization_id,
                                 "project_id": scope.project_id, "stream_id": scope.run_id}
        configuration = body["configuration_digest"]
        if (body["scope"] != wire_scope or body["task_id"] != task_id
                or not isinstance(configuration, str) or not _DIGEST.fullmatch(configuration)):
            raise ValueError("Task execution owner scope or configuration differs")
        machine = _object(body["machine"], {"name", "reducer_version", "configuration_digest"})
        if (not isinstance(machine["name"], str) or not machine["name"]
                or _integer(machine["reducer_version"], maximum=2**32 - 1) == 0):
            raise ValueError("Task execution machine identity invalid")
        machine_digest = machine["configuration_digest"]
        if not isinstance(machine_digest, str) or not _DIGEST.fullmatch(machine_digest):
            raise ValueError("Task execution machine digest invalid")
        position = _object(body["position"], {"sequence", "controller_generation"})
        sequence = _integer(position["sequence"])
        generation = _integer(position["controller_generation"])
        if generation == 0:
            raise ValueError("Task execution controller generation invalid")
        retention = ExecutionReadRetention.from_wire(body["retention"])
        if not retention.floor_sequence <= sequence <= retention.head_sequence:
            raise ValueError("Task execution position outside retained range")
        payload = _object(body["view"], {"kind", "schema", "sha256", "canonical_json"})
        raw = payload["canonical_json"]
        if payload["kind"] != "inline" or payload["schema"] != SCHEMA or not isinstance(raw, str):
            raise ValueError("Task execution view payload invalid")
        encoded = raw.encode("utf-8")
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if len(encoded) > 131072 or payload["sha256"] != digest:
            raise ValueError("Task execution view original bytes differ")
        view = _object(cast(JsonValue, json.loads(raw)), {"schema_version", "scope", "task_id",
                        "configuration_digest", "graph", "task", "assignment",
                        "planned_actor_class", "planned_profile", "attempts"})
        if (view["schema_version"] != SCHEMA or view["scope"] != wire_scope
                or view["task_id"] != task_id or view["configuration_digest"] != configuration):
            raise ValueError("Task execution view binding differs")
        for field in ("graph", "task", "assignment", "planned_profile"):
            _reference(view[field])
        rows = view["attempts"]
        if not isinstance(rows, list):
            raise ValueError("Task execution attempt inventory invalid")
        attempts = tuple(_attempt(row, wire_scope, configuration) for row in rows)
        identities = [cast(str, attempt["attempt_id"]) for attempt in attempts]
        if identities != sorted(set(identities)):
            raise ValueError("Task execution attempts are repeated or unordered")
        return cls(scope, task_id, machine, sequence, generation, configuration,
                   raw, digest, view, attempts, retention)
