"""OpenResponses-aligned event schema definitions (Rust-backed)."""

from __future__ import annotations

from typing import Any, Dict

from .types import CandidateStatus, EventLevel, JobEventType, JobStatus

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.base.") from exc


def _require_rust():
    if synth_ai_py is None or not hasattr(synth_ai_py, "orchestration_base_event_schemas"):
        raise RuntimeError("Rust core base event schemas required; synth_ai_py is unavailable.")
    return synth_ai_py


_rust = _require_rust()
_BASE_SCHEMAS = _rust.orchestration_base_event_schemas()

# Event classes (Rust-backed)
BaseJobEvent = _rust.BaseJobEvent
JobEvent = _rust.JobEvent
CandidateEvent = _rust.CandidateEvent

# JSON Schemas
BASE_JOB_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get("BASE_JOB_EVENT_SCHEMA", {})
JOB_STARTED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get("JOB_STARTED_EVENT_SCHEMA", {})
JOB_COMPLETED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get("JOB_COMPLETED_EVENT_SCHEMA", {})
JOB_FAILED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get("JOB_FAILED_EVENT_SCHEMA", {})
CANDIDATE_ADDED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get("CANDIDATE_ADDED_EVENT_SCHEMA", {})
CANDIDATE_EVALUATED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get(
    "CANDIDATE_EVALUATED_EVENT_SCHEMA", {}
)
CANDIDATE_COMPLETED_EVENT_SCHEMA: Dict[str, Any] = _BASE_SCHEMAS.get(
    "CANDIDATE_COMPLETED_EVENT_SCHEMA", {}
)
BASE_EVENT_SCHEMAS: Dict[str, Dict[str, Any]] = _BASE_SCHEMAS.get("BASE_EVENT_SCHEMAS", {})

__all__ = [
    "EventLevel",
    "JobEventType",
    "CandidateStatus",
    "JobStatus",
    "BaseJobEvent",
    "JobEvent",
    "CandidateEvent",
    "BASE_JOB_EVENT_SCHEMA",
    "JOB_STARTED_EVENT_SCHEMA",
    "JOB_COMPLETED_EVENT_SCHEMA",
    "JOB_FAILED_EVENT_SCHEMA",
    "CANDIDATE_ADDED_EVENT_SCHEMA",
    "CANDIDATE_EVALUATED_EVENT_SCHEMA",
    "CANDIDATE_COMPLETED_EVENT_SCHEMA",
    "BASE_EVENT_SCHEMAS",
]
