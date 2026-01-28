"""OpenResponses-aligned event schema definitions (Rust-backed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .types import CandidateStatus, EventLevel, JobEventType, JobStatus

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.base.") from exc


def _load_base_schemas_fallback() -> Dict[str, Dict[str, Any]]:
    asset = (
        Path(__file__).resolve().parents[5] / "synth_ai_core" / "assets" / "event_base_schemas.json"
    )
    if asset.exists():
        return json.loads(asset.read_text())
    return {}


if synth_ai_py is None or not hasattr(synth_ai_py, "orchestration_base_event_schemas"):
    _BASE_SCHEMAS = _load_base_schemas_fallback()
    BaseJobEvent = object  # type: ignore[misc,assignment]
    JobEvent = object  # type: ignore[misc,assignment]
    CandidateEvent = object  # type: ignore[misc,assignment]
else:
    _BASE_SCHEMAS = synth_ai_py.orchestration_base_event_schemas()
    # Event classes (Rust-backed)
    BaseJobEvent = synth_ai_py.BaseJobEvent
    JobEvent = synth_ai_py.JobEvent
    CandidateEvent = synth_ai_py.CandidateEvent

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
