from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from synth_ai.core.levers import MiproLeverSummary
from synth_ai.core.sensors import SensorFrameSummary


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _from_data_or_metadata(data: Dict[str, Any], key: str) -> Optional[Any]:
    value = data.get(key)
    if value is not None:
        return value
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get(key)
        if nested is not None:
            return nested
    job_metadata = data.get("job_metadata")
    if isinstance(job_metadata, dict):
        nested = job_metadata.get(key)
        if nested is not None:
            return nested
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_candidate_reward(candidate: Dict[str, Any]) -> Optional[float]:
    for key in ("mean_reward", "reward", "avg_reward", "train_reward", "validation_reward"):
        parsed = _coerce_float(candidate.get(key))
        if parsed is not None:
            return parsed
    objectives = candidate.get("instance_objectives")
    if isinstance(objectives, list):
        values = [
            parsed
            for parsed in (
                _coerce_float(obj.get("reward")) for obj in objectives if isinstance(obj, dict)
            )
            if parsed is not None
        ]
        if values:
            return sum(values) / len(values)
    for value in candidate.values():
        if not isinstance(value, dict):
            continue
        parsed = _coerce_float(value.get("reward"))
        if parsed is not None:
            return parsed
    return None


def _extract_candidate_objective(candidate: Dict[str, Any]) -> Optional[float]:
    objective = candidate.get("objective")
    if isinstance(objective, dict):
        reward = _coerce_float(objective.get("reward"))
        if reward is not None:
            return reward
    return _coerce_float(candidate.get("objective"))


def _extract_best_reward_value(data: Dict[str, Any], include_train: bool = True) -> Optional[float]:
    if include_train:
        reward_keys = ("best_reward", "best_score", "best_train_reward", "best_train_score")
    else:
        reward_keys = ("best_reward", "best_score")
    parsed = _coerce_float(_first_present(data, reward_keys))
    if parsed is not None:
        return parsed

    for key in ("metadata", "job_metadata"):
        metadata = data.get(key)
        if not isinstance(metadata, dict):
            continue
        parsed = _coerce_float(_first_present(metadata, reward_keys))
        if parsed is not None:
            return parsed

    candidate_collections: list[Any] = [data.get("candidates"), data.get("frontier"), data.get("archive")]
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        candidate_collections.append(metadata.get("candidates"))
    job_metadata = data.get("job_metadata")
    if isinstance(job_metadata, dict):
        candidate_collections.append(job_metadata.get("candidates"))

    reward_values: list[float] = []
    for collection in candidate_collections:
        if not isinstance(collection, list):
            continue
        for candidate in collection:
            if not isinstance(candidate, dict):
                continue
            candidate_reward = _extract_candidate_reward(candidate)
            if candidate_reward is not None:
                reward_values.append(candidate_reward)
    if reward_values:
        return max(reward_values)
    return None


def _parse_lever_versions(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    versions: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            versions[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return versions


def _normalize_status(status: str) -> str:
    return status.strip().lower().replace(" ", "_")


def _extract_system_prompt_from_dict(prompt: Dict[str, Any]) -> Optional[str]:
    """Extract system prompt from a structured prompt dict."""
    # Try messages format (most common for GEPA)
    messages = prompt.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return msg.get("pattern") or msg.get("content")

    # Try sections format (canonical)
    sections = prompt.get("sections", [])
    for sec in sections:
        if isinstance(sec, dict) and sec.get("role") == "system":
            return sec.get("content") or sec.get("pattern")

    # Try text_replacements format (transformation)
    text_replacements = prompt.get("text_replacements", [])
    for tr in text_replacements:
        if isinstance(tr, dict) and tr.get("apply_to_role") == "system":
            return tr.get("new_text")

    return None


def _extract_candidate_content_from_dict(candidate: Dict[str, Any]) -> Optional[str]:
    """Extract generic candidate artifact text from structured candidate payloads."""
    for key in (
        "candidate_content",
        "candidate_code",
        "solver_code",
        "program_text",
        "code",
        "instruction_text",
        "instruction",
        "prompt_text",
        "text",
        "system_prompt",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    program_candidate = candidate.get("program_candidate")
    if isinstance(program_candidate, dict):
        nested = _extract_candidate_content_from_dict(program_candidate)
        if nested:
            return nested

    stage_payloads = candidate.get("stage_payloads")
    if isinstance(stage_payloads, dict):
        for payload in stage_payloads.values():
            if not isinstance(payload, dict):
                continue
            nested = _extract_candidate_content_from_dict(payload)
            if nested:
                return nested

    messages = candidate.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content") or message.get("pattern") or message.get("text")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            return "\n\n".join(parts)

    nested_candidate = candidate.get("candidate")
    if isinstance(nested_candidate, dict):
        nested = _extract_candidate_content_from_dict(nested_candidate)
        if nested:
            return nested

    return None


def _extract_candidate_content(
    best_candidate: Optional[str | Dict[str, Any]],
    raw: Dict[str, Any],
) -> Optional[str]:
    """Extract generic best-candidate content for prompt and non-prompt artifacts."""
    if isinstance(best_candidate, str) and best_candidate.strip():
        return best_candidate.strip()
    if isinstance(best_candidate, dict):
        result = _extract_candidate_content_from_dict(best_candidate)
        if result:
            return result

    for key in ("best_candidate", "best_prompt"):
        raw_best = raw.get(key)
        if isinstance(raw_best, str) and raw_best.strip():
            return raw_best.strip()
        if isinstance(raw_best, dict):
            result = _extract_candidate_content_from_dict(raw_best)
            if result:
                return result

    for key in ("optimized_candidates", "frontier", "candidates", "archive"):
        candidates = raw.get(key)
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            result = _extract_candidate_content_from_dict(candidate)
            if result:
                return result
    return None


def _extract_system_prompt(
    best_candidate: Optional[str | Dict[str, Any]],
    raw: Dict[str, Any],
) -> Optional[str]:
    """Extract system prompt from result data, trying multiple sources."""
    # Direct string
    if isinstance(best_candidate, str) and best_candidate:
        return best_candidate

    # Structured dict
    if isinstance(best_candidate, dict):
        result = _extract_system_prompt_from_dict(best_candidate)
        if result:
            return result

    # Try raw response fields
    raw_best = raw.get("best_candidate")
    if raw_best is None:
        raw_best = raw.get("best_prompt")
    if isinstance(raw_best, str) and raw_best:
        return raw_best
    if isinstance(raw_best, dict):
        result = _extract_system_prompt_from_dict(raw_best)
        if result:
            return result

    # Try candidates in raw data
    for key in ("optimized_candidates", "frontier", "candidates", "archive"):
        candidates = raw.get(key, [])
        if isinstance(candidates, list):
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                pattern = cand.get("pattern") or cand.get("object", {}).get("pattern")
                if isinstance(pattern, dict):
                    result = _extract_system_prompt_from_dict(pattern)
                    if result:
                        return result

    # Strict for non-prompt artifacts that still expose candidate text.
    return _extract_candidate_content(best_candidate, raw)


def _normalize_error_message(error_text: Any) -> Optional[str]:
    text = str(error_text or "").strip()
    if not text:
        return None
    lower = text.lower()
    is_container_health_failure = (
        (
            "container health check failed for" in lower
            or "container health check failed:" in lower
            or "health check failed for" in lower
        )
        and "/health" in lower
    )
    if not is_container_health_failure:
        return text
    if "skip_health_check=true only skips sdk pre-submit checks" in lower:
        return text
    return (
        f"{text} "
        "Hint: This health check runs in backend workers. "
        "skip_health_check=True only skips SDK pre-submit checks. "
        "Ensure container_url is reachable from backend workers and that the eval server is running."
    )


class PolicyJobStatus(str, Enum):
    """Status of a policy optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> PolicyJobStatus:
        normalized = _normalize_status(status)
        if normalized in ("success", "succeeded", "completed", "complete"):
            return cls.SUCCEEDED
        if normalized in ("cancelled", "canceled", "cancel"):
            return cls.CANCELLED
        if normalized in ("failed", "failure", "error"):
            return cls.FAILED
        if normalized in ("running", "in_progress"):
            return cls.RUNNING
        if normalized == "paused":
            return cls.PAUSED
        if normalized == "queued":
            return cls.QUEUED
        if normalized == "pending":
            return cls.PENDING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (
            PolicyJobStatus.SUCCEEDED,
            PolicyJobStatus.FAILED,
            PolicyJobStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        return self == PolicyJobStatus.SUCCEEDED


class GraphJobStatus(str, Enum):
    """Status of a graph optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> GraphJobStatus:
        normalized = _normalize_status(status)
        if normalized in ("success", "succeeded"):
            return cls.SUCCEEDED
        if normalized in ("completed", "complete"):
            return cls.COMPLETED
        if normalized in ("cancelled", "canceled", "cancel"):
            return cls.CANCELLED
        if normalized in ("failed", "failure", "error"):
            return cls.FAILED
        if normalized in ("running", "in_progress"):
            return cls.RUNNING
        if normalized == "queued":
            return cls.QUEUED
        if normalized == "pending":
            return cls.PENDING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (
            GraphJobStatus.COMPLETED,
            GraphJobStatus.SUCCEEDED,
            GraphJobStatus.FAILED,
            GraphJobStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        return self in (GraphJobStatus.COMPLETED, GraphJobStatus.SUCCEEDED)


@dataclass
class PolicyCandidate:
    """Canonical typed candidate artifact model."""

    candidate_id: str
    candidate_type: Optional[str] = None
    artifact_kind: str = "unknown"
    artifact_payload: Optional[Any] = None
    artifact_preview: Optional[str] = None
    candidate_content: Optional[str] = None
    status: Optional[str] = None
    optimization_mode: Optional[str] = None
    score: Optional[float] = None
    reward: Optional[float] = None
    objective: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCandidate":
        candidate_id = str(data.get("candidate_id") or "").strip()
        artifact_kind = str(data.get("artifact_kind") or "").strip() or "unknown"
        artifact_payload = data.get("artifact_payload")
        if artifact_payload is None:
            artifact_payload = data.get("candidate_artifact")
        artifact_preview = data.get("artifact_preview")
        if isinstance(artifact_preview, str):
            artifact_preview = artifact_preview.strip() or None
        else:
            artifact_preview = None

        candidate_content = data.get("candidate_content")
        if not isinstance(candidate_content, str) or not candidate_content.strip():
            candidate_content = None
            for key in ("candidate_code", "prompt_text", "instruction", "text"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_content = value.strip()
                    break
        if candidate_content is None and isinstance(artifact_payload, dict):
            for key in ("candidate_code", "candidate_content", "prompt_text", "instruction", "text"):
                value = artifact_payload.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_content = value.strip()
                    break

        objective = _extract_candidate_objective(data)
        reward = _extract_candidate_reward(data)
        score = _coerce_float(data.get("score")) or objective or reward
        return cls(
            candidate_id=candidate_id,
            candidate_type=(
                str(data.get("candidate_type")).strip() if data.get("candidate_type") is not None else None
            ),
            artifact_kind=artifact_kind,
            artifact_payload=artifact_payload,
            artifact_preview=artifact_preview,
            candidate_content=candidate_content,
            status=str(data.get("status")).strip() if data.get("status") is not None else None,
            optimization_mode=(
                str(data.get("optimization_mode")).strip()
                if data.get("optimization_mode") is not None
                else (
                    str(data.get("mode")).strip()
                    if data.get("mode") is not None
                    else None
                )
            ),
            score=score,
            reward=reward,
            objective=objective,
            raw=dict(data),
        )


@dataclass
class PolicyCandidatePage:
    """Typed page of canonical candidates."""

    items: list[PolicyCandidate] = field(default_factory=list)
    next_cursor: Optional[str] = None
    job_id: Optional[str] = None
    system_id: Optional[str] = None
    algorithm: Optional[str] = None
    mode: Optional[str] = None
    sort: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyCandidatePage":
        raw_items = payload.get("items")
        items = []
        if isinstance(raw_items, list):
            items = [
                PolicyCandidate.from_dict(item)
                for item in raw_items
                if isinstance(item, dict)
            ]
        next_cursor = payload.get("next_cursor")
        return cls(
            items=items,
            next_cursor=next_cursor if isinstance(next_cursor, str) and next_cursor.strip() else None,
            job_id=payload.get("job_id") if isinstance(payload.get("job_id"), str) else None,
            system_id=payload.get("system_id") if isinstance(payload.get("system_id"), str) else None,
            algorithm=payload.get("algorithm") if isinstance(payload.get("algorithm"), str) else None,
            mode=payload.get("mode") if isinstance(payload.get("mode"), str) else None,
            sort=payload.get("sort") if isinstance(payload.get("sort"), str) else None,
            raw=dict(payload),
        )


@dataclass
class PolicyOptimizationResult:
    """Typed result from a policy optimization job."""

    job_id: str
    status: PolicyJobStatus
    algorithm: Optional[str] = None
    best_reward: Optional[float] = None
    best_candidate: Optional[str | Dict[str, Any]] = None
    lever_summary: Optional[Dict[str, Any]] = None
    sensor_frames: list[Dict[str, Any]] = field(default_factory=list)
    lever_versions: Dict[str, int] = field(default_factory=dict)
    best_lever_version: Optional[int] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, job_id: str, data: Dict[str, Any], *, algorithm: Optional[str] = None
    ) -> PolicyOptimizationResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=True)
        lever_summary_raw = _from_data_or_metadata(data, "lever_summary")
        lever_summary = lever_summary_raw if isinstance(lever_summary_raw, dict) else None
        sensor_frames_raw = _from_data_or_metadata(data, "sensor_frames")
        sensor_frames = sensor_frames_raw if isinstance(sensor_frames_raw, list) else []
        lever_versions = _parse_lever_versions(_from_data_or_metadata(data, "lever_versions"))
        best_lever_version_raw = _from_data_or_metadata(data, "best_lever_version")
        best_lever_version = None
        if best_lever_version_raw is not None:
            try:
                best_lever_version = int(best_lever_version_raw)
            except (TypeError, ValueError):
                best_lever_version = None
        if lever_versions:
            best_lever_version = best_lever_version or max(int(v) for v in lever_versions.values())
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm or data.get("algorithm"),
            best_reward=best_reward,
            best_candidate=_from_data_or_metadata(data, "best_candidate")
            or _from_data_or_metadata(data, "best_prompt"),
            lever_summary=lever_summary,
            sensor_frames=[frame for frame in sensor_frames if isinstance(frame, dict)],
            lever_versions=lever_versions,
            best_lever_version=best_lever_version,
            error=_normalize_error_message(
                _first_present(data, ("error", "error_message", "failure_reason", "message"))
            ),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == PolicyJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def best_prompt(self) -> Optional[str | Dict[str, Any]]:
        """Backward-compatible alias for `best_candidate`."""
        return self.best_candidate

    @property
    def best_candidate_content(self) -> Optional[str]:
        """Generic best-candidate content (prompt text or non-prompt artifact text)."""
        return _extract_candidate_content(self.best_candidate, self.raw)

    @property
    def lever_summary_typed(self) -> Optional[MiproLeverSummary]:
        """Best-effort typed parsing of `lever_summary` for MIPRO runs."""
        return MiproLeverSummary.from_dict(self.lever_summary) if self.lever_summary else None

    @property
    def sensor_frame_summaries_typed(self) -> list[SensorFrameSummary]:
        """Best-effort typed parsing of `sensor_frames` summaries."""
        out: list[SensorFrameSummary] = []
        for frame in self.sensor_frames:
            parsed = SensorFrameSummary.from_dict(frame)
            if parsed is not None:
                out.append(parsed)
        return out


@dataclass
class PromptLearningResult:
    """Typed result from a prompt learning job."""

    job_id: str
    status: PolicyJobStatus
    best_reward: Optional[float] = None
    best_candidate: Optional[str | Dict[str, Any]] = None
    lever_summary: Optional[Dict[str, Any]] = None
    sensor_frames: list[Dict[str, Any]] = field(default_factory=list)
    lever_versions: Dict[str, int] = field(default_factory=dict)
    best_lever_version: Optional[int] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> PromptLearningResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=True)
        lever_summary_raw = _from_data_or_metadata(data, "lever_summary")
        lever_summary = lever_summary_raw if isinstance(lever_summary_raw, dict) else None
        sensor_frames_raw = _from_data_or_metadata(data, "sensor_frames")
        sensor_frames = sensor_frames_raw if isinstance(sensor_frames_raw, list) else []
        lever_versions = _parse_lever_versions(_from_data_or_metadata(data, "lever_versions"))
        best_lever_version_raw = _from_data_or_metadata(data, "best_lever_version")
        best_lever_version = None
        if best_lever_version_raw is not None:
            try:
                best_lever_version = int(best_lever_version_raw)
            except (TypeError, ValueError):
                best_lever_version = None
        if lever_versions:
            best_lever_version = best_lever_version or max(int(v) for v in lever_versions.values())
        return cls(
            job_id=job_id,
            status=status,
            best_reward=best_reward,
            best_candidate=_from_data_or_metadata(data, "best_candidate")
            or _from_data_or_metadata(data, "best_prompt"),
            lever_summary=lever_summary,
            sensor_frames=[frame for frame in sensor_frames if isinstance(frame, dict)],
            lever_versions=lever_versions,
            best_lever_version=best_lever_version,
            error=_normalize_error_message(
                _first_present(data, ("error", "error_message", "failure_reason", "message"))
            ),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == PolicyJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def get_system_prompt(self) -> Optional[str]:
        """Extract the optimized system prompt text.

        Handles various prompt formats (pattern, template, transformation)
        and returns the system prompt as a clean string.

        Returns:
            System prompt text, or None if extraction fails
        """
        return _extract_system_prompt(self.best_candidate, self.raw)

    @property
    def best_prompt(self) -> Optional[str | Dict[str, Any]]:
        """Backward-compatible alias for `best_candidate`."""
        return self.best_candidate

    @property
    def best_candidate_content(self) -> Optional[str]:
        """Generic best-candidate content (prompt text or non-prompt artifact text)."""
        return _extract_candidate_content(self.best_candidate, self.raw)

    @property
    def lever_summary_typed(self) -> Optional[MiproLeverSummary]:
        """Best-effort typed parsing of `lever_summary` for MIPRO runs."""
        return MiproLeverSummary.from_dict(self.lever_summary) if self.lever_summary else None

    @property
    def sensor_frame_summaries_typed(self) -> list[SensorFrameSummary]:
        """Best-effort typed parsing of `sensor_frames` summaries."""
        out: list[SensorFrameSummary] = []
        for frame in self.sensor_frames:
            parsed = SensorFrameSummary.from_dict(frame)
            if parsed is not None:
                out.append(parsed)
        return out


@dataclass
class GraphOptimizationResult:
    """Typed result from a graph optimization job."""

    job_id: str
    status: GraphJobStatus
    algorithm: Optional[str] = None
    best_reward: Optional[float] = None
    best_yaml: Optional[str] = None
    best_snapshot_id: Optional[str] = None
    generations_completed: Optional[int] = None
    total_candidates_evaluated: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, job_id: str, data: Dict[str, Any], *, algorithm: Optional[str] = None
    ) -> GraphOptimizationResult:
        status_str = data.get("status", "pending")
        status = GraphJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=False)
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm,
            best_reward=best_reward,
            best_yaml=data.get("best_yaml"),
            best_snapshot_id=data.get("best_snapshot_id"),
            generations_completed=data.get("generations_completed"),
            total_candidates_evaluated=data.get("total_candidates_evaluated"),
            duration_seconds=data.get("duration_seconds"),
            error=data.get("error"),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == GraphJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


__all__ = [
    "PolicyJobStatus",
    "GraphJobStatus",
    "PolicyCandidate",
    "PolicyCandidatePage",
    "PolicyOptimizationResult",
    "PromptLearningResult",
    "GraphOptimizationResult",
]
