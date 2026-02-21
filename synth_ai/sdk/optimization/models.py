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

    # Try sections format (legacy)
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

    return None


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
        best_reward = _first_present(
            data,
            (
                "best_score",
                "best_reward",
                "best_train_score",
                "best_train_reward",
            ),
        )
        if best_reward is None:
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                best_reward = _first_present(
                    metadata,
                    (
                        "best_score",
                        "best_reward",
                        "best_train_score",
                        "best_train_reward",
                    ),
                )
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
        best_reward = _first_present(
            data,
            (
                "best_score",
                "best_reward",
                "best_train_score",
                "best_train_reward",
            ),
        )
        if best_reward is None:
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                best_reward = _first_present(
                    metadata,
                    (
                        "best_score",
                        "best_reward",
                        "best_train_score",
                        "best_train_reward",
                    ),
                )
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
        best_reward = _first_present(data, ("best_score", "best_reward"))
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
    "PolicyOptimizationResult",
    "PromptLearningResult",
    "GraphOptimizationResult",
]
