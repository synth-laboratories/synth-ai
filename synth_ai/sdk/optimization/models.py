from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _normalize_status(status: str) -> str:
    return status.strip().lower().replace(" ", "_")


class PolicyJobStatus(str, Enum):
    """Status of a policy optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
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
    best_score: Optional[float] = None
    best_prompt: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, job_id: str, data: Dict[str, Any], *, algorithm: Optional[str] = None
    ) -> PolicyOptimizationResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_score = _first_present(
            data,
            (
                "best_score",
                "best_reward",
                "best_train_score",
                "best_train_reward",
            ),
        )
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm or data.get("algorithm"),
            best_score=best_score,
            best_prompt=data.get("best_prompt"),
            error=data.get("error"),
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


@dataclass
class PromptLearningResult:
    """Typed result from a prompt learning job."""

    job_id: str
    status: PolicyJobStatus
    best_score: Optional[float] = None
    best_prompt: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> PromptLearningResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_score = _first_present(
            data,
            (
                "best_score",
                "best_reward",
                "best_train_score",
                "best_train_reward",
            ),
        )
        return cls(
            job_id=job_id,
            status=status,
            best_score=best_score,
            best_prompt=data.get("best_prompt"),
            error=data.get("error"),
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


@dataclass
class GraphOptimizationResult:
    """Typed result from a graph optimization job."""

    job_id: str
    status: GraphJobStatus
    algorithm: Optional[str] = None
    best_score: Optional[float] = None
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
        best_score = _first_present(data, ("best_score", "best_reward"))
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm,
            best_score=best_score,
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
