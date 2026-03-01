"""Backward-compatible eval job models used by external tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str) -> EvalStatus:
        normalized = (value or "").strip().lower()
        for status in cls:
            if status.value == normalized:
                return status
        raise ValueError(f"Unknown eval status: {value}")


@dataclass
class EvalJobConfig:
    container_url: str
    api_key: str
    backend_url: str
    env_name: str
    seeds: list[int]
    policy_config: dict[str, Any]
    container_api_key: Optional[str] = None

    def __init__(
        self,
        *,
        container_url: str,
        api_key: str,
        backend_url: str,
        env_name: str,
        seeds: list[int],
        policy_config: dict[str, Any],
        container_api_key: Optional[str] = None,
        container_key: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.container_url = container_url
        self.api_key = api_key
        self.backend_url = backend_url
        self.env_name = env_name
        self.seeds = seeds
        self.policy_config = policy_config
        self.container_api_key = container_api_key or container_key


@dataclass
class EvalResult:
    eval_id: str
    status: EvalStatus
    mean_reward: float
    total_tokens: int
    total_cost_usd: float
    num_completed: int
    num_total: int

    @property
    def succeeded(self) -> bool:
        return self.status == EvalStatus.SUCCEEDED

    @classmethod
    def from_response(cls, eval_id: str, response: dict[str, Any]) -> EvalResult:
        results = dict(response.get("results") or {})
        return cls(
            eval_id=eval_id,
            status=EvalStatus.from_string(str(response.get("status") or "")),
            mean_reward=float(results.get("mean_reward") or 0.0),
            total_tokens=int(results.get("total_tokens") or 0),
            total_cost_usd=float(results.get("total_cost_usd") or 0.0),
            num_completed=int(results.get("completed") or 0),
            num_total=int(results.get("total") or 0),
        )
