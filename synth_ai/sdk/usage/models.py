"""Data models for usage tracking.

These dataclasses represent the usage and limits data returned by the
GET /api/v1/usage endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class UsageMetric:
    """A single usage metric with its limit and remaining capacity.

    Attributes:
        used: Current usage value
        limit: Maximum allowed value
        remaining: Remaining capacity (limit - used)
    """

    used: int | float
    limit: int | float
    remaining: int | float

    @property
    def percent_used(self) -> float:
        """Return percentage of limit used (0-100)."""
        if self.limit == 0:
            return 0.0
        return (self.used / self.limit) * 100

    @property
    def is_exhausted(self) -> bool:
        """Return True if the limit has been reached."""
        return self.remaining <= 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageMetric:
        """Create from API response dict."""
        return cls(
            used=data.get("used", 0),
            limit=data.get("limit", 0),
            remaining=data.get("remaining", 0),
        )


@dataclass
class UsagePeriod:
    """Time period for usage tracking."""

    daily_start: datetime
    monthly_start: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsagePeriod:
        """Create from API response dict."""
        return cls(
            daily_start=datetime.fromisoformat(data["daily_start"].replace("Z", "+00:00")),
            monthly_start=datetime.fromisoformat(data["monthly_start"].replace("Z", "+00:00")),
        )


@dataclass
class InferenceUsage:
    """Usage metrics for the inference API."""

    requests_per_min: UsageMetric
    tokens_per_day: UsageMetric
    spend_cents_per_month: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceUsage:
        """Create from API response dict."""
        return cls(
            requests_per_min=UsageMetric.from_dict(data.get("requests_per_min", {})),
            tokens_per_day=UsageMetric.from_dict(data.get("tokens_per_day", {})),
            spend_cents_per_month=UsageMetric.from_dict(data.get("spend_cents_per_month", {})),
        )


@dataclass
class JudgesUsage:
    """Usage metrics for the judges API."""

    evaluations_per_day: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgesUsage:
        """Create from API response dict."""
        return cls(
            evaluations_per_day=UsageMetric.from_dict(data.get("evaluations_per_day", {})),
        )


@dataclass
class PromptOptUsage:
    """Usage metrics for prompt optimization (GEPA/MIPRO)."""

    jobs_per_day: UsageMetric
    rollouts_per_day: UsageMetric
    spend_cents_per_day: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptOptUsage:
        """Create from API response dict."""
        return cls(
            jobs_per_day=UsageMetric.from_dict(data.get("jobs_per_day", {})),
            rollouts_per_day=UsageMetric.from_dict(data.get("rollouts_per_day", {})),
            spend_cents_per_day=UsageMetric.from_dict(data.get("spend_cents_per_day", {})),
        )


@dataclass
class RLUsage:
    """Usage metrics for RL training."""

    jobs_per_month: UsageMetric
    gpu_hours_per_month: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLUsage:
        """Create from API response dict."""
        return cls(
            jobs_per_month=UsageMetric.from_dict(data.get("jobs_per_month", {})),
            gpu_hours_per_month=UsageMetric.from_dict(data.get("gpu_hours_per_month", {})),
        )


@dataclass
class SFTUsage:
    """Usage metrics for SFT training."""

    jobs_per_month: UsageMetric
    gpu_hours_per_month: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTUsage:
        """Create from API response dict."""
        return cls(
            jobs_per_month=UsageMetric.from_dict(data.get("jobs_per_month", {})),
            gpu_hours_per_month=UsageMetric.from_dict(data.get("gpu_hours_per_month", {})),
        )


@dataclass
class ResearchUsage:
    """Usage metrics for research agents."""

    jobs_per_month: UsageMetric
    agent_spend_cents_per_month: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchUsage:
        """Create from API response dict."""
        return cls(
            jobs_per_month=UsageMetric.from_dict(data.get("jobs_per_month", {})),
            agent_spend_cents_per_month=UsageMetric.from_dict(data.get("agent_spend_cents_per_month", {})),
        )


@dataclass
class TotalUsage:
    """Total usage across all APIs."""

    spend_cents_per_month: UsageMetric

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TotalUsage:
        """Create from API response dict."""
        return cls(
            spend_cents_per_month=UsageMetric.from_dict(data.get("spend_cents_per_month", {})),
        )


@dataclass
class APIUsage:
    """Container for all API usage metrics."""

    inference: InferenceUsage
    judges: JudgesUsage
    prompt_opt: PromptOptUsage
    rl: RLUsage
    sft: SFTUsage
    research: ResearchUsage

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIUsage:
        """Create from API response dict."""
        return cls(
            inference=InferenceUsage.from_dict(data.get("inference", {})),
            judges=JudgesUsage.from_dict(data.get("judges", {})),
            prompt_opt=PromptOptUsage.from_dict(data.get("prompt_opt", {})),
            rl=RLUsage.from_dict(data.get("rl", {})),
            sft=SFTUsage.from_dict(data.get("sft", {})),
            research=ResearchUsage.from_dict(data.get("research", {})),
        )


@dataclass
class OrgUsage:
    """Complete org usage report.

    This is the top-level object returned by UsageClient.get().

    Attributes:
        org_id: The organization ID
        tier: The org's tier (free, starter, growth, enterprise)
        period: Time period info for daily/monthly resets
        apis: Per-API usage metrics
        totals: Aggregate totals across all APIs
    """

    org_id: str
    tier: str
    period: UsagePeriod
    apis: APIUsage
    totals: TotalUsage

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrgUsage:
        """Create from API response dict."""
        return cls(
            org_id=data.get("org_id", ""),
            tier=data.get("tier", "free"),
            period=UsagePeriod.from_dict(data.get("period", {})),
            apis=APIUsage.from_dict(data.get("apis", {})),
            totals=TotalUsage.from_dict(data.get("totals", {})),
        )

    def get_metric(self, api: str, metric: str) -> UsageMetric | None:
        """Get a specific metric by API and metric name.

        Args:
            api: API name (inference, judges, prompt_opt, rl, sft, research)
            metric: Metric name (e.g., requests_per_min, jobs_per_day)

        Returns:
            UsageMetric if found, None otherwise
        """
        api_usage = getattr(self.apis, api, None)
        if api_usage is None:
            return None
        return getattr(api_usage, metric, None)


__all__ = [
    "UsageMetric",
    "UsagePeriod",
    "InferenceUsage",
    "JudgesUsage",
    "PromptOptUsage",
    "RLUsage",
    "SFTUsage",
    "ResearchUsage",
    "TotalUsage",
    "APIUsage",
    "OrgUsage",
]
