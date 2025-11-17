"""Pydantic schemas for backend API responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ProgressEventData(BaseModel):
    """Data payload for prompt.learning.progress events.
    
    Validates and coerces progress data from backend events.
    """

    rollouts_completed: int | None = Field(None, ge=0, description="Number of rollouts completed")
    rollouts_total: int | None = Field(None, gt=0, description="Total rollouts planned")
    rollouts_remaining: int | None = Field(None, description="Rollouts remaining (may be negative due to timing/calculation issues)")
    rollout_budget_display: int | None = Field(None, gt=0, description="Alternative field name for rollouts_total")
    percent_rollouts: float | None = Field(None, ge=0.0, le=1.0, description="Rollout progress (0-1)")
    percent_overall: float | None = Field(None, ge=0.0, le=1.0, description="Overall progress (0-1)")
    transformations_planned: int | None = Field(None, ge=0, description="Total transformations planned")
    transformations_tried: int | None = Field(None, ge=0, description="Transformations tried")
    percent_transformations: float | None = Field(None, ge=0.0, le=1.0, description="Transformation progress (0-1)")
    elapsed_seconds: int | None = Field(None, ge=0, description="Elapsed time in seconds")
    eta_seconds: float | None = Field(None, ge=0.0, description="Estimated time remaining in seconds")
    rollout_tokens_used: int | None = Field(None, ge=0, description="Tokens used for rollouts")
    rollout_tokens_budget: int | None = Field(None, ge=0, description="Token budget for rollouts")
    best_score: float | None = Field(None, ge=0.0, le=1.0, description="Best score achieved (0-1)")
    accuracy: float | None = Field(None, ge=0.0, le=1.0, description="Alternative field name for best_score")

    @field_validator(
        "rollouts_completed",
        "rollouts_total",
        "rollouts_remaining",
        "rollout_budget_display",
        "transformations_planned",
        "transformations_tried",
        "elapsed_seconds",
        "rollout_tokens_used",
        "rollout_tokens_budget",
        mode="before",
    )
    @classmethod
    def coerce_int_fields(cls, v: Any) -> int | None:
        """Coerce int fields from various types."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float | str):
            try:
                coerced = int(float(v))
                assert coerced >= 0, f"Cannot coerce negative value to int: {v}"
                return coerced
            except (ValueError, TypeError, AssertionError):
                return None
        return None

    @field_validator("eta_seconds", "best_score", "accuracy", "percent_rollouts", "percent_overall", "percent_transformations", mode="before")
    @classmethod
    def coerce_float_fields(cls, v: Any) -> float | None:
        """Coerce float fields from various types."""
        if v is None:
            return None
        if isinstance(v, int | float):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None
        return None

    @model_validator(mode="after")
    def validate_consistency(self) -> ProgressEventData:
        """Validate data consistency after initialization."""
        # Validate rollouts_completed <= rollouts_total
        if self.rollouts_completed is not None and self.effective_rollouts_total is not None:
            assert self.rollouts_completed <= self.effective_rollouts_total, (
                f"rollouts_completed ({self.rollouts_completed}) > rollouts_total ({self.effective_rollouts_total})"
            )
        
        # Validate rollouts_remaining consistency (but allow negative values due to timing/calculation issues)
        if (
            self.rollouts_completed is not None
            and self.effective_rollouts_total is not None
            and self.rollouts_remaining is not None
        ):
            expected_remaining = self.effective_rollouts_total - self.rollouts_completed
            # Allow small discrepancy due to timing, and allow negative values (backend calculation may be off)
            # Negative values can happen if backend thinks more rollouts were done than budget allows
            if abs(self.rollouts_remaining - expected_remaining) > 5:
                # Log warning but don't fail - backend calculation may be using different tracking
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "rollouts_remaining (%s) inconsistent with rollouts_completed (%s) and total (%s), "
                    "expected remaining: %s. Backend calculation may be using different tracking.",
                    self.rollouts_remaining,
                    self.rollouts_completed,
                    self.effective_rollouts_total,
                    expected_remaining,
                )
        
        return self

    @property
    def effective_rollouts_total(self) -> int | None:
        """Get rollouts_total, falling back to rollout_budget_display."""
        return self.rollouts_total or self.rollout_budget_display

    @property
    def effective_best_score(self) -> float | None:
        """Get best_score, falling back to accuracy."""
        return self.best_score or self.accuracy


class BackendJobEvent(BaseModel):
    """A single event from the backend API."""

    seq: int = Field(ge=0, description="Sequence number (monotonically increasing)")
    type: str = Field(description="Event type (e.g., 'prompt.learning.progress')")
    message: str = Field(description="Human-readable message")
    data: dict[str, Any] | None = Field(None, description="Optional event data payload")
    timestamp: str | None = Field(None, description="ISO timestamp when event was emitted")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate event type is not empty."""
        if not v or not v.strip():
            raise ValueError("Event type cannot be empty")
        return v.strip()

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is not empty."""
        if not v or not v.strip():
            raise ValueError("Event message cannot be empty")
        return v.strip()

    def get_progress_data(self) -> ProgressEventData | None:
        """Extract and validate progress data if this is a progress event."""
        if self.type != "prompt.learning.progress":
            return None
        if not self.data:
            return None
        try:
            return ProgressEventData(**self.data)
        except Exception as e:
            # Log but don't fail - allow partial data
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Failed to parse progress event data: %s", e)
            # Try to create with partial data
            return ProgressEventData.model_validate(self.data, strict=False)


class BackendEventsResponse(BaseModel):
    """Response from backend /jobs/{job_id}/events endpoint.
    
    The backend API can return either:
    1. A list of events directly: [{"seq": 1, "type": "...", ...}, ...]
    2. A dict with "events" key: {"events": [...], "has_more": false, ...}
    
    This model handles both formats.
    """

    events: list[BackendJobEvent] = Field(default_factory=list)
    has_more: bool = False
    next_seq: int | None = None

    @classmethod
    def parse_response(cls, data: Any) -> BackendEventsResponse:
        """Parse API response, handling both list and dict formats.
        
        Args:
            data: Raw JSON response (list or dict)
            
        Returns:
            BackendEventsResponse with parsed events
            
        Raises:
            ValueError: If response format is invalid
            AssertionError: If response structure is unexpected
        """
        # Assert we got valid JSON
        assert data is not None, "API response is None"
        
        # Handle list format (direct list of events)
        if isinstance(data, list):
            # Validate all items are dicts
            assert all(isinstance(item, dict) for item in data), (
                f"Expected list of dicts, got list with types: {[type(item).__name__ for item in data[:5]]}"
            )
            events = [BackendJobEvent.model_validate(item) for item in data]
            return cls(events=events, has_more=False, next_seq=None)
        
        # Handle dict format (with "events" key)
        if isinstance(data, dict):
            events_data = data.get("events", [])
            assert isinstance(events_data, list), (
                f"Expected 'events' to be a list, got {type(events_data).__name__}"
            )
            assert all(isinstance(item, dict) for item in events_data), (
                f"Expected list of dicts in 'events', got types: {[type(item).__name__ for item in events_data[:5]]}"
            )
            events = [BackendJobEvent.model_validate(item) for item in events_data]
            has_more = data.get("has_more", False)
            next_seq = data.get("next_seq")
            return cls(events=events, has_more=bool(has_more), next_seq=next_seq)
        
        # Invalid format
        raise ValueError(
            f"Invalid API response format: expected list or dict, got {type(data).__name__}. "
            f"Response preview: {str(data)[:200]}"
        )

