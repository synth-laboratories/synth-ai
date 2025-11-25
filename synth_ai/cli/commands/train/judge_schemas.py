"""
Pydantic schemas for judge/rubric configuration.

These models define the ACTUAL fields used by the backend judge service,
with all dead code removed. This is the single source of truth for what
gets sent in HTTP requests.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, model_validator

from synth_ai.sdk.api.train.configs.shared import ExtraModel

__all__ = [
    "RubricWeightsConfig",
    "RubricConfig",
    "JudgeOptionsConfig",
    "JudgeConfig",
    "JudgeRequestPayload",
]


class RubricWeightsConfig(ExtraModel):
    """
    Reward blending weights (client-side only, not sent to backend).
    
    These weights control how env rewards, event judge scores, and outcome
    judge scores are combined into a final reward signal for policy gradients.
    
    Formula:
        total_reward = (env * env_return) + (event * sum(event_scores)) + (outcome * outcome_score)
    """
    env: float = Field(
        default=1.0,
        description="Weight for environment rewards (task app native rewards)",
        ge=0.0,
    )
    event: float = Field(
        default=0.0,
        description="Weight for per-event judge scores (step-level judging)",
        ge=0.0,
    )
    outcome: float = Field(
        default=0.0,
        description="Weight for outcome judge score (episode-level judging)",
        ge=0.0,
    )

    @model_validator(mode="after")
    def _validate_weights_sum(self) -> RubricWeightsConfig:
        """Ensure at least one weight is non-zero."""
        if self.env == 0.0 and self.event == 0.0 and self.outcome == 0.0:
            raise ValueError("At least one reward weight must be non-zero")
        return self


class RubricConfig(ExtraModel):
    """
    Top-level rubric configuration.
    
    Controls whether rubric-based judging is enabled and how rewards are blended.
    """
    enabled: bool = Field(
        default=False,
        description="Master switch for rubric-based judging",
    )
    weights: RubricWeightsConfig = Field(
        default_factory=RubricWeightsConfig,
        description="Reward blending weights (env/event/outcome)",
    )


class JudgeOptionsConfig(ExtraModel):
    """
    Judge provider options (sent to backend in HTTP request).
    
    These fields are sent in the "options" object of the judge score request.
    All fields here map directly to the backend JudgeOptions schema.
    """
    provider: str = Field(
        ...,
        description="Judge provider type ('openai', 'groq', 'gemini')",
        pattern=r"^(openai|groq|gemini)$",
    )
    model: str = Field(
        ...,
        description="Model identifier (e.g., 'openai/gpt-oss-120b', 'gpt-5')",
        min_length=1,
    )
    rubric_id: Optional[str] = Field(
        default=None,
        description="Base rubric identifier (e.g., 'crafter/bundle@v1')",
    )
    event: bool = Field(
        default=True,
        description="Enable per-event (step-level) judging",
    )
    outcome: bool = Field(
        default=True,
        description="Enable outcome (episode-level) judging",
    )
    timeout_s: Optional[float] = Field(
        default=None,
        description="Request timeout in seconds",
        gt=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., {'async': true, 'custom_field': 'value'})",
    )
    rubric_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Static rubric criteria overrides (rarely used - TaskInfo overrides take priority). "
            "Format: {'event': {'criteria': [...]}, 'outcome': {'criteria': [...]}}"
        ),
    )

    @model_validator(mode="after")
    def _validate_at_least_one_enabled(self) -> JudgeOptionsConfig:
        """Ensure at least one judging type is enabled."""
        if not self.event and not self.outcome:
            raise ValueError("At least one of 'event' or 'outcome' must be enabled")
        return self


class JudgeConfig(ExtraModel):
    """
    Top-level judge configuration.
    
    This is parsed from TOML [judge] section and contains all judge-related settings.
    """
    options: JudgeOptionsConfig = Field(
        ...,
        description="Judge provider options (sent to backend)",
    )


# HTTP Request Payload Structures (for documentation/type safety)

class JudgeRequestPayload(ExtraModel):
    """
    HTTP request payload structure for POST /api/judge/v1/score.
    
    This is the ACTUAL payload sent to the backend judge service.
    Used for type safety and documentation only.
    """
    policy_name: str = Field(..., description="Name of the policy being evaluated")
    task_app: dict[str, Any] = Field(..., description="Task app metadata (id, base_url)")
    trace: dict[str, Any] = Field(..., description="Tracing v3 payload (event_history, metadata)")
    options: dict[str, Any] = Field(..., description="Judge options (provider, model, etc.)")

    class Config:
        extra = "allow"  # Backend might add extra fields


# Helper to convert to backend request format

def build_judge_http_options(
    options_config: JudgeOptionsConfig,
    *,
    rubric_overrides_from_task_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build the 'options' dict for HTTP request to backend judge.
    
    Args:
        options_config: Validated judge options from TOML
        rubric_overrides_from_task_info: Dynamic overrides fetched from TaskInfo (takes priority)
    
    Returns:
        Dict ready to send in HTTP request payload
    """
    payload = {
        "provider": options_config.provider,
        "model": options_config.model,
        "event": options_config.event,
        "outcome": options_config.outcome,
    }
    
    # Optional fields
    if options_config.rubric_id:
        payload["rubric_id"] = options_config.rubric_id
    
    if options_config.timeout_s is not None:
        payload["timeout_s"] = options_config.timeout_s
    
    if options_config.metadata:
        payload["metadata"] = options_config.metadata
    
    # Rubric overrides: TaskInfo takes priority over static config
    if rubric_overrides_from_task_info:
        payload["rubric_overrides"] = rubric_overrides_from_task_info
    elif options_config.rubric_overrides:
        payload["rubric_overrides"] = options_config.rubric_overrides
    
    return payload

