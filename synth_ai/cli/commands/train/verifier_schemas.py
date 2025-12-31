"""
Pydantic schemas for verifier/rubric configuration.

These models define the ACTUAL fields used by the backend verifier service,
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
    "VerifierOptionsConfig",
    "VerifierConfig",
    "VerifierRequestPayload",
]


class RubricWeightsConfig(ExtraModel):
    """
    Reward blending weights (client-side only, not sent to backend).
    
    These weights control how env rewards, event verifier scores, and outcome
    verifier scores are combined into a final reward signal for policy gradients.
    
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
        description="Weight for per-event verifier scores (step-level verification)",
        ge=0.0,
    )
    outcome: float = Field(
        default=0.0,
        description="Weight for outcome verifier score (episode-level verification)",
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
    
    Controls whether rubric-based verification is enabled and how rewards are blended.
    """
    enabled: bool = Field(
        default=False,
        description="Master switch for rubric-based verification",
    )
    weights: RubricWeightsConfig = Field(
        default_factory=RubricWeightsConfig,
        description="Reward blending weights (env/event/outcome)",
    )


class VerifierOptionsConfig(ExtraModel):
    """
    Verifier provider options (sent to backend in HTTP request).
    
    These fields are sent in the "options" object of the verifier request.
    All fields here map directly to the backend verifier options schema.
    """
    provider: str = Field(
        ...,
        description="Verifier provider type ('openai', 'groq', 'gemini')",
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
        description="Enable per-event (step-level) verification",
    )
    outcome: bool = Field(
        default=True,
        description="Enable outcome (episode-level) verification",
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
    def _validate_at_least_one_enabled(self) -> VerifierOptionsConfig:
        """Ensure at least one verification type is enabled."""
        if not self.event and not self.outcome:
            raise ValueError("At least one of 'event' or 'outcome' must be enabled")
        return self


class VerifierConfig(ExtraModel):
    """
    Top-level verifier configuration.
    
    This is parsed from TOML [verifier] section and contains all verifier-related settings.
    """
    options: VerifierOptionsConfig = Field(
        ...,
        description="Verifier provider options (sent to backend)",
    )


# HTTP Request Payload Structures (for documentation/type safety)

class VerifierRequestPayload(ExtraModel):
    """
    HTTP request payload structure for POST /api/graphs/verifiers/completions.
    
    This is the ACTUAL payload sent to the backend verifier service.
    Used for type safety and documentation only.
    """
    policy_name: str = Field(..., description="Name of the policy being evaluated")
    task_app: dict[str, Any] = Field(..., description="Task app metadata (id, base_url)")
    trace: dict[str, Any] = Field(..., description="Tracing v3 payload (event_history, metadata)")
    options: dict[str, Any] = Field(..., description="Verifier options (provider, model, etc.)")

    class Config:
        extra = "allow"  # Backend might add extra fields


# Helper to convert to backend request format

def build_verifier_http_options(
    options_config: VerifierOptionsConfig,
    *,
    rubric_overrides_from_task_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build the 'options' dict for HTTP request to backend verifier.
    
    Args:
        options_config: Validated verifier options from TOML
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
