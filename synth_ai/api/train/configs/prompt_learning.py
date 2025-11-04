"""Prompt Learning configuration models for MIPRO and GEPA."""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from ..utils import load_toml
from .shared import ExtraModel


class InferenceMode(str, Enum):
    synth_hosted = "synth_hosted"


class ProviderName(str, Enum):
    openai = "openai"
    groq = "groq"
    google = "google"


class PromptLearningPolicyConfig(ExtraModel):
    """Policy configuration for prompt learning (model, provider, etc.)."""
    model: str
    provider: ProviderName
    inference_url: str
    inference_mode: InferenceMode = InferenceMode.synth_hosted
    temperature: float = 0.0
    max_completion_tokens: int = 512
    policy_name: str | None = None

    @field_validator("inference_url")
    @classmethod
    def _normalize_inference_url(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("inference_url must be a string")
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("inference_url must start with http:// or https://")
        return v


class MessagePatternConfig(ExtraModel):
    """Configuration for a single message pattern."""
    role: str
    pattern: str
    order: int = 0


class PromptPatternConfig(ExtraModel):
    """Initial prompt pattern configuration."""
    id: str | None = None
    name: str | None = None
    messages: list[MessagePatternConfig] = []
    wildcards: dict[str, str] = Field(default_factory=dict)


class MIPROConfig(ExtraModel):
    """MIPRO-specific configuration."""
    num_iterations: int = 20
    num_evaluations_per_iteration: int = 5
    batch_size: int = 32
    max_concurrent: int = 20
    env_name: str = "banking77"
    env_config: dict[str, Any] | None = None
    meta_model: str = "gpt-4o-mini"
    meta_model_provider: str = "openai"
    meta_model_inference_url: str | None = None
    few_shot_score_threshold: float = 0.8
    results_file: str | None = None
    max_wall_clock_seconds: float | None = None
    max_total_tokens: int | None = None
    
    # TPE configuration
    tpe: dict[str, Any] | None = None
    
    # Demo configuration
    demo: dict[str, Any] | None = None
    
    # Grounding configuration
    grounding: dict[str, Any] | None = None
    
    # Meta-update configuration
    meta_update: dict[str, Any] | None = None
    
    # Bootstrap seeds (for few-shot examples)
    bootstrap_train_seeds: list[int] | None = None
    
    # Online pool (for mini-batch evaluation)
    online_pool: list[int] | None = None
    
    # Test pool (held-out seeds)
    test_pool: list[int] | None = None


class GEPAConfig(ExtraModel):
    """GEPA-specific configuration."""
    env_name: str = "banking77"
    env_config: dict[str, Any] | None = None
    rollout_budget: int | None = None
    feedback_fraction: float = 0.5
    pareto_set_size: int = 64
    minibatch_size: int = 8
    rng_seed: int | None = None
    initial_population_size: int = 20
    num_generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    selection_pressure: float = 1.0
    max_token_limit: int | None = None
    token_counting_model: str = "gpt-4"
    mutation_llm_model: str | None = None
    mutation_llm_provider: str = "groq"
    mutation_llm_inference_url: str | None = None
    mutation_prompt: str | None = None
    enforce_pattern_token_limit: bool = True
    archive_size: int = 64
    pareto_eps: float = 1e-6
    patience_generations: int = 3
    children_per_generation: int = 5
    max_concurrent_rollouts: int = 20
    max_spend_usd: float | None = None
    
    # Evaluation seeds
    evaluation_seeds: list[int] | None = None
    
    # Test pool (held-out seeds)
    test_pool: list[int] | None = None


class PromptLearningConfig(ExtraModel):
    """Top-level prompt learning configuration."""
    algorithm: str  # "mipro" or "gepa"
    task_app_url: str
    task_app_api_key: str | None = None
    task_app_id: str | None = None
    initial_prompt: PromptPatternConfig | None = None
    policy: PromptLearningPolicyConfig | None = None
    mipro: MIPROConfig | None = None
    gepa: GEPAConfig | None = None
    env_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for API payload."""
        result = self.model_dump(mode="python", exclude_none=True)
        # Ensure prompt_learning section wraps everything
        if "prompt_learning" not in result:
            pl_data = dict(result.items())
            result = {"prompt_learning": pl_data}
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PromptLearningConfig:
        """Load prompt learning config from dict/TOML mapping."""
        # Handle both [prompt_learning] section and flat structure
        pl_data = data.get("prompt_learning", {})
        if not pl_data:
            # If no prompt_learning section, assume top-level is prompt_learning
            pl_data = dict(data)
        
        return cls.model_validate(pl_data)

    @classmethod
    def from_path(cls, path: Path) -> PromptLearningConfig:
        """Load prompt learning config from TOML file."""
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "GEPAConfig",
    "MIPROConfig",
    "MessagePatternConfig",
    "PromptLearningConfig",
    "PromptLearningPolicyConfig",
    "PromptPatternConfig",
]

