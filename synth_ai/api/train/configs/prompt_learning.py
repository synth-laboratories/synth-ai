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
    """MIPRO-specific configuration.
    
    MIPROv2 uses meta-learning with bootstrap phase, TPE optimization, and mini-batch evaluation
    to efficiently optimize prompts with fewer evaluations than genetic algorithms.
    """
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
    
    # Token and budget configuration (mirrors GEPA pattern)
    max_token_limit: int | None = None  # Total tokens across all rollouts (policy + proposer)
    max_spend_usd: float | None = None  # Maximum spend in USD
    token_counting_model: str = "gpt-4"  # Model for token estimation (tiktoken)
    enforce_token_limit: bool = True  # Halt optimization if limit exceeded
    
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


# GEPA nested configs (mirroring RL structure)
class GEPARolloutConfig(ExtraModel):
    """GEPA rollout configuration (mirrors RL [rollout] section)."""
    budget: int | None = None  # Total rollout budget
    max_concurrent: int = 20  # Maximum concurrent rollouts
    minibatch_size: int = 8  # Minibatch size for evaluation


class GEPAEvaluationConfig(ExtraModel):
    """GEPA evaluation configuration (mirrors RL [evaluation] section)."""
    seeds: list[int] | None = None  # Evaluation seeds (training set)
    validation_seeds: list[int] | None = None  # Validation seeds (held-out)
    test_pool: list[int] | None = None  # Test pool (final evaluation)
    validation_pool: str | None = None  # Pool name for validation (e.g., "validation")
    validation_top_k: int | None = None  # Top-K prompts to validate


class GEPAMutationConfig(ExtraModel):
    """GEPA mutation configuration (LLM-guided mutation settings)."""
    rate: float = 0.3  # Mutation rate
    llm_model: str | None = None  # Model for generating mutations
    llm_provider: str = "groq"  # Provider for mutation LLM
    llm_inference_url: str | None = None  # Custom inference URL
    prompt: str | None = None  # Custom mutation prompt


class GEPAPopulationConfig(ExtraModel):
    """GEPA population configuration (evolution parameters)."""
    initial_size: int = 20  # Initial population size
    num_generations: int = 10  # Number of generations
    children_per_generation: int = 5  # Children generated per generation
    crossover_rate: float = 0.5  # Crossover rate
    selection_pressure: float = 1.0  # Pareto selection pressure
    patience_generations: int = 3  # Early stopping patience


class GEPAArchiveConfig(ExtraModel):
    """GEPA archive configuration (Pareto archive settings)."""
    size: int = 64  # Archive size
    pareto_set_size: int = 64  # Pareto set size
    pareto_eps: float = 1e-6  # Pareto epsilon
    feedback_fraction: float = 0.5  # Fraction of archive for feedback


class GEPATokenConfig(ExtraModel):
    """GEPA token and budget configuration."""
    max_limit: int | None = None  # Maximum tokens allowed in prompt
    counting_model: str = "gpt-4"  # Model for token counting
    enforce_pattern_limit: bool = True  # Enforce token limit on patterns
    max_spend_usd: float | None = None  # Maximum spend in USD


class GEPAConfig(ExtraModel):
    """GEPA-specific configuration with nested subsections."""
    # Top-level fields (for backwards compatibility)
    env_name: str = "banking77"
    env_config: dict[str, Any] | None = None
    rng_seed: int | None = None
    proposer_type: str = "dspy"  # "dspy" or "synth"
    
    # Nested subsections (preferred, mirrors RL structure)
    rollout: GEPARolloutConfig | None = None
    evaluation: GEPAEvaluationConfig | None = None
    mutation: GEPAMutationConfig | None = None
    population: GEPAPopulationConfig | None = None
    archive: GEPAArchiveConfig | None = None
    token: GEPATokenConfig | None = None
    
    # Backwards compatibility: flat fields (deprecated, prefer nested)
    # These will be flattened from nested configs if provided
    rollout_budget: int | None = None
    max_concurrent_rollouts: int | None = None
    minibatch_size: int | None = None
    evaluation_seeds: list[int] | None = None
    validation_seeds: list[int] | None = None
    test_pool: list[int] | None = None
    validation_pool: str | None = None
    validation_top_k: int | None = None
    mutation_rate: float | None = None
    mutation_llm_model: str | None = None
    mutation_llm_provider: str | None = None
    mutation_llm_inference_url: str | None = None
    mutation_prompt: str | None = None
    initial_population_size: int | None = None
    num_generations: int | None = None
    children_per_generation: int | None = None
    crossover_rate: float | None = None
    selection_pressure: float | None = None
    patience_generations: int | None = None
    archive_size: int | None = None
    pareto_set_size: int | None = None
    pareto_eps: float | None = None
    feedback_fraction: float | None = None
    max_token_limit: int | None = None
    token_counting_model: str | None = None
    enforce_pattern_token_limit: bool | None = None
    max_spend_usd: float | None = None
    
    def _get_rollout_budget(self) -> int | None:
        """Get rollout budget from nested or flat structure."""
        if self.rollout and self.rollout.budget is not None:
            return self.rollout.budget
        return self.rollout_budget
    
    def _get_max_concurrent_rollouts(self) -> int:
        """Get max concurrent rollouts from nested or flat structure."""
        if self.rollout and self.rollout.max_concurrent is not None:
            return self.rollout.max_concurrent
        return self.max_concurrent_rollouts or 20
    
    def _get_minibatch_size(self) -> int:
        """Get minibatch size from nested or flat structure."""
        if self.rollout and self.rollout.minibatch_size is not None:
            return self.rollout.minibatch_size
        return self.minibatch_size or 8
    
    def _get_evaluation_seeds(self) -> list[int] | None:
        """Get evaluation seeds from nested or flat structure."""
        if self.evaluation and self.evaluation.seeds is not None:
            return self.evaluation.seeds
        return self.evaluation_seeds
    
    def _get_validation_seeds(self) -> list[int] | None:
        """Get validation seeds from nested or flat structure."""
        if self.evaluation and self.evaluation.validation_seeds is not None:
            return self.evaluation.validation_seeds
        return self.validation_seeds
    
    def _get_test_pool(self) -> list[int] | None:
        """Get test pool from nested or flat structure."""
        if self.evaluation and self.evaluation.test_pool is not None:
            return self.evaluation.test_pool
        return self.test_pool
    
    def _get_mutation_rate(self) -> float:
        """Get mutation rate from nested or flat structure."""
        if self.mutation and self.mutation.rate is not None:
            return self.mutation.rate
        return self.mutation_rate or 0.3
    
    def _get_mutation_llm_model(self) -> str | None:
        """Get mutation LLM model from nested or flat structure."""
        if self.mutation and self.mutation.llm_model is not None:
            return self.mutation.llm_model
        return self.mutation_llm_model
    
    def _get_mutation_llm_provider(self) -> str:
        """Get mutation LLM provider from nested or flat structure."""
        if self.mutation and self.mutation.llm_provider is not None:
            return self.mutation.llm_provider
        return self.mutation_llm_provider or "groq"
    
    def _get_mutation_llm_inference_url(self) -> str | None:
        """Get mutation LLM inference URL from nested or flat structure."""
        if self.mutation and self.mutation.llm_inference_url is not None:
            return self.mutation.llm_inference_url
        return self.mutation_llm_inference_url
    
    def _get_mutation_prompt(self) -> str | None:
        """Get mutation prompt from nested or flat structure."""
        if self.mutation and self.mutation.prompt is not None:
            return self.mutation.prompt
        return self.mutation_prompt
    
    def _get_initial_population_size(self) -> int:
        """Get initial population size from nested or flat structure."""
        if self.population and self.population.initial_size is not None:
            return self.population.initial_size
        return self.initial_population_size or 20
    
    def _get_num_generations(self) -> int:
        """Get num generations from nested or flat structure."""
        if self.population and self.population.num_generations is not None:
            return self.population.num_generations
        return self.num_generations or 10
    
    def _get_children_per_generation(self) -> int:
        """Get children per generation from nested or flat structure."""
        if self.population and self.population.children_per_generation is not None:
            return self.population.children_per_generation
        return self.children_per_generation or 5
    
    def _get_crossover_rate(self) -> float:
        """Get crossover rate from nested or flat structure."""
        if self.population and self.population.crossover_rate is not None:
            return self.population.crossover_rate
        return self.crossover_rate or 0.5
    
    def _get_selection_pressure(self) -> float:
        """Get selection pressure from nested or flat structure."""
        if self.population and self.population.selection_pressure is not None:
            return self.population.selection_pressure
        return self.selection_pressure or 1.0
    
    def _get_patience_generations(self) -> int:
        """Get patience generations from nested or flat structure."""
        if self.population and self.population.patience_generations is not None:
            return self.population.patience_generations
        return self.patience_generations or 3
    
    def _get_archive_size(self) -> int:
        """Get archive size from nested or flat structure."""
        if self.archive and self.archive.size is not None:
            return self.archive.size
        return self.archive_size or 64
    
    def _get_pareto_set_size(self) -> int:
        """Get pareto set size from nested or flat structure."""
        if self.archive and self.archive.pareto_set_size is not None:
            return self.archive.pareto_set_size
        return self.pareto_set_size or 64
    
    def _get_pareto_eps(self) -> float:
        """Get pareto eps from nested or flat structure."""
        if self.archive and self.archive.pareto_eps is not None:
            return self.archive.pareto_eps
        return self.pareto_eps or 1e-6
    
    def _get_feedback_fraction(self) -> float:
        """Get feedback fraction from nested or flat structure."""
        if self.archive and self.archive.feedback_fraction is not None:
            return self.archive.feedback_fraction
        return self.feedback_fraction or 0.5
    
    def _get_max_token_limit(self) -> int | None:
        """Get max token limit from nested or flat structure."""
        if self.token and self.token.max_limit is not None:
            return self.token.max_limit
        return self.max_token_limit
    
    def _get_token_counting_model(self) -> str:
        """Get token counting model from nested or flat structure."""
        if self.token and self.token.counting_model is not None:
            return self.token.counting_model
        return self.token_counting_model or "gpt-4"
    
    def _get_enforce_pattern_token_limit(self) -> bool:
        """Get enforce pattern token limit from nested or flat structure."""
        if self.token and self.token.enforce_pattern_limit is not None:
            return self.token.enforce_pattern_limit
        return self.enforce_pattern_token_limit if self.enforce_pattern_token_limit is not None else True
    
    def _get_max_spend_usd(self) -> float | None:
        """Get max spend USD from nested or flat structure."""
        if self.token and self.token.max_spend_usd is not None:
            return self.token.max_spend_usd
        return self.max_spend_usd
    
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GEPAConfig:
        """Load GEPA config from dict/TOML, handling both nested and flat structures."""
        # Check for nested structure first
        nested_data = {}
        flat_data = {}
        
        for key, value in data.items():
            if key in ("rollout", "evaluation", "mutation", "population", "archive", "token"):
                nested_data[key] = value
            else:
                flat_data[key] = value
        
        # If we have nested data, create nested configs
        if nested_data:
            if "rollout" in nested_data:
                nested_data["rollout"] = GEPARolloutConfig.model_validate(nested_data["rollout"])
            if "evaluation" in nested_data:
                nested_data["evaluation"] = GEPAEvaluationConfig.model_validate(nested_data["evaluation"])
            if "mutation" in nested_data:
                nested_data["mutation"] = GEPAMutationConfig.model_validate(nested_data["mutation"])
            if "population" in nested_data:
                nested_data["population"] = GEPAPopulationConfig.model_validate(nested_data["population"])
            if "archive" in nested_data:
                nested_data["archive"] = GEPAArchiveConfig.model_validate(nested_data["archive"])
            if "token" in nested_data:
                nested_data["token"] = GEPATokenConfig.model_validate(nested_data["token"])
        
        # Merge nested and flat data
        merged_data = {**flat_data, **nested_data}
        return cls.model_validate(merged_data)


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
        
        # Handle gepa config specially to support nested structure
        if "gepa" in pl_data and isinstance(pl_data["gepa"], dict):
            gepa_data = pl_data["gepa"]
            pl_data["gepa"] = GEPAConfig.from_mapping(gepa_data)
        
        return cls.model_validate(pl_data)

    @classmethod
    def from_path(cls, path: Path) -> PromptLearningConfig:
        """Load prompt learning config from TOML file."""
        content = load_toml(path)
        return cls.from_mapping(content)


__all__ = [
    "GEPAConfig",
    "GEPARolloutConfig",
    "GEPAEvaluationConfig",
    "GEPAMutationConfig",
    "GEPAPopulationConfig",
    "GEPAArchiveConfig",
    "GEPATokenConfig",
    "MIPROConfig",
    "MessagePatternConfig",
    "PromptLearningConfig",
    "PromptLearningPolicyConfig",
    "PromptPatternConfig",
]
