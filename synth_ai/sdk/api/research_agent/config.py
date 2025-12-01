"""Typed configuration models for Research Agent jobs.

These models mirror the backend Pydantic models in:
backend/app/routes/research_agent/models.py

This provides type safety and IDE autocomplete for SDK users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class ModelProvider(str, Enum):
    """Supported model providers for prompt optimization."""

    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"


class OptimizationTool(str, Enum):
    """Available optimization tools."""

    MIPRO = "mipro"
    GEPA = "gepa"


# Type aliases for Literal types
ProposerEffort = Literal["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"]
ProposerOutputTokens = Literal["RAPID", "FAST", "SLOW"]
ReasoningEffort = Literal["low", "medium", "high"]
DatasetSourceType = Literal["huggingface", "upload", "inline"]


@dataclass
class PermittedModel:
    """A single permitted model configuration."""

    model: str
    """Model name (e.g., 'gpt-4o-mini', 'llama-3.3-70b-versatile')"""

    provider: ModelProvider
    """Model provider: openai, groq, or google"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider.value if isinstance(self.provider, Enum) else self.provider,
        }


@dataclass
class PermittedModelsConfig:
    """Configuration for permitted models in the optimization pipeline.

    The user specifies which models the agent is ALLOWED to use during optimization.
    The agent decides which models to use for which pipeline stages.
    """

    models: List[PermittedModel] = field(default_factory=list)
    """List of models the agent is permitted to use in the pipeline"""

    default_temperature: float = 0.7
    """Default sampling temperature"""

    default_max_tokens: int = 4096
    """Default max tokens per response"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": [m.to_dict() for m in self.models],
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }


@dataclass
class DatasetSource:
    """Configuration for dataset injection into the sandbox."""

    source_type: DatasetSourceType
    """Type of dataset source: huggingface, upload, or inline"""

    description: Optional[str] = None
    """Optional description of the dataset"""

    # For source_type="huggingface"
    hf_repo_id: Optional[str] = None
    """HuggingFace dataset repo ID (e.g., 'PolyAI/banking77')"""

    hf_split: str = "train"
    """Dataset split to use"""

    hf_subset: Optional[str] = None
    """Dataset subset/config name"""

    # For source_type="upload"
    file_ids: Optional[List[str]] = None
    """List of uploaded file IDs"""

    # For source_type="inline"
    inline_data: Optional[Dict[str, str]] = None
    """Dict of filename -> content"""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"source_type": self.source_type}
        if self.description:
            result["description"] = self.description
        if self.source_type == "huggingface":
            if self.hf_repo_id:
                result["hf_repo_id"] = self.hf_repo_id
            result["hf_split"] = self.hf_split
            if self.hf_subset:
                result["hf_subset"] = self.hf_subset
        elif self.source_type == "upload":
            if self.file_ids:
                result["file_ids"] = self.file_ids
        elif self.source_type == "inline":
            if self.inline_data:
                result["inline_data"] = self.inline_data
        return result


@dataclass
class GEPAConfig:
    """GEPA-specific model configuration.

    GEPA uses a mutation model to generate prompt variations/mutations.
    """

    # Mutation model (for generating prompt mutations)
    mutation_model: str = "openai/gpt-oss-120b"
    """Model for generating prompt mutations"""

    mutation_provider: ModelProvider = ModelProvider.GROQ
    """Provider for mutation model"""

    mutation_temperature: float = 0.7
    """Temperature for mutation generation"""

    mutation_max_tokens: int = 8192
    """Max tokens for mutation responses"""

    # Advanced GEPA settings
    population_size: int = 20
    """Population size for genetic algorithm"""

    num_generations: int = 10
    """Number of generations to evolve"""

    elite_fraction: float = 0.2
    """Fraction of population to keep as elite"""

    # Proposer settings
    proposer_type: Literal["dspy", "spec"] = "dspy"
    """Type of proposer to use"""

    proposer_effort: ProposerEffort = "MEDIUM"
    """Effort level for proposal generation"""

    proposer_output_tokens: ProposerOutputTokens = "FAST"
    """Output token budget for proposer"""

    spec_path: Optional[str] = None
    """Path to spec file (for proposer_type='spec')"""

    # Seed pool sizes (optional - agent decides if not set)
    train_size: Optional[int] = None
    """Training set size"""

    val_size: Optional[int] = None
    """Validation set size"""

    reference_size: Optional[int] = None
    """Reference set size"""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "mutation_model": self.mutation_model,
            "mutation_provider": self.mutation_provider.value if isinstance(self.mutation_provider, Enum) else self.mutation_provider,
            "mutation_temperature": self.mutation_temperature,
            "mutation_max_tokens": self.mutation_max_tokens,
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "elite_fraction": self.elite_fraction,
            "proposer_type": self.proposer_type,
            "proposer_effort": self.proposer_effort,
            "proposer_output_tokens": self.proposer_output_tokens,
        }
        if self.spec_path:
            result["spec_path"] = self.spec_path
        if self.train_size is not None:
            result["train_size"] = self.train_size
        if self.val_size is not None:
            result["val_size"] = self.val_size
        if self.reference_size is not None:
            result["reference_size"] = self.reference_size
        return result


@dataclass
class MIPROConfig:
    """MIPRO-specific model configuration.

    MIPRO uses a meta model to generate instruction/prompt proposals.
    """

    # Meta model (for generating instruction proposals)
    meta_model: str = "llama-3.3-70b-versatile"
    """Model for generating instruction proposals"""

    meta_provider: ModelProvider = ModelProvider.GROQ
    """Provider for meta model"""

    meta_temperature: float = 0.7
    """Temperature for proposal generation"""

    meta_max_tokens: int = 4096
    """Max tokens for proposal responses"""

    # Advanced MIPRO settings
    num_candidates: int = 20
    """Number of instruction candidates to generate"""

    num_trials: int = 10
    """Number of optimization trials"""

    # Proposer settings
    proposer_effort: ProposerEffort = "MEDIUM"
    """Effort level for proposal generation"""

    proposer_output_tokens: ProposerOutputTokens = "FAST"
    """Output token budget for proposer"""

    # Seed pool sizes (optional - agent decides if not set)
    train_size: Optional[int] = None
    """Training set size"""

    val_size: Optional[int] = None
    """Validation set size"""

    reference_size: Optional[int] = None
    """Reference set size"""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "meta_model": self.meta_model,
            "meta_provider": self.meta_provider.value if isinstance(self.meta_provider, Enum) else self.meta_provider,
            "meta_temperature": self.meta_temperature,
            "meta_max_tokens": self.meta_max_tokens,
            "num_candidates": self.num_candidates,
            "num_trials": self.num_trials,
            "proposer_effort": self.proposer_effort,
            "proposer_output_tokens": self.proposer_output_tokens,
        }
        if self.train_size is not None:
            result["train_size"] = self.train_size
        if self.val_size is not None:
            result["val_size"] = self.val_size
        if self.reference_size is not None:
            result["reference_size"] = self.reference_size
        return result


@dataclass
class ResearchConfig:
    """Configuration for prompt/pipeline research optimization.

    This is the main configuration for the "research" algorithm, which uses
    MIPRO or GEPA to optimize prompts/pipelines.
    """

    task_description: str
    """What to optimize (e.g., 'Improve accuracy on banking intent classification')"""

    tools: List[OptimizationTool] = field(default_factory=lambda: [OptimizationTool.MIPRO])
    """Optimization tools to use (mipro, gepa)"""

    # Datasets
    datasets: List[DatasetSource] = field(default_factory=list)
    """Datasets for training/evaluation"""

    # Metrics
    primary_metric: str = "accuracy"
    """Main metric to optimize"""

    secondary_metrics: List[str] = field(default_factory=list)
    """Additional metrics to track"""

    # Optimization parameters
    num_iterations: int = 10
    """Number of optimization iterations"""

    population_size: int = 20
    """Population size (GEPA) or candidates (MIPRO)"""

    timeout_minutes: int = 60
    """Maximum runtime in minutes"""

    max_eval_samples: Optional[int] = None
    """Max samples to evaluate per iteration"""

    # Model configurations
    permitted_models: Optional[PermittedModelsConfig] = None
    """Models the agent is allowed to use in the pipeline"""

    gepa_config: Optional[GEPAConfig] = None
    """GEPA-specific settings"""

    mipro_config: Optional[MIPROConfig] = None
    """MIPRO-specific settings"""

    # Initial prompt/pipeline
    initial_prompt: Optional[str] = None
    """Initial prompt template to optimize"""

    pipeline_entrypoint: Optional[str] = None
    """Path to pipeline script (e.g., 'pipeline.py')"""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "task_description": self.task_description,
            "tools": [t.value if isinstance(t, Enum) else t for t in self.tools],
            "primary_metric": self.primary_metric,
            "num_iterations": self.num_iterations,
            "population_size": self.population_size,
            "timeout_minutes": self.timeout_minutes,
        }

        if self.datasets:
            result["datasets"] = [d.to_dict() for d in self.datasets]

        if self.secondary_metrics:
            result["secondary_metrics"] = self.secondary_metrics

        if self.max_eval_samples is not None:
            result["max_eval_samples"] = self.max_eval_samples

        if self.permitted_models:
            result["permitted_models"] = self.permitted_models.to_dict()

        if self.gepa_config:
            result["gepa_config"] = self.gepa_config.to_dict()

        if self.mipro_config:
            result["mipro_config"] = self.mipro_config.to_dict()

        if self.initial_prompt:
            result["initial_prompt"] = self.initial_prompt

        if self.pipeline_entrypoint:
            result["pipeline_entrypoint"] = self.pipeline_entrypoint

        return result
