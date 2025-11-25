from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class ExtraModel(BaseModel):
    """Base model that tolerates unknown keys so configs keep forward compatibility."""

    model_config = ConfigDict(extra="allow")


class AlgorithmConfig(ExtraModel):
    type: str
    method: str
    variety: str


class TopologyConfig(ExtraModel):
    """Compute topology configuration - how GPUs are distributed across processes."""
    type: str | None = None  # e.g., "single_node_split"
    gpus_for_vllm: int | None = None
    gpus_for_training: int | None = None
    gpus_for_ref: int | None = None
    tensor_parallel: int | None = None
    reference_placement: str | None = None  # NEW: e.g., "none", "shared", "dedicated"


class LoraConfig(ExtraModel):
    """LoRA (Low-Rank Adaptation) training configuration."""
    r: int | None = None  # Rank
    alpha: int | None = None
    dropout: float | None = None
    target_modules: list[str] | None = None


class ComputeConfig(ExtraModel):
    gpu_type: str
    gpu_count: int
    nodes: int | None = None
    topology: TopologyConfig | None = None  # NEW: nested topology


class PolicyConfig(ExtraModel):
    """Unified policy configuration for both SFT and RL.
    
    This is the SINGLE SOURCE OF TRUTH for:
    - What model to use (model_name or source)
    - How to sample from it (temperature, max_tokens, etc.)
    - How to train it (trainer_mode, label)
    """
    
    # Model specification (exactly one required)
    model_name: str | None = None  # e.g., "Qwen/Qwen3-4B"
    source: str | None = None       # e.g., "ft:abc123" for checkpoints
    
    # Sampling parameters (with sensible defaults)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0
    stop_sequences: list[str] | None = None
    
    # Training-specific
    trainer_mode: str  # "lora", "full", "qlora"
    label: str         # Model identifier/name
    
    # Optional - for distributed inference
    inference_url: str | None = None
    
    @model_validator(mode="after")
    def _ensure_exactly_one_source(self) -> PolicyConfig:
        """Ensure exactly one of model_name or source is set."""
        if not (bool(self.model_name) ^ bool(self.source)):
            raise ValueError(
                "Must set exactly one: [policy].model_name OR [policy].source"
            )
        return self


__all__ = ["ExtraModel", "AlgorithmConfig", "ComputeConfig", "PolicyConfig", "TopologyConfig", "LoraConfig"]
