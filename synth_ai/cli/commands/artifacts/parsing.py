"""Artifact ID parsing and resolution utilities.

This module provides centralized logic for parsing and validating artifact identifiers,
including model IDs (peft:, ft:, rl:) and prompt job IDs (pl_, job_pl_).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ParsedModelId:
    """Parsed model identifier components."""
    
    prefix: str  # peft, ft, or rl
    base_model: str
    job_id: str
    full_id: str  # Original full model ID
    
    @property
    def is_fine_tuned(self) -> bool:
        """Check if this is a fine-tuned model."""
        return self.prefix in ("peft", "ft")
    
    @property
    def is_rl(self) -> bool:
        """Check if this is an RL model."""
        return self.prefix == "rl"


@dataclass
class ParsedPromptId:
    """Parsed prompt job identifier components."""
    
    job_id: str
    full_id: str  # Original full ID (may be same as job_id)


ArtifactType = Literal["model", "prompt", "unknown"]


def parse_model_id(model_id: str) -> ParsedModelId:
    """Parse a model identifier into its components.
    
    Supports:
    - peft:BASE_MODEL:JOB_ID (canonical fine-tuned model)
    - ft:BASE_MODEL:JOB_ID (legacy fine-tuned model)
    - rl:BASE_MODEL:JOB_ID (RL model)
    
    Args:
        model_id: Model identifier string
        
    Returns:
        ParsedModelId with prefix, base_model, job_id, and full_id
        
    Raises:
        ValueError: If model_id format is invalid
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError(f"Invalid model_id: must be a non-empty string, got {type(model_id)}")
    
    model_id = model_id.strip()
    
    # Check for fine-tuned model prefixes (peft: or ft:)
    if model_id.startswith("peft:"):
        prefix = "peft"
        parts = model_id.split(":", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid peft: model ID format. Expected 'peft:BASE_MODEL:JOB_ID', got: {model_id}")
        return ParsedModelId(
            prefix=prefix,
            base_model=parts[1],
            job_id=parts[2],
            full_id=model_id,
        )
    
    elif model_id.startswith("ft:"):
        prefix = "ft"
        parts = model_id.split(":", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid ft: model ID format. Expected 'ft:BASE_MODEL:JOB_ID', got: {model_id}")
        return ParsedModelId(
            prefix=prefix,
            base_model=parts[1],
            job_id=parts[2],
            full_id=model_id,
        )
    
    # Check for RL model prefix
    elif model_id.startswith("rl:"):
        prefix = "rl"
        parts = model_id.split(":", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid rl: model ID format. Expected 'rl:BASE_MODEL:JOB_ID', got: {model_id}")
        return ParsedModelId(
            prefix=prefix,
            base_model=parts[1],
            job_id=parts[2],
            full_id=model_id,
        )
    
    raise ValueError(
        f"Unsupported model ID prefix. Expected 'peft:', 'ft:', or 'rl:', got: {model_id}"
    )


def parse_prompt_id(prompt_id: str) -> ParsedPromptId:
    """Parse a prompt job identifier.
    
    Supports:
    - pl_JOB_ID (canonical prompt learning job ID)
    - job_pl_JOB_ID (alternative format)
    - JOB_ID (bare job ID, assumed to be prompt learning)
    
    Args:
        prompt_id: Prompt job identifier string
        
    Returns:
        ParsedPromptId with job_id and full_id
        
    Raises:
        ValueError: If prompt_id format is invalid
    """
    if not prompt_id or not isinstance(prompt_id, str):
        raise ValueError(f"Invalid prompt_id: must be a non-empty string, got {type(prompt_id)}")
    
    prompt_id = prompt_id.strip()
    
    # Handle job_pl_ prefix
    if prompt_id.startswith("job_pl_"):
        job_id = prompt_id
        return ParsedPromptId(job_id=job_id, full_id=prompt_id)
    
    # Handle pl_ prefix
    if prompt_id.startswith("pl_"):
        job_id = prompt_id
        return ParsedPromptId(job_id=job_id, full_id=prompt_id)
    
    # Assume it's a bare job ID (could be prompt learning)
    # We'll let the backend validate this
    return ParsedPromptId(job_id=prompt_id, full_id=prompt_id)


def detect_artifact_type(artifact_id: str) -> ArtifactType:
    """Detect the type of artifact from its ID.
    
    Args:
        artifact_id: Artifact identifier string
        
    Returns:
        "model", "prompt", or "unknown"
    """
    if not artifact_id or not isinstance(artifact_id, str):
        return "unknown"
    
    artifact_id = artifact_id.strip()
    
    # Check for model prefixes
    if artifact_id.startswith(("peft:", "ft:", "rl:")):
        return "model"
    
    # Check for prompt prefixes
    if artifact_id.startswith(("pl_", "job_pl_")):
        return "prompt"
    
    return "unknown"


def is_model_id(artifact_id: str) -> bool:
    """Check if an artifact ID is a model ID."""
    return detect_artifact_type(artifact_id) == "model"


def is_prompt_id(artifact_id: str) -> bool:
    """Check if an artifact ID is a prompt job ID."""
    return detect_artifact_type(artifact_id) == "prompt"


def resolve_wasabi_key_for_model(
    parsed: ParsedModelId,
    prefer_merged: bool = True,
) -> str:
    """Resolve Wasabi storage key for a parsed model ID.
    
    This constructs the expected Wasabi key based on naming conventions.
    Note: This is a fallback - the backend should provide the actual key.
    
    Args:
        parsed: Parsed model ID
        prefer_merged: For fine-tuned models, prefer merged checkpoint over adapter
        
    Returns:
        Wasabi storage key path
        
    Raises:
        ValueError: If model type is unsupported
    """
    if parsed.is_fine_tuned:
        base_model = parsed.base_model
        job_id = parsed.job_id
        
        if prefer_merged:
            # Merged checkpoint format: models/BASE-JOB_ID-fp16.tar.gz
            safe_base = base_model.replace("/", "-").replace(":", "-").replace(".", "-")
            return f"models/{safe_base}-{job_id}-fp16.tar.gz"
        else:
            # Adapter format: models/BASE/ft-JOB_ID/adapter.tar.gz
            return f"models/{base_model}/ft-{job_id}/adapter.tar.gz"
    
    elif parsed.is_rl:
        # RL model format: models/BASE-JOB_ID-rl.tar.gz
        safe_base = parsed.base_model.replace("/", "-").replace(":", "-").replace(".", "-")
        return f"models/{safe_base}-{parsed.job_id}-rl.tar.gz"
    
    raise ValueError(f"Unsupported model type for Wasabi key resolution: {parsed.prefix}")


def validate_model_id(model_id: str) -> bool:
    """Validate that a model ID has the correct format.
    
    Args:
        model_id: Model identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_model_id(model_id)
        return True
    except ValueError:
        return False


def validate_prompt_id(prompt_id: str) -> bool:
    """Validate that a prompt ID has a reasonable format.
    
    Args:
        prompt_id: Prompt identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_prompt_id(prompt_id)
        return True
    except ValueError:
        return False

