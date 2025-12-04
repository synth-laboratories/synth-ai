"""TOML validation logic for train commands (SFT and RL)."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from synth_ai.core.telemetry import log_info
from synth_ai.sdk.api.train.configs.rl import RLConfig
from synth_ai.sdk.api.train.configs.sft import SFTConfig
from synth_ai.sdk.api.train.utils import load_toml

from .errors import (
    InvalidJudgeConfigError,
    InvalidRLConfigError,
    InvalidRubricConfigError,
    InvalidSFTConfigError,
    MissingAlgorithmError,
    MissingComputeError,
    MissingDatasetError,
    MissingModelError,
    TomlParseError,
    UnsupportedAlgorithmError,
)
from .judge_validation import extract_and_validate_judge_rubric

__all__ = [
    "validate_sft_config",
    "validate_rl_config",
    "load_and_validate_sft",
    "load_and_validate_rl",
]


def validate_sft_config(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Validate SFT configuration from TOML.

    Args:
        config: Raw configuration dictionary from TOML

    Returns:
        Validated configuration dictionary

    Raises:
        InvalidSFTConfigError: If validation fails
        MissingAlgorithmError: If algorithm section is missing or invalid
        MissingModelError: If model is not specified
        MissingDatasetError: If dataset path is not specified
        MissingComputeError: If compute section is missing required fields
    """
    ctx: dict[str, Any] = {"config_keys": list(config.keys())[:10]}
    log_info("validate_sft_config invoked", ctx=ctx)
    # Check for required top-level sections
    if "algorithm" not in config or not config["algorithm"]:
        raise MissingAlgorithmError(
            detail="[algorithm] section is required for SFT configs"
        )
    
    if "job" not in config or not config["job"]:
        raise InvalidSFTConfigError(
            detail="[job] section is required for SFT configs"
        )
    
    job = config.get("job", {})
    if not job.get("model"):
        raise MissingModelError(
            detail="[job].model is required (e.g., 'Qwen/Qwen3-4B')"
        )
    
    # Check that at least one dataset source is specified
    if not (job.get("data") or job.get("data_path")):
        raise MissingDatasetError(
            detail="[job].data or [job].data_path must be specified",
            hint="Provide path to training JSONL file"
        )
    
    # Validate algorithm type, method, and variety
    algorithm = config.get("algorithm", {})
    if algorithm.get("type") not in {"offline", None}:
        raise UnsupportedAlgorithmError(
            algorithm_type=algorithm.get("type", "unknown"),
            expected="offline",
            hint="SFT requires algorithm.type = 'offline'"
        )
    
    method = algorithm.get("method", "")
    if method and method not in {"sft", "supervised_finetune"}:
        raise UnsupportedAlgorithmError(
            algorithm_type=method,
            expected="sft or supervised_finetune",
            hint="SFT requires algorithm.method = 'sft' or 'supervised_finetune'"
        )
    
    # Validate variety is present
    if not algorithm.get("variety"):
        raise MissingAlgorithmError(
            detail="[algorithm].variety is required (e.g., 'fft', 'lora', 'qlora')"
        )
    
    # Validate compute section
    compute = config.get("compute", {})
    if not compute:
        raise MissingComputeError(
            detail="[compute] section is required",
            hint="Specify gpu_type, gpu_count, and nodes"
        )
    
    if not compute.get("gpu_type"):
        raise MissingComputeError(
            detail="[compute].gpu_type is required (e.g., 'H100', 'A100')"
        )
    
    if not compute.get("gpu_count"):
        raise MissingComputeError(
            detail="[compute].gpu_count is required"
        )
    
    # Validate using Pydantic model
    try:
        validated = SFTConfig.from_mapping(config)
        return validated.to_dict()
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • {loc}: {msg}")
        raise InvalidSFTConfigError(
            detail="Pydantic validation failed:\n" + "\n".join(errors)
        ) from exc


def validate_rl_config(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Validate RL configuration from TOML.

    Args:
        config: Raw configuration dictionary from TOML

    Returns:
        Validated configuration dictionary

    Raises:
        InvalidRLConfigError: If validation fails
        MissingAlgorithmError: If algorithm section is missing or invalid
        MissingModelError: If model is not specified
        MissingComputeError: If compute section is missing required fields
    """
    ctx: dict[str, Any] = {"config_keys": list(config.keys())[:10]}
    log_info("validate_rl_config invoked", ctx=ctx)
    # Check for required top-level sections
    if "algorithm" not in config or not config["algorithm"]:
        raise MissingAlgorithmError(
            detail="[algorithm] section is required for RL configs"
        )
    
    # Check for model OR policy (policy is the new format)
    if "policy" not in config and "model" not in config:
        raise MissingModelError(
            detail="[policy] or [model] section is required for RL configs"
        )
    
    # Validate algorithm type, method, and variety
    algorithm = config.get("algorithm", {})
    if algorithm.get("type") not in {"online", None}:
        raise UnsupportedAlgorithmError(
            algorithm_type=algorithm.get("type", "unknown"),
            expected="online",
            hint="RL requires algorithm.type = 'online'"
        )
    
    method = algorithm.get("method", "")
    if method and method not in {"policy_gradient", "ppo", "gspo"}:
        raise UnsupportedAlgorithmError(
            algorithm_type=method,
            expected="policy_gradient",
            hint="RL requires algorithm.method = 'policy_gradient'"
        )
    
    # Validate variety is present
    if not algorithm.get("variety"):
        raise MissingAlgorithmError(
            detail="[algorithm].variety is required (e.g., 'gspo', 'ppo')"
        )
    
    # Validate model/policy section
    model = config.get("model", {})
    policy = config.get("policy", {})
    
    # Use policy if available, otherwise fall back to model
    if policy:
        if not policy.get("model_name") and not policy.get("source"):
            raise MissingModelError(
                detail="[policy].model_name or [policy].source must be specified",
                hint="Provide base model (e.g., 'Qwen/Qwen3-4B') or source checkpoint"
            )
        
        if not policy.get("trainer_mode"):
            raise InvalidRLConfigError(
                detail="[policy].trainer_mode is required (e.g., 'full', 'lora')"
            )
        
        if not policy.get("label"):
            raise InvalidRLConfigError(
                detail="[policy].label is required (e.g., 'my-rl-model')",
                hint="Provide a descriptive label for this model"
            )
    elif model:
        if not model.get("base") and not model.get("source"):
            raise MissingModelError(
                detail="[model].base or [model].source must be specified",
                hint="Provide base model (e.g., 'Qwen/Qwen3-4B') or source checkpoint"
            )
        
        if not model.get("trainer_mode"):
            raise InvalidRLConfigError(
                detail="[model].trainer_mode is required (e.g., 'full', 'lora')"
            )
        
        if not model.get("label"):
            raise InvalidRLConfigError(
                detail="[model].label is required (e.g., 'my-rl-model')",
                hint="Provide a descriptive label for this model"
            )
    
    # Validate compute section
    compute = config.get("compute", {})
    if not compute:
        raise MissingComputeError(
            detail="[compute] section is required",
            hint="Specify gpu_type and gpu_count"
        )
    
    if not compute.get("gpu_type"):
        raise MissingComputeError(
            detail="[compute].gpu_type is required (e.g., 'H100', 'A100')"
        )
    
    if not compute.get("gpu_count"):
        raise MissingComputeError(
            detail="[compute].gpu_count is required"
        )
    
    # Check for rollout configuration
    rollout = config.get("rollout", {})
    if not rollout:
        raise InvalidRLConfigError(
            detail="[rollout] section is required for RL configs",
            hint="Specify env_name, policy_name, max_turns, etc."
        )
    
    if not rollout.get("env_name"):
        raise InvalidRLConfigError(
            detail="[rollout].env_name is required (e.g., 'math', 'crafter')"
        )
    
    if not rollout.get("policy_name"):
        raise InvalidRLConfigError(
            detail="[rollout].policy_name is required"
        )
    
    # Validate topology section (can be top-level or under compute)
    topology = config.get("topology") or compute.get("topology", {})
    if not topology:
        raise InvalidRLConfigError(
            detail="[topology] or [compute.topology] section is required",
            hint="Specify gpus_for_vllm, gpus_for_training, etc."
        )
    
    # Check for training section and its required fields
    training = config.get("training", {})
    if training:
        required_training_fields = {
            "num_epochs": "number of training epochs",
            "iterations_per_epoch": "iterations per epoch",
            "max_turns": "maximum turns",
            "batch_size": "batch size",
            "group_size": "group size",
            "learning_rate": "learning rate",
        }
        
        for field, description in required_training_fields.items():
            if field not in training:
                raise InvalidRLConfigError(
                    detail=f"[training].{field} is required ({description})",
                    hint=f"Add {field} to the [training] section"
                )
    
    # Check for evaluation section
    evaluation = config.get("evaluation", {})
    if evaluation:
        required_eval_fields = {
            "instances": "number of evaluation instances",
            "every_n_iters": "evaluation frequency",
            "seeds": "evaluation seeds",
        }
        
        for field, description in required_eval_fields.items():
            if field not in evaluation:
                raise InvalidRLConfigError(
                    detail=f"[evaluation].{field} is required ({description})",
                    hint=f"Add {field} to the [evaluation] section"
                )
    
    # Inject services section if not present (will be populated at runtime)
    if "services" not in config:
        config["services"] = {
            "task_url": "placeholder",  # Will be resolved at runtime
        }
    
    # Inject reference placement if not present (like builders.py does)
    # Reference is now under compute.topology.reference_placement
    if "compute" not in config:
        config["compute"] = {}
    if "topology" not in config["compute"]:
        config["compute"]["topology"] = {}
    if "reference_placement" not in config["compute"]["topology"]:
        config["compute"]["topology"]["reference_placement"] = "none"
    
    # Validate judge/rubric configuration with formalized Pydantic models
    # This will emit deprecation warnings for dead fields and validate structure
    try:
        rubric_config, judge_config = extract_and_validate_judge_rubric(config)
        # Validation passed - configs are clean and ready for use
        # The validated Pydantic models can be used by training code if needed
    except (InvalidJudgeConfigError, InvalidRubricConfigError) as exc:
        raise InvalidRLConfigError(
            detail=f"Judge/Rubric validation failed: {exc.detail}",
            hint="Check JUDGE_RUBRIC_CLEANUP_GUIDE.md for migration help."
        ) from exc
    
    # Validate using Pydantic model
    try:
        validated = RLConfig.from_mapping(config)
        return validated.to_dict()
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  • {loc}: {msg}")
        raise InvalidRLConfigError(
            detail="Pydantic validation failed:\n" + "\n".join(errors)
        ) from exc


def load_and_validate_sft(config_path: Path) -> dict[str, Any]:
    """Load and validate an SFT TOML configuration file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        TomlParseError: If TOML parsing fails
        InvalidSFTConfigError: If validation fails
    """
    try:
        raw_config = load_toml(config_path)
    except Exception as exc:
        raise TomlParseError(
            path=str(config_path),
            detail=str(exc)
        ) from exc
    
    return validate_sft_config(raw_config)


def load_and_validate_rl(config_path: Path) -> dict[str, Any]:
    """Load and validate an RL TOML configuration file.
    
    Args:
        config_path: Path to TOML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        TomlParseError: If TOML parsing fails
        InvalidRLConfigError: If validation fails
    """
    try:
        raw_config = load_toml(config_path)
    except Exception as exc:
        raise TomlParseError(
            path=str(config_path),
            detail=str(exc)
        ) from exc
    
    return validate_rl_config(raw_config)
