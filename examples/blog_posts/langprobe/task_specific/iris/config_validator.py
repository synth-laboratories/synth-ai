#!/usr/bin/env python3
"""Rich validation for GEPA and MIPRO prompt learning configs.

This module provides comprehensive validation that catches errors early,
before submission to the backend, with clear error messages.
"""

import toml
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_gepa_config(config_path: Path) -> Tuple[bool, List[str]]:
    """Validate GEPA config with comprehensive checks.
    
    Returns:
        (is_valid, errors) tuple where errors is a list of error messages
    """
    errors = []
    
    try:
        with open(config_path, 'r') as f:
            config_dict = toml.load(f)
    except Exception as e:
        return False, [f"Failed to parse TOML: {e}"]
    
    pl_section = config_dict.get("prompt_learning", {})
    if not isinstance(pl_section, dict):
        errors.append("❌ [prompt_learning] section is missing or invalid")
        return False, errors
    
    # Check algorithm
    algorithm = pl_section.get("algorithm")
    if algorithm != "gepa":
        errors.append(f"❌ Expected algorithm='gepa', got '{algorithm}'")
    
    # Check required top-level fields (env_name is now in gepa section)
    required_top_level = ["task_app_url", "task_app_api_key"]
    for field in required_top_level:
        if not pl_section.get(field):
            errors.append(f"❌ [prompt_learning].{field} is required")
    
    # Check GEPA section
    gepa_section = pl_section.get("gepa", {})
    if not isinstance(gepa_section, dict):
        errors.append("❌ [prompt_learning.gepa] section is missing or invalid")
        return False, errors
    
    # Check env_name in gepa section (required)
    if not gepa_section.get("env_name"):
        errors.append("❌ [prompt_learning.gepa].env_name is required")
    
    # Check required GEPA subsections
    required_sections = ["evaluation", "rollout", "mutation", "population", "archive", "token"]
    missing_sections = [s for s in required_sections if not gepa_section.get(s)]
    if missing_sections:
        errors.append(
            f"❌ Missing required GEPA sections: {', '.join(f'[prompt_learning.gepa.{s}]' for s in missing_sections)}"
        )
    
    # Validate evaluation section
    eval_section = gepa_section.get("evaluation", {})
    if isinstance(eval_section, dict):
        # Check train_seeds (required, can be in eval section or top-level)
        train_seeds = (
            eval_section.get("train_seeds") or 
            eval_section.get("seeds") or 
            pl_section.get("train_seeds")
        )
        if not train_seeds:
            errors.append(
                "❌ train_seeds is required. "
                "Must be in [prompt_learning.gepa.evaluation].train_seeds or [prompt_learning].train_seeds"
            )
        elif not isinstance(train_seeds, list):
            errors.append(f"❌ train_seeds must be a list, got {type(train_seeds).__name__}")
        elif len(train_seeds) == 0:
            errors.append("❌ train_seeds cannot be empty")
        elif not all(isinstance(s, int) for s in train_seeds):
            errors.append("❌ train_seeds must contain only integers")
        
        # Check val_seeds (required)
        val_seeds = eval_section.get("val_seeds") or eval_section.get("validation_seeds")
        if not val_seeds:
            errors.append(
                "❌ val_seeds is required in [prompt_learning.gepa.evaluation].val_seeds"
            )
        elif not isinstance(val_seeds, list):
            errors.append(f"❌ val_seeds must be a list, got {type(val_seeds).__name__}")
        elif len(val_seeds) == 0:
            errors.append("❌ val_seeds cannot be empty")
        elif not all(isinstance(s, int) for s in val_seeds):
            errors.append("❌ val_seeds must contain only integers")
        
        # Check validation_pool (optional but should be valid if present)
        validation_pool = eval_section.get("validation_pool")
        if validation_pool is not None:
            if not isinstance(validation_pool, str):
                errors.append(f"❌ validation_pool must be a string, got {type(validation_pool).__name__}")
            elif validation_pool not in ("train", "test", "val", "validation"):
                errors.append(
                    f"❌ validation_pool must be one of: train, test, val, validation. Got '{validation_pool}'"
                )
        
        # Check validation_top_k (optional but should be valid if present)
        validation_top_k = eval_section.get("validation_top_k")
        if validation_top_k is not None:
            if not isinstance(validation_top_k, int):
                errors.append(f"❌ validation_top_k must be an integer, got {type(validation_top_k).__name__}")
            elif validation_top_k <= 0:
                errors.append(f"❌ validation_top_k must be > 0, got {validation_top_k}")
    
    # Validate rollout section
    rollout_section = gepa_section.get("rollout", {})
    if isinstance(rollout_section, dict):
        budget = rollout_section.get("budget")
        if budget is None:
            errors.append("❌ [prompt_learning.gepa.rollout].budget is required")
        elif not isinstance(budget, int):
            errors.append(f"❌ rollout.budget must be an integer, got {type(budget).__name__}")
        elif budget <= 0:
            errors.append(f"❌ rollout.budget must be > 0, got {budget}")
        
        max_concurrent = rollout_section.get("max_concurrent")
        if max_concurrent is not None:
            if not isinstance(max_concurrent, int):
                errors.append(f"❌ rollout.max_concurrent must be an integer, got {type(max_concurrent).__name__}")
            elif max_concurrent <= 0:
                errors.append(f"❌ rollout.max_concurrent must be > 0, got {max_concurrent}")
    
    # Validate mutation section
    mutation_section = gepa_section.get("mutation", {})
    if isinstance(mutation_section, dict):
        required_mutation_fields = ["llm_model", "llm_provider"]
        for field in required_mutation_fields:
            if not mutation_section.get(field):
                errors.append(f"❌ [prompt_learning.gepa.mutation].{field} is required")
        
        rate = mutation_section.get("rate")
        if rate is not None:
            if not isinstance(rate, (int, float)):
                errors.append(f"❌ mutation.rate must be a number, got {type(rate).__name__}")
            elif not (0.0 <= rate <= 1.0):
                errors.append(f"❌ mutation.rate must be between 0.0 and 1.0, got {rate}")
    
    # Validate population section
    population_section = gepa_section.get("population", {})
    if isinstance(population_section, dict):
        initial_size = population_section.get("initial_size")
        if initial_size is not None:
            if not isinstance(initial_size, int):
                errors.append(f"❌ population.initial_size must be an integer, got {type(initial_size).__name__}")
            elif initial_size <= 0:
                errors.append(f"❌ population.initial_size must be > 0, got {initial_size}")
        
        num_generations = population_section.get("num_generations")
        if num_generations is not None:
            if not isinstance(num_generations, int):
                errors.append(f"❌ population.num_generations must be an integer, got {type(num_generations).__name__}")
            elif num_generations <= 0:
                errors.append(f"❌ population.num_generations must be > 0, got {num_generations}")
    
    # Validate archive section
    archive_section = gepa_section.get("archive", {})
    if isinstance(archive_section, dict):
        max_size = archive_section.get("max_size")
        if max_size is not None:
            if not isinstance(max_size, int):
                errors.append(f"❌ archive.max_size must be an integer, got {type(max_size).__name__}")
            elif max_size < 0:
                errors.append(f"❌ archive.max_size must be >= 0, got {max_size}")
    
    # Validate token section
    token_section = gepa_section.get("token", {})
    if isinstance(token_section, dict):
        max_limit = token_section.get("max_limit")
        if max_limit is not None:
            if not isinstance(max_limit, int):
                errors.append(f"❌ token.max_limit must be an integer, got {type(max_limit).__name__}")
            elif max_limit <= 0:
                errors.append(f"❌ token.max_limit must be > 0, got {max_limit}")
    
    # Check initial_prompt section
    initial_prompt = pl_section.get("initial_prompt", {})
    if not isinstance(initial_prompt, dict):
        errors.append("❌ [prompt_learning.initial_prompt] section is missing or invalid")
    else:
        if not initial_prompt.get("id"):
            errors.append("❌ [prompt_learning.initial_prompt].id is required")
        if not initial_prompt.get("messages"):
            errors.append("❌ [prompt_learning.initial_prompt].messages is required (must be a list)")
        elif not isinstance(initial_prompt.get("messages"), list):
            errors.append("❌ [prompt_learning.initial_prompt].messages must be a list")
        elif len(initial_prompt.get("messages", [])) == 0:
            errors.append("❌ [prompt_learning.initial_prompt].messages cannot be empty")
    
    # Check policy section
    policy_section = pl_section.get("policy", {})
    if not isinstance(policy_section, dict):
        errors.append("❌ [prompt_learning.policy] section is missing or invalid")
    else:
        if not policy_section.get("model"):
            errors.append("❌ [prompt_learning.policy].model is required")
        if not policy_section.get("provider"):
            errors.append("❌ [prompt_learning.policy].provider is required")
    
    return len(errors) == 0, errors


def validate_mipro_config(config_path: Path) -> Tuple[bool, List[str]]:
    """Validate MIPRO config with comprehensive checks.
    
    Returns:
        (is_valid, errors) tuple where errors is a list of error messages
    """
    errors = []
    
    try:
        with open(config_path, 'r') as f:
            config_dict = toml.load(f)
    except Exception as e:
        return False, [f"Failed to parse TOML: {e}"]
    
    pl_section = config_dict.get("prompt_learning", {})
    if not isinstance(pl_section, dict):
        errors.append("❌ [prompt_learning] section is missing or invalid")
        return False, errors
    
    # Check algorithm
    algorithm = pl_section.get("algorithm")
    if algorithm != "mipro":
        errors.append(f"❌ Expected algorithm='mipro', got '{algorithm}'")
    
    # Check required top-level fields
    required_top_level = ["task_app_url", "task_app_api_key", "env_name"]
    for field in required_top_level:
        if not pl_section.get(field):
            errors.append(f"❌ [prompt_learning].{field} is required")
    
    # Check MIPRO section
    mipro_section = pl_section.get("mipro", {})
    if not isinstance(mipro_section, dict):
        errors.append("❌ [prompt_learning.mipro] section is missing or invalid")
        return False, errors
    
    # Check required MIPRO fields
    rollout_budget = mipro_section.get("rollout_budget")
    if rollout_budget is None:
        errors.append("❌ [prompt_learning.mipro].rollout_budget is required")
    elif not isinstance(rollout_budget, int):
        errors.append(f"❌ mipro.rollout_budget must be an integer, got {type(rollout_budget).__name__}")
    elif rollout_budget <= 0:
        errors.append(f"❌ mipro.rollout_budget must be > 0, got {rollout_budget}")
    
    # Check seeds section
    seeds_section = mipro_section.get("seeds", {})
    if not isinstance(seeds_section, dict):
        errors.append("❌ [prompt_learning.mipro.seeds] section is missing or invalid")
    else:
        # At least one seed type must be provided
        seed_types = ["bootstrap", "online", "test", "reference"]
        provided_seeds = [st for st in seed_types if seeds_section.get(st)]
        if len(provided_seeds) == 0:
            errors.append(
                "❌ At least one seed type must be provided in [prompt_learning.mipro.seeds]. "
                f"Expected one or more of: {', '.join(seed_types)}"
            )
        
        # Validate each seed type if present
        for seed_type in seed_types:
            seed_list = seeds_section.get(seed_type)
            if seed_list is not None:
                if not isinstance(seed_list, list):
                    errors.append(f"❌ mipro.seeds.{seed_type} must be a list, got {type(seed_list).__name__}")
                elif len(seed_list) == 0:
                    errors.append(f"❌ mipro.seeds.{seed_type} cannot be empty")
                elif not all(isinstance(s, int) for s in seed_list):
                    errors.append(f"❌ mipro.seeds.{seed_type} must contain only integers")
    
    # Check modules section
    modules_section = mipro_section.get("modules", [])
    if not isinstance(modules_section, list):
        errors.append("❌ [prompt_learning.mipro].modules must be a list")
    elif len(modules_section) == 0:
        errors.append("❌ [prompt_learning.mipro].modules cannot be empty")
    else:
        for i, module in enumerate(modules_section):
            if not isinstance(module, dict):
                errors.append(f"❌ mipro.modules[{i}] must be a dict")
                continue
            
            if not module.get("module_id"):
                errors.append(f"❌ mipro.modules[{i}].module_id is required")
            
            stages = module.get("stages", [])
            if not isinstance(stages, list):
                errors.append(f"❌ mipro.modules[{i}].stages must be a list")
            elif len(stages) == 0:
                errors.append(f"❌ mipro.modules[{i}].stages cannot be empty")
            else:
                for j, stage in enumerate(stages):
                    if not isinstance(stage, dict):
                        errors.append(f"❌ mipro.modules[{i}].stages[{j}] must be a dict")
                        continue
                    
                    if not stage.get("stage_id"):
                        errors.append(f"❌ mipro.modules[{i}].stages[{j}].stage_id is required")
                    
                    max_instruction_slots = stage.get("max_instruction_slots")
                    if max_instruction_slots is not None:
                        if not isinstance(max_instruction_slots, int):
                            errors.append(
                                f"❌ mipro.modules[{i}].stages[{j}].max_instruction_slots must be an integer, "
                                f"got {type(max_instruction_slots).__name__}"
                            )
                        elif max_instruction_slots <= 0:
                            errors.append(
                                f"❌ mipro.modules[{i}].stages[{j}].max_instruction_slots must be > 0, "
                                f"got {max_instruction_slots}"
                            )
                    
                    max_demo_slots = stage.get("max_demo_slots")
                    if max_demo_slots is not None:
                        if not isinstance(max_demo_slots, int):
                            errors.append(
                                f"❌ mipro.modules[{i}].stages[{j}].max_demo_slots must be an integer, "
                                f"got {type(max_demo_slots).__name__}"
                            )
                        elif max_demo_slots < 0:
                            errors.append(
                                f"❌ mipro.modules[{i}].stages[{j}].max_demo_slots must be >= 0, "
                                f"got {max_demo_slots}"
                            )
    
    # Check optional MIPRO fields for validity
    num_iterations = mipro_section.get("num_iterations")
    if num_iterations is not None:
        if not isinstance(num_iterations, int):
            errors.append(f"❌ mipro.num_iterations must be an integer, got {type(num_iterations).__name__}")
        elif num_iterations <= 0:
            errors.append(f"❌ mipro.num_iterations must be > 0, got {num_iterations}")
    
    num_evaluations_per_iteration = mipro_section.get("num_evaluations_per_iteration")
    if num_evaluations_per_iteration is not None:
        if not isinstance(num_evaluations_per_iteration, int):
            errors.append(
                f"❌ mipro.num_evaluations_per_iteration must be an integer, "
                f"got {type(num_evaluations_per_iteration).__name__}"
            )
        elif num_evaluations_per_iteration <= 0:
            errors.append(
                f"❌ mipro.num_evaluations_per_iteration must be > 0, "
                f"got {num_evaluations_per_iteration}"
            )
    
    batch_size = mipro_section.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int):
            errors.append(f"❌ mipro.batch_size must be an integer, got {type(batch_size).__name__}")
        elif batch_size <= 0:
            errors.append(f"❌ mipro.batch_size must be > 0, got {batch_size}")
    
    max_concurrent = mipro_section.get("max_concurrent")
    if max_concurrent is not None:
        if not isinstance(max_concurrent, int):
            errors.append(f"❌ mipro.max_concurrent must be an integer, got {type(max_concurrent).__name__}")
        elif max_concurrent <= 0:
            errors.append(f"❌ mipro.max_concurrent must be > 0, got {max_concurrent}")
    
    # Check meta section (optional but should be valid if present)
    meta_section = mipro_section.get("meta", {})
    if isinstance(meta_section, dict):
        if meta_section.get("model") and not meta_section.get("provider"):
            errors.append("❌ If mipro.meta.model is set, mipro.meta.provider must also be set")
        if meta_section.get("provider") and not meta_section.get("model"):
            errors.append("❌ If mipro.meta.provider is set, mipro.meta.model must also be set")
        
        temperature = meta_section.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append(f"❌ mipro.meta.temperature must be a number, got {type(temperature).__name__}")
            elif not (0.0 <= temperature <= 2.0):
                errors.append(f"❌ mipro.meta.temperature must be between 0.0 and 2.0, got {temperature}")
        
        max_tokens = meta_section.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                errors.append(f"❌ mipro.meta.max_tokens must be an integer, got {type(max_tokens).__name__}")
            elif max_tokens <= 0:
                errors.append(f"❌ mipro.meta.max_tokens must be > 0, got {max_tokens}")
    
    # Check policy section
    policy_section = pl_section.get("policy", {})
    if not isinstance(policy_section, dict):
        errors.append("❌ [prompt_learning.policy] section is missing or invalid")
    else:
        if not policy_section.get("model"):
            errors.append("❌ [prompt_learning.policy].model is required")
        if not policy_section.get("provider"):
            errors.append("❌ [prompt_learning.policy].provider is required")
    
    return len(errors) == 0, errors


def validate_config(config_path: Path, algorithm: str) -> None:
    """Validate config and raise ConfigValidationError with all errors if invalid.
    
    Args:
        config_path: Path to TOML config file
        algorithm: Either 'gepa' or 'mipro'
    
    Raises:
        ConfigValidationError: If validation fails, with detailed error messages
    """
    if algorithm == "gepa":
        is_valid, errors = validate_gepa_config(config_path)
    elif algorithm == "mipro":
        is_valid, errors = validate_mipro_config(config_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be 'gepa' or 'mipro'")
    
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ConfigValidationError(
            f"\n{'=' * 80}\n"
            f"❌ Config Validation Failed ({algorithm.upper()})\n"
            f"{'=' * 80}\n"
            f"{error_msg}\n"
            f"{'=' * 80}\n"
        )

