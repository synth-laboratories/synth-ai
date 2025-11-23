#!/usr/bin/env python3
"""Run GEPA in parallel for Crafter and Verilog tasks with gpt-oss-120b via Groq.

This script:
- Sets rollout limit to 200 and time limit to 3 minutes
- Runs Crafter and Verilog tasks in parallel
- Uses gpt-oss-120b via Groq
- Shows only aggregate stats table (masks everything else)
- Shows only candidate 1 lift (not candidate 2)
"""

import os
DEBUG_CONFIG = os.getenv("DEBUG_CONFIG", "false").lower() == "true"
DEBUG_TERMINATION = os.getenv("DEBUG_TERMINATION", "true").lower() == "true"

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

# Paths to config files
# Script is in langprobe/comparisons/, so go up to synth-ai root
# comparisons/ -> langprobe/ -> blog_posts/ -> examples/ -> synth-ai/
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
CONFIGS = {
    "Crafter": REPO_ROOT / "examples/blog_posts/langprobe/task_specific_agentic/crafter/crafter_gepa.toml",
    "Verilog": REPO_ROOT / "examples/blog_posts/langprobe/task_specific_agentic/verilog/verilog_gepa.toml",
}

# Load configuration from YAML file
COMPARISONS_DIR = Path(__file__).parent
CONFIG_FILE = COMPARISONS_DIR / "synth_gepa_config.yaml"

def load_config() -> Dict:
    """Load configuration from YAML file."""
    assert CONFIG_FILE.exists(), f"CRITICAL: Config file not found: {CONFIG_FILE}"
    
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    assert config is not None, f"CRITICAL: Failed to parse YAML config: {CONFIG_FILE}"
    
    # Set defaults if missing
    limits = config.get("limits", {})
    gepa_population = config.get("gepa_population", {})
    display = config.get("display", {})
    backend = config.get("backend", {})
    
    # ASSERT: All required limit values must be present
    assert "rollout_limit" in limits, f"CRITICAL: rollout_limit missing from limits section!"
    assert "time_limit_seconds" in limits, f"CRITICAL: time_limit_seconds missing from limits section!"
    assert "max_trials" in limits, f"CRITICAL: max_trials missing from limits section!"
    assert "max_cost_usd" in limits, f"CRITICAL: max_cost_usd missing from limits section!"
    
    # ASSERT: All required gepa_population values must be present
    assert "num_generations" in gepa_population, f"CRITICAL: num_generations missing from gepa_population section!"
    assert "children_per_generation" in gepa_population, f"CRITICAL: children_per_generation missing from gepa_population section!"
    
    # ASSERT: Values must be positive
    assert limits["rollout_limit"] > 0, f"CRITICAL: rollout_limit must be > 0, got {limits['rollout_limit']}"
    assert limits["time_limit_seconds"] > 0, f"CRITICAL: time_limit_seconds must be > 0, got {limits['time_limit_seconds']}"
    assert limits["max_trials"] > 0, f"CRITICAL: max_trials must be > 0, got {limits['max_trials']}"
    assert limits["max_cost_usd"] > 0, f"CRITICAL: max_cost_usd must be > 0, got {limits['max_cost_usd']}"
    assert gepa_population["num_generations"] > 0, f"CRITICAL: num_generations must be > 0, got {gepa_population['num_generations']}"
    assert gepa_population["children_per_generation"] > 0, f"CRITICAL: children_per_generation must be > 0, got {gepa_population['children_per_generation']}"
    
    result = {
        "rollout_limit": limits["rollout_limit"],
        "time_limit_seconds": limits["time_limit_seconds"],
        "max_trials": limits["max_trials"],
        "max_cost_usd": limits["max_cost_usd"],
        "num_generations": gepa_population["num_generations"],
        "children_per_generation": gepa_population["children_per_generation"],
        "tui": display.get("tui", False),
        "show_curve": display.get("show_curve", False),
        "verbose_summary": display.get("verbose_summary", False),
        "local_backend": backend.get("local_backend", True),
    }
    
    if DEBUG_TERMINATION:
        print(f"[DEBUG] Loaded config from {CONFIG_FILE}:")
        for key, value in result.items():
            print(f"  {key} = {value}")
    
    return result


def modify_config_for_limits(
    config_path: Path,
    rollout_limit: int,
    time_limit_seconds: int,
    max_trials: int,
    max_cost_usd: float,
    num_generations: int = 1,
    children_per_generation: int = 5,
    tui: bool = False,
    show_curve: bool = False,
    verbose_summary: bool = False,
    local_backend: bool = True,
) -> Path:
    """Create a temporary modified config with rollout and time limits, and set model to gpt-oss-120b via Groq."""
    # Read original config as text
    with open(config_path, "r") as f:
        config_text = f.read()
    
    # Resolve env_file_path relative to original config directory
    config_dir = config_path.parent.resolve()
    repo_root = config_path.parent
    # Go up to find repo root (where .env is)
    for _ in range(6):  # Max depth
        if (repo_root / ".env").exists():
            break
        repo_root = repo_root.parent
    else:
        # Fallback: use synth-ai repo root
        repo_root = Path(__file__).parent
    
    env_file_abs = (repo_root / ".env").resolve()
    
    # Resolve results_folder to absolute path (relative to original config directory)
    results_folder_abs = None
    results_folder_updated = False
    
    # Extract initial_size from config to adjust max_trials
    initial_size = 5  # Default
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None
    
    if tomllib:
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            pl_config = config.get("prompt_learning", {})
            gepa_config = pl_config.get("gepa", {})
            gepa_population = gepa_config.get("population", {})
            initial_size = gepa_population.get("initial_size", 5)
        except Exception:
            pass
    
    # Adjust max_trials to account for initial population
    effective_max_trials = max(max_trials, initial_size + (num_generations * children_per_generation))
    
    # Add/modify termination_config section
    lines = config_text.split("\n")
    new_lines = []
    in_termination_config = False
    termination_config_added = False
    in_prompt_learning = False
    in_gepa_policy = False
    in_policy = False
    env_file_path_updated = False
    policy_model_updated = False
    policy_provider_updated = False
    
    for i, line in enumerate(lines):
        # Update env_file_path to absolute path
        if "[prompt_learning]" in line:
            in_prompt_learning = True
            new_lines.append(line)
            continue
        
        if in_prompt_learning and "env_file_path" in line and "=" in line and not env_file_path_updated:
            # Replace relative path with absolute path
            new_lines.append(f'env_file_path = "{env_file_abs}"')
            env_file_path_updated = True
            continue
        
        if in_prompt_learning and "results_folder" in line and "=" in line and not results_folder_updated:
            # Extract current results_folder value
            match = re.search(r'results_folder\s*=\s*["\']?([^"\']+)["\']?', line)
            if match:
                results_folder_rel = match.group(1).strip()
                # Resolve relative to original config directory
                if not Path(results_folder_rel).is_absolute():
                    results_folder_abs = (config_dir / results_folder_rel).resolve()
                else:
                    results_folder_abs = Path(results_folder_rel).expanduser().resolve()
                # Replace with absolute path
                new_lines.append(f'results_folder = "{results_folder_abs}"')
                results_folder_updated = True
                continue
        
        # Track policy sections
        if "[prompt_learning.gepa.policy]" in line:
            in_gepa_policy = True
            in_policy = False
            new_lines.append(line)
            continue
        
        if "[prompt_learning.policy]" in line:
            in_policy = True
            in_gepa_policy = False
            new_lines.append(line)
            continue
        
        # Update policy model to gpt-oss-120b (via Groq, but model name is openai/gpt-oss-120b)
        if in_gepa_policy or in_policy:
            if "model" in line and "=" in line and not policy_model_updated:
                new_lines.append('model = "openai/gpt-oss-120b"')
                policy_model_updated = True
                continue
            if "provider" in line and "=" in line and not policy_provider_updated:
                new_lines.append('provider = "groq"')
                policy_provider_updated = True
                continue
            # Exit policy section when we hit a new section
            if line.strip().startswith("[") and "[prompt_learning.gepa.policy" not in line and "[prompt_learning.policy" not in line:
                in_gepa_policy = False
                in_policy = False
                # Fall through to append the new section header
        
        if in_prompt_learning and line.strip().startswith("[") and "[prompt_learning" not in line:
            in_prompt_learning = False
            in_gepa_policy = False
            in_policy = False
        
        if "[prompt_learning.termination_config]" in line:
            in_termination_config = True
            new_lines.append(line)
            continue
        
        if in_termination_config:
            if line.strip().startswith("[") and not line.strip().startswith("[prompt_learning"):
                # End of termination_config section
                # Add our limits if not already present
                recent_lines = "\n".join(new_lines[-15:])
                if f"max_rollouts" not in recent_lines or f"max_rollouts = {rollout_limit}" not in recent_lines:
                    new_lines.append(f"max_rollouts = {rollout_limit}")
                if f"max_seconds" not in recent_lines or f"max_seconds = {time_limit_seconds}" not in recent_lines:
                    new_lines.append(f"max_seconds = {time_limit_seconds}")
                if f"max_trials" not in recent_lines or f"max_trials = {effective_max_trials}" not in recent_lines:
                    new_lines.append(f"max_trials = {effective_max_trials}")
                if f"max_cost_usd" not in recent_lines or f"max_cost_usd = {max_cost_usd}" not in recent_lines:
                    new_lines.append(f"max_cost_usd = {max_cost_usd}")
                in_termination_config = False
                termination_config_added = True
            elif "max_rollouts" in line or "max_seconds" in line or "max_trials" in line or "max_cost_usd" in line:
                # Replace existing values - match any format (with or without spaces around =)
                if "max_rollouts" in line:
                    new_lines.append(f"max_rollouts = {rollout_limit}")
                elif "max_seconds" in line:
                    new_lines.append(f"max_seconds = {time_limit_seconds}")
                elif "max_trials" in line:
                    new_lines.append(f"max_trials = {effective_max_trials}")
                elif "max_cost_usd" in line:
                    new_lines.append(f"max_cost_usd = {max_cost_usd}")
                continue  # Skip adding the original line
        
        new_lines.append(line)
    
    # Ensure env_file_path is set if not found
    if not env_file_path_updated:
        # Find [prompt_learning] section and add env_file_path after it
        for i, line in enumerate(new_lines):
            if "[prompt_learning]" in line:
                # Insert after the section header
                new_lines.insert(i + 1, f'env_file_path = "{env_file_abs}"')
                break
    
    # Ensure policy model and provider are set to openai/gpt-oss-120b (via Groq)
    if not policy_model_updated or not policy_provider_updated:
        # Find gepa.policy section or policy section
        gepa_policy_found = False
        policy_found = False
        for i, line in enumerate(new_lines):
            if "[prompt_learning.gepa.policy]" in line:
                gepa_policy_found = True
                # Insert model and provider after section header
                insert_idx = i + 1
                if not policy_model_updated:
                    new_lines.insert(insert_idx, 'model = "openai/gpt-oss-120b"')
                    policy_model_updated = True
                    insert_idx += 1
                if not policy_provider_updated:
                    new_lines.insert(insert_idx, 'provider = "groq"')
                    policy_provider_updated = True
                break
            elif "[prompt_learning.policy]" in line:
                policy_found = True
                # Insert model and provider after section header
                insert_idx = i + 1
                if not policy_model_updated:
                    new_lines.insert(insert_idx, 'model = "openai/gpt-oss-120b"')
                    policy_model_updated = True
                    insert_idx += 1
                if not policy_provider_updated:
                    new_lines.insert(insert_idx, 'provider = "groq"')
                    policy_provider_updated = True
                break
        
        # If neither section exists, add gepa.policy section
        if not gepa_policy_found and not policy_found:
            # Find [prompt_learning.gepa] section
            for i, line in enumerate(new_lines):
                if "[prompt_learning.gepa]" in line:
                    # Insert gepa.policy section after gepa section
                    insert_idx = i + 1
                    new_lines.insert(insert_idx, "")
                    new_lines.insert(insert_idx + 1, "[prompt_learning.gepa.policy]")
                    new_lines.insert(insert_idx + 2, 'model = "openai/gpt-oss-120b"')
                    new_lines.insert(insert_idx + 3, 'provider = "groq"')
                    policy_model_updated = True
                    policy_provider_updated = True
                    break
    
    # Ensure results_folder is set to absolute path if not found or not updated
    if not results_folder_updated:
        # Try to find existing results_folder in original config
        match = re.search(r'results_folder\s*=\s*["\']?([^"\']+)["\']?', config_text)
        if match:
            results_folder_rel = match.group(1).strip()
            if not Path(results_folder_rel).is_absolute():
                results_folder_abs = (config_dir / results_folder_rel).resolve()
            else:
                results_folder_abs = Path(results_folder_rel).expanduser().resolve()
        else:
            # Default: results folder relative to original config
            results_folder_abs = (config_dir / "results").resolve()
        
        # Find [prompt_learning] section and add/update results_folder
        for i, line in enumerate(new_lines):
            if "[prompt_learning]" in line:
                # Check if results_folder already exists in next few lines
                found = False
                for j in range(i + 1, min(i + 10, len(new_lines))):
                    if "results_folder" in new_lines[j] and "=" in new_lines[j]:
                        # Update existing line
                        new_lines[j] = f'results_folder = "{results_folder_abs}"'
                        found = True
                        break
                if not found:
                    # Insert after env_file_path or section header
                    insert_idx = i + 1
                    if env_file_path_updated:
                        # Find env_file_path line
                        for j in range(i + 1, min(i + 10, len(new_lines))):
                            if "env_file_path" in new_lines[j]:
                                insert_idx = j + 1
                                break
                    new_lines.insert(insert_idx, f'results_folder = "{results_folder_abs}"')
                break
    
    # If termination_config section doesn't exist, add it
    if not termination_config_added:
        # Find where to insert it (after gepa section or at end)
        insert_idx = len(new_lines)
        for i, line in enumerate(new_lines):
            if "[prompt_learning.gepa.token]" in line or "[display]" in line:
                insert_idx = i
                break
        
        # Insert termination_config section
        new_lines.insert(insert_idx, "")
        new_lines.insert(insert_idx + 1, "[prompt_learning.termination_config]")
        new_lines.insert(insert_idx + 2, f"max_rollouts = {rollout_limit}")
        new_lines.insert(insert_idx + 3, f"max_seconds = {time_limit_seconds}")
        # Keep existing max_cost_usd and max_trials if present
        if "max_cost_usd" not in config_text:
            new_lines.insert(insert_idx + 4, f"max_cost_usd = {max_cost_usd}")
        if "max_trials" not in config_text:
            new_lines.insert(insert_idx + 5, f"max_trials = {effective_max_trials}")
    
    # Update gepa.population section with configured values
    gepa_population_updated = False
    for i, line in enumerate(new_lines):
        if "[prompt_learning.gepa.population]" in line:
            gepa_population_updated = True
            # Update num_generations and children_per_generation in this section
            for j in range(i + 1, min(i + 10, len(new_lines))):
                next_line = new_lines[j]
                if next_line.strip().startswith("[") and "[prompt_learning.gepa.population" not in next_line:
                    break
                if "num_generations" in next_line and "=" in next_line:
                    new_lines[j] = f"num_generations = {num_generations}"
                elif "children_per_generation" in next_line and "=" in next_line:
                    new_lines[j] = f"children_per_generation = {children_per_generation}"
            break
    
    # Add gepa.population section if it doesn't exist
    if not gepa_population_updated:
        # Find where to insert it (after gepa section or before termination_config)
        insert_idx = len(new_lines)
        for i, line in enumerate(new_lines):
            if "[prompt_learning.gepa.token]" in line or "[prompt_learning.termination_config]" in line:
                insert_idx = i
                break
        
        # Insert gepa.population section
        new_lines.insert(insert_idx, "")
        new_lines.insert(insert_idx + 1, "[prompt_learning.gepa.population]")
        new_lines.insert(insert_idx + 2, f"num_generations = {num_generations}")
        new_lines.insert(insert_idx + 3, f"children_per_generation = {children_per_generation}")
    
    # Also update rollout budget
    for i, line in enumerate(new_lines):
        if "[prompt_learning.gepa.rollout]" in line:
            # Find budget line and update it
            for j in range(i + 1, min(i + 5, len(new_lines))):
                if "budget" in new_lines[j] and "=" in new_lines[j]:
                    new_lines[j] = f"budget = {rollout_limit}"
                    break
            break
    
    # Ensure display section exists and uses config values
    display_section_exists = False
    for i, line in enumerate(new_lines):
        if "[display]" in line:
            display_section_exists = True
            # Update display settings from config
            for j in range(i + 1, min(i + 10, len(new_lines))):
                next_line = new_lines[j]
                if next_line.strip().startswith("[") and "[display" not in next_line:
                    break
                if "tui" in next_line and "=" in next_line:
                    new_lines[j] = f"tui = {str(tui).lower()}"
                elif "show_curve" in next_line and "=" in next_line:
                    new_lines[j] = f"show_curve = {str(show_curve).lower()}"
                elif "verbose_summary" in next_line and "=" in next_line:
                    new_lines[j] = f"verbose_summary = {str(verbose_summary).lower()}"
                elif "local_backend" in next_line and "=" in next_line:
                    new_lines[j] = f"local_backend = {str(local_backend).lower()}"
            break
    
    # Add display section if it doesn't exist
    if not display_section_exists:
        # Insert before END or at end
        insert_idx = len(new_lines)
        for i, line in enumerate(new_lines):
            if line.strip().startswith("[") and "[prompt_learning" not in line:
                insert_idx = i
                break
        new_lines.insert(insert_idx, "")
        new_lines.insert(insert_idx + 1, "[display]")
        new_lines.insert(insert_idx + 2, f"tui = {str(tui).lower()}")
        new_lines.insert(insert_idx + 3, f"show_curve = {str(show_curve).lower()}")
        new_lines.insert(insert_idx + 4, f"verbose_summary = {str(verbose_summary).lower()}")
        new_lines.insert(insert_idx + 5, f"local_backend = {str(local_backend).lower()}")
    
    # Write to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write("\n".join(new_lines))
    temp_file.close()
    
    # Debug: verify limits were set
    config_content = "\n".join(new_lines)
    
    # ASSERT: Validate that all termination config values are present
    validation_errors = []
    if f"max_rollouts = {rollout_limit}" not in config_content:
        validation_errors.append(f"max_rollouts = {rollout_limit}")
    if f"max_seconds = {time_limit_seconds}" not in config_content:
        validation_errors.append(f"max_seconds = {time_limit_seconds}")
    if f"max_trials = {effective_max_trials}" not in config_content:
        validation_errors.append(f"max_trials = {effective_max_trials}")
    if f"max_cost_usd = {max_cost_usd}" not in config_content:
        validation_errors.append(f"max_cost_usd = {max_cost_usd}")
    
    # ASSERT: All termination config values must be present
    assert not validation_errors, f"CRITICAL: Missing termination config values: {validation_errors}\nConfig preview:\n{config_content[-1000:]}"
    
    # ASSERT: Verify model is set to openai/gpt-oss-120b (via Groq)
    assert 'model = "openai/gpt-oss-120b"' in config_content or 'model = "openai\\/gpt-oss-120b"' in config_content, \
        f"CRITICAL: Model not set to openai/gpt-oss-120b!\nConfig preview:\n{config_content[-1000:]}"
    assert 'provider = "groq"' in config_content, \
        f"CRITICAL: Provider not set to groq!\nConfig preview:\n{config_content[-1000:]}"
    
    if DEBUG_TERMINATION or validation_errors:
        print(f"[DEBUG] ✅ Termination config validation passed for {config_path.name}")
        print(f"  max_rollouts = {rollout_limit}")
        print(f"  max_seconds = {time_limit_seconds}")
        print(f"  max_trials = {effective_max_trials} (adjusted from {max_trials} to account for initial_size={initial_size})")
        print(f"  max_cost_usd = {max_cost_usd}")
        print(f"  model = openai/gpt-oss-120b")
        print(f"  provider = groq")
    
    # ASSERT: Validate gepa.population values
    assert f"num_generations = {num_generations}" in config_content, \
        f"CRITICAL: num_generations = {num_generations} not found in config!\nConfig preview:\n{config_content[-1000:]}"
    assert f"children_per_generation = {children_per_generation}" in config_content, \
        f"CRITICAL: children_per_generation = {children_per_generation} not found in config!\nConfig preview:\n{config_content[-1000:]}"
    
    if DEBUG_TERMINATION:
        print(f"  num_generations = {num_generations}")
        print(f"  children_per_generation = {children_per_generation}")
    
    # ASSERT: Verify termination_config section exists
    assert "[prompt_learning.termination_config]" in config_content, \
        f"CRITICAL: [prompt_learning.termination_config] section not found!\nConfig preview:\n{config_content[-1000:]}"
    
    # Extract and verify actual values from config
    termination_section_match = re.search(
        r'\[prompt_learning\.termination_config\](.*?)(?=\n\[|\Z)',
        config_content,
        re.DOTALL
    )
    assert termination_section_match, "CRITICAL: Could not extract termination_config section!"
    termination_section = termination_section_match.group(1)
    
    # ASSERT: Verify each value appears exactly once in termination_config section
    max_rollouts_count = termination_section.count(f"max_rollouts = {rollout_limit}")
    max_seconds_count = termination_section.count(f"max_seconds = {time_limit_seconds}")
    max_trials_count = termination_section.count(f"max_trials = {effective_max_trials}")
    max_cost_count = termination_section.count(f"max_cost_usd = {max_cost_usd}")
    
    assert max_rollouts_count == 1, f"CRITICAL: max_rollouts appears {max_rollouts_count} times (expected 1)!\nSection:\n{termination_section}"
    assert max_seconds_count == 1, f"CRITICAL: max_seconds appears {max_seconds_count} times (expected 1)!\nSection:\n{termination_section}"
    assert max_trials_count == 1, f"CRITICAL: max_trials = {effective_max_trials} appears {max_trials_count} times (expected 1)!\nSection:\n{termination_section}"
    assert max_cost_count == 1, f"CRITICAL: max_cost_usd appears {max_cost_count} times (expected 1)!\nSection:\n{termination_section}"
    
    # ASSERT: Verify no old/high values remain
    old_max_trials_pattern = r'max_trials\s*=\s*(?!' + str(effective_max_trials) + r')\d+'
    old_matches = re.findall(old_max_trials_pattern, termination_section)
    assert not old_matches, f"CRITICAL: Found old max_trials values in termination_config: {old_matches}\nSection:\n{termination_section}"
    
    if DEBUG_CONFIG:
        print(f"\n[DEBUG] Full generated config for {config_path.name}:")
        print("=" * 80)
        print(config_content)
        print("=" * 80)
    
    return temp_path


# Import the rest of the functions from the original file
# (extract_results_from_file, extract_results_from_events, run_gepa_job, main)
# We'll copy them as-is since they don't need modification

def extract_results_from_file(task_name: str, job_id: str, config_path: Optional[Path] = None) -> Dict:
    """Extract results from results file by job_id. Never reads old files."""
    print(f"[{task_name}] Extracting results from file for job_id: {job_id}")
    
    if not job_id:
        print(f"[{task_name}] ERROR: No job_id provided")
        return {"error": "No job_id provided"}
    
    if not config_path:
        # Try to find config by task name
        config_path = CONFIGS.get(task_name)
    
    if not config_path or not config_path.exists():
        print(f"[{task_name}] ERROR: Config not found: {config_path}")
        return {"error": "Config not found"}
    
    # Extract policy model from config
    policy_model = None
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        
        if tomllib:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            pl_config = config.get("prompt_learning", {})
            # Try gepa.policy first, then policy
            gepa_config = pl_config.get("gepa", {})
            gepa_policy_config = gepa_config.get("policy", {})
            if gepa_policy_config and isinstance(gepa_policy_config, dict) and gepa_policy_config.get("model"):
                policy_config = gepa_policy_config
            else:
                policy_config = pl_config.get("policy", {})
            policy_model = policy_config.get("model") if isinstance(policy_config, dict) else None
            if policy_model:
                print(f"[{task_name}] Found policy_model: {policy_model}")
        else:
            # Fallback: try to parse manually - check both gepa.policy and policy
            with open(config_path, "r") as f:
                content = f.read()
            # Try gepa.policy first
            match = re.search(r'\[prompt_learning\.gepa\.policy\].*?model\s*=\s*["\']?([^"\'\n]+)', content, re.DOTALL)
            if not match:
                # Fallback to policy
                match = re.search(r'\[prompt_learning\.policy\].*?model\s*=\s*["\']?([^"\'\n]+)', content, re.DOTALL)
            if match:
                policy_model = match.group(1).strip()
                print(f"[{task_name}] Found policy_model (manual parse): {policy_model}")
    except Exception as e:
        print(f"[{task_name}] WARNING: Could not extract policy model: {e}")
    
    results_folder = config_path.parent / "results"
    if not results_folder.exists():
        print(f"[{task_name}] ERROR: Results folder not found: {results_folder}")
        return {"error": "Results folder not found"}
    
    # Find results file by job_id (format: gepa_results_pl_xxxxx_timestamp.txt)
    result_file = None
    all_files = list(results_folder.glob("gepa_results_*.txt"))
    print(f"[{task_name}] Searching {len(all_files)} results files for job_id {job_id}")
    
    for file in all_files:
        # Extract job_id from filename
        match = re.search(r'pl_[a-f0-9]+', file.name)
        if match:
            file_job_id = match.group(0)
            print(f"[{task_name}] Checking file {file.name} with job_id {file_job_id}")
            if file_job_id == job_id:
                result_file = file
                print(f"[{task_name}] Found matching file: {file.name}")
                break
    
    if not result_file:
        print(f"[{task_name}] ERROR: No results file found for job_id {job_id}")
        print(f"[{task_name}] Available files: {[f.name for f in all_files[:5]]}")
        return {"error": f"No results file found for job_id {job_id}"}
    
    # Parse results file (same as original)
    try:
        print(f"[{task_name}] Reading results file: {result_file}")
        with open(result_file, "r") as f:
            content = f.read()
        
        print(f"[{task_name}] File size: {len(content)} chars")
        
        # Extract baseline and candidate scores
        baseline_score = None
        candidate1_score = None
        candidate1_lift = None
        total_cost = None
        total_rollouts = None
        total_time = None
        total_tokens = None
        eval_seeds_n = None
        
        # Look for baseline score (format: "📊 Baseline Score: 0.2600 (26.0%)")
        baseline_match = re.search(r'Baseline Score:\s*([\d.]+)', content)
        if baseline_match:
            baseline_score = float(baseline_match.group(1))
            print(f"[{task_name}] Found baseline_score: {baseline_score}")
        else:
            print(f"[{task_name}] WARNING: No baseline score found")
        
        # Extract cost (format: "Total Cost: $0.1234")
        cost_match = re.search(r'Total Cost:\s*\$?([\d.]+)', content)
        if cost_match:
            total_cost = float(cost_match.group(1))
            print(f"[{task_name}] Found total_cost: ${total_cost}")
        
        # Extract rollouts (format: "Rollouts: N: 50 | Tokens: ..." or "Rollouts: 50")
        rollouts_match = re.search(r'Rollouts:.*?N:\s*(\d+)', content, re.IGNORECASE)
        if not rollouts_match:
            # Fallback: try simple format
            rollouts_match = re.search(r'(?:Total\s+)?Rollouts:\s*(\d+)', content)
        if rollouts_match:
            total_rollouts = int(rollouts_match.group(1))
            print(f"[{task_name}] Found total_rollouts: {total_rollouts}")
        
        # Extract time (format: "Time: 45.2s" or "Total Time: 45.2s (0.8 min)")
        time_match = re.search(r'Time:\s*([\d.]+)s', content)
        if not time_match:
            # Fallback: try with "Total"
            time_match = re.search(r'Total\s+Time:\s*([\d.]+)s', content)
        if time_match:
            total_time = float(time_match.group(1))
            print(f"[{task_name}] Found total_time: {total_time}s")
        
        # Extract tokens (format: "Rollouts: N: 50 | Tokens: 1.2345M" or "Tokens: 1.2345M")
        tokens_match = re.search(r'Rollouts:.*?Tokens:\s*([\d.]+)([KMkm]?)', content, re.IGNORECASE)
        if not tokens_match:
            # Fallback: try standalone tokens
            tokens_match = re.search(r'(?:Total\s+)?Tokens:\s*([\d.]+)([KMkm]?)', content)
        if tokens_match:
            tokens_val = float(tokens_match.group(1))
            unit = tokens_match.group(2).upper() if tokens_match.group(2) else ""
            if unit == "K":
                total_tokens = int(tokens_val * 1000)
            elif unit == "M":
                total_tokens = int(tokens_val * 1000000)
            else:
                total_tokens = int(tokens_val)
            print(f"[{task_name}] Found total_tokens: {total_tokens}")
        
        # Extract eval seeds N (look for "Heldout Evaluation" section with N= or seeds count)
        eval_n_match = re.search(r'(?:Heldout Evaluation|Validation Summary|FINAL SUMMARY).*?N\s*=\s*(\d+)', content, re.DOTALL | re.IGNORECASE)
        if not eval_n_match:
            # Try to find in baseline section
            baseline_n_match = re.search(r'Baseline.*?N\s*=\s*(\d+)', content, re.DOTALL | re.IGNORECASE)
            if baseline_n_match:
                eval_seeds_n = int(baseline_n_match.group(1))
                print(f"[{task_name}] Found eval_seeds_n from baseline: {eval_seeds_n}")
            else:
                # Try validation summary format: "Baseline: 0.3000\nN=2"
                validation_n_match = re.search(r'Baseline:\s*[\d.]+\s*\n\s*N\s*=\s*(\d+)', content, re.IGNORECASE)
                if validation_n_match:
                    eval_seeds_n = int(validation_n_match.group(1))
                    print(f"[{task_name}] Found eval_seeds_n from validation summary: {eval_seeds_n}")
        else:
            eval_seeds_n = int(eval_n_match.group(1))
            print(f"[{task_name}] Found eval_seeds_n: {eval_seeds_n}")
        
        # Look for validation section with Candidate 1 in FINAL SUMMARY
        validation_section = re.search(r'FINAL SUMMARY.*?Candidate 1.*?Accuracy:\s*([\d.]+).*?\(Δ([+-]?[\d.]+)', content, re.DOTALL)
        if validation_section:
            candidate1_score = float(validation_section.group(1))
            candidate1_lift = float(validation_section.group(2))
            print(f"[{task_name}] Found candidate1_score from FINAL SUMMARY: {candidate1_score}, lift: {candidate1_lift}")
        else:
            # Also try Heldout Evaluation section format
            heldout_section = re.search(r'Heldout Evaluation.*?Candidate 1.*?Accuracy:\s*([\d.]+).*?\(Δ([+-]?[\d.]+)', content, re.DOTALL)
            if heldout_section:
                candidate1_score = float(heldout_section.group(1))
                candidate1_lift = float(heldout_section.group(2))
                print(f"[{task_name}] Found candidate1_score from Heldout Evaluation: {candidate1_score}, lift: {candidate1_lift}")
            else:
                # Fallback: look for best score if no validation section
                best_match = re.search(r'Best Score:\s*([\d.]+)', content)
                if best_match:
                    candidate1_score = float(best_match.group(1))
                    if baseline_score is not None:
                        candidate1_lift = candidate1_score - baseline_score
                    print(f"[{task_name}] Found candidate1_score from Best Score: {candidate1_score}, lift: {candidate1_lift}")
                else:
                    print(f"[{task_name}] WARNING: No candidate score found")
        
        result = {
            "baseline_score": baseline_score,
            "candidate1_score": candidate1_score,
            "candidate1_lift": candidate1_lift,
            "total_cost": total_cost,
            "total_rollouts": total_rollouts,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "eval_seeds_n": eval_seeds_n,
            "policy_model": policy_model,
        }
        print(f"[{task_name}] Extracted results: {result}")
        return result
    except Exception as e:
        print(f"[{task_name}] ERROR parsing results file: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to parse results file: {e}"}


async def run_gepa_job(task_name: str, config_path: Path, config: Dict) -> Dict:
    """Run a single GEPA job and return results."""
    import time
    
    print(f"[{task_name}] Starting job...")
    
    # Record start time to find new results files
    start_time = time.time()
    
    # ASSERT: Config values are valid before creating temp config
    assert config["rollout_limit"] > 0, f"CRITICAL: Invalid rollout_limit: {config['rollout_limit']}"
    assert config["time_limit_seconds"] > 0, f"CRITICAL: Invalid time_limit_seconds: {config['time_limit_seconds']}"
    assert config["max_trials"] > 0, f"CRITICAL: Invalid max_trials: {config['max_trials']}"
    assert config["max_cost_usd"] > 0, f"CRITICAL: Invalid max_cost_usd: {config['max_cost_usd']}"
    assert config["num_generations"] > 0, f"CRITICAL: Invalid num_generations: {config['num_generations']}"
    assert config["children_per_generation"] > 0, f"CRITICAL: Invalid children_per_generation: {config['children_per_generation']}"
    
    if DEBUG_TERMINATION:
        print(f"[{task_name}] Creating temp config with limits:")
        print(f"  rollout_limit={config['rollout_limit']}")
        print(f"  time_limit_seconds={config['time_limit_seconds']}")
        print(f"  max_trials={config['max_trials']}")
        print(f"  max_cost_usd={config['max_cost_usd']}")
        print(f"  num_generations={config['num_generations']}")
        print(f"  children_per_generation={config['children_per_generation']}")
    
    # Create modified config with limits from YAML config and gpt-oss-120b via Groq
    temp_config = modify_config_for_limits(
        config_path,
        rollout_limit=config["rollout_limit"],
        time_limit_seconds=config["time_limit_seconds"],
        max_trials=config["max_trials"],
        max_cost_usd=config["max_cost_usd"],
        num_generations=config["num_generations"],
        children_per_generation=config["children_per_generation"],
        tui=config["tui"],
        show_curve=config["show_curve"],
        verbose_summary=config["verbose_summary"],
        local_backend=config["local_backend"],
    )
    
    # ASSERT: Temp config file must exist
    assert temp_config.exists(), f"CRITICAL: Temp config file not created: {temp_config}"
    
    # ASSERT: Verify temp config contains our values by reading it back
    with open(temp_config, "r") as f:
        temp_config_content = f.read()
    
    assert f"max_rollouts = {config['rollout_limit']}" in temp_config_content, \
        f"CRITICAL: Temp config missing max_rollouts!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert f"max_seconds = {config['time_limit_seconds']}" in temp_config_content, \
        f"CRITICAL: Temp config missing max_seconds!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    # Note: effective_max_trials might be >= config['max_trials'] due to initial_size adjustment
    max_trials_match = re.search(r'max_trials\s*=\s*(\d+)', temp_config_content)
    assert max_trials_match, f"CRITICAL: Temp config missing max_trials!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    actual_max_trials = int(max_trials_match.group(1))
    assert actual_max_trials >= config['max_trials'], \
        f"CRITICAL: Temp config max_trials ({actual_max_trials}) < requested ({config['max_trials']})!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert f"max_cost_usd = {config['max_cost_usd']}" in temp_config_content, \
        f"CRITICAL: Temp config missing max_cost_usd!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert f"num_generations = {config['num_generations']}" in temp_config_content, \
        f"CRITICAL: Temp config missing num_generations!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert f"children_per_generation = {config['children_per_generation']}" in temp_config_content, \
        f"CRITICAL: Temp config missing children_per_generation!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert 'model = "openai/gpt-oss-120b"' in temp_config_content or 'model = "openai\\/gpt-oss-120b"' in temp_config_content, \
        f"CRITICAL: Temp config missing openai/gpt-oss-120b model!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    assert 'provider = "groq"' in temp_config_content, \
        f"CRITICAL: Temp config missing groq provider!\nFile: {temp_config}\nContent preview:\n{temp_config_content[-500:]}"
    
    print(f"[{task_name}] ✅ Created and validated temp config: {temp_config}")
    
    # Get timestamp before running to identify new results files
    results_folder = config_path.parent / "results"
    existing_files = set()
    if results_folder.exists():
        existing_files = {f.name for f in results_folder.glob("gepa_results_*.txt")}
        print(f"[{task_name}] Found {len(existing_files)} existing results files")
    
    try:
        # Find the .env file path from the temp config
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None
        
        if not tomllib:
            raise ImportError("tomllib or tomli not available")
        
        with temp_config.open("rb") as f:
            config_data = tomllib.load(f)
        env_file_path = None
        if "prompt_learning" in config_data:
            env_file_path = config_data["prompt_learning"].get("env_file_path")
        if not env_file_path:
            env_file_path = config_data.get("env_file_path")
        
        # Submit job and get job_id, then poll in background
        # Use uv run to ensure correct Python environment
        cmd = [
            "uv", "run", "python", "-m", "synth_ai", "train",
            str(temp_config),  # Config path is positional argument
            "--no-poll",  # Don't wait, just submit
            "--local-backend",  # Use local backend (localhost:8000)
        ]
        
        # Add --env if we have an env_file_path to avoid prompts
        if env_file_path:
            cmd.extend(["--env", str(env_file_path)])
        
        print(f"[{task_name}] Submitting job: {' '.join(cmd[:3])} ...")
        print(f"[{task_name}] Full command: {' '.join(cmd)}")
        
        # Submit job and capture job_id
        stdout_lines = []
        stderr_lines = []
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # Stream stdout and stderr to capture job_id
        async def read_stream(stream, lines_list, prefix):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode('utf-8', errors='replace').rstrip()
                lines_list.append(decoded)
                # Print important lines
                if "job_id" in decoded.lower() or "pl_" in decoded or "submitted" in decoded.lower():
                    print(f"[{task_name}] {prefix}: {decoded}")
        
        # Read both streams concurrently
        await asyncio.gather(
            read_stream(process.stdout, stdout_lines, "STDOUT"),
            read_stream(process.stderr, stderr_lines, "STDERR"),
        )
        
        return_code = await process.wait()
        
        stdout = "\n".join(stdout_lines)
        stderr = "\n".join(stderr_lines)
        
        # Extract job_id from stdout
        job_id = None
        if stdout:
            # Look for job_id in response JSON
            match = re.search(r'"job_id"\s*:\s*"([^"]+)"', stdout)
            if match:
                job_id = match.group(1)
                print(f"[{task_name}] ✅ Found job_id from JSON: {job_id}")
            else:
                # Try pattern pl_xxxxx
                match = re.search(r'pl_[a-f0-9]+', stdout)
                if match:
                    job_id = match.group(0)
                    print(f"[{task_name}] ✅ Found job_id from pattern: {job_id}")
        
        if not job_id:
            error_msg = stderr[:500] if stderr else "Unknown error"
            print(f"[{task_name}] ❌ Could not extract job_id")
            print(f"[{task_name}] Stderr (first 500 chars): {error_msg}")
            if stdout:
                print(f"[{task_name}] Stdout (last 500 chars): {stdout[-500:]}")
            return {
                "task": task_name,
                "status": "failed",
                "error": "Could not extract job_id from submission",
            }
        
        print(f"[{task_name}] ✅ Job submitted with job_id: {job_id}")
        print(f"[{task_name}] 🔄 Polling job status until completion...")
        
        # Poll job status until completion using async HTTP
        import httpx
        import time as time_module
        
        # Get API key and backend URL from env
        api_key = os.getenv("SYNTH_API_KEY")
        if not api_key:
            # Try to read from env file
            if env_file_path:
                env_path = Path(env_file_path)
                if env_path.exists():
                    from synth_ai.api.train.utils import read_env_file
                    env_data = read_env_file(env_path)
                    api_key = env_data.get("SYNTH_API_KEY")
        
        if not api_key:
            return {
                "task": task_name,
                "status": "failed",
                "error": "SYNTH_API_KEY not found",
            }
        
        # Use local backend (localhost:8000)
        backend_base = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").strip()
        if not backend_base.startswith("http"):
            backend_base = f"http://{backend_base}"
        if not backend_base.endswith("/api"):
            backend_base = f"{backend_base}/api"
        
        url_status = f"{backend_base}/prompt-learning/online/jobs/{job_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        max_wait = config["time_limit_seconds"] + 300  # Add 5 min buffer
        poll_interval = 5.0  # Poll every 5 seconds
        start_time = time_module.time()
        last_status = None
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            while time_module.time() - start_time < max_wait:
                try:
                    resp = await client.get(url_status, headers=headers)
                    if resp.status_code == 200:
                        job_data = resp.json()
                        status = job_data.get("status", "").lower()
                        
                        if status != last_status:
                            print(f"[{task_name}] Status: {status}")
                            last_status = status
                        
                        if status in ("succeeded", "failed", "canceled", "cancelled"):
                            print(f"[{task_name}] ✅ Job reached terminal state: {status}")
                            if status == "failed":
                                # Extract detailed error message
                                error_msg = (
                                    job_data.get("error") 
                                    or job_data.get("error_message")
                                    or job_data.get("metadata", {}).get("error")
                                    or job_data.get("metadata", {}).get("error_message")
                                    or "Job failed"
                                )
                                # Print full error details for debugging
                                print(f"[{task_name}] ❌ Error: {error_msg}")
                                print(f"[{task_name}] Full job data: {json.dumps(job_data, indent=2)}")
                                return {
                                    "task": task_name,
                                    "status": "failed",
                                    "job_id": job_id,
                                    "error": error_msg,
                                    "error_details": job_data,
                                    "config_path": config_path,
                                }
                            break
                    else:
                        print(f"[{task_name}] ⚠️  Status check failed with code {resp.status_code}")
                    
                    await asyncio.sleep(poll_interval)
                except Exception as e:
                    print(f"[{task_name}] ⚠️  Exception polling status: {e}")
                    await asyncio.sleep(poll_interval)
            else:
                print(f"[{task_name}] ⚠️  Polling timed out after {max_wait}s")
                return {
                    "task": task_name,
                    "status": "timeout",
                    "job_id": job_id,
                    "error": f"Job did not complete within {max_wait}s",
                    "config_path": config_path,
                }
        
        # Give it a moment to ensure file I/O is complete
        await asyncio.sleep(1.0)
        
        # Check if results file exists (CLI saves it at the end)
        result_file_found = False
        if results_folder.exists() and job_id:
            # Retry a few times in case file is still being written
            for attempt in range(3):
                all_results_files = list(results_folder.glob("gepa_results_*.txt"))
                print(f"[{task_name}] Attempt {attempt+1}: Checking {len(all_results_files)} results files in {results_folder}")
                print(f"[{task_name}] Looking for job_id: {job_id}")
                
                for file in all_results_files:
                    # Extract job_id from filename (format: gepa_results_pl_xxxxx_timestamp.txt)
                    match = re.search(r'pl_[a-f0-9]+', file.name)
                    if match:
                        file_job_id = match.group(0)
                        if file_job_id == job_id:
                            result_file_found = True
                            print(f"[{task_name}] ✅ Found matching results file: {file.name}")
                            break
                
                if result_file_found:
                    break
                
                if attempt < 2:
                    print(f"[{task_name}] File not found yet, waiting 0.5s...")
                    time.sleep(0.5)
            
            if not result_file_found:
                print(f"[{task_name}] ❌ No matching results file found after retries")
                all_results_files = list(results_folder.glob("gepa_results_*.txt"))
                print(f"[{task_name}] All files: {[f.name for f in all_results_files[:10]]}")
        elif not results_folder.exists():
            print(f"[{task_name}] ⚠️  Results folder doesn't exist: {results_folder}")
        elif not job_id:
            print(f"[{task_name}] ⚠️  No job_id to search for")
        
        if not result_file_found and job_id:
            print(f"[{task_name}] Results file not found (will extract from events)")
        
        print(f"[{task_name}] Job completed with job_id: {job_id}")
        return {
            "task": task_name,
            "status": "completed",
            "job_id": job_id,
            "config_path": config_path,
        }
    finally:
        # Clean up temp config
        try:
            temp_config.unlink()
            print(f"[{task_name}] Cleaned up temp config")
        except Exception as e:
            print(f"[{task_name}] Failed to clean up temp config: {e}")


# Import extract_results_from_events from original (it's long, so we'll reference it)
# For now, we'll use a simplified version that calls the original function
# Actually, let's just copy the whole thing since we need it

def extract_results_from_events(job_id: str, backend_base: str = "http://localhost:8000/api", api_key: Optional[str] = None, config_path: Optional[Path] = None) -> Dict:
    """Extract results from job events."""
    import os
    import requests
    import time
    from pathlib import Path
    
    print(f"[extract_events] Extracting results from events for job_id: {job_id}")
    
    if not api_key:
        # Try environment first
        api_key = os.getenv("SYNTH_API_KEY")
        
        # Fallback: read from repo root .env file
        if not api_key:
            repo_root = Path(__file__).resolve().parents[4]  # Go up from comparisons/ to synth-ai root
            env_file = repo_root / ".env"
            if env_file.exists():
                from synth_ai.api.train.utils import read_env_file
                env_data = read_env_file(env_file)
                api_key = env_data.get("SYNTH_API_KEY")
    
    if not api_key:
        print(f"[extract_events] ERROR: No API key found")
        return {"error": "No API key found"}
    
    # Check job status first
    url_status = f"{backend_base}/prompt-learning/online/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print(f"[extract_events] Checking job status at {url_status}")
    
    try:
        resp = requests.get(url_status, headers=headers, timeout=10)
        resp.raise_for_status()
        job_data = resp.json()
        job_status = job_data.get("status")
        print(f"[extract_events] Job status: {job_status}")
        
        if job_status == "failed":
            error_msg = job_data.get("error") or job_data.get("metadata", {}).get("error") or "Job failed"
            print(f"[extract_events] Job failed: {error_msg}")
            return {"error": "Job failed", "status": "failed", "error_message": error_msg}
    except Exception as e:
        print(f"[extract_events] Failed to check job status: {e}")
        # Continue anyway to try to extract from events
    
    # Wait for job to complete (poll status if still running)
    print(f"[extract_events] Polling job status at {url_status}")
    max_wait = 600  # 10 minutes max wait
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(url_status, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                job_data = resp.json()
                status = job_data.get("status", "")
                print(f"[extract_events] Job status: {status}")
                if status in ("succeeded", "failed"):
                    print(f"[extract_events] Job completed with status: {status}")
                    if status == "failed":
                        error_msg = job_data.get("error") or job_data.get("metadata", {}).get("error") or "Job failed"
                        return {"error": "Job failed", "status": "failed", "error_message": error_msg}
                    break
            else:
                print(f"[extract_events] Status check failed with code {resp.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"[extract_events] Exception polling status: {e}")
            time.sleep(2)
    
    # Fetch events
    url_events = f"{backend_base}/prompt-learning/online/jobs/{job_id}/events?limit=1000"
    
    print(f"[extract_events] Fetching events from {url_events}")
    try:
        resp = requests.get(url_events, headers=headers, timeout=30.0)
        if resp.status_code != 200:
            print(f"[extract_events] Failed to fetch events: status {resp.status_code}")
            return {"error": f"Failed to fetch events: status {resp.status_code}"}
        
        events = resp.json()
        if isinstance(events, dict):
            events = events.get("events", [])
        
        print(f"[extract_events] Fetched {len(events)} events")
        
        # Extract key metrics (same as original)
        baseline_score = None
        candidate1_score = None
        candidate1_lift = None
        total_cost = None
        total_rollouts = None
        total_tokens = None
        total_time = None
        eval_seeds_n = None
        policy_model = None
        trials_tried = None
        
        # Extract policy model from config file FIRST (most reliable)
        if config_path and config_path.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    try:
                        import tomli as tomllib
                    except ImportError:
                        tomllib = None
                
                if tomllib:
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)
                    pl_config = config.get("prompt_learning", {})
                    gepa_config = pl_config.get("gepa", {})
                    gepa_policy_config = gepa_config.get("policy", {})
                    if gepa_policy_config and isinstance(gepa_policy_config, dict) and gepa_policy_config.get("model"):
                        policy_model = gepa_policy_config.get("model")
                    else:
                        policy_config = pl_config.get("policy", {})
                        if isinstance(policy_config, dict) and policy_config.get("model"):
                            policy_model = policy_config.get("model")
                    if policy_model:
                        print(f"[extract_events] Found policy_model from config file: {policy_model}")
            except Exception as e:
                print(f"[extract_events] WARNING: Could not extract policy model from config: {e}")
        
        # Log event types found
        event_types = [e.get("type", "unknown") for e in events if isinstance(e, dict)]
        print(f"[extract_events] Event types found: {set(event_types)}")
        
        # Extract from job status for time and policy model (fallback)
        try:
            resp_status = requests.get(url_status, headers=headers, timeout=10)
            if resp_status.status_code == 200:
                job_data = resp_status.json()
                # Extract time from job status
                created_at = job_data.get("created_at")
                finished_at = job_data.get("finished_at")
                if created_at and finished_at:
                    from datetime import datetime
                    try:
                        start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                        total_time = (end - start).total_seconds()
                        print(f"[extract_events] Total time from job status: {total_time}s")
                    except Exception as e:
                        print(f"[extract_events] Failed to parse time: {e}")
                
                # Extract policy model from metadata (only if not already set from config)
                if not policy_model:
                    metadata = job_data.get("metadata", {})
                    if metadata:
                        # Policy model might be in config_body or request_metadata
                        config_body = metadata.get("config_body", {})
                        pl_config = config_body.get("prompt_learning", {})
                        gepa_config = pl_config.get("gepa", {})
                        # Try gepa.policy.model first
                        gepa_policy = gepa_config.get("policy", {})
                        if isinstance(gepa_policy, dict) and gepa_policy.get("model"):
                            policy_model = gepa_policy.get("model")
                        else:
                            # Fallback to top-level policy
                            policy_config = pl_config.get("policy", {})
                            if isinstance(policy_config, dict) and policy_config.get("model"):
                                policy_model = policy_config.get("model")
                        if not policy_model:
                            # Last resort: check gepa_config directly
                            policy_model = gepa_config.get("policy_model") or pl_config.get("policy_model")
                        if policy_model:
                            print(f"[extract_events] Policy model from metadata: {policy_model}")
        except Exception as e:
            print(f"[extract_events] Failed to get job status for time/model: {e}")
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            
            if event_type == "prompt.learning.validation.summary":
                print(f"[extract_events] Found validation.summary event")
                validation = event_data
                baseline = validation.get("baseline", {})
                results = validation.get("results", [])
                
                if baseline:
                    baseline_score = baseline.get("accuracy")
                    # Extract N from seeds list length (baseline has "seeds" list, not "num_seeds" or "n")
                    baseline_seeds = baseline.get("seeds", [])
                    eval_seeds_n = len(baseline_seeds) if baseline_seeds else baseline.get("num_seeds") or baseline.get("n")
                    print(f"[extract_events] Baseline score: {baseline_score}, N: {eval_seeds_n}")
                
                if results and len(results) > 0:
                    # Only get candidate 1
                    candidate1 = results[0]
                    candidate1_score = candidate1.get("accuracy")
                    print(f"[extract_events] Candidate 1 score: {candidate1_score}")
                    if baseline_score is not None and candidate1_score is not None:
                        candidate1_lift = candidate1_score - baseline_score
                        print(f"[extract_events] Candidate 1 lift: {candidate1_lift}")
                else:
                    print(f"[extract_events] No results found in validation summary")
            
            elif event_type == "prompt.learning.progress":
                # Extract rollouts from progress events (use max value across all progress events)
                progress_rollouts = event_data.get("rollouts_completed")
                if progress_rollouts is not None:
                    if total_rollouts is None or progress_rollouts > total_rollouts:
                        total_rollouts = progress_rollouts
                        print(f"[extract_events] Rollouts from progress: {total_rollouts}")
            
            elif event_type == "prompt.learning.completed":
                print(f"[extract_events] Found completed event")
                total_cost = event_data.get("total_cost_usd")
                # Only use rollouts from completed event if not already set from progress events
                completed_rollouts = event_data.get("total_rollouts")
                if completed_rollouts is not None and total_rollouts is None:
                    total_rollouts = completed_rollouts
                
                # Extract tokens from completed event
                rollouts_prompt = event_data.get("rollouts_prompt_tokens", 0) or 0
                rollouts_completion = event_data.get("rollouts_completion_tokens", 0) or 0
                rollouts_unknown = event_data.get("rollouts_unknown_tokens", 0) or 0
                total_tokens = rollouts_prompt + rollouts_completion + rollouts_unknown
                
                print(f"[extract_events] Total cost: {total_cost}, rollouts: {total_rollouts}, tokens: {total_tokens}")
            
            elif event_type == "prompt.learning.billing.end":
                # Extract time from billing.end (more accurate than job status)
                billing_time = event_data.get("seconds")
                if billing_time is not None:
                    total_time = billing_time
                    print(f"[extract_events] Time from billing.end: {total_time}s")
            
            elif event_type == "prompt.learning.results.summary":
                # Extract trials tried count from results summary message
                message = event.get("message", "")
                tried_match = re.search(r'tried=(\d+)', message)
                if tried_match:
                    trials_tried = int(tried_match.group(1))
                    print(f"[extract_events] Found results.summary event: tried={trials_tried}")
            
            elif event_type == "prompt.learning.failed":
                # Extract error message from failed event
                error_message = event_data.get("message") or event_data.get("error") or "Job failed"
                print(f"[extract_events] Found failed event: {error_message}")
                # Return error result
                return {
                    "error": error_message,
                    "status": "failed",
                    "error_message": error_message,
                }
        
        result = {
            "baseline_score": baseline_score,
            "candidate1_score": candidate1_score,
            "candidate1_lift": candidate1_lift,
            "total_cost": total_cost,
            "total_rollouts": total_rollouts,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "eval_seeds_n": eval_seeds_n,
            "policy_model": policy_model,
            "trials_tried": trials_tried,
        }
        print(f"[extract_events] Final extracted result: {result}")
        return result
    except Exception as e:
        print(f"[extract_events] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def main():
    """Run all GEPA jobs in parallel and show aggregate stats."""
    print("=" * 80)
    print("Starting parallel GEPA jobs for Crafter and Verilog (gpt-oss-120b via Groq)...")
    print("=" * 80)
    
    # Load configuration from YAML
    try:
        config = load_config()
        print(f"✅ Loaded config from {CONFIG_FILE}")
        print(f"  Rollout limit: {config['rollout_limit']}")
        print(f"  Time limit: {config['time_limit_seconds']}s")
        print(f"  Max trials: {config['max_trials']}")
        print(f"  Max cost: ${config['max_cost_usd']}")
        print(f"  Num generations: {config['num_generations']}")
        print(f"  Children per generation: {config['children_per_generation']}")
        print(f"  Model: openai/gpt-oss-120b (via Groq)")
        
    except AssertionError as e:
        print(f"❌ CRITICAL ASSERTION FAILED: {e}")
        raise
    except Exception as e:
        print(f"⚠️  Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Run all jobs in parallel
    tasks = []
    for task_name, config_path in CONFIGS.items():
        if not config_path.exists():
            print(f"⚠️  Config not found: {config_path}")
            continue
        tasks.append(run_gepa_job(task_name, config_path, config))
    
    if not tasks:
        print("No valid configs found!")
        return
    
    print(f"\n🚀 Running {len(tasks)} jobs in parallel...\n")
    results = await asyncio.gather(*tasks)
    print(f"\n✅ All {len(tasks)} jobs completed\n")
    
    # Extract detailed results from events or files
    print("\n" + "=" * 80)
    print("Extracting results from completed jobs...")
    print("=" * 80)
    
    detailed_results = []
    for result in results:
        task_name = result.get("task", "Unknown")
        job_id = result.get("job_id")
        config_path = result.get("config_path")
        status = result.get("status")
        
        print(f"\n[{task_name}] Processing result: status={status}, job_id={job_id}")
        
        if status == "completed" and job_id:
            # First check if job actually succeeded (might have failed but CLI returned success)
            # Try to extract from events first to get job status
            print(f"[{task_name}] Trying to extract from events...")
            details = extract_results_from_events(job_id, config_path=config_path)
            print(f"[{task_name}] Events extraction result: {details}")
            
            # Check if job actually failed (from events or stdout)
            if details.get("error") == "Job failed" or details.get("status") == "failed":
                print(f"[{task_name}] ⚠️  Job failed - skipping result extraction")
                detailed_results.append({
                    "task": task_name,
                    "status": "failed",
                    "error": details.get("error", "Job failed"),
                })
            elif "error" not in details and details.get("baseline_score") is not None:
                details["task"] = task_name
                detailed_results.append(details)
                print(f"[{task_name}] Successfully extracted from events")
            else:
                # Fallback: extract from results file by job_id
                print(f"[{task_name}] Falling back to file extraction...")
                details = extract_results_from_file(task_name, job_id, config_path)
                if "error" in details:
                    # Job likely failed - no results file created
                    print(f"[{task_name}] ⚠️  {details.get('error')}")
                    detailed_results.append({
                        "task": task_name,
                        "status": "failed",
                        "error": details.get("error", "No results file found"),
                    })
                else:
                    details["task"] = task_name
                    detailed_results.append(details)
                    print(f"[{task_name}] File extraction result: {details}")
        else:
            error_msg = result.get("error", "Failed to complete or no job_id")
            print(f"[{task_name}] Job failed: {error_msg}")
            detailed_results.append({
                "task": task_name,
                "error": error_msg,
            })
    
    # Calculate aggregates
    valid_results = [r for r in detailed_results if "error" not in r and r.get("baseline_score") is not None]
    
    print("\n" + "=" * 80)
    print(f"Found {len(valid_results)} valid results out of {len(detailed_results)} total")
    print("=" * 80)
    
    if not valid_results:
        # Debug: show what we got
        print("\nNo valid results to aggregate!")
        print("\nDebug info:")
        for r in detailed_results:
            print(f"  {r.get('task')}: {r}")
        return
    
    # Aggregate stats
    total_cost = sum(r.get("total_cost", 0) or 0 for r in valid_results)
    total_rollouts = sum(r.get("total_rollouts", 0) or 0 for r in valid_results)
    total_trials = sum(r.get("trials_tried", 0) or 0 for r in valid_results)
    total_time = sum(r.get("total_time", 0) or 0 for r in valid_results)
    total_tokens = sum(r.get("total_tokens", 0) or 0 for r in valid_results)
    
    avg_baseline = sum(r.get("baseline_score", 0) or 0 for r in valid_results) / len(valid_results)
    avg_candidate1 = sum(r.get("candidate1_score", 0) or 0 for r in valid_results) / len(valid_results)
    avg_lift = sum(r.get("candidate1_lift", 0) or 0 for r in valid_results) / len(valid_results)
    
    # Generate aggregate table output
    from datetime import datetime
    output_lines = []
    output_lines.append("\n" + "=" * 150)
    output_lines.append("AGGREGATE STATS ACROSS ALL TASKS (synth_gepa_agentic - openai/gpt-oss-120b via Groq)")
    output_lines.append("=" * 150)
    output_lines.append("")
    output_lines.append(f"{'Task':<20} {'Policy Model':<25} {'Baseline':<12} {'Candidate 1':<14} {'Lift':<12} {'Rollouts':<10} {'Trials':<10} {'Tokens':<12} {'Time':<10} {'Eval N':<8}")
    output_lines.append("-" * 150)
    
    for result in detailed_results:
        task = result.get("task", "Unknown")
        if "error" in result:
            output_lines.append(f"{task:<20} {'ERROR':<25} {'ERROR':<12} {'ERROR':<14} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8}")
        else:
            baseline = result.get("baseline_score")
            candidate1 = result.get("candidate1_score")
            lift = result.get("candidate1_lift")
            rollouts = result.get("total_rollouts", 0) or 0
            trials = result.get("trials_tried", 0) or 0
            tokens = result.get("total_tokens", 0) or 0
            time_sec = result.get("total_time", 0) or 0
            eval_n = result.get("eval_seeds_n", 0) or 0
            policy_model = result.get("policy_model", "N/A")
            
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            candidate1_str = f"{candidate1:.4f}" if candidate1 is not None else "N/A"
            lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
            rollouts_str = str(rollouts) if rollouts > 0 else "N/A"
            trials_str = str(trials) if trials > 0 else "N/A"
            tokens_str = f"{tokens/1e6:.2f}M" if tokens >= 1e6 else (f"{tokens/1e3:.1f}K" if tokens >= 1e3 else (str(tokens) if tokens > 0 else "N/A"))
            time_str = f"{time_sec:.1f}s" if time_sec < 60 else f"{time_sec/60:.1f}m" if time_sec > 0 else "N/A"
            eval_n_str = str(eval_n) if eval_n > 0 else "N/A"
            policy_model_str = str(policy_model) if policy_model else "N/A"
            
            output_lines.append(f"{task:<20} {policy_model_str:<25} {baseline_str:<12} {candidate1_str:<14} {lift_str:<12} {rollouts_str:<10} {trials_str:<10} {tokens_str:<12} {time_str:<10} {eval_n_str:<8}")
    
    output_lines.append("-" * 150)
    tokens_str = f"{total_tokens/1e6:.2f}M" if total_tokens >= 1e6 else (f"{total_tokens/1e3:.1f}K" if total_tokens >= 1e3 else str(total_tokens))
    time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time/60:.1f}m"
    output_lines.append(f"{'TOTAL':<20} {'':<25} {'':<12} {'':<14} {'':<12} {total_rollouts:<10} {total_trials:<10} {tokens_str:<12} {time_str:<10} {'':<8}")
    output_lines.append(f"{'AVERAGE':<20} {'':<25} {avg_baseline:.4f}     {avg_candidate1:.4f}     {avg_lift:+.4f} {'':<10} {'':<10} {'':<12} {'':<10} {'':<8}")
    output_lines.append("")
    output_lines.append(f"Total Cost: ${total_cost:.4f}")
    output_lines.append("=" * 150)
    
    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)
    
    # Save to file (script is already in comparisons/ directory)
    comparisons_dir = Path(__file__).parent
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = comparisons_dir / f"synth_gepa_agentic_comparison_readout_{timestamp}.txt"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n📄 Comparison results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save comparison results file: {e}")


if __name__ == "__main__":
    asyncio.run(main())

