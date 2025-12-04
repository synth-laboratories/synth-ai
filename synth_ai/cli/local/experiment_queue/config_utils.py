"""Helpers for preparing TOML configs for experiment jobs."""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import tomli_w

from synth_ai.core.telemetry import log_info

from .validation import validate_path


def _load_toml(config_path: Path) -> dict[str, Any]:
    """Load TOML using tomllib/tomli depending on runtime.
    
    Args:
        config_path: Path to TOML config file
        
    Returns:
        Parsed TOML as dictionary
        
    Raises:
        AssertionError: If config_path is invalid or file cannot be parsed
        FileNotFoundError: If config file doesn't exist
    """
    # Validate input
    assert config_path is not None, "config_path cannot be None"
    path = validate_path(config_path, "config_path", must_exist=True)
    
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
        assert isinstance(config, dict), (
            f"TOML config must be dict, got {type(config).__name__}"
        )
        return config


def _ensure_prompt_learning_section(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure prompt_learning section exists and is a dict.
    
    Args:
        config: TOML config dictionary
        
    Returns:
        prompt_learning section dictionary
        
    Raises:
        AssertionError: If config is invalid or section is wrong type
    """
    assert isinstance(config, dict), (
        f"config must be dict, got {type(config).__name__}"
    )
    section = config.setdefault("prompt_learning", {})
    assert isinstance(section, dict), (
        f"Expected [prompt_learning] section to be a dict, got {type(section).__name__}"
    )
    return section


def _find_similar_keys(data: dict[str, Any], search_key: str, results: list[str], prefix: str = "") -> None:
    """Recursively find keys similar to search_key in nested dict structure."""
    if not isinstance(data, dict):
        return
    
    for key, value in data.items():
        current_path = f"{prefix}.{key}" if prefix else key
        if search_key.lower() in key.lower() or key.lower() in search_key.lower():
            results.append(current_path)
        if isinstance(value, dict):
            _find_similar_keys(value, search_key, results, current_path)


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep update with support for dot-notation keys (e.g., 'prompt_learning.gepa.rollout.budget').
    
    Dot-notation keys are split and create nested dictionaries.
    Regular keys are updated normally.
    """
    for key, value in overrides.items():
        # Handle dot-notation keys (e.g., "prompt_learning.gepa.rollout.budget")
        if "." in key:
            keys = key.split(".")
            # Navigate/create nested structure
            current = base
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], MutableMapping):
                    current[k] = {}
                current = current[k]
            # Set final value
            final_key = keys[-1]
            if (
                isinstance(value, Mapping)
                and isinstance(current.get(final_key), MutableMapping)
                and not isinstance(value, str | bytes)
            ):
                nested = copy.deepcopy(dict(current[final_key]))
                current[final_key] = _deep_update(nested, value)
            else:
                current[final_key] = copy.deepcopy(value)
        else:
            # Regular key (no dot notation)
            if (
                isinstance(value, Mapping)
                and isinstance(base.get(key), MutableMapping)
                and not isinstance(value, str | bytes)
            ):
                nested = copy.deepcopy(dict(base[key]))
                base[key] = _deep_update(nested, value)
            else:
                base[key] = copy.deepcopy(value)
    return base


def _resolve_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (relative_to / value).resolve()
    else:
        path = path.expanduser().resolve()
    return path


def resolve_results_folder(config: dict[str, Any], config_path: Path) -> Path:
    """Resolve and ensure the results folder exists."""
    config_dir = config_path.parent.resolve()
    prompt_section = _ensure_prompt_learning_section(config)
    raw = prompt_section.get("results_folder") or config.get("results_folder")
    if raw:
        results_path = _resolve_path(str(raw), relative_to=config_dir)
    else:
        results_path = (config_dir / "results").resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    prompt_section["results_folder"] = str(results_path)
    return results_path


def normalize_env_file_path(config: dict[str, Any], config_path: Path) -> None:
    """Ensure env_file_path fields are absolute so subprocesses inherit the right file."""
    config_dir = config_path.parent.resolve()
    prompt_section = _ensure_prompt_learning_section(config)
    raw = prompt_section.get("env_file_path") or config.get("env_file_path")
    if not raw:
        return
    env_path = _resolve_path(str(raw), relative_to=config_dir)
    prompt_section["env_file_path"] = str(env_path)
    config["env_file_path"] = str(env_path)


@dataclass(slots=True)
class PreparedConfig:
    """Container for a temporary config file plus metadata."""

    path: Path
    results_folder: Path
    workdir: Path | None = None

    def cleanup(self) -> None:
        if self.workdir and self.workdir.exists():
            import shutil

            shutil.rmtree(self.workdir, ignore_errors=True)


def prepare_config_file(config_path: str | Path, overrides: Mapping[str, Any] | None = None) -> PreparedConfig:
    """
    Load a TOML config, apply overrides, and materialize a temporary file.

    Args:
        config_path: Path to source TOML config file
        overrides: Optional dictionary of config overrides to apply

    Returns:
        PreparedConfig with the path to the merged TOML and resolved results_folder.

    Raises:
        AssertionError: If inputs are invalid
        FileNotFoundError: If config file doesn't exist
    """
    ctx: dict[str, Any] = {"config_path": str(config_path), "has_overrides": overrides is not None}
    log_info("prepare_config_file invoked", ctx=ctx)
    # Validate inputs
    assert config_path is not None, "config_path cannot be None"
    source_path = validate_path(config_path, "config_path", must_exist=True)
    
    if overrides is not None:
        assert isinstance(overrides, Mapping), (
            f"overrides must be Mapping, got {type(overrides).__name__}"
        )

    data = _load_toml(source_path)
    assert isinstance(data, dict), (
        f"_load_toml must return dict, got {type(data).__name__}"
    )
    
    if overrides:
        _deep_update(data, overrides)
        # Validate after merge
        assert isinstance(data, dict), (
            f"Config after merge must be dict, got {type(data).__name__}"
        )
        
        # VALIDATION: Verify critical overrides were actually applied
        # This prevents silent failures where overrides don't match expected paths
        for override_key, override_value in overrides.items():
            # Navigate the nested dict structure (created by _deep_update)
            keys = override_key.split(".")
            current = data
            found = True
            
            # Navigate through nested structure
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    found = False
                    break
                current = current[key]
            
            if not found:
                # Try to find similar keys for better error message
                similar_keys = []
                _find_similar_keys(data, keys[0], similar_keys)
                similar_msg = f" Similar keys found: {similar_keys[:5]}" if similar_keys else ""
                raise ValueError(
                    f"Config override validation failed: '{override_key}' was not found in config after merge.{similar_msg} "
                    f"Override value: {override_value!r}. This indicates the override path is incorrect or the config structure doesn't match."
                )
            elif isinstance(current, dict) and isinstance(override_value, dict):
                # For dict overrides, check that override_value is a subset of current
                # (i.e., all keys in override_value match current)
                for key, val in override_value.items():
                    if key not in current or current[key] != val:
                        raise ValueError(
                            f"Config override validation failed: '{override_key}.{key}' was set to {current.get(key)!r} "
                            f"but override specified {val!r}. Override may not have been applied correctly."
                        )
            elif current != override_value:
                # Value exists but doesn't match - this is a problem
                # This is critical for rollout limits - if base config has 300 but override specifies 100,
                # we MUST ensure 100 is applied, not 300
                raise ValueError(
                    f"Config override validation failed: '{override_key}' was set to {current!r} "
                    f"but override specified {override_value!r}. Override may not have been applied correctly. "
                    f"This indicates the base config value ({current!r}) was not overridden by the specified value ({override_value!r})."
                )

    normalize_env_file_path(data, source_path)
    results_folder = resolve_results_folder(data, source_path)
    assert results_folder.exists(), (
        f"results_folder must exist after creation: {results_folder}"
    )

    workdir = Path(tempfile.mkdtemp(prefix="experiment_queue_cfg_"))
    assert workdir.exists(), f"workdir must exist after creation: {workdir}"
    
    tmp_config_path = workdir / source_path.name
    with open(tmp_config_path, "wb") as fh:
        tomli_w.dump(data, fh)
    
    assert tmp_config_path.exists(), (
        f"Temporary config file must exist after writing: {tmp_config_path}"
    )

    return PreparedConfig(path=tmp_config_path, results_folder=results_folder, workdir=workdir)
