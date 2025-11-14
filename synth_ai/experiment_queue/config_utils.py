"""Helpers for preparing TOML configs for experiment jobs."""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import tomli_w


def _load_toml(config_path: Path) -> dict[str, Any]:
    """Load TOML using tomllib/tomli depending on runtime."""
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(config_path, "rb") as fh:
        return tomllib.load(fh)


def _ensure_prompt_learning_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.setdefault("prompt_learning", {})
    if not isinstance(section, dict):
        raise ValueError("Expected [prompt_learning] section to be a table.")
    return section


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if (
            isinstance(value, Mapping)
            and isinstance(base.get(key), Mapping)
            and not isinstance(value, (str, bytes))
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

    Returns:
        PreparedConfig with the path to the merged TOML and resolved results_folder.
    """
    source_path = Path(config_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Config path not found: {source_path}")

    data = _load_toml(source_path)
    if overrides:
        _deep_update(data, overrides)

    normalize_env_file_path(data, source_path)
    results_folder = resolve_results_folder(data, source_path)

    workdir = Path(tempfile.mkdtemp(prefix="experiment_queue_cfg_"))
    tmp_config_path = workdir / source_path.name
    with open(tmp_config_path, "wb") as fh:
        tomli_w.dump(data, fh)

    return PreparedConfig(path=tmp_config_path, results_folder=results_folder, workdir=workdir)
