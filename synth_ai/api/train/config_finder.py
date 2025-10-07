from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click

from .utils import REPO_ROOT, load_toml, preview_json

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}


@dataclass(slots=True)
class ConfigCandidate:
    path: Path
    train_type: str  # "rl", "sft", or "unknown"


def _iter_candidate_paths() -> Iterable[Path]:
    seen: set[Path] = set()
    
    # Prioritize current working directory first
    try:
        cwd = Path.cwd().resolve()
    except Exception:
        cwd = None
    if cwd and cwd.exists():
        for path in cwd.rglob("*.toml"):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved

    # Then look in explicit config directories
    preferred = [
        REPO_ROOT / "configs",
        REPO_ROOT / "examples",
        REPO_ROOT / "training",
    ]
    for base in preferred:
        if not base.exists():
            continue
        for path in base.rglob("*.toml"):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _infer_config_type(data: dict) -> str:
    # 1) Strong signals from [algorithm]
    algo = data.get("algorithm")
    if isinstance(algo, dict):
        method = str(algo.get("method") or "").lower()
        algo_type = str(algo.get("type") or "").lower()
        variety = str(algo.get("variety") or "").lower()

        # RL indicators
        if method in {"policy_gradient", "ppo", "reinforce"}:
            return "rl"
        if algo_type == "online":
            return "rl"
        if variety in {"gspo", "grpo", "ppo"}:
            return "rl"

        # SFT indicators
        if method in {"supervised_finetune", "sft"}:
            return "sft"
        if algo_type == "offline":
            return "sft"
        if variety in {"fft"}:
            return "sft"

    # 2) Other RL signals
    if data.get("job_type") == "rl":
        return "rl"
    services = data.get("services")
    if isinstance(services, dict) and ("task_url" in services or "environment" in services):
        return "rl"

    # 3) Other SFT signals
    training = data.get("training")
    if isinstance(training, dict):
        mode = str(training.get("mode") or "").lower()
        if mode.startswith("sft") or mode == "sft_offline":
            return "sft"
    hyper = data.get("hyperparameters")
    if isinstance(hyper, dict):
        kind = str(hyper.get("train_kind") or "").lower()
        if kind in {"sft", "fft"}:
            return "sft"

    # 4) Fallback
    return "unknown"


def discover_configs(explicit: list[str], *, requested_type: str | None) -> list[ConfigCandidate]:
    candidates: list[ConfigCandidate] = []
    seen: set[Path] = set()

    for raw in explicit:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise click.ClickException(f"Config not found: {path}")
        data = load_toml(path)
        cfg_type = _infer_config_type(data)
        candidates.append(ConfigCandidate(path=path, train_type=cfg_type))
        seen.add(path)

    if explicit:
        return candidates

    for path in _iter_candidate_paths():
        if path in seen:
            continue
        try:
            data = load_toml(path)
        except Exception:
            continue
        cfg_type = _infer_config_type(data)
        candidates.append(ConfigCandidate(path=path, train_type=cfg_type))

    if requested_type and requested_type != "auto":
        candidates = [c for c in candidates if c.train_type in {requested_type, "unknown"}]

    # De-dupe by path and keep deterministic ordering by directory depth then name
    candidates.sort(key=lambda c: (len(c.path.parts), str(c.path)))
    return candidates


def prompt_for_config(candidates: list[ConfigCandidate], *, requested_type: str | None) -> ConfigCandidate:
    if not candidates:
        raise click.ClickException("No training configs found. Pass --config explicitly.")

    click.echo("Select a training config:")
    for idx, cand in enumerate(candidates, start=1):
        label = cand.train_type if cand.train_type != "unknown" else "?"
        click.echo(f"  {idx}) [{label}] {cand.path}")
    click.echo("  0) Abort")

    choice = click.prompt("Enter choice", type=int)
    if choice == 0:
        raise click.ClickException("Aborted by user")
    if choice < 0 or choice > len(candidates):
        raise click.ClickException("Invalid selection")

    selection = candidates[choice - 1]
    try:
        data = load_toml(selection.path)
        preview = preview_json({k: data.get(k) for k in list(data.keys())[:4]}, limit=320)
        click.echo(f"Loaded {selection.path}: {preview}")
    except Exception:
        pass
    return selection


__all__ = [
    "ConfigCandidate",
    "discover_configs",
    "prompt_for_config",
]
