from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click

from .utils import REPO_ROOT, load_toml, preview_json

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}

_STATE_DIR = Path.home() / ".synth-ai"
_STATE_FILE = _STATE_DIR / "train_cli.json"


@dataclass(slots=True)
class ConfigCandidate:
    path: Path
    train_type: str  # "rl", "sft", "prompt_learning", or "unknown"


def _load_last_config() -> Path | None:
    """Load the last used training config path from state file."""
    try:
        if _STATE_FILE.is_file():
            with _STATE_FILE.open() as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    last_config = data.get("LAST_CONFIG")
                    if last_config:
                        path = Path(last_config).resolve()
                        if path.exists():
                            return path
    except Exception:
        pass
    return None


def _save_last_config(config_path: Path) -> None:
    """Save the last used training config path to state file."""
    try:
        data = {}
        if _STATE_FILE.is_file():
            with _STATE_FILE.open() as fh:
                data = json.load(fh) or {}
        if not isinstance(data, dict):
            data = {}
        data["LAST_CONFIG"] = str(config_path.resolve())
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _STATE_FILE.open("w") as fh:
            json.dump(data, fh)
    except Exception:
        pass


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
        REPO_ROOT / "synth_ai" / "demos",
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
    # 0) Check for prompt_learning section (highest priority)
    pl_section = data.get("prompt_learning")
    if isinstance(pl_section, dict):
        algorithm = pl_section.get("algorithm", "").lower()
        if algorithm in {"mipro", "gepa"}:
            return "prompt_learning"
    # Also check if top-level has prompt_learning indicators
    algorithm = data.get("algorithm")
    if isinstance(algorithm, str) and algorithm.lower() in {"mipro", "gepa"}:
        return "prompt_learning"
    
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
        if cfg_type == "unknown":
            raise click.ClickException(
                f"Config {path} is missing algorithm.type/method metadata. Add type = 'rl', 'sft', or 'prompt_learning'."
            )
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
        if cfg_type == "unknown":
            continue
        candidates.append(ConfigCandidate(path=path, train_type=cfg_type))

    if requested_type and requested_type != "auto":
        candidates = [c for c in candidates if c.train_type == requested_type]

    # De-dupe by path and keep deterministic ordering by directory depth then name
    candidates.sort(key=lambda c: (len(c.path.parts), str(c.path)))
    return candidates


def prompt_for_config(
    candidates: list[ConfigCandidate], *, requested_type: str | None, allow_autoselect: bool = False
) -> ConfigCandidate:
    import sys
    
    if not candidates:
        raise click.ClickException("No training configs found. Pass --config explicitly.")

    # Check for last used config and move it to the top if found
    last_config = _load_last_config()
    default_idx = 1

    # Auto-select if only one candidate OR if allow_autoselect=True and we're in non-interactive mode
    is_interactive = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
    
    if allow_autoselect and (len(candidates) == 1 or not is_interactive):
        # In non-interactive mode with allow_autoselect, use the first candidate (or last used if available)
        if last_config:
            for cand in candidates:
                if cand.path.resolve() == last_config:
                    chosen = cand
                    _save_last_config(chosen.path)
                    return chosen
        chosen = candidates[0]
        _save_last_config(chosen.path)
        if not is_interactive:
            click.echo(f"[AUTO-SELECT] Non-interactive mode: using config {chosen.path}", err=True)
        return chosen

    if last_config:
        for idx, cand in enumerate(candidates):
            if cand.path.resolve() == last_config:
                # Move last used config to the front
                candidates.insert(0, candidates.pop(idx))
                break

    # If not interactive and allow_autoselect is False, we can't prompt - use first candidate
    if not is_interactive:
        click.echo(f"[AUTO-SELECT] Non-interactive mode: using first config {candidates[0].path}", err=True)
        chosen = candidates[0]
        _save_last_config(chosen.path)
        return chosen

    click.echo("Select a training config:")
    for idx, cand in enumerate(candidates, start=1):
        last_marker = " (last used)" if last_config and cand.path.resolve() == last_config else ""
        click.echo(f"  {idx}) {cand.path}{last_marker}")
    click.echo("  0) Abort")

    choice = click.prompt("Enter choice", type=int, default=default_idx)
    if choice == 0:
        raise click.ClickException("Aborted by user")
    if choice < 0 or choice > len(candidates):
        raise click.ClickException("Invalid selection")

    selection = candidates[choice - 1]

    # Save this config as the last used
    _save_last_config(selection.path)

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
