from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Dict, Tuple

from synth_ai.api.train.env_resolver import KeySpec, resolve_env

__all__ = ["validate_train_environment"]


def validate_train_environment(
    *,
    config_path: Path | None,
    explicit_env_paths: Iterable[str],
    required_keys: list[KeySpec],
) -> Tuple[Path, Dict[str, str]]:
    """Validate and resolve environment secrets used by the train command."""
    resolved_path, resolved_keys = resolve_env(
        config_path=config_path,
        explicit_env_paths=explicit_env_paths,
        required_keys=required_keys,
    )
    return resolved_path, resolved_keys
