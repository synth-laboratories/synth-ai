"""Compatibility helpers sharing the core credential configuration authority."""

from pathlib import Path

from synth_ai.core.utils.env import _candidate_config_paths, get_api_key


def config_search_paths() -> tuple[Path, ...]:
    return tuple(_candidate_config_paths())


__all__ = ["config_search_paths", "get_api_key"]
