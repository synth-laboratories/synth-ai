"""Synth AI canonical SDK surface.

# See: specifications/daily/feb24_2026/tinker_synth_final.md
"""

from __future__ import annotations

import importlib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

# Install log filter as early as possible to suppress noisy codex_otel logs
try:
    from synth_ai.core.utils.log_filter import install_log_filter

    install_log_filter()
except Exception:
    pass

# Version resolution
try:
    __version__ = _metadata.version("synth-ai")
except PackageNotFoundError:
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as _toml  # type: ignore[no-redef]

    try:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as fh:
            _pyproject = _toml.load(fh)
        __version__ = str(_pyproject["project"]["version"])
    except Exception:
        __version__ = "0.0.0.dev0"


__all__ = [
    "AsyncSynthClient",
    "Client",
    "ContainerClient",
    "GraphsClient",
    "InProcessContainer",
    "JobsClient",
    "OfflineJob",
    "OnlineSession",
    "PoolsClient",
    "SynthClient",
    "System",
    "VerifiersClient",
    "container",
    "graphs",
    "inference",
    "optimization",
    "pools",
    "recipes",
    "verifiers",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "SynthClient": ("synth_ai.client", "SynthClient"),
    "AsyncSynthClient": ("synth_ai.client", "AsyncSynthClient"),
    "System": ("synth_ai.optimization", "System"),
    "OfflineJob": ("synth_ai.optimization", "OfflineJob"),
    "OnlineSession": ("synth_ai.optimization", "OnlineSession"),
    "Client": ("synth_ai.inference", "Client"),
    "JobsClient": ("synth_ai.inference", "JobsClient"),
    "GraphsClient": ("synth_ai.graphs", "GraphsClient"),
    "VerifiersClient": ("synth_ai.verifiers", "VerifiersClient"),
    "PoolsClient": ("synth_ai.pools", "PoolsClient"),
    "InProcessContainer": ("synth_ai.container", "InProcessContainer"),
    "ContainerClient": ("synth_ai.container", "ContainerClient"),
}

_NAMESPACE_MODULES = {
    "optimization",
    "inference",
    "graphs",
    "verifiers",
    "pools",
    "container",
    "recipes",
}


def __getattr__(name: str) -> Any:
    if name in _NAMESPACE_MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
