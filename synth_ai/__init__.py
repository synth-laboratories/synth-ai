"""Python-only Synth SDK surface."""

from __future__ import annotations

import importlib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

try:
    from synth_ai.core.utils.log_filter import install_log_filter

    install_log_filter()
except Exception:
    pass

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
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncSynthClient",
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "SynthClient",
    "TunnelsClient",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "SynthClient": ("synth_ai.client", "SynthClient"),
    "AsyncSynthClient": ("synth_ai.client", "AsyncSynthClient"),
    "ContainersClient": ("synth_ai.sdk.containers", "ContainersClient"),
    "AsyncContainersClient": ("synth_ai.sdk.containers", "AsyncContainersClient"),
    "ContainerPoolsClient": ("synth_ai.sdk.pools", "ContainerPoolsClient"),
    "AsyncContainerPoolsClient": ("synth_ai.sdk.pools", "AsyncContainerPoolsClient"),
    "TunnelsClient": ("synth_ai.sdk.tunnels", "TunnelsClient"),
    "AsyncTunnelsClient": ("synth_ai.sdk.tunnels", "AsyncTunnelsClient"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
