"""Synth Python SDK: typed clients for Index, Swarms and hosted MCP."""

from __future__ import annotations

import re as _re
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from synth_ai.client import AsyncSynthClient, SynthClient

try:
    __version__ = _installed_version("synth-ai")
except _PackageNotFoundError as _version_error:
    _project = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    _match = _re.search(r'^version = "([^"]+)"$', _project.read_text(), _re.MULTILINE)
    if _match is None:
        raise RuntimeError("SDK source metadata has no version") from _version_error
    __version__ = _match.group(1)

__all__ = ["AsyncSynthClient", "SynthClient"]


def __getattr__(name: str) -> object:
    """Load client composition only when requested, keeping release tooling independent."""
    if name in __all__:
        from synth_ai.client import AsyncSynthClient, SynthClient

        clients = {"AsyncSynthClient": AsyncSynthClient, "SynthClient": SynthClient}
        globals().update(clients)
        return clients[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
