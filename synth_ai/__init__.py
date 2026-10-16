"""Reduced SDK entry point; the 0.23 replacement surface is under construction."""

import re as _re
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path as _Path

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
