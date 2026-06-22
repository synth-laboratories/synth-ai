"""Package version derived from package metadata or pyproject.toml.

Order: installed distribution metadata (normal installs), then parsing this repo's
``pyproject.toml`` (editable checkouts), then ``0.0.0.dev0`` only when both fail.
The last case is explicitly degraded; enable logging at DEBUG to see the cause.

# See: Synth Style — avoid silent failure; degradation is logged, not swallowed quietly.
"""

from __future__ import annotations

import logging
import tomllib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path

_logger = logging.getLogger(__name__)

try:
    __version__ = _metadata.version("synth-ai")
except PackageNotFoundError:
    try:
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as fh:
            _pyproject = tomllib.load(fh)
        __version__ = str(_pyproject["project"]["version"])
    except (
        OSError,
        TypeError,
        KeyError,
        ValueError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        _logger.debug(
            "synth-ai managed research: could not read version from pyproject.toml; "
            "using dev placeholder",
            exc_info=exc,
        )
        __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
