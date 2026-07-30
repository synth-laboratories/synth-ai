"""Where the SDK looks for user-scoped configuration."""

from __future__ import annotations

import os
from pathlib import Path

SYNTH_HOME_DIR = Path(os.environ.get("SYNTH_HOME", Path.home() / ".synth_ai")).expanduser()

__all__ = ["SYNTH_HOME_DIR"]
