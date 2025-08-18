"""Synth Environment Package - A framework for reinforcement learning environments."""

__version__ = "0.1.5"

# Import key modules for easier access
from . import environment, examples, service, stateful, tasks

__all__ = [
    "environment",
    "service",
    "stateful",
    "tasks",
    "examples",
]

import sys
import types

# ---------------------------------------------------------------------------
# Compatibility Shim
# Some older code imports synth_env via the namespace prefix 'src.synth_env'.
# Rather than requiring consumers to update import paths (or adding try/except
# imports which violates project style rules), we register an alias module so
# that both import styles resolve to the same package instance.
# ---------------------------------------------------------------------------
if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")

# Expose this package as src.synth_env
sys.modules["src.synth_env"] = sys.modules[__name__]
sys.modules["src"].synth_env = sys.modules[__name__]
