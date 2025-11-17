"""GEPA benchmark task apps (HotpotQA, IFBench, HoVer, PUPA, Crafter, Sokoban, Verilog)."""

# Import modules for side effects (task app registration) when package is imported.
from . import hotpotqa_task_app  # noqa: F401
from . import hover_task_app  # noqa: F401
from . import ifbench_task_app  # noqa: F401
from . import pupa_task_app  # noqa: F401
from . import crafter_task_app  # noqa: F401

# Lazy import for agentic tasks that may have heavy dependencies
try:
    from . import sokoban_task_app  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import sokoban_task_app: {e}", ImportWarning)

try:
    from . import verilog_task_app  # noqa: F401
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import verilog_task_app: {e}", ImportWarning)
