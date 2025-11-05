"""GEPA benchmark task apps (HotpotQA, IFBench, HoVer, PUPA)."""

# Import modules for side effects (task app registration) when package is imported.
from . import hotpotqa_task_app  # noqa: F401
from . import hover_task_app  # noqa: F401
from . import ifbench_task_app  # noqa: F401
from . import pupa_task_app  # noqa: F401
