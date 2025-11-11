"""Integration components for LangProBe blog post comparisons."""

from .learning_curve_tracker import (
    Checkpoint,
    LearningCurve,
    LearningCurveTracker,
)
from .task_app_client import TaskAppClient

# Try to import in-process adapter classes (requires backend access)
try:
    from .synth_gepa_adapter_inprocess import SynthGEPAAdapterInProcess
    from .synth_mipro_adapter_inprocess import SynthMIPROAdapterInProcess
    INPROCESS_AVAILABLE = True
except ImportError:
    INPROCESS_AVAILABLE = False
    SynthGEPAAdapterInProcess = None  # type: ignore
    SynthMIPROAdapterInProcess = None  # type: ignore

__all__ = [
    "TaskAppClient",
    "LearningCurveTracker",
    "LearningCurve",
    "Checkpoint",
]

if INPROCESS_AVAILABLE:
    __all__.extend([
        "SynthGEPAAdapterInProcess",
        "SynthMIPROAdapterInProcess",
    ])

