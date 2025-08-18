import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "synth_ai.tracing (now tracing_v1) is deprecated. "
    "Please use synth_ai.tracing_v2 instead. "
    "Backend upload functionality is no longer supported in v1.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the main components with deprecation warnings
from .abstractions import *
from .config import *
from .decorators import *
from .trackers import *
