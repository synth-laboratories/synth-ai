"""
DEPRECATED: This module has been moved to synth_ai.lm

The synth_ai.zyk module is deprecated and will be removed in a future version.
Please update your imports:

OLD: from synth_ai.zyk import LM
NEW: from synth_ai.lm.core.main import LM

or

NEW: from synth_ai import LM  # (recommended)
"""

# ruff: noqa: E402
import warnings

# Issue deprecation warning
warnings.warn(
    "synth_ai.zyk is deprecated and will be removed in a future version. "
    "Please use 'from synth_ai import LM' or 'from synth_ai.lm.core.main import LM' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from new location for backward compatibility
from synth_ai.lm.core.main import LM
from synth_ai.lm.vendors.base import BaseLMResponse

__all__ = ["LM", "BaseLMResponse"]
