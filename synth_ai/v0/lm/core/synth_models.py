"""
Synth-supported models registry.

This module defines the specific models that are supported by Synth's infrastructure.
Models are organized by family and size for easy maintenance and extension.

MAINTENANCE GUIDE:
1. Add new base models to synth_ai.v0.api.models.supported.QWEN3_MODELS
2. Fine-tuned models (ft:) are automatically detected by regex
3. Update SYNTH_SUPPORTED_MODELS set when adding new models
4. Test changes with: pytest tests/lms/test_qwen_chat_completions.py

WHY THIS EXISTS:
- The previous regex (^.*\/.*$) was too broad and caught unintended models
- This provides explicit control over which models use Synth infrastructure
- Easier to maintain and debug model routing issues
"""

from typing import List, Set

from synth_ai.v0.api.models.supported import QWEN3_MODELS

# Fine-tuned models pattern - any model starting with "ft:" is considered Synth-compatible
# These are dynamically detected, but we can add specific known ones here
FINE_TUNED_MODELS: List[str] = [
    # Add specific fine-tuned models that are known to work with Synth
    # Examples:
    # "ft:Qwen/Qwen3-4B-2507:ftjob-22",
]

# Combine all Synth-supported models
SYNTH_SUPPORTED_MODELS: Set[str] = set(QWEN3_MODELS + FINE_TUNED_MODELS)

# Export the main set for easy import
__all__ = ["SYNTH_SUPPORTED_MODELS", "QWEN3_MODELS", "FINE_TUNED_MODELS"]
