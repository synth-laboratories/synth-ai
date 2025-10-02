"""
Synth-supported models registry.

This module defines the specific models that are supported by Synth's infrastructure.
Models are organized by family and size for easy maintenance and extension.

MAINTENANCE GUIDE:
1. Add new model families to the appropriate lists (QWEN_MODELS, OTHER_SYNTH_MODELS)
2. Fine-tuned models (ft:) are automatically detected by regex
3. Update SYNTH_SUPPORTED_MODELS set when adding new models
4. Test changes with: pytest tests/lms/test_qwen_chat_completions.py

WHY THIS EXISTS:
- The previous regex (^.*\/.*$) was too broad and caught unintended models
- This provides explicit control over which models use Synth infrastructure
- Easier to maintain and debug model routing issues
"""

from typing import List, Set

# Qwen3 model families supported by Synth
QWEN3_MODELS: List[str] = [
    # Qwen3 base models
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",

    # Qwen3 specialized variants
    "Qwen/Qwen3-4B-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
]

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
