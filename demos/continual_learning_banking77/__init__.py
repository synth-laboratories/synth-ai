"""Banking77 Continual Learning Comparison Demo.

This module demonstrates the comparison between classic GEPA (non-continual)
and MIPRO continual learning approaches on the Banking77 dataset with
progressive data splits.

Splits:
    Split 1: 2 intents
    Split 2: 7 intents (superset of Split 1)
    Split 3: 27 intents (superset of Split 2)
    Split 4: 77 intents (full dataset)

Usage:
    python run_comparison.py  # Run full comparison
    python run_classic_gepa.py  # Run classic GEPA only
    python run_mipro_continual.py  # Run MIPRO continual only
    python analyze_results.py  # Analyze existing results
"""

from .data_splits import (
    SPLITS,
    SPLIT_1_INTENTS,
    SPLIT_2_INTENTS,
    SPLIT_3_INTENTS,
    SPLIT_4_INTENTS,
    Banking77SplitDataset,
    get_split_intents,
    get_split_size,
    format_available_intents,
)

__all__ = [
    "SPLITS",
    "SPLIT_1_INTENTS",
    "SPLIT_2_INTENTS",
    "SPLIT_3_INTENTS",
    "SPLIT_4_INTENTS",
    "Banking77SplitDataset",
    "get_split_intents",
    "get_split_size",
    "format_available_intents",
]
