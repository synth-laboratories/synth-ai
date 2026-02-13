"""Rubric schema, loading, and evaluation helpers for Task Apps.

This module provides:
- Flexible rubric models (Criterion, Rubric) for general task app use
- Strict validators (StrictCriterion, StrictRubric) for step-wise verifiers
- Loading utilities supporting JSON, YAML, and HTTP sources
- Blending utilities for composing rubrics
- Evaluation utilities for events and outcomes
"""

# Core models (flexible validation)
# Loading and blending
# Evaluation
from .evaluation import evaluate_events_against_rubric, evaluate_outcome_against_rubric
from .loaders import blend_rubrics, load_rubric
from .models import Criterion, Rubric

# Strict validators (for verifier configs)
from .strict import (
    StrictCriterion,
    StrictRubric,
    ValidationError,
    validate_rubric_dict,
    validate_rubric_file,
    validate_rubric_files,
)

__all__ = [
    # Flexible models
    "Criterion",
    "Rubric",
    # Loaders
    "load_rubric",
    "blend_rubrics",
    # Evaluation
    "evaluate_events_against_rubric",
    "evaluate_outcome_against_rubric",
    # Strict validators
    "StrictCriterion",
    "StrictRubric",
    "ValidationError",
    "validate_rubric_dict",
    "validate_rubric_file",
    "validate_rubric_files",
]

# Breaking change: legacy rubric aliases removed. Use Criterion/Rubric directly.
