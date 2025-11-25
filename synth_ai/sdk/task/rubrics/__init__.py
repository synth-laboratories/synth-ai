"""Rubric schema, loading, and scoring helpers for Task Apps.

This module provides:
- Flexible rubric models (Criterion, Rubric) for general task app use
- Strict validators (StrictCriterion, StrictRubric) for step-wise judges
- Loading utilities supporting JSON, YAML, and HTTP sources
- Blending utilities for composing rubrics
- Scoring utilities for events and outcomes
"""

# Core models (flexible validation)
# Loading and blending
from .loaders import blend_rubrics, load_rubric
from .models import Criterion, Rubric

# Scoring
from .scoring import score_events_against_rubric, score_outcome_against_rubric

# Strict validators (for judge configs)
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
    # Scoring
    "score_events_against_rubric",
    "score_outcome_against_rubric",
    # Strict validators
    "StrictCriterion",
    "StrictRubric",
    "ValidationError",
    "validate_rubric_dict",
    "validate_rubric_file",
    "validate_rubric_files",
]

# Maintain backwards compatibility
# Old code may import these names expecting the flexible variants
RubricCriterion = StrictCriterion
RubricSpec = StrictRubric




