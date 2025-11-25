"""Specs SDK - system specification loading and validation.

This module provides utilities for working with system specifications:
- Loading specs from files or dicts
- Validating spec structure
- Converting specs to prompt context

Example:
    from synth_ai.sdk.specs import load_spec_from_file, spec_to_prompt_context
    
    spec = load_spec_from_file("my_spec.yaml")
    context = spec_to_prompt_context(spec)
"""

from __future__ import annotations

# Re-export from existing location
from synth_ai.spec import (
    Spec,
    load_spec_from_dict,
    load_spec_from_file,
    spec_to_compact_context,
    spec_to_prompt_context,
    SpecValidationError,
    SpecValidator,
    validate_spec_dict,
    validate_spec_file,
)

__all__ = [
    # Types
    "Spec",
    # Loading
    "load_spec_from_dict",
    "load_spec_from_file",
    # Serialization
    "spec_to_prompt_context",
    "spec_to_compact_context",
    # Validation
    "SpecValidator",
    "SpecValidationError",
    "validate_spec_dict",
    "validate_spec_file",
]

