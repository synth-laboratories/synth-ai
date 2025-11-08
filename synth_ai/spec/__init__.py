"""System specification abstractions for synth-ai.

Provides hierarchical specification format inspired by Sean Grove's "spec as code" pattern.
Specs encode intent, policies, and rules as versioned, testable artifacts.
"""

from synth_ai.spec.dataclasses import (
    Spec,
    Metadata,
    Principle,
    Rule,
    Constraints,
    Example,
    TestCase,
    Interfaces,
    GlossaryItem,
)
from synth_ai.spec.loader import load_spec_from_file, load_spec_from_dict
from synth_ai.spec.serializer import spec_to_prompt_context, spec_to_compact_context
from synth_ai.spec.validation import (
    SpecValidator,
    SpecValidationError,
    validate_spec_dict,
    validate_spec_file,
)

__all__ = [
    "Spec",
    "Metadata",
    "Principle",
    "Rule",
    "Constraints",
    "Example",
    "TestCase",
    "Interfaces",
    "GlossaryItem",
    "load_spec_from_file",
    "load_spec_from_dict",
    "spec_to_prompt_context",
    "spec_to_compact_context",
    "SpecValidator",
    "SpecValidationError",
    "validate_spec_dict",
    "validate_spec_file",
]

