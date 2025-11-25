"""System specification abstractions for synth-ai.

Provides hierarchical specification format inspired by Sean Grove's "spec as code" pattern.
Specs encode intent, policies, and rules as versioned, testable artifacts.
"""

from synth_ai.sdk.specs.dataclasses import (
    Constraints,
    Example,
    GlossaryItem,
    Interfaces,
    Metadata,
    Principle,
    Rule,
    Spec,
    TestCase,
)
from synth_ai.sdk.specs.loader import load_spec_from_dict, load_spec_from_file
from synth_ai.sdk.specs.serializer import spec_to_compact_context, spec_to_prompt_context
from synth_ai.sdk.specs.validation import (
    SpecValidationError,
    SpecValidator,
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

