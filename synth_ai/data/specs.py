"""System specification data types.

This module re-exports spec dataclasses from their original location
to provide a cleaner import path: `from synth_ai.data.specs import Spec`

Based on Sean Grove's "spec as code" pattern from AI Engineer World's Fair.
Specs are the source of truth that encode intent, policies, and rules.
"""

from __future__ import annotations

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
]


