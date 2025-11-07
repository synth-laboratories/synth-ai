"""Dataclasses for hierarchical system specifications.

Based on Sean Grove's "spec as code" pattern from AI Engineer World's Fair.
Specs are the source of truth that encode intent, policies, and rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Principle:
    """A high-level principle or value that guides behavior."""

    id: str
    text: str
    rationale: Optional[str] = None


@dataclass
class Example:
    """An example demonstrating good or bad behavior."""

    kind: str  # "good" | "bad"
    prompt: str
    response: str
    description: Optional[str] = None


@dataclass
class TestCase:
    """A test case to verify adherence to a rule."""

    id: str
    challenge: str
    asserts: List[str] = field(default_factory=list)
    expected_behavior: Optional[str] = None


@dataclass
class Constraints:
    """Positive and negative constraints for a rule."""

    must: List[str] = field(default_factory=list)
    must_not: List[str] = field(default_factory=list)
    should: List[str] = field(default_factory=list)
    should_not: List[str] = field(default_factory=list)


@dataclass
class Rule:
    """A specific policy or rule with constraints, examples, and tests."""

    id: str
    title: str
    rationale: Optional[str] = None
    constraints: Constraints = field(default_factory=Constraints)
    examples: List[Example] = field(default_factory=list)
    tests: List[TestCase] = field(default_factory=list)
    priority: Optional[int] = None  # Higher = more important


@dataclass
class Metadata:
    """Metadata about the specification."""

    id: str
    title: str
    version: str
    owner: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    scope: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Interfaces:
    """Interface definitions for the system."""

    io_modes: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class GlossaryItem:
    """A term definition in the glossary."""

    term: str
    definition: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class Spec:
    """A complete system specification.
    
    Hierarchical structure:
    - Metadata (versioning, ownership, imports)
    - Principles (high-level values)
    - Rules (specific policies with constraints, examples, tests)
    - Interfaces (capabilities, modes)
    - Glossary (domain terminology)
    - Changelog (version history)
    """

    metadata: Metadata
    principles: List[Principle] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    interfaces: Interfaces = field(default_factory=Interfaces)
    glossary: List[GlossaryItem] = field(default_factory=list)
    changelog: List[Dict[str, Any]] = field(default_factory=list)

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_principle(self, principle_id: str) -> Optional[Principle]:
        """Get a principle by ID."""
        for principle in self.principles:
            if principle.id == principle_id:
                return principle
        return None

    def get_glossary_term(self, term: str) -> Optional[GlossaryItem]:
        """Get a glossary item by term or alias."""
        term_lower = term.lower()
        for item in self.glossary:
            if item.term.lower() == term_lower:
                return item
            if any(alias.lower() == term_lower for alias in item.aliases):
                return item
        return None

    def get_high_priority_rules(self, min_priority: int = 8) -> List[Rule]:
        """Get rules with priority >= min_priority."""
        return [
            rule for rule in self.rules
            if rule.priority is not None and rule.priority >= min_priority
        ]


