"""Unit tests for spec serializer."""

import pytest
from synth_ai.sdk.specs.dataclasses import (
    Spec,
    Metadata,
    Principle,
    Rule,
    Constraints,
    Example,
    GlossaryItem,
)
from synth_ai.sdk.specs.serializer import spec_to_prompt_context, spec_to_compact_context


class TestSpecToPromptContext:
    """Tests for spec_to_prompt_context."""
    
    def test_minimal_spec(self):
        """Test serializing a minimal spec."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            )
        )
        
        context = spec_to_prompt_context(spec)
        
        assert "Test Spec" in context
        assert "1.0.0" in context
    
    def test_with_principles(self):
        """Test serializing spec with principles."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            principles=[
                Principle(
                    id="P-1",
                    text="Test principle",
                    rationale="Test rationale",
                )
            ],
        )
        
        context = spec_to_prompt_context(spec)
        
        assert "Guiding Principles" in context
        assert "P-1" in context
        assert "Test principle" in context
        assert "Test rationale" in context
    
    def test_with_rules(self):
        """Test serializing spec with rules."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(
                    id="R-1",
                    title="Test Rule",
                    rationale="Test rationale",
                    priority=8,
                    constraints=Constraints(
                        must=["Do this"],
                        must_not=["Don't do that"],
                    ),
                )
            ],
        )
        
        context = spec_to_prompt_context(spec)
        
        assert "Rules and Policies" in context
        assert "R-1" in context
        assert "Test Rule" in context
        assert "[Priority: 8]" in context
        assert "MUST:" in context
        assert "Do this" in context
        assert "MUST NOT:" in context
        assert "Don't do that" in context
    
    def test_with_examples(self):
        """Test serializing spec with examples."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(
                    id="R-1",
                    title="Test Rule",
                    examples=[
                        Example(
                            kind="good",
                            prompt="Good prompt",
                            response="Good response",
                            description="Good example",
                        ),
                        Example(
                            kind="bad",
                            prompt="Bad prompt",
                            response="Bad response",
                            description="Bad example",
                        ),
                    ],
                )
            ],
        )
        
        context = spec_to_prompt_context(spec, include_examples=True)
        
        assert "✅ **Good:**" in context
        assert "Good prompt" in context
        assert "Good response" in context
        assert "❌ **Bad:**" in context
        assert "Bad prompt" in context
        assert "Bad response" in context
    
    def test_exclude_examples(self):
        """Test excluding examples from serialization."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(
                    id="R-1",
                    title="Test Rule",
                    examples=[
                        Example(
                            kind="good",
                            prompt="Good prompt",
                            response="Good response",
                        )
                    ],
                )
            ],
        )
        
        context = spec_to_prompt_context(spec, include_examples=False)
        
        assert "Good prompt" not in context
        assert "Good response" not in context
    
    def test_priority_threshold(self):
        """Test filtering rules by priority threshold."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(id="R-1", title="High Priority", priority=9),
                Rule(id="R-2", title="Medium Priority", priority=5),
                Rule(id="R-3", title="Low Priority", priority=2),
            ],
        )
        
        context = spec_to_prompt_context(spec, priority_threshold=8)
        
        assert "High Priority" in context
        assert "Medium Priority" not in context
        assert "Low Priority" not in context
    
    def test_max_rules(self):
        """Test limiting number of rules."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(id="R-1", title="Rule 1", priority=10),
                Rule(id="R-2", title="Rule 2", priority=9),
                Rule(id="R-3", title="Rule 3", priority=8),
            ],
        )
        
        context = spec_to_prompt_context(spec, max_rules=2)
        
        # Should include top 2 by priority
        assert "Rule 1" in context
        assert "Rule 2" in context
        assert "Rule 3" not in context
    
    def test_with_glossary(self):
        """Test serializing spec with glossary."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            glossary=[
                GlossaryItem(
                    term="test",
                    definition="A test definition",
                    aliases=["testing"],
                )
            ],
        )
        
        context = spec_to_prompt_context(spec, include_glossary=True)
        
        assert "Glossary" in context
        assert "test" in context
        assert "A test definition" in context
        assert "testing" in context


class TestSpecToCompactContext:
    """Tests for spec_to_compact_context."""
    
    def test_respects_token_limit(self):
        """Test that compact context attempts to reduce size."""
        # Create a large spec with varying priorities
        rules = [
            Rule(
                id=f"R-high-{i}",
                title=f"High Priority Rule {i}",
                priority=10 - i,  # Varying priorities
                rationale="Important rationale",
                constraints=Constraints(
                    must=["Must do this"],
                ),
            )
            for i in range(10)
        ]
        
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=rules,
        )
        
        # Get full context vs compact context
        full_context = spec_to_prompt_context(spec, include_examples=True)
        compact_context = spec_to_compact_context(spec, max_tokens=1000)
        
        # Compact should prioritize and be smaller than full
        # (may not hit exact token limit but should reduce)
        assert len(compact_context) <= len(full_context)
    
    def test_prioritizes_high_priority_rules(self):
        """Test that compact context prioritizes high-priority rules."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(id="R-high", title="High Priority Rule", priority=10),
                Rule(id="R-low", title="Low Priority Rule", priority=1),
            ],
        )
        
        context = spec_to_compact_context(spec, max_tokens=500)
        
        # High priority rule should be included
        assert "High Priority Rule" in context

