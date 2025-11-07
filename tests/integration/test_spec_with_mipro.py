"""Integration test for system specs with MIPRO."""

import pytest
from pathlib import Path

from synth_ai.spec import load_spec_from_file, spec_to_compact_context, validate_spec_file


class TestSpecIntegration:
    """Integration tests for spec with MIPRO."""
    
    def test_banking77_spec_loads(self):
        """Test that Banking77 spec loads correctly."""
        spec_path = Path("examples/blog_posts/mipro/configs/banking77_spec.json")
        
        # Should not raise
        spec = load_spec_from_file(spec_path)
        
        assert spec.metadata.id == "spec.banking77.v1"
        assert len(spec.principles) > 0
        assert len(spec.rules) > 0
    
    def test_banking77_spec_validates(self):
        """Test that Banking77 spec passes validation."""
        spec_path = Path("examples/blog_posts/mipro/configs/banking77_spec.json")
        
        # Should not raise
        assert validate_spec_file(str(spec_path), strict=False) is True
    
    def test_banking77_pipeline_spec_loads(self):
        """Test that Banking77 pipeline spec loads correctly."""
        spec_path = Path("examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json")
        
        # Should not raise
        spec = load_spec_from_file(spec_path)
        
        assert spec.metadata.id == "spec.banking77_pipeline.v1"
        assert "pipeline" in spec.metadata.scope.lower()
        assert len(spec.principles) >= 3
        assert len(spec.rules) >= 6
        
        # Check for pipeline-specific rules
        rule_ids = [r.id for r in spec.rules]
        assert "R-analyzer-complexity" in rule_ids
        assert "R-stage-coordination" in rule_ids
    
    def test_banking77_pipeline_spec_validates(self):
        """Test that Banking77 pipeline spec passes validation."""
        spec_path = Path("examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json")
        
        # Should not raise
        assert validate_spec_file(str(spec_path), strict=False) is True
    
    def test_spec_serialization_to_context(self):
        """Test spec serialization for use in MIPRO meta-prompts."""
        spec_path = Path("examples/blog_posts/mipro/configs/banking77_spec.json")
        spec = load_spec_from_file(spec_path)
        
        # Full context
        full_context = spec_to_compact_context(spec, max_tokens=10000)
        assert len(full_context) > 0
        assert spec.metadata.title in full_context
        assert "Principles" in full_context or "principles" in full_context.lower()
        
        # Compact context
        compact_context = spec_to_compact_context(spec, max_tokens=2000)
        assert len(compact_context) > 0
        assert len(compact_context) <= len(full_context)
    
    def test_spec_high_priority_rules_only(self):
        """Test filtering to high-priority rules only."""
        spec_path = Path("examples/blog_posts/mipro/configs/banking77_spec.json")
        spec = load_spec_from_file(spec_path)
        
        high_priority_rules = spec.get_high_priority_rules(min_priority=8)
        
        # Should have some high-priority rules
        assert len(high_priority_rules) > 0
        
        # All should have priority >= 8
        for rule in high_priority_rules:
            assert rule.priority is not None
            assert rule.priority >= 8
    
    def test_spec_glossary_lookup(self):
        """Test glossary term lookup."""
        spec_path = Path("examples/blog_posts/mipro/configs/banking77_spec.json")
        spec = load_spec_from_file(spec_path)
        
        # Look up a term
        intent_term = spec.get_glossary_term("intent")
        assert intent_term is not None
        assert intent_term.definition
        
        # Look up by alias
        category_term = spec.get_glossary_term("category")
        assert category_term is not None
        assert category_term.term == "intent"  # Should resolve to main term
    
    def test_pipeline_spec_covers_both_stages(self):
        """Test that pipeline spec addresses both analyzer and classifier stages."""
        spec_path = Path("examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json")
        spec = load_spec_from_file(spec_path)
        
        # Should have analyzer-specific content
        context = spec_to_compact_context(spec, max_tokens=10000)
        assert "query_analyzer" in context or "analyzer" in context.lower()
        assert "classifier" in context.lower()
        assert "stage" in context.lower()
        
        # Should have coordination rule
        coord_rule = spec.get_rule("R-stage-coordination")
        assert coord_rule is not None
        assert "stages" in coord_rule.title.lower() or "coordinate" in coord_rule.title.lower()
    
    @pytest.mark.skip(reason="Requires backend to be running")
    def test_spec_with_mipro_backend(self):
        """Integration test with actual MIPRO backend.
        
        This test is skipped by default but documents the expected integration pattern.
        """
        # This would require:
        # 1. Backend server running
        # 2. Task app deployed
        # 3. Valid API keys
        # 4. TOML config with spec_path set
        pass

