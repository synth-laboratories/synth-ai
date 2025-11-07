"""Unit tests for spec validation."""

import pytest
from synth_ai.spec.dataclasses import (
    Spec,
    Metadata,
    Principle,
    Rule,
    Constraints,
    Example,
    GlossaryItem,
)
from synth_ai.spec.validation import SpecValidator, SpecValidationError, validate_spec_dict


class TestSpecValidator:
    """Tests for SpecValidator."""
    
    def test_valid_minimal_spec(self):
        """Test validation of a minimal valid spec."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            )
        )
        
        validator = SpecValidator()
        assert validator.validate(spec) is True
        assert len(validator.errors) == 0
    
    def test_missing_metadata_id(self):
        """Test that missing metadata ID raises error."""
        spec = Spec(
            metadata=Metadata(
                id="",
                title="Test Spec",
                version="1.0.0",
            )
        )
        
        validator = SpecValidator()
        with pytest.raises(SpecValidationError, match="'id' is required"):
            validator.validate(spec)
    
    def test_invalid_semver_warning(self):
        """Test that invalid semver generates warning."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0",  # Invalid semver
            )
        )
        
        validator = SpecValidator()
        assert validator.validate(spec) is True
        assert any("semver" in w.lower() for w in validator.warnings)
    
    def test_duplicate_rule_ids(self):
        """Test that duplicate rule IDs raise error."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(id="R-1", title="Rule 1"),
                Rule(id="R-1", title="Rule 1 Duplicate"),
            ],
        )
        
        validator = SpecValidator()
        with pytest.raises(SpecValidationError, match="duplicate id"):
            validator.validate(spec)
    
    def test_rule_priority_validation(self):
        """Test that invalid rule priorities raise errors."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(id="R-1", title="Rule 1", priority=15),  # Invalid: > 10
            ],
        )
        
        validator = SpecValidator()
        with pytest.raises(SpecValidationError, match="priority must be"):
            validator.validate(spec)
    
    def test_example_kind_validation(self):
        """Test that invalid example kinds raise errors."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[
                Rule(
                    id="R-1",
                    title="Rule 1",
                    examples=[
                        Example(
                            kind="invalid",  # Should be 'good' or 'bad'
                            prompt="test",
                            response="test",
                        )
                    ],
                )
            ],
        )
        
        validator = SpecValidator()
        with pytest.raises(SpecValidationError, match="kind must be 'good' or 'bad'"):
            validator.validate(spec)
    
    def test_warnings_become_errors_in_strict_mode(self):
        """Test that warnings become errors in strict mode."""
        spec = Spec(
            metadata=Metadata(
                id="test.v1",  # Missing 'spec.' prefix (warning)
                title="Test Spec",
                version="1.0.0",
            )
        )
        
        # Non-strict: should pass with warning
        validator = SpecValidator()
        assert validator.validate(spec, strict=False) is True
        assert len(validator.warnings) > 0
        
        # Strict: should fail
        validator_strict = SpecValidator()
        with pytest.raises(SpecValidationError):
            validator_strict.validate(spec, strict=True)
    
    def test_glossary_duplicate_terms(self):
        """Test that duplicate glossary terms raise errors."""
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            glossary=[
                GlossaryItem(term="test", definition="Definition 1"),
                GlossaryItem(term="test", definition="Definition 2"),
            ],
        )
        
        validator = SpecValidator()
        with pytest.raises(SpecValidationError, match="duplicate term"):
            validator.validate(spec)


class TestValidateSpecDict:
    """Tests for validate_spec_dict function."""
    
    def test_valid_dict(self):
        """Test validation of valid spec dictionary."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            }
        }
        
        errors = validate_spec_dict(data)
        assert len(errors) == 0
    
    def test_missing_metadata(self):
        """Test that missing metadata key raises error."""
        data = {"rules": []}
        
        errors = validate_spec_dict(data)
        assert any("metadata" in e.lower() for e in errors)
    
    def test_missing_metadata_fields(self):
        """Test that missing metadata fields raise errors."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                # Missing title and version
            }
        }
        
        errors = validate_spec_dict(data)
        assert any("title" in e for e in errors)
        assert any("version" in e for e in errors)
    
    def test_invalid_field_types(self):
        """Test that invalid field types raise errors."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            },
            "rules": "not a list",  # Should be a list
            "principles": {},  # Should be a list
        }
        
        errors = validate_spec_dict(data)
        assert any("rules" in e and "list" in e for e in errors)
        assert any("principles" in e and "list" in e for e in errors)


class TestSpecDataclasses:
    """Tests for spec dataclass methods."""
    
    def test_get_rule_by_id(self):
        """Test getting a rule by ID."""
        rule1 = Rule(id="R-1", title="Rule 1")
        rule2 = Rule(id="R-2", title="Rule 2")
        
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=[rule1, rule2],
        )
        
        assert spec.get_rule("R-1") == rule1
        assert spec.get_rule("R-2") == rule2
        assert spec.get_rule("R-3") is None
    
    def test_get_principle_by_id(self):
        """Test getting a principle by ID."""
        p1 = Principle(id="P-1", text="Principle 1")
        p2 = Principle(id="P-2", text="Principle 2")
        
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            principles=[p1, p2],
        )
        
        assert spec.get_principle("P-1") == p1
        assert spec.get_principle("P-2") == p2
        assert spec.get_principle("P-3") is None
    
    def test_get_high_priority_rules(self):
        """Test filtering rules by priority."""
        rules = [
            Rule(id="R-1", title="Rule 1", priority=5),
            Rule(id="R-2", title="Rule 2", priority=8),
            Rule(id="R-3", title="Rule 3", priority=10),
            Rule(id="R-4", title="Rule 4"),  # No priority
        ]
        
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            rules=rules,
        )
        
        high_priority = spec.get_high_priority_rules(min_priority=8)
        assert len(high_priority) == 2
        assert all(r.priority >= 8 for r in high_priority if r.priority)
    
    def test_get_glossary_term(self):
        """Test getting glossary terms by name or alias."""
        item = GlossaryItem(
            term="Test",
            definition="A test definition",
            aliases=["testing", "test-case"],
        )
        
        spec = Spec(
            metadata=Metadata(
                id="spec.test.v1",
                title="Test Spec",
                version="1.0.0",
            ),
            glossary=[item],
        )
        
        # By term (case-insensitive)
        assert spec.get_glossary_term("test") == item
        assert spec.get_glossary_term("Test") == item
        assert spec.get_glossary_term("TEST") == item
        
        # By alias
        assert spec.get_glossary_term("testing") == item
        assert spec.get_glossary_term("test-case") == item
        
        # Not found
        assert spec.get_glossary_term("nonexistent") is None

