"""Unit tests for spec loader."""

import json
import pytest
from pathlib import Path
from synth_ai.sdk.specs.loader import load_spec_from_dict, load_spec_from_file
from synth_ai.sdk.specs.dataclasses import Spec


class TestLoadSpecFromDict:
    """Tests for load_spec_from_dict."""
    
    def test_load_minimal_spec(self):
        """Test loading a minimal spec."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            }
        }
        
        spec = load_spec_from_dict(data)
        
        assert isinstance(spec, Spec)
        assert spec.metadata.id == "spec.test.v1"
        assert spec.metadata.title == "Test Spec"
        assert spec.metadata.version == "1.0.0"
        assert len(spec.rules) == 0
        assert len(spec.principles) == 0
    
    def test_load_spec_with_principles(self):
        """Test loading a spec with principles."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            },
            "principles": [
                {
                    "id": "P-1",
                    "text": "Principle 1",
                    "rationale": "Reason 1",
                }
            ],
        }
        
        spec = load_spec_from_dict(data)
        
        assert len(spec.principles) == 1
        assert spec.principles[0].id == "P-1"
        assert spec.principles[0].text == "Principle 1"
        assert spec.principles[0].rationale == "Reason 1"
    
    def test_load_spec_with_rules(self):
        """Test loading a spec with rules."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            },
            "rules": [
                {
                    "id": "R-1",
                    "title": "Rule 1",
                    "rationale": "Reason",
                    "priority": 8,
                    "constraints": {
                        "must": ["Do X"],
                        "must_not": ["Do Y"],
                    },
                    "examples": [
                        {
                            "kind": "good",
                            "prompt": "test prompt",
                            "response": "test response",
                        }
                    ],
                    "tests": [
                        {
                            "id": "T-1",
                            "challenge": "test challenge",
                            "asserts": ["assertion1"],
                        }
                    ],
                }
            ],
        }
        
        spec = load_spec_from_dict(data)
        
        assert len(spec.rules) == 1
        rule = spec.rules[0]
        assert rule.id == "R-1"
        assert rule.title == "Rule 1"
        assert rule.priority == 8
        assert len(rule.constraints.must) == 1
        assert len(rule.constraints.must_not) == 1
        assert len(rule.examples) == 1
        assert len(rule.tests) == 1
    
    def test_load_spec_with_glossary(self):
        """Test loading a spec with glossary."""
        data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            },
            "glossary": [
                {
                    "term": "test",
                    "definition": "A test definition",
                    "aliases": ["testing"],
                }
            ],
        }
        
        spec = load_spec_from_dict(data)
        
        assert len(spec.glossary) == 1
        assert spec.glossary[0].term == "test"
        assert spec.glossary[0].definition == "A test definition"
        assert spec.glossary[0].aliases == ["testing"]


class TestLoadSpecFromFile:
    """Tests for load_spec_from_file."""
    
    def test_load_from_nonexistent_file(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_spec_from_file("nonexistent.json")
    
    def test_load_from_valid_file(self, tmp_path):
        """Test loading from a valid JSON file."""
        spec_data = {
            "metadata": {
                "id": "spec.test.v1",
                "title": "Test Spec",
                "version": "1.0.0",
            }
        }
        
        spec_file = tmp_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)
        
        spec = load_spec_from_file(spec_file)
        
        assert isinstance(spec, Spec)
        assert spec.metadata.id == "spec.test.v1"
    
    def test_load_from_invalid_json(self, tmp_path):
        """Test that loading from invalid JSON raises JSONDecodeError."""
        spec_file = tmp_path / "invalid.json"
        with open(spec_file, "w") as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            load_spec_from_file(spec_file)

