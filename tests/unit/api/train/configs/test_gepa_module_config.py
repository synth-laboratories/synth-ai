"""Unit tests for GEPA multi-stage module configuration."""
import pytest
from synth_ai.api.train.configs.prompt_learning import (
    GEPAConfig,
    GEPAModuleConfig,
)


class TestGEPAModuleConfig:
    """Test GEPA module configuration."""
    
    def test_valid_module_config(self):
        """Test valid module configuration."""
        config = GEPAModuleConfig(
            module_id="classifier",
            max_instruction_slots=3,
            max_tokens=1024,
            allowed_tools=["classify", "format"],
            policy={"model": "gpt-4o-mini", "provider": "openai"},
        )
        assert config.module_id == "classifier"
        assert config.max_instruction_slots == 3
        assert config.max_tokens == 1024
        assert config.allowed_tools == ["classify", "format"]
    
    def test_module_config_defaults(self):
        """Test module configuration with defaults."""
        config = GEPAModuleConfig(
            module_id="stage1",
            policy={"model": "gpt-4o-mini", "provider": "openai"},
        )
        assert config.module_id == "stage1"
        assert config.max_instruction_slots == 3  # Default
        assert config.max_tokens is None
        assert config.allowed_tools is None
    
    def test_module_id_validation_empty(self):
        """Test that empty module_id is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GEPAModuleConfig(
                module_id="  ",
                policy={"model": "gpt-4o-mini", "provider": "openai"},
            )
    
    def test_max_instruction_slots_validation(self):
        """Test that invalid max_instruction_slots is rejected."""
        with pytest.raises(ValueError, match="must be >= 1"):
            GEPAModuleConfig(
                module_id="stage1",
                max_instruction_slots=0,
                policy={"model": "gpt-4o-mini", "provider": "openai"},
            )
    
    def test_module_id_stripped(self):
        """Test that module_id is stripped of whitespace."""
        config = GEPAModuleConfig(
            module_id="  stage1  ",
            policy={"model": "gpt-4o-mini", "provider": "openai"},
        )
        assert config.module_id == "stage1"


class TestGEPAConfigWithModules:
    """Test GEPA config with multi-stage modules."""
    
    def test_single_stage_config(self):
        """Test GEPA config without modules (single-stage, backwards compatible)."""
        config = GEPAConfig(
            env_name="banking77",
            proposer_type="dspy",
        )
        assert config.env_name == "banking77"
        assert config.modules is None  # No modules = single-stage
    
    def test_multi_stage_config(self):
        """Test GEPA config with multiple modules."""
        modules = [
            GEPAModuleConfig(
                module_id="query_analyzer",
                max_instruction_slots=2,
                policy={"model": "gpt-4o-mini", "provider": "openai"},
            ),
            GEPAModuleConfig(
                module_id="classifier",
                max_instruction_slots=3,
                policy={"model": "gpt-4o-mini", "provider": "openai"},
            ),
        ]
        config = GEPAConfig(
            env_name="banking77_pipeline",
            modules=modules,
        )
        assert config.env_name == "banking77_pipeline"
        assert config.modules is not None
        assert len(config.modules) == 2
        assert config.modules[0].module_id == "query_analyzer"
        assert config.modules[1].module_id == "classifier"
    
    def test_from_mapping_with_modules(self):
        """Test loading GEPA config from dict with modules."""
        data = {
            "env_name": "banking77_pipeline",
            "proposer_type": "dspy",
            "modules": [
                {
                    "module_id": "query_analyzer",
                    "max_instruction_slots": 2,
                    "max_tokens": 512,
                    "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                },
                {
                    "module_id": "classifier",
                    "max_instruction_slots": 3,
                    "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                },
            ],
            "rollout": {"budget": 1000, "max_concurrent": 20},
        }
        config = GEPAConfig.from_mapping(data)
        
        assert config.env_name == "banking77_pipeline"
        assert config.modules is not None
        assert len(config.modules) == 2
        
        # Check first module
        assert config.modules[0].module_id == "query_analyzer"
        assert config.modules[0].max_instruction_slots == 2
        assert config.modules[0].max_tokens == 512
        
        # Check second module
        assert config.modules[1].module_id == "classifier"
        assert config.modules[1].max_instruction_slots == 3
        assert config.modules[1].max_tokens is None
        
        # Check rollout config also loaded
        assert config.rollout is not None
        assert config.rollout.budget == 1000
    
    def test_from_mapping_without_modules(self):
        """Test loading GEPA config from dict without modules (backwards compatible)."""
        data = {
            "env_name": "banking77",
            "rollout": {"budget": 500},
        }
        config = GEPAConfig.from_mapping(data)
        
        assert config.env_name == "banking77"
        assert config.modules is None  # No modules provided
        assert config.rollout is not None
        assert config.rollout.budget == 500
    
    def test_modules_in_nested_data(self):
        """Test that modules are recognized as nested data."""
        data = {
            "env_name": "test",
            "rng_seed": 42,
            "modules": [
                {
                    "module_id": "stage1",
                    "max_instruction_slots": 1,
                    "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                },
            ],
        }
        config = GEPAConfig.from_mapping(data)
        
        assert config.env_name == "test"
        assert config.rng_seed == 42
        assert config.modules is not None
        assert len(config.modules) == 1
        assert isinstance(config.modules[0], GEPAModuleConfig)


