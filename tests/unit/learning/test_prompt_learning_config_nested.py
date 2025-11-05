"""Unit tests for nested GEPA config structure in SDK."""
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from synth_ai.api.train.configs.prompt_learning import (
    GEPAConfig,
    GEPARolloutConfig,
    GEPAEvaluationConfig,
    GEPAMutationConfig,
    GEPAPopulationConfig,
    GEPAArchiveConfig,
    GEPATokenConfig,
    PromptLearningConfig,
)


pytestmark = pytest.mark.unit


def _create_test_toml(config_dict: Dict[str, Any]) -> Path:
    """Helper to create a temporary TOML config file."""
    try:
        import tomli_w
    except ImportError:
        import tomllib as tomli_w  # type: ignore
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.toml', delete=False) as f:
        tomli_w.dump(config_dict, f)
        return Path(f.name)


class TestNestedGEPAConfigDataclasses:
    """Tests for nested GEPA config dataclasses."""
    
    def test_rollout_config_creation(self):
        """Test GEPARolloutConfig creation."""
        config = GEPARolloutConfig(
            budget=1000,
            max_concurrent=20,
            minibatch_size=8,
        )
        assert config.budget == 1000
        assert config.max_concurrent == 20
        assert config.minibatch_size == 8
    
    def test_evaluation_config_creation(self):
        """Test GEPAEvaluationConfig creation."""
        config = GEPAEvaluationConfig(
            seeds=[1, 2, 3],
            validation_seeds=[10, 11, 12],
            test_pool=[20, 21, 22],
            validation_pool="validation",
            validation_top_k=5,
        )
        assert config.seeds == [1, 2, 3]
        assert config.validation_seeds == [10, 11, 12]
        assert config.test_pool == [20, 21, 22]
        assert config.validation_pool == "validation"
        assert config.validation_top_k == 5
    
    def test_mutation_config_creation(self):
        """Test GEPAMutationConfig creation."""
        config = GEPAMutationConfig(
            rate=0.4,
            llm_model="gpt-4",
            llm_provider="openai",
            llm_inference_url="https://api.openai.com/v1",
            prompt="Custom mutation prompt",
        )
        assert config.rate == 0.4
        assert config.llm_model == "gpt-4"
        assert config.llm_provider == "openai"
        assert config.llm_inference_url == "https://api.openai.com/v1"
        assert config.prompt == "Custom mutation prompt"
    
    def test_population_config_creation(self):
        """Test GEPAPopulationConfig creation."""
        config = GEPAPopulationConfig(
            initial_size=30,
            num_generations=20,
            children_per_generation=15,
            crossover_rate=0.6,
            selection_pressure=1.5,
            patience_generations=5,
        )
        assert config.initial_size == 30
        assert config.num_generations == 20
        assert config.children_per_generation == 15
        assert config.crossover_rate == 0.6
        assert config.selection_pressure == 1.5
        assert config.patience_generations == 5
    
    def test_archive_config_creation(self):
        """Test GEPAArchiveConfig creation."""
        config = GEPAArchiveConfig(
            size=100,
            pareto_set_size=50,
            pareto_eps=1e-5,
            feedback_fraction=0.6,
        )
        assert config.size == 100
        assert config.pareto_set_size == 50
        assert config.pareto_eps == 1e-5
        assert config.feedback_fraction == 0.6
    
    def test_token_config_creation(self):
        """Test GEPATokenConfig creation."""
        config = GEPATokenConfig(
            max_limit=2000,
            counting_model="gpt-3.5-turbo",
            enforce_pattern_limit=False,
            max_spend_usd=50.0,
        )
        assert config.max_limit == 2000
        assert config.counting_model == "gpt-3.5-turbo"
        assert config.enforce_pattern_limit is False
        assert config.max_spend_usd == 50.0


class TestGEPAConfigNested:
    """Tests for GEPAConfig with nested structure."""
    
    def test_gepa_config_with_nested_rollout(self):
        """Test GEPAConfig with nested rollout config."""
        rollout = GEPARolloutConfig(budget=500, max_concurrent=10, minibatch_size=4)
        config = GEPAConfig(
            env_name="test",
            rollout=rollout,
        )
        
        # Should access via helper methods
        assert config._get_rollout_budget() == 500
        assert config._get_max_concurrent_rollouts() == 10
        assert config._get_minibatch_size() == 4
    
    def test_gepa_config_with_nested_evaluation(self):
        """Test GEPAConfig with nested evaluation config."""
        evaluation = GEPAEvaluationConfig(
            seeds=[1, 2, 3],
            validation_seeds=[10, 11],
            test_pool=[20, 21],
        )
        config = GEPAConfig(
            env_name="test",
            evaluation=evaluation,
        )
        
        assert config._get_evaluation_seeds() == [1, 2, 3]
        assert config._get_validation_seeds() == [10, 11]
        assert config._get_test_pool() == [20, 21]
    
    def test_gepa_config_with_nested_mutation(self):
        """Test GEPAConfig with nested mutation config."""
        mutation = GEPAMutationConfig(
            rate=0.5,
            llm_model="gpt-4",
            llm_provider="groq",
        )
        config = GEPAConfig(
            env_name="test",
            mutation=mutation,
        )
        
        assert config._get_mutation_rate() == 0.5
        assert config._get_mutation_llm_model() == "gpt-4"
        assert config._get_mutation_llm_provider() == "groq"
    
    def test_gepa_config_nested_overrides_flat(self):
        """Test that nested config values override flat ones."""
        rollout = GEPARolloutConfig(budget=1000, max_concurrent=20, minibatch_size=8)
        config = GEPAConfig(
            env_name="test",
            rollout=rollout,
            rollout_budget=500,  # Flat value (should be overridden)
            max_concurrent_rollouts=10,  # Flat value (should be overridden)
        )
        
        # Nested should win
        assert config._get_rollout_budget() == 1000
        assert config._get_max_concurrent_rollouts() == 20
    
    def test_gepa_config_flat_fallback(self):
        """Test that flat values are used when nested is not provided."""
        config = GEPAConfig(
            env_name="test",
            rollout_budget=300,
            max_concurrent_rollouts=15,
            mutation_rate=0.35,
        )
        
        # Should fall back to flat values
        assert config._get_rollout_budget() == 300
        assert config._get_max_concurrent_rollouts() == 15
        assert config._get_mutation_rate() == 0.35
    
    def test_gepa_config_defaults(self):
        """Test that defaults are used when neither nested nor flat provided."""
        config = GEPAConfig(env_name="test")
        
        # Should use defaults
        assert config._get_max_concurrent_rollouts() == 20
        assert config._get_minibatch_size() == 8
        assert config._get_mutation_rate() == 0.3
        assert config._get_initial_population_size() == 20
        assert config._get_num_generations() == 10


class TestGEPAConfigFromMapping:
    """Tests for GEPAConfig.from_mapping with nested structure."""
    
    def test_from_mapping_nested_structure(self):
        """Test loading nested structure from mapping."""
        data = {
            "env_name": "banking77",
            "rollout": {
                "budget": 1000,
                "max_concurrent": 20,
                "minibatch_size": 8,
            },
            "evaluation": {
                "seeds": [1, 2, 3, 4, 5],
                "validation_seeds": [10, 11, 12],
            },
            "mutation": {
                "rate": 0.4,
                "llm_model": "gpt-4",
            },
        }
        
        config = GEPAConfig.from_mapping(data)
        
        assert config.rollout is not None
        assert isinstance(config.rollout, GEPARolloutConfig)
        assert config.rollout.budget == 1000
        
        assert config.evaluation is not None
        assert isinstance(config.evaluation, GEPAEvaluationConfig)
        assert config.evaluation.seeds == [1, 2, 3, 4, 5]
        
        assert config.mutation is not None
        assert isinstance(config.mutation, GEPAMutationConfig)
        assert config.mutation.rate == 0.4
    
    def test_from_mapping_flat_structure(self):
        """Test loading flat structure from mapping."""
        data = {
            "env_name": "banking77",
            "rollout_budget": 500,
            "max_concurrent_rollouts": 10,
            "mutation_rate": 0.35,
            "evaluation_seeds": [1, 2, 3],
        }
        
        config = GEPAConfig.from_mapping(data)
        
        # Flat values should be set
        assert config.rollout_budget == 500
        assert config.max_concurrent_rollouts == 10
        assert config.mutation_rate == 0.35
        assert config.evaluation_seeds == [1, 2, 3]
    
    def test_from_mapping_mixed_structure(self):
        """Test loading mixed nested and flat structure."""
        data = {
            "env_name": "banking77",
            "rollout_budget": 300,  # Flat
            "rollout": {  # Nested (should override flat via helper methods)
                "budget": 1000,
                "max_concurrent": 20,
            },
            "evaluation_seeds": [99, 98],  # Flat
        }
        
        config = GEPAConfig.from_mapping(data)
        
        # Both should be set, helpers prefer nested
        assert config._get_rollout_budget() == 1000
        assert config._get_max_concurrent_rollouts() == 20
        
        # Flat evaluation_seeds should be set
        assert config.evaluation_seeds == [99, 98]


class TestPromptLearningConfigNested:
    """Tests for PromptLearningConfig with nested GEPA."""
    
    def test_prompt_learning_config_with_nested_gepa(self):
        """Test that PromptLearningConfig correctly handles nested GEPA."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8102",
                "gepa": {
                    "env_name": "test",
                    "rollout": {
                        "budget": 1000,
                    },
                    "evaluation": {
                        "seeds": [1, 2, 3],
                    },
                },
            }
        }
        
        config = PromptLearningConfig.from_mapping(data)
        
        assert config.algorithm == "gepa"
        assert config.gepa is not None
        assert isinstance(config.gepa, GEPAConfig)
        assert config.gepa._get_rollout_budget() == 1000
        assert config.gepa._get_evaluation_seeds() == [1, 2, 3]

