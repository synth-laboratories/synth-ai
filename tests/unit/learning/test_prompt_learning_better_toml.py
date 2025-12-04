"""Tests for better-toml improvements: range notation and deprecated field validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_ai.sdk.api.train.configs.prompt_learning import (
    GEPAConfig,
    GEPAEvaluationConfig,
    GEPAMutationConfig,
    MIPROConfig,
    PromptLearningConfig,
    SeedRange,
)

pytestmark = pytest.mark.unit


class TestSeedRangeNotation:
    """Tests for seed range notation feature."""

    def test_seed_range_basic(self) -> None:
        """Test basic seed range creation."""
        seed_range = SeedRange(start=0, end=10)
        assert seed_range.start == 0
        assert seed_range.end == 10
        assert seed_range.step == 1

    def test_seed_range_with_step(self) -> None:
        """Test seed range with custom step."""
        seed_range = SeedRange(start=0, end=100, step=2)
        assert seed_range.start == 0
        assert seed_range.end == 100
        assert seed_range.step == 2

    def test_seed_range_to_list_basic(self) -> None:
        """Test converting seed range to list."""
        seed_range = SeedRange(start=0, end=5)
        result = seed_range.to_list()
        assert result == [0, 1, 2, 3, 4]

    def test_seed_range_to_list_with_step(self) -> None:
        """Test converting seed range with step to list."""
        seed_range = SeedRange(start=0, end=10, step=2)
        result = seed_range.to_list()
        assert result == [0, 2, 4, 6, 8]

    def test_seed_range_large_range(self) -> None:
        """Test seed range with large range."""
        seed_range = SeedRange(start=0, end=1000)
        result = seed_range.to_list()
        assert len(result) == 1000
        assert result[0] == 0
        assert result[-1] == 999

    def test_gepa_evaluation_seeds_with_range(self) -> None:
        """Test GEPA evaluation config accepts range notation for seeds."""
        config = GEPAEvaluationConfig(
            seeds={"start": 0, "end": 30},
            validation_seeds={"start": 30, "end": 50},
        )
        assert config.seeds == list(range(0, 30))
        assert config.validation_seeds == list(range(30, 50))

    def test_gepa_evaluation_seeds_with_array(self) -> None:
        """Test GEPA evaluation config still accepts array notation."""
        config = GEPAEvaluationConfig(
            seeds=[0, 1, 2, 3, 4],
            validation_seeds=[5, 6, 7, 8, 9],
        )
        assert config.seeds == [0, 1, 2, 3, 4]
        assert config.validation_seeds == [5, 6, 7, 8, 9]

    def test_mipro_seeds_with_range(self) -> None:
        """Test MIPRO config accepts range notation for seed pools."""
        config = MIPROConfig(
            bootstrap_train_seeds={"start": 0, "end": 10},
            online_pool={"start": 10, "end": 25},
            test_pool={"start": 25, "end": 30},
            reference_pool={"start": 30, "end": 35},
        )
        assert config.bootstrap_train_seeds == list(range(0, 10))
        assert config.online_pool == list(range(10, 25))
        assert config.test_pool == list(range(25, 30))
        assert config.reference_pool == list(range(30, 35))

    def test_mipro_seeds_with_array(self) -> None:
        """Test MIPRO config still accepts array notation."""
        config = MIPROConfig(
            bootstrap_train_seeds=[0, 1, 2],
            online_pool=[3, 4, 5],
        )
        assert config.bootstrap_train_seeds == [0, 1, 2]
        assert config.online_pool == [3, 4, 5]

    def test_seed_range_from_toml(self) -> None:
        """Test loading seed range from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
env_name = "test"

[prompt_learning.gepa.evaluation]
seeds = { start = 0, end = 30 }
validation_seeds = { start = 30, end = 50 }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.gepa is not None
            assert config.gepa.evaluation is not None
            assert config.gepa.evaluation.seeds == list(range(0, 30))
            assert config.gepa.evaluation.validation_seeds == list(range(30, 50))
        finally:
            path.unlink()

    def test_seed_range_invalid_type(self) -> None:
        """Test that invalid seed type raises error."""
        with pytest.raises(ValueError, match="Seeds must be"):
            GEPAEvaluationConfig(seeds="invalid")  # type: ignore


class TestDeprecatedFieldValidation:
    """Tests for deprecated field validation."""

    def test_deprecated_display_section_raises_error(self) -> None:
        """Test that [display] section raises error."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "gepa": {},
            },
            "display": {
                "show_progress": True,
            },
        }
        with pytest.raises(ValueError, match="Deprecated field 'display'"):
            PromptLearningConfig.from_mapping(data)

    def test_deprecated_results_folder_raises_error(self) -> None:
        """Test that results_folder field raises error."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "results_folder": "results",
                "gepa": {},
            },
        }
        with pytest.raises(ValueError, match="Deprecated field 'results_folder'"):
            PromptLearningConfig.from_mapping(data)

    def test_deprecated_env_file_path_raises_error(self) -> None:
        """Test that env_file_path field raises error."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "env_file_path": ".env",
                "gepa": {},
            },
        }
        with pytest.raises(ValueError, match="Deprecated field 'env_file_path'"):
            PromptLearningConfig.from_mapping(data)

    def test_deprecated_flat_gepa_rollout_budget_raises_error(self) -> None:
        """Test that flat GEPA rollout_budget field raises error."""
        data = {
            "rollout_budget": 100,
        }
        with pytest.raises(ValueError, match="Deprecated flat GEPA format field 'rollout_budget'"):
            GEPAConfig.model_validate(data)

    def test_deprecated_flat_gepa_evaluation_seeds_raises_error(self) -> None:
        """Test that flat GEPA evaluation_seeds field raises error."""
        data = {
            "evaluation_seeds": [0, 1, 2, 3, 4],
        }
        with pytest.raises(ValueError, match="Deprecated flat GEPA format field 'evaluation_seeds'"):
            GEPAConfig.model_validate(data)

    def test_deprecated_flat_gepa_mutation_rate_raises_error(self) -> None:
        """Test that flat GEPA mutation_rate field raises error."""
        data = {
            "mutation_rate": 0.3,
        }
        with pytest.raises(ValueError, match="Deprecated flat GEPA format field 'mutation_rate'"):
            GEPAConfig.model_validate(data)

    def test_deprecated_mipro_meta_model_raises_error(self) -> None:
        """Test that MIPRO meta_model field raises error."""
        data = {
            "meta_model": "gpt-4o-mini",
        }
        with pytest.raises(ValueError, match="Deprecated field 'meta_model'"):
            MIPROConfig.model_validate(data)

    def test_deprecated_mipro_meta_model_provider_raises_error(self) -> None:
        """Test that MIPRO meta_model_provider field raises error."""
        data = {
            "meta_model_provider": "openai",
        }
        with pytest.raises(ValueError, match="Deprecated field 'meta_model_provider'"):
            MIPROConfig.model_validate(data)

    def test_deprecated_gepa_mutation_llm_model_raises_error(self) -> None:
        """Test that GEPA mutation.llm_model field raises error."""
        data = {
            "rate": 0.3,
            "llm_model": "gpt-4o-mini",
        }
        with pytest.raises(ValueError, match="Deprecated field 'llm_model'"):
            GEPAMutationConfig.model_validate(data)

    def test_deprecated_gepa_mutation_llm_provider_raises_error(self) -> None:
        """Test that GEPA mutation.llm_provider field raises error."""
        data = {
            "rate": 0.3,
            "llm_provider": "openai",
        }
        with pytest.raises(ValueError, match="Deprecated field 'llm_provider'"):
            GEPAMutationConfig.model_validate(data)

    def test_nested_gepa_format_accepted(self) -> None:
        """Test that nested GEPA format is accepted."""
        data = {
            "env_name": "test",
            "rollout": {"budget": 100},
            "evaluation": {"seeds": [0, 1, 2]},
            "mutation": {"rate": 0.3},
            "population": {"initial_size": 10},
            "archive": {"size": 20},
            "token": {"max_limit": 4096},
            "proposer_effort": "LOW",
            "proposer_output_tokens": "FAST",
        }
        # Should not raise
        config = GEPAConfig.model_validate(data)
        assert config.env_name == "test"

    def test_termination_config_is_supported(self) -> None:
        """Test that termination_config is still supported (not deprecated)."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "termination_config": {
                    "max_cost_usd": 10.0,
                    "max_trials": 100,
                },
                "gepa": {},
            },
        }
        # Should not raise - termination_config is supported
        config = PromptLearningConfig.from_mapping(data)
        assert config.termination_config is not None


class TestProposerEffortValidation:
    """Tests for proposer_effort and proposer_output_tokens validation."""

    def test_gepa_with_proposer_effort(self) -> None:
        """Test GEPA config with proposer_effort."""
        config = GEPAConfig(
            env_name="test",
            proposer_effort="LOW",
            proposer_output_tokens="FAST",
        )
        assert config.proposer_effort == "LOW"
        assert config.proposer_output_tokens == "FAST"

    def test_gepa_proposer_effort_values(self) -> None:
        """Test that proposer_effort accepts valid values."""
        valid_efforts = ["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"]
        for effort in valid_efforts:
            config = GEPAConfig(
                env_name="test",
                proposer_effort=effort,
            )
            assert config.proposer_effort == effort

    def test_gepa_proposer_output_tokens_values(self) -> None:
        """Test that proposer_output_tokens accepts valid values."""
        valid_tokens = ["RAPID", "FAST", "SLOW"]
        for tokens in valid_tokens:
            config = GEPAConfig(
                env_name="test",
                proposer_output_tokens=tokens,
            )
            assert config.proposer_output_tokens == tokens

    def test_mipro_with_proposer_effort(self) -> None:
        """Test MIPRO config with proposer_effort."""
        config = MIPROConfig(
            proposer_effort="MEDIUM",
            proposer_output_tokens="SLOW",
        )
        assert config.proposer_effort == "MEDIUM"
        assert config.proposer_output_tokens == "SLOW"

    def test_proposer_effort_from_toml(self) -> None:
        """Test loading proposer_effort from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
env_name = "test"
proposer_effort = "MEDIUM"
proposer_output_tokens = "FAST"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.gepa is not None
            assert config.gepa.proposer_effort == "MEDIUM"
            assert config.gepa.proposer_output_tokens == "FAST"
        finally:
            path.unlink()


class TestFullConfigWithBetterToml:
    """Integration tests for full configs using better-toml features."""

    def test_modern_gepa_config(self) -> None:
        """Test loading a modern GEPA config with all new features."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_id = "banking77"

[prompt_learning.policy]
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 0.0
max_completion_tokens = 512
policy_name = "banking77-classifier"

[prompt_learning.gepa]
env_name = "banking77"
proposer_effort = "LOW"
proposer_output_tokens = "FAST"

[prompt_learning.gepa.rollout]
budget = 100
max_concurrent = 20
minibatch_size = 10

[prompt_learning.gepa.evaluation]
seeds = { start = 0, end = 30 }
validation_seeds = { start = 30, end = 50 }
validation_pool = "validation"
validation_top_k = 3

[prompt_learning.gepa.mutation]
rate = 0.3

[prompt_learning.gepa.population]
initial_size = 10
num_generations = 5
children_per_generation = 12

[prompt_learning.gepa.archive]
size = 40
pareto_set_size = 32

[prompt_learning.gepa.token]
max_spend_usd = 10.0
counting_model = "gpt-4"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "gepa"
            assert config.gepa is not None
            assert config.gepa.proposer_effort == "LOW"
            assert config.gepa.evaluation is not None
            assert config.gepa.evaluation.seeds == list(range(0, 30))
            assert config.gepa.mutation is not None
            assert config.gepa.mutation.rate == 0.3
        finally:
            path.unlink()

    def test_modern_mipro_config(self) -> None:
        """Test loading a modern MIPRO config with all new features."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
task_app_id = "banking77"

[prompt_learning.policy]
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 0.0
max_completion_tokens = 128
policy_name = "banking77-mipro"

[prompt_learning.mipro]
env_name = "banking77"
num_iterations = 16
num_evaluations_per_iteration = 6
batch_size = 6
max_concurrent = 16
few_shot_score_threshold = 0.85
proposer_effort = "LOW"
proposer_output_tokens = "FAST"

bootstrap_train_seeds = { start = 0, end = 5 }
online_pool = { start = 5, end = 15 }
test_pool = { start = 15, end = 20 }
reference_pool = { start = 20, end = 25 }

max_token_limit = 100000
max_spend_usd = 5.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "mipro"
            assert config.mipro is not None
            assert config.mipro.proposer_effort == "LOW"
            assert config.mipro.bootstrap_train_seeds == list(range(0, 5))
            assert config.mipro.online_pool == list(range(5, 15))
        finally:
            path.unlink()
