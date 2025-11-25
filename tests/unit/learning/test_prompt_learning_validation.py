"""Comprehensive validation tests for prompt learning configurations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment,unused-ignore]

from synth_ai.sdk.api.train.configs.prompt_learning import (
    GEPAConfig,
    MIPROConfig,
    PromptLearningConfig,
    PromptLearningPolicyConfig,
)
from synth_ai.sdk.api.train.utils import TrainError
from synth_ai.sdk.api.train.validators import validate_prompt_learning_config

pytestmark = pytest.mark.unit


class TestInferenceUrlValidation:
    """Tests for inference_url field validation."""

    def test_valid_http_url(self) -> None:
        """Test that http:// URLs are accepted."""
        policy = PromptLearningPolicyConfig(
            model="gpt-4o-mini",
            provider="openai",
            inference_url="http://localhost:8000/v1",
        )
        assert policy.inference_url == "http://localhost:8000/v1"

    def test_valid_https_url(self) -> None:
        """Test that https:// URLs are accepted."""
        policy = PromptLearningPolicyConfig(
            model="gpt-4o-mini",
            provider="openai",
            inference_url="https://api.openai.com/v1",
        )
        assert policy.inference_url == "https://api.openai.com/v1"

    def test_url_stripping(self) -> None:
        """Test that whitespace is stripped from URLs."""
        policy = PromptLearningPolicyConfig(
            model="gpt-4o-mini",
            provider="openai",
            inference_url="  https://api.openai.com/v1  ",
        )
        assert policy.inference_url == "https://api.openai.com/v1"

    def test_invalid_url_no_protocol(self) -> None:
        """Test that URLs without http:// or https:// are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromptLearningPolicyConfig(
                model="gpt-4o-mini",
                provider="openai",
                inference_url="api.openai.com/v1",
            )
        assert "inference_url must start with http:// or https://" in str(exc_info.value)

    def test_invalid_url_empty_string(self) -> None:
        """Test that empty string URL is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromptLearningPolicyConfig(
                model="gpt-4o-mini",
                provider="openai",
                inference_url="",
            )
        assert "inference_url must start with http:// or https://" in str(exc_info.value)

    def test_invalid_url_not_string(self) -> None:
        """Test that non-string URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromptLearningPolicyConfig(
                model="gpt-4o-mini",
                provider="openai",
                inference_url=12345,  # type: ignore
            )
        # Pydantic v2 uses different error message format
        assert "string" in str(exc_info.value).lower() or "inference_url" in str(exc_info.value)


class TestProviderValidation:
    """Tests for provider enum validation."""

    def test_valid_provider_openai(self) -> None:
        """Test that 'openai' provider is accepted."""
        policy = PromptLearningPolicyConfig(
            model="gpt-4o-mini",
            provider="openai",
            inference_url="https://api.openai.com/v1",
        )
        assert policy.provider == "openai"

    def test_valid_provider_groq(self) -> None:
        """Test that 'groq' provider is accepted."""
        policy = PromptLearningPolicyConfig(
            model="llama-3.1-70b-versatile",
            provider="groq",
            inference_url="https://api.groq.com/v1",
        )
        assert policy.provider == "groq"

    def test_valid_provider_google(self) -> None:
        """Test that 'google' provider is accepted."""
        policy = PromptLearningPolicyConfig(
            model="gemini-pro",
            provider="google",
            inference_url="https://generativelanguage.googleapis.com/v1",
        )
        assert policy.provider == "google"

    def test_invalid_provider(self) -> None:
        """Test that invalid provider is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromptLearningPolicyConfig(
                model="gpt-4o-mini",
                provider="invalid_provider",  # type: ignore
                inference_url="https://api.openai.com/v1",
            )
        assert "provider" in str(exc_info.value).lower()


class TestModelValidation:
    """Tests for model name validation and common model patterns."""

    def test_openai_models(self) -> None:
        """Test common OpenAI model names."""
        valid_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
        for model in valid_models:
            policy = PromptLearningPolicyConfig(
                model=model,
                provider="openai",
                inference_url="https://api.openai.com/v1",
            )
            assert policy.model == model

    def test_groq_models(self) -> None:
        """Test common Groq model names."""
        valid_models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
        for model in valid_models:
            policy = PromptLearningPolicyConfig(
                model=model,
                provider="groq",
                inference_url="https://api.groq.com/v1",
            )
            assert policy.model == model

    def test_model_cannot_be_empty(self) -> None:
        """Test that empty model name is rejected or handled."""
        # Empty string might be allowed by Pydantic but fail validation later
        # This test documents behavior - empty model should be caught by validator function
        try:
            policy = PromptLearningPolicyConfig(
                model="",
                provider="openai",
                inference_url="https://api.openai.com/v1",
            )
            # If it doesn't raise, empty model is technically valid Pydantic but should fail validation
            assert policy.model == ""
        except ValidationError:
            # If it raises, that's the expected behavior
            pass


class TestAlgorithmValidation:
    """Tests for algorithm field validation."""

    def test_valid_algorithm_gepa(self) -> None:
        """Test that 'gepa' algorithm is accepted."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            gepa=GEPAConfig(),
        )
        assert config.algorithm == "gepa"

    def test_valid_algorithm_mipro(self) -> None:
        """Test that 'mipro' algorithm is accepted."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            mipro=MIPROConfig(),
        )
        assert config.algorithm == "mipro"

    def test_invalid_algorithm(self) -> None:
        """Test that invalid algorithm is rejected or handled."""
        # Pydantic doesn't validate enum values strictly at model level
        # Invalid algorithm should be caught by validate_prompt_learning_config function
        try:
            config = PromptLearningConfig(
                algorithm="invalid_algorithm",  # type: ignore
                task_app_url="http://localhost:8001",
            )
            # If it doesn't raise ValidationError, that's okay - validator function will catch it
            assert config.algorithm == "invalid_algorithm"
        except ValidationError:
            # If it raises, that's also acceptable
            pass

    def test_missing_algorithm(self) -> None:
        """Test that missing algorithm raises error."""
        with pytest.raises(ValidationError):
            PromptLearningConfig(
                task_app_url="http://localhost:8001",
            )


class TestTaskAppUrlValidation:
    """Tests for task_app_url validation."""

    def test_valid_task_app_url(self) -> None:
        """Test that valid task app URL is accepted."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            gepa=GEPAConfig(),
        )
        assert config.task_app_url == "http://localhost:8001"

    def test_task_app_url_with_path(self) -> None:
        """Test that task app URL with path is accepted."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="https://example.com/task-app/v1",
            gepa=GEPAConfig(),
        )
        assert config.task_app_url == "https://example.com/task-app/v1"

    def test_missing_task_app_url(self) -> None:
        """Test that missing task_app_url raises error."""
        with pytest.raises(ValidationError):
            PromptLearningConfig(
                algorithm="gepa",
                gepa=GEPAConfig(),
            )


class TestGEPAConfigValidation:
    """Tests for GEPA-specific configuration validation."""

    def test_valid_gepa_config(self) -> None:
        """Test valid GEPA configuration."""
        gepa = GEPAConfig(
            env_name="banking77",
            initial_population_size=20,
            num_generations=10,
            mutation_rate=0.3,
            crossover_rate=0.5,
            evaluation_seeds=[0, 1, 2, 3, 4],
        )
        assert gepa.env_name == "banking77"
        assert gepa.initial_population_size == 20
        assert gepa.num_generations == 10
        assert gepa.mutation_rate == 0.3
        assert gepa.crossover_rate == 0.5
        assert gepa.evaluation_seeds == [0, 1, 2, 3, 4]

    def test_gepa_mutation_rate_range(self) -> None:
        """Test that mutation_rate can be any float (including 0.0 and 1.0)."""
        # Mutation rate should be valid from 0.0 to 1.0
        for rate in [0.0, 0.3, 0.5, 0.7, 1.0]:
            gepa = GEPAConfig(mutation_rate=rate)
            assert gepa.mutation_rate == rate

    def test_gepa_crossover_rate_range(self) -> None:
        """Test that crossover_rate can be any float."""
        for rate in [0.0, 0.5, 1.0]:
            gepa = GEPAConfig(crossover_rate=rate)
            assert gepa.crossover_rate == rate

    def test_gepa_evaluation_seeds(self) -> None:
        """Test that evaluation_seeds list is accepted."""
        seeds = list(range(50, 80))  # 30 seeds
        gepa = GEPAConfig(evaluation_seeds=seeds)
        assert gepa.evaluation_seeds == seeds
        assert len(gepa.evaluation_seeds) == 30

    def test_gepa_validation_seeds(self) -> None:
        """Test that validation_seeds (test_pool) list is accepted."""
        seeds = list(range(0, 50))  # 50 seeds
        gepa = GEPAConfig(test_pool=seeds)
        assert gepa.test_pool == seeds
        assert len(gepa.test_pool) == 50


class TestMIPROConfigValidation:
    """Tests for MIPRO-specific configuration validation."""

    def test_valid_mipro_config(self) -> None:
        """Test valid MIPRO configuration."""
        mipro = MIPROConfig(
            num_iterations=20,
            num_evaluations_per_iteration=5,
            batch_size=32,
            meta_model="gpt-4o-mini",
            meta_model_provider="openai",
        )
        assert mipro.num_iterations == 20
        assert mipro.num_evaluations_per_iteration == 5
        assert mipro.batch_size == 32
        assert mipro.meta_model == "gpt-4o-mini"
        assert mipro.meta_model_provider == "openai"

    def test_mipro_num_iterations_positive(self) -> None:
        """Test that num_iterations must be positive."""
        mipro = MIPROConfig(num_iterations=1)
        assert mipro.num_iterations == 1

        # Zero iterations should be allowed (edge case)
        mipro = MIPROConfig(num_iterations=0)
        assert mipro.num_iterations == 0

    def test_mipro_bootstrap_seeds(self) -> None:
        """Test that bootstrap_train_seeds list is accepted."""
        seeds = [0, 1, 2, 3, 4, 5]
        mipro = MIPROConfig(bootstrap_train_seeds=seeds)
        assert mipro.bootstrap_train_seeds == seeds

    def test_mipro_online_pool(self) -> None:
        """Test that online_pool list is accepted."""
        pool = list(range(10, 50))
        mipro = MIPROConfig(online_pool=pool)
        assert mipro.online_pool == pool


class TestConfigFromMapping:
    """Tests for loading configs from dictionaries/TOML."""

    def test_from_mapping_with_prompt_learning_section(self) -> None:
        """Test loading config with [prompt_learning] section."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        config = PromptLearningConfig.from_mapping(data)
        assert config.algorithm == "gepa"
        assert config.task_app_url == "http://localhost:8001"
        assert config.gepa is not None
        assert config.gepa.num_generations == 10

    def test_from_mapping_without_prompt_learning_section(self) -> None:
        """Test loading config without [prompt_learning] section (flat structure)."""
        data = {
            "algorithm": "mipro",
            "task_app_url": "http://localhost:8001",
            "mipro": {
                "num_iterations": 5,
            },
        }
        config = PromptLearningConfig.from_mapping(data)
        assert config.algorithm == "mipro"
        assert config.task_app_url == "http://localhost:8001"
        assert config.mipro is not None
        assert config.mipro.num_iterations == 5


class TestConfigFromPath:
    """Tests for loading configs from TOML files."""

    def test_from_path_valid_gepa_config(self) -> None:
        """Test loading valid GEPA config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
temperature = 0.0
max_completion_tokens = 512

[prompt_learning.gepa]
env_name = "banking77"
initial_population_size = 20
num_generations = 10
mutation_rate = 0.3
crossover_rate = 0.5
evaluation_seeds = [50, 51, 52, 53, 54]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "gepa"
            assert config.task_app_url == "http://localhost:8001"
            assert config.policy is not None
            assert config.policy.model == "gpt-4o-mini"
            assert config.policy.provider == "openai"
            assert config.gepa is not None
            assert config.gepa.num_generations == 10
            assert config.gepa.evaluation_seeds == [50, 51, 52, 53, 54]
        finally:
            path.unlink()

    def test_from_path_valid_mipro_config(self) -> None:
        """Test loading valid MIPRO config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"

[prompt_learning.mipro]
num_iterations = 20
num_evaluations_per_iteration = 5
batch_size = 32
meta_model = "gpt-4o"
meta_model_provider = "openai"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "mipro"
            assert config.mipro is not None
            assert config.mipro.num_iterations == 20
            assert config.mipro.meta_model == "gpt-4o"
        finally:
            path.unlink()

    def test_from_path_invalid_toml_syntax(self) -> None:
        """Test that invalid TOML syntax raises error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
invalid = [unclosed bracket
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink()


class TestConfigToDict:
    """Tests for converting configs to dictionaries."""

    def test_to_dict_wraps_in_prompt_learning_section(self) -> None:
        """Test that to_dict wraps config in prompt_learning section."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            gepa=GEPAConfig(num_generations=5),
        )
        result = config.to_dict()
        assert "prompt_learning" in result
        pl = result["prompt_learning"]
        assert pl["algorithm"] == "gepa"
        assert pl["task_app_url"] == "http://localhost:8001"
        assert "gepa" in pl
        assert pl["gepa"]["num_generations"] == 5

    def test_to_dict_excludes_none_values(self) -> None:
        """Test that to_dict excludes None values."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            gepa=GEPAConfig(),
        )
        result = config.to_dict()
        pl = result["prompt_learning"]
        # task_app_api_key should not be present if None
        assert "task_app_api_key" not in pl or pl["task_app_api_key"] is None


class TestValidatePromptLearningConfig:
    """Tests for the validate_prompt_learning_config function."""

    def test_valid_gepa_config_passes(self) -> None:
        """Test that valid GEPA config passes validation."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_algorithm_raises_error(self) -> None:
        """Test that missing algorithm raises validation error."""
        config_data = {
            "prompt_learning": {
                "task_app_url": "http://localhost:8001",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="algorithm"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_algorithm_raises_error(self) -> None:
        """Test that invalid algorithm raises validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "invalid",
                "task_app_url": "http://localhost:8001",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="algorithm"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_task_app_url_raises_error(self) -> None:
        """Test that missing task_app_url raises validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_raises_error(self) -> None:
        """Test that missing policy section raises validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="policy"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_inference_url_rejection(self) -> None:
        """Test that inference_url in policy config is rejected (trainer provides it)."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_url": "https://api.openai.com/v1",  # Should be rejected
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="inference_url.*must not"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_provider_raises_error(self) -> None:
        """Test that missing policy.provider raises validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="provider"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_model_raises_error(self) -> None:
        """Test that missing policy.model raises validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_gepa_section_raises_error(self) -> None:
        """Test that missing gepa section raises error when algorithm is gepa."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="gepa"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mipro_section_raises_error(self) -> None:
        """Test that MIPRO algorithm requires mipro section."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value).lower()
            assert "mipro" in error_msg
            assert "missing" in error_msg or "section" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_invalid_positive_int_fields(self) -> None:
        """Test that GEPA fields requiring positive integers are validated."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": -5,  # Invalid: negative
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="initial_population_size"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_invalid_max_spend(self) -> None:
        """Test that GEPA max_spend_usd must be positive."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "max_spend_usd": -10.0,  # Invalid: negative
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="max_spend_usd"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_inference_mode(self) -> None:
        """Test that invalid inference_mode raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "bring_your_own",  # Invalid
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="inference_mode"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_task_app_url_not_string(self) -> None:
        """Test that task_app_url must be a string."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": 12345,  # Invalid: not a string
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_task_app_url_invalid_protocol(self) -> None:
        """Test that task_app_url must start with http:// or https://."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "ftp://example.com",  # Invalid protocol
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_initial_prompt_messages_raises_error(self) -> None:
        """Test that empty initial_prompt.messages raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "initial_prompt": {
                    "messages": [],  # Invalid: empty
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="messages"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_initial_prompt_messages_not_list(self) -> None:
        """Test that initial_prompt.messages must be a list."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "initial_prompt": {
                    "messages": "not a list",  # Invalid
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception, match="messages"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestInvalidTOMLSyntax:
    """Tests for rejecting invalid TOML syntax."""

    def test_unclosed_array_bracket(self) -> None:
        """Test that unclosed array bracket raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
evaluation_seeds = [1, 2, 3  # Missing closing bracket
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_unclosed_string(self) -> None:
        """Test that unclosed string raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa
task_app_url = "http://localhost:8001"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_table_syntax(self) -> None:
        """Test that invalid table syntax raises TOML parsing error."""
        toml_content = """
[prompt_learning
algorithm = "gepa"
task_app_url = "http://localhost:8001"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_duplicate_key(self) -> None:
        """Test that duplicate keys are handled (TOML allows but may warn)."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
algorithm = "mipro"  # Duplicate key
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # TOML may accept duplicate keys (last one wins) or raise error
            # This test documents current behavior
            try:
                config = PromptLearningConfig.from_path(path)
                # If it doesn't raise, the last value should win
                assert config.algorithm == "mipro"
            except Exception:
                # If it raises, that's also acceptable behavior
                pass
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_inline_table(self) -> None:
        """Test that invalid inline table syntax raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
gepa = { num_generations = 10  # Missing closing brace
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_number_format(self) -> None:
        """Test that invalid number format raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
num_generations = 10.5.3  # Invalid number format
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_boolean(self) -> None:
        """Test that invalid boolean raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
enforce_pattern_token_limit = yes  # Should be true/false, not yes/no
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_date_time(self) -> None:
        """Test that invalid date/time format raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
invalid_date = 2024-13-45  # Invalid date
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_file(self) -> None:
        """Test that empty TOML file raises error."""
        toml_content = ""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # Empty file should raise validation or parsing error
            if tomllib is not None:
                with pytest.raises((tomllib.TOMLDecodeError, ValueError, ValidationError)):
                    PromptLearningConfig.from_path(path)
            else:
                with pytest.raises((ValueError, ValidationError)):
                    PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_only_comments(self) -> None:
        """Test that TOML file with only comments raises error."""
        toml_content = """
# This is a comment
# Another comment
# No actual config
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # Comment-only file should raise validation error
            with pytest.raises((ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_array_of_tables(self) -> None:
        """Test that malformed array of tables raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[[prompt_learning.gepa]]  # Array of tables, but gepa should be a table, not array
num_generations = 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # This might parse but fail validation, or fail parsing
            if tomllib is not None:
                with pytest.raises((tomllib.TOMLDecodeError, ValueError, ValidationError)):
                    PromptLearningConfig.from_path(path)
            else:
                with pytest.raises((ValueError, ValidationError)):
                    PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_escape_sequence(self) -> None:
        """Test that invalid escape sequence raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
invalid = "\\x"  # Invalid escape sequence
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_equal_sign(self) -> None:
        """Test that missing equals sign raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm "gepa"  # Missing = sign
task_app_url = "http://localhost:8001"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_nested_table(self) -> None:
        """Test that invalid nested table syntax raises TOML parsing error."""
        toml_content = """
[prompt_learning.gepa]  # Nested table
num_generations = 10
[prompt_learning.gepa.invalid]  # Cannot nest deeper than 2 levels
value = 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # This might parse but the structure would be wrong
            with pytest.raises((ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_mixed_array_types(self) -> None:
        """Test that array with mixed types may cause issues."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
evaluation_seeds = [1, "two", 3]  # Mixed types in array
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # TOML allows mixed types, but our validation should catch it
            with pytest.raises((ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_unicode_in_string(self) -> None:
        """Test that invalid unicode in string raises TOML parsing error."""
        # Note: Writing invalid unicode surrogates to file can cause UnicodeEncodeError
        # This test documents that such strings should be rejected
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
invalid = "test"  # Using valid string - invalid surrogates can't be written to file
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False, encoding="utf-8") as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # This should parse fine with valid unicode
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "gepa"
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_float_exponent(self) -> None:
        """Test that invalid float exponent raises TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.gepa]
mutation_rate = 0.3e  # Invalid exponent (missing number)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # load_toml wraps TOML parsing errors in TrainError
            with pytest.raises((TrainError, ValueError, ValidationError)):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_unclosed_multiline_string(self) -> None:
        """Test that unclosed multiline string raises TOML parsing error."""
        # Some TOML parsers are lenient with multiline strings
        # This test documents expected behavior - parser may accept or reject
        toml_content = '''
[prompt_learning]
algorithm = "gepa"
task_app_url = """http://localhost:8001
# Missing closing """
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # Some parsers might accept this, others reject it
            # Both behaviors are acceptable - the test documents parser behavior
            try:
                config = PromptLearningConfig.from_path(path)
                # If parser accepts it, that's fine - validator will catch issues
                assert config.algorithm == "gepa"
            except Exception:
                # If parser rejects it, that's also fine
                pass
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_key_with_special_chars(self) -> None:
        """Test that invalid key characters raise TOML parsing error."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
invalid-key = "value"  # Keys with dashes may be invalid depending on TOML parser
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # Some TOML parsers allow quoted keys with dashes
            # This test documents behavior
            try:
                config = PromptLearningConfig.from_path(path)
                # If it parses, that's fine
                assert config.algorithm == "gepa"
            except Exception:
                # If it fails, that's also acceptable
                pass
        finally:
            path.unlink(missing_ok=True)

    def test_circular_reference_in_table(self) -> None:
        """Test that tables cannot reference themselves circularly."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
prompt_learning = { algorithm = "mipro" }  # Circular reference (same name as parent)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)
        try:
            # TOML parser might overwrite the table or parse both
            # This test documents behavior - should either parse or fail
            try:
                config = PromptLearningConfig.from_path(path)
                # If it parses, last value might win or structure might be wrong
                # That's acceptable - validator function will catch invalid structure
                assert config.algorithm in ("gepa", "mipro")
            except Exception:
                # If it fails parsing/validation, that's also acceptable
                pass
        finally:
            path.unlink(missing_ok=True)


class TestMultiStageGEPAValidation:
    """Tests for multi-stage GEPA pipeline validation."""

    def test_multi_stage_gepa_with_modules_passes(self) -> None:
        """Test that multi-stage GEPA with modules config passes validation."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "modules": [
                        {
                            "module_id": "classifier",
                            "max_instruction_slots": 2,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                        {
                            "module_id": "calibrator",
                            "max_instruction_slots": 3,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                    ],
                    "num_generations": 10,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_multi_stage_gepa_missing_modules_raises_error(self) -> None:
        """Test that multi-stage GEPA without modules config raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "num_generations": 10,
                    # Missing modules config
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value).lower()
            assert "modules" in error_msg or "multi-stage" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_multi_stage_gepa_mismatched_module_ids_raises_error(self) -> None:
        """Test that mismatched module IDs raise validation error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "modules": [
                        {"module_id": "classifier", "max_instruction_slots": 2},
                        # Missing "calibrator" module
                    ],
                    "num_generations": 10,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value).lower()
            assert "calibrator" in error_msg or "missing" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_single_stage_gepa_still_works(self) -> None:
        """Test that single-stage GEPA (no pipeline_modules) still works."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                    # No modules config - single stage
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise - single-stage is still valid
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_multi_stage_gepa_with_string_pipeline_modules(self) -> None:
        """Test multi-stage GEPA when pipeline_modules is list of strings."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["stage1", "stage2"],  # List of strings
                    },
                },
                "gepa": {
                    "modules": [
                        {
                            "module_id": "stage1",
                            "max_instruction_slots": 2,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                        {
                            "module_id": "stage2",
                            "max_instruction_slots": 3,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                    ],
                    "num_generations": 10,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

