"""Unit tests for prompt learning configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from synth_ai.api.train.configs.prompt_learning import (
    GEPAConfig,
    MIPROConfig,
    MessagePatternConfig,
    PromptLearningConfig,
    PromptLearningPolicyConfig,
    PromptPatternConfig,
)
from synth_ai.api.train.builders import build_prompt_learning_payload


class TestPromptPatternConfig:
    """Test PromptPatternConfig model."""

    def test_message_pattern_creation(self) -> None:
        """Test creating a message pattern."""
        msg = MessagePatternConfig(role="user", pattern="Answer this: {question}")
        assert msg.role == "user"
        assert msg.pattern == "Answer this: {question}"
        assert msg.order == 0

    def test_prompt_pattern_with_messages(self) -> None:
        """Test prompt pattern with messages."""
        pattern = PromptPatternConfig(
            messages=[
                MessagePatternConfig(role="system", pattern="You are helpful"),
                MessagePatternConfig(role="user", pattern="{input}"),
            ]
        )
        assert len(pattern.messages) == 2
        assert pattern.messages[0].role == "system"
        assert pattern.messages[0].pattern == "You are helpful"

    def test_prompt_pattern_empty(self) -> None:
        """Test prompt pattern with no messages (defaults)."""
        pattern = PromptPatternConfig()
        assert pattern.messages == []
        assert pattern.wildcards == {}


class TestPromptLearningPolicyConfig:
    """Test PromptLearningPolicyConfig model."""

    def test_default_policy(self) -> None:
        """Test policy with defaults."""
        policy = PromptLearningPolicyConfig(
            model="gpt-4o-mini",
            provider="openai",
            inference_url="https://api.openai.com/v1"
        )
        assert policy.model == "gpt-4o-mini"
        assert policy.temperature == 0.0
        assert policy.max_completion_tokens == 512
        assert policy.provider == "openai"

    def test_policy_with_parameters(self) -> None:
        """Test policy with generation parameters."""
        policy = PromptLearningPolicyConfig(
            model="gemini-2.0-flash-exp",
            provider="google",
            inference_url="https://generativelanguage.googleapis.com/v1beta",
            temperature=0.7,
            max_completion_tokens=1024,
        )
        assert policy.model == "gemini-2.0-flash-exp"
        assert policy.provider == "google"
        assert policy.temperature == 0.7
        assert policy.max_completion_tokens == 1024


class TestMIPROConfig:
    """Test MIPROConfig model."""

    def test_minimal_mipro_config(self) -> None:
        """Test MIPRO config with minimal settings."""
        config = MIPROConfig(num_iterations=3)
        assert config.num_iterations == 3
        assert config.env_name == "banking77"  # default
        assert config.meta_model == "gpt-4o-mini"  # default

    def test_full_mipro_config(self) -> None:
        """Test MIPRO config with all settings."""
        config = MIPROConfig(
            num_iterations=5,
            num_evaluations_per_iteration=10,
            batch_size=64,
            meta_model="gpt-4o",
            meta_model_provider="openai",
            bootstrap_train_seeds=[0, 1, 2],
            online_pool=list(range(10, 20)),
        )
        assert config.num_iterations == 5
        assert config.num_evaluations_per_iteration == 10
        assert config.batch_size == 64
        assert config.meta_model == "gpt-4o"
        assert config.meta_model_provider == "openai"
        assert config.bootstrap_train_seeds == [0, 1, 2]


class TestGEPAConfig:
    """Test GEPAConfig model."""

    def test_minimal_gepa_config(self) -> None:
        """Test GEPA config with minimal settings."""
        config = GEPAConfig()
        assert config.env_name == "banking77"  # default
        # Use helper methods instead of accessing flat attributes
        assert config._get_num_generations() == 10  # default
        assert config._get_mutation_rate() == 0.3  # default

    def test_full_gepa_config(self) -> None:
        """Test GEPA config with all settings."""
        config = GEPAConfig(
            env_name="my_env",
            initial_population_size=20,
            num_generations=10,
            mutation_rate=0.3,
            crossover_rate=0.7,
            selection_pressure=1.5,
            rng_seed=42,
            evaluation_seeds=list(range(100)),
            children_per_generation=10,
        )
        assert config.env_name == "my_env"
        assert config.initial_population_size == 20
        assert config.num_generations == 10
        assert config.mutation_rate == 0.3
        assert config.crossover_rate == 0.7
        assert config.selection_pressure == 1.5
        assert config.rng_seed == 42
        assert len(config.evaluation_seeds) == 100


class TestPromptLearningConfig:
    """Test PromptLearningConfig model."""

    def test_minimal_mipro_config(self) -> None:
        """Test minimal MIPRO configuration."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            mipro=MIPROConfig(num_iterations=3),
        )
        assert config.algorithm == "mipro"
        assert config.task_app_url == "http://localhost:8001"
        assert config.mipro is not None
        assert config.mipro.num_iterations == 3
        assert config.gepa is None

    def test_minimal_gepa_config(self) -> None:
        """Test minimal GEPA configuration."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            gepa=GEPAConfig(
                num_generations=5,
                mutation_rate=0.2,
            ),
        )
        assert config.algorithm == "gepa"
        assert config.task_app_url == "http://localhost:8001"
        assert config.gepa is not None
        assert config.gepa.num_generations == 5
        assert config.mipro is None

    def test_config_with_initial_prompt(self) -> None:
        """Test config with initial prompt pattern."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            initial_prompt=PromptPatternConfig(
                id="banking77_prompt",
                name="Banking77 Classifier",
                messages=[
                    MessagePatternConfig(role="system", pattern="You are a banking assistant"),
                    MessagePatternConfig(role="user", pattern="Classify: {input}"),
                ],
            ),
            mipro=MIPROConfig(num_iterations=3),
        )
        assert config.initial_prompt is not None
        assert config.initial_prompt.id == "banking77_prompt"
        assert len(config.initial_prompt.messages) == 2

    def test_config_with_policy(self) -> None:
        """Test config with policy settings."""
        config = PromptLearningConfig(
            algorithm="gepa",
            task_app_url="http://localhost:8001",
            policy=PromptLearningPolicyConfig(
                model="gpt-4o-mini",
                provider="openai",
                inference_url="https://api.openai.com/v1",
                temperature=0.7,
                max_completion_tokens=512,
            ),
            gepa=GEPAConfig(
                num_generations=5,
                mutation_rate=0.2,
            ),
        )
        assert config.policy is not None
        assert config.policy.model == "gpt-4o-mini"
        assert config.policy.temperature == 0.7

    def test_config_with_task_app_credentials(self) -> None:
        """Test config with task app authentication."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            task_app_api_key="secret-key",
            task_app_id="banking77",
            mipro=MIPROConfig(num_iterations=3),
        )
        assert config.task_app_api_key == "secret-key"
        assert config.task_app_id == "banking77"

    def test_config_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            task_app_api_key="key",
            mipro=MIPROConfig(num_iterations=3),
        )
        result = config.to_dict()
        assert "prompt_learning" in result
        pl = result["prompt_learning"]
        assert pl["algorithm"] == "mipro"
        assert pl["task_app_url"] == "http://localhost:8001"
        assert "mipro" in pl
        assert pl["mipro"]["num_iterations"] == 3

    def test_config_from_mapping(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "algorithm": "gepa",
            "task_app_url": "http://localhost:8001",
            "gepa": {
                "initial_population_size": 10,
                "num_generations": 5,
                "mutation_rate": 0.2,
            },
        }
        config = PromptLearningConfig.from_mapping(data)
        assert config.algorithm == "gepa"
        assert config.gepa is not None
        assert config.gepa.initial_population_size == 10

    def test_config_from_path_mipro(self) -> None:
        """Test loading MIPRO config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.mipro]
num_evaluations_per_iteration = 10
num_iterations = 5
meta_model = "gpt-4o"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "mipro"
            assert config.task_app_url == "http://localhost:8001"
            assert config.mipro is not None
            assert config.mipro.num_evaluations_per_iteration == 10
            assert config.mipro.meta_model == "gpt-4o"
        finally:
            path.unlink()

    def test_config_from_path_gepa(self) -> None:
        """Test loading GEPA config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"
temperature = 0.7
max_completion_tokens = 512

[prompt_learning.gepa]
initial_population_size = 20
num_generations = 10
mutation_rate = 0.3
crossover_rate = 0.7
rng_seed = 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "gepa"
            assert config.policy is not None
            assert config.policy.model == "gpt-4o-mini"
            assert config.gepa is not None
            assert config.gepa.initial_population_size == 20
            assert config.gepa.rng_seed == 42
        finally:
            path.unlink()


class TestBuildPromptLearningPayload:
    """Test build_prompt_learning_payload function."""

    @pytest.mark.skip(reason="MIPRO not yet implemented")
    def test_build_payload_mipro(self) -> None:
        """Test building payload for MIPRO job."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"

[prompt_learning.mipro]
num_candidates = 5
num_iterations = 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={},
            )
            assert result.task_url == "http://localhost:8001"
            assert result.payload["algorithm"] == "mipro"
            assert "config_body" in result.payload
            assert result.payload["auto_start"] is True
        finally:
            path.unlink()

    def test_build_payload_gepa(self) -> None:
        """Test building payload for GEPA job."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
population_size = 10
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={},
            )
            assert result.task_url == "http://localhost:8001"
            assert result.payload["algorithm"] == "gepa"
            config_body = result.payload["config_body"]
            assert "prompt_learning" in config_body
            pl = config_body["prompt_learning"]
            assert pl["gepa"]["population_size"] == 10
        finally:
            path.unlink()

    def test_build_payload_with_task_url_override(self) -> None:
        """Test building payload with task_url override."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            # Note: task_url parameter is currently ignored (TOML is source of truth)
            # The override would need to be in overrides dict or TOML itself
            result = build_prompt_learning_payload(
                config_path=path,
                task_url="http://override:9000",  # This is ignored per current implementation
                overrides={},
            )
            # Builder uses TOML value, not task_url parameter
            assert result.task_url == "http://localhost:8001"
            config_body = result.payload["config_body"]
            assert config_body["prompt_learning"]["task_app_url"] == "http://localhost:8001"
        finally:
            path.unlink()

    def test_build_payload_with_backend_override(self) -> None:
        """Test building payload with backend override."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={"backend": "http://custom-backend:8000"},
            )
            assert "metadata" in result.payload
            assert "backend_base_url" in result.payload["metadata"]
            assert result.payload["metadata"]["backend_base_url"] == "http://custom-backend:8000/api"
        finally:
            path.unlink()

    def test_build_payload_with_metadata_override(self) -> None:
        """Test building payload with metadata override."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
population_size = 10
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={"metadata": {"experiment": "test-1", "user": "alice"}},
            )
            assert "metadata" in result.payload
            assert result.payload["metadata"]["experiment"] == "test-1"
            assert result.payload["metadata"]["user"] == "alice"
        finally:
            path.unlink()

    def test_build_payload_missing_task_url(self) -> None:
        """Test that missing task_app_url raises validation error."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_api_key = "test-key"

[prompt_learning.mipro]
num_iterations = 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            # Should raise error about missing task_app_url during validation
            from click import ClickException

            with pytest.raises(ClickException, match="(Config validation failed|task_app_url)"):
                build_prompt_learning_payload(
                    config_path=path,
                    task_url=None,
                    overrides={},
                )
        finally:
            path.unlink()

    def test_build_payload_with_api_key_in_env(self, monkeypatch) -> None:
        """Test that API key can come from environment."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_url = "https://api.openai.com/v1"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            # Set env var for API key
            monkeypatch.setenv("ENVIRONMENT_API_KEY", "env-key-123")
            
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={},
            )
            # Should succeed with env var providing the API key
            assert result.payload["algorithm"] == "gepa"
            config_body = result.payload["config_body"]
            assert config_body["prompt_learning"]["task_app_api_key"] == "env-key-123"
        finally:
            path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

