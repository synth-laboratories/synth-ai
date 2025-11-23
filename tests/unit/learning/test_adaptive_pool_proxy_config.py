"""Comprehensive unit tests for adaptive pool and proxy models config handling.

Tests cover:
- Default behavior (LOW for adaptive pool, None for proxy models)
- Config parsing for both MIPRO and GEPA
- Top-level vs algorithm-specific config precedence
- Level resolution (NONE, LOW, MODERATE, HIGH)
- Overrides handling
- dev_pool_size extraction
- Error handling
- Edge cases
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.configs.prompt_learning import (
    AdaptiveCurriculumLevel,
    AdaptivePoolConfig,
    GEPAConfig,
    MIPROConfig,
    PromptLearningConfig,
    ProxyModelsConfig,
    resolve_adaptive_pool_config,
)

pytestmark = pytest.mark.unit


class TestAdaptivePoolDefaults:
    """Test default behavior for adaptive pool configuration."""

    def test_adaptive_pool_config_defaults_to_low(self) -> None:
        """Test that AdaptivePoolConfig defaults to LOW level."""
        config = AdaptivePoolConfig()
        assert config.level == AdaptiveCurriculumLevel.LOW

    def test_resolve_adaptive_pool_config_defaults_to_low(self) -> None:
        """Test that resolve_adaptive_pool_config defaults to LOW when level is None."""
        config = resolve_adaptive_pool_config(level=None)
        assert config.level == AdaptiveCurriculumLevel.LOW

    def test_resolve_adaptive_pool_config_with_low_level(self) -> None:
        """Test resolving adaptive pool config with LOW level."""
        config = resolve_adaptive_pool_config(level="LOW")
        assert config.level == AdaptiveCurriculumLevel.LOW
        assert config.anchor_size == 50  # LOW level default
        assert config.pool_init_size == 150  # LOW level default

    def test_resolve_adaptive_pool_config_with_moderate_level(self) -> None:
        """Test resolving adaptive pool config with MODERATE level."""
        config = resolve_adaptive_pool_config(level="MODERATE")
        assert config.level == AdaptiveCurriculumLevel.MODERATE
        assert config.anchor_size == 30  # MODERATE level default

    def test_resolve_adaptive_pool_config_with_high_level(self) -> None:
        """Test resolving adaptive pool config with HIGH level."""
        config = resolve_adaptive_pool_config(level="HIGH")
        assert config.level == AdaptiveCurriculumLevel.HIGH
        assert config.anchor_size == 20  # HIGH level default

    def test_resolve_adaptive_pool_config_with_none_level(self) -> None:
        """Test resolving adaptive pool config with NONE level."""
        config = resolve_adaptive_pool_config(level="NONE")
        assert config.level == AdaptiveCurriculumLevel.NONE
        assert config.anchor_size == 0  # NONE level default

    def test_resolve_adaptive_pool_config_with_overrides(self) -> None:
        """Test resolving adaptive pool config with parameter overrides."""
        config = resolve_adaptive_pool_config(
            level="LOW",
            overrides={"anchor_size": 75, "pool_init_size": 200},
        )
        assert config.level == AdaptiveCurriculumLevel.LOW
        assert config.anchor_size == 75  # Overridden
        assert config.pool_init_size == 200  # Overridden
        assert config.pool_min_size == 100  # Still LOW default

    def test_resolve_adaptive_pool_config_with_dev_pool_size(self) -> None:
        """Test resolving adaptive pool config with dev_pool_size capping."""
        config = resolve_adaptive_pool_config(
            level="LOW",
            dev_pool_size=100,  # Smaller than default pool_init_size (150)
        )
        assert config.level == AdaptiveCurriculumLevel.LOW
        assert config.pool_init_size == 100  # Capped by dev_pool_size

    def test_resolve_adaptive_pool_config_dev_pool_size_no_cap(self) -> None:
        """Test that dev_pool_size doesn't cap if pool_init_size is smaller."""
        config = resolve_adaptive_pool_config(
            level="HIGH",
            dev_pool_size=200,  # Larger than default pool_init_size (60)
        )
        assert config.pool_init_size == 60  # Not capped, uses HIGH default


class TestProxyModelsDefaults:
    """Test default behavior for proxy models configuration."""

    def test_proxy_models_defaults_to_none_in_mipro(self) -> None:
        """Test that MIPROConfig defaults proxy_models to None."""
        config = MIPROConfig()
        assert config.proxy_models is None

    def test_proxy_models_defaults_to_none_in_gepa(self) -> None:
        """Test that GEPAConfig defaults proxy_models to None."""
        config = GEPAConfig()
        assert config.proxy_models is None

    def test_proxy_models_defaults_to_none_in_prompt_learning_config(self) -> None:
        """Test that PromptLearningConfig defaults proxy_models to None."""
        config = PromptLearningConfig(
            algorithm="mipro",
            task_app_url="http://localhost:8001",
            mipro=MIPROConfig(),
        )
        assert config.proxy_models is None

    def test_proxy_models_can_be_set(self) -> None:
        """Test that proxy_models can be explicitly set."""
        proxy_config = ProxyModelsConfig(
            hi_provider="groq",
            hi_model="openai/gpt-oss-120b",
            lo_provider="groq",
            lo_model="openai/gpt-oss-20b",
        )
        config = MIPROConfig(proxy_models=proxy_config)
        assert config.proxy_models is not None
        assert config.proxy_models.hi_model == "openai/gpt-oss-120b"
        assert config.proxy_models.lo_model == "openai/gpt-oss-20b"


class TestGEPAAdaptivePoolParsing:
    """Test adaptive pool parsing in GEPA configs."""

    def test_gepa_config_without_adaptive_pool(self) -> None:
        """Test GEPA config without adaptive_pool section defaults to None."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is None

    def test_gepa_config_with_adaptive_pool_no_level(self) -> None:
        """Test GEPA config with adaptive_pool but no level defaults to LOW."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "anchor_size": 40,  # Override
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
        assert config.adaptive_pool.anchor_size == 40  # Override applied

    def test_gepa_config_with_adaptive_pool_low_level(self) -> None:
        """Test GEPA config with adaptive_pool level=LOW."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "level": "LOW",
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
        assert config.adaptive_pool.anchor_size == 50  # LOW default

    def test_gepa_config_with_adaptive_pool_moderate_level(self) -> None:
        """Test GEPA config with adaptive_pool level=MODERATE."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "level": "MODERATE",
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.MODERATE
        assert config.adaptive_pool.anchor_size == 30  # MODERATE default

    def test_gepa_config_with_adaptive_pool_dev_pool_size_extraction(self) -> None:
        """Test that GEPA extracts dev_pool_size from evaluation.seeds."""
        config_data = {
            "evaluation": {
                "seeds": list(range(50)),  # 50 seeds
            },
            "adaptive_pool": {
                "level": "LOW",  # Default pool_init_size is 150
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        # pool_init_size should be capped at 50 (dev_pool_size)
        assert config.adaptive_pool.pool_init_size == 50

    def test_gepa_config_with_adaptive_pool_overrides(self) -> None:
        """Test GEPA config with adaptive_pool overrides."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "level": "LOW",
                "anchor_size": 60,
                "pool_min_size": 80,
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
        assert config.adaptive_pool.anchor_size == 60  # Override
        assert config.adaptive_pool.pool_min_size == 80  # Override
        assert config.adaptive_pool.pool_init_size == 150  # Still LOW default


class TestMIPROAdaptivePoolParsing:
    """Test adaptive pool parsing in MIPRO configs."""

    def test_mipro_config_without_adaptive_pool(self) -> None:
        """Test MIPRO config without adaptive_pool section defaults to None."""
        config_data = {
            "num_iterations": 20,
        }
        config = MIPROConfig.model_validate(config_data)
        assert config.adaptive_pool is None

    def test_mipro_config_with_adaptive_pool_no_level(self) -> None:
        """Test MIPRO config with adaptive_pool but no level defaults to LOW."""
        config_data = {
            "num_iterations": 20,
            "adaptive_pool": {
                "anchor_size": 40,  # Override
            },
        }
        # Use PromptLearningConfig.from_mapping to trigger full parsing
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.adaptive_pool is not None
        assert config.mipro.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
        assert config.mipro.adaptive_pool.anchor_size == 40  # Override applied

    def test_mipro_config_with_adaptive_pool_low_level(self) -> None:
        """Test MIPRO config with adaptive_pool level=LOW."""
        config_data = {
            "num_iterations": 20,
            "adaptive_pool": {
                "level": "LOW",
            },
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.adaptive_pool is not None
        assert config.mipro.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
        assert config.mipro.adaptive_pool.anchor_size == 50  # LOW default

    def test_mipro_config_with_adaptive_pool_dev_pool_size_extraction(self) -> None:
        """Test that MIPRO extracts dev_pool_size from online_pool."""
        config_data = {
            "num_iterations": 20,
            "online_pool": list(range(75)),  # 75 seeds
            "adaptive_pool": {
                "level": "LOW",  # Default pool_init_size is 150
            },
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.adaptive_pool is not None
        # pool_init_size should be capped at 75 (dev_pool_size)
        assert config.mipro.adaptive_pool.pool_init_size == 75

    def test_mipro_config_with_adaptive_pool_from_seeds_section(self) -> None:
        """Test that MIPRO extracts dev_pool_size from seeds.online."""
        config_data = {
            "num_iterations": 20,
            "seeds": {
                "online": list(range(60)),  # 60 seeds
            },
            "adaptive_pool": {
                "level": "LOW",
            },
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.adaptive_pool is not None
        # pool_init_size should be capped at 60 (dev_pool_size from seeds.online)
        assert config.mipro.adaptive_pool.pool_init_size == 60


class TestProxyModelsParsing:
    """Test proxy models parsing in configs."""

    def test_gepa_config_without_proxy_models(self) -> None:
        """Test GEPA config without proxy_models section defaults to None."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.proxy_models is None

    def test_gepa_config_with_proxy_models(self) -> None:
        """Test GEPA config with proxy_models section."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "proxy_models": {
                "hi_provider": "groq",
                "hi_model": "openai/gpt-oss-120b",
                "lo_provider": "groq",
                "lo_model": "openai/gpt-oss-20b",
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.proxy_models is not None
        assert config.proxy_models.hi_provider == "groq"
        assert config.proxy_models.hi_model == "openai/gpt-oss-120b"
        assert config.proxy_models.lo_provider == "groq"
        assert config.proxy_models.lo_model == "openai/gpt-oss-20b"

    def test_mipro_config_without_proxy_models(self) -> None:
        """Test MIPRO config without proxy_models section defaults to None."""
        config_data = {
            "num_iterations": 20,
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.proxy_models is None

    def test_mipro_config_with_proxy_models(self) -> None:
        """Test MIPRO config with proxy_models section."""
        config_data = {
            "num_iterations": 20,
            "proxy_models": {
                "hi_provider": "groq",
                "hi_model": "openai/gpt-oss-120b",
                "lo_provider": "groq",
                "lo_model": "openai/gpt-oss-20b",
            },
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.proxy_models is not None
        assert config.mipro.proxy_models.hi_model == "openai/gpt-oss-120b"
        assert config.mipro.proxy_models.lo_model == "openai/gpt-oss-20b"


class TestTopLevelPrecedence:
    """Test that top-level proxy_models takes precedence over algorithm-specific."""

    def test_top_level_proxy_models_overrides_gepa_specific(self) -> None:
        """Test that top-level proxy_models overrides gepa-specific proxy_models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "proxy_models": {
                    "hi_provider": "groq",
                    "hi_model": "openai/gpt-oss-120b",
                    "lo_provider": "groq",
                    "lo_model": "openai/gpt-oss-20b",
                },
                "gepa": {
                    "proxy_models": {
                        "hi_provider": "openai",
                        "hi_model": "gpt-4o",
                        "lo_provider": "openai",
                        "lo_model": "gpt-4o-mini",
                    },
                },
            },
        }
        config = PromptLearningConfig.from_mapping(config_data)
        # Top-level proxy_models should be used
        assert config.proxy_models is not None
        assert config.proxy_models.hi_model == "openai/gpt-oss-120b"
        # GEPA-specific proxy_models should be removed
        assert config.gepa is not None
        assert config.gepa.proxy_models is None

    def test_top_level_proxy_models_overrides_mipro_specific(self) -> None:
        """Test that top-level proxy_models overrides mipro-specific proxy_models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "proxy_models": {
                    "hi_provider": "groq",
                    "hi_model": "openai/gpt-oss-120b",
                    "lo_provider": "groq",
                    "lo_model": "openai/gpt-oss-20b",
                },
                "mipro": {
                    "proxy_models": {
                        "hi_provider": "openai",
                        "hi_model": "gpt-4o",
                        "lo_provider": "openai",
                        "lo_model": "gpt-4o-mini",
                    },
                },
            },
        }
        config = PromptLearningConfig.from_mapping(config_data)
        # Top-level proxy_models should be used
        assert config.proxy_models is not None
        assert config.proxy_models.hi_model == "openai/gpt-oss-120b"
        # MIPRO-specific proxy_models should be removed
        assert config.mipro is not None
        assert config.mipro.proxy_models is None


class TestTOMLConfigParsing:
    """Test parsing adaptive pool and proxy models from TOML files."""

    def test_gepa_toml_with_adaptive_pool_low(self) -> None:
        """Test GEPA TOML config with adaptive_pool level=LOW."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_mode = "synth_hosted"

[prompt_learning.gepa.evaluation]
seeds = [0, 1, 2, 3, 4]

[prompt_learning.gepa.adaptive_pool]
level = "LOW"
anchor_size = 40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.gepa is not None
            assert config.gepa.adaptive_pool is not None
            assert config.gepa.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
            assert config.gepa.adaptive_pool.anchor_size == 40
        finally:
            path.unlink()

    def test_gepa_toml_with_adaptive_pool_no_level(self) -> None:
        """Test GEPA TOML config with adaptive_pool but no level (defaults to LOW)."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_mode = "synth_hosted"

[prompt_learning.gepa.evaluation]
seeds = [0, 1, 2, 3, 4]

[prompt_learning.gepa.adaptive_pool]
anchor_size = 40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.gepa is not None
            assert config.gepa.adaptive_pool is not None
            assert config.gepa.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
            assert config.gepa.adaptive_pool.anchor_size == 40
        finally:
            path.unlink()

    def test_gepa_toml_with_proxy_models(self) -> None:
        """Test GEPA TOML config with proxy_models."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_mode = "synth_hosted"

[prompt_learning.gepa.evaluation]
seeds = [0, 1, 2, 3, 4]

[prompt_learning.proxy_models]
hi_provider = "groq"
hi_model = "openai/gpt-oss-120b"
lo_provider = "groq"
lo_model = "openai/gpt-oss-20b"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.proxy_models is not None
            assert config.proxy_models.hi_model == "openai/gpt-oss-120b"
            assert config.proxy_models.lo_model == "openai/gpt-oss-20b"
        finally:
            path.unlink()

    def test_mipro_toml_with_adaptive_pool_low(self) -> None:
        """Test MIPRO TOML config with adaptive_pool level=LOW."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_mode = "synth_hosted"

[prompt_learning.mipro]
num_iterations = 20
online_pool = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

[prompt_learning.mipro.adaptive_pool]
level = "LOW"
anchor_size = 40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.mipro is not None
            assert config.mipro.adaptive_pool is not None
            assert config.mipro.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
            assert config.mipro.adaptive_pool.anchor_size == 40
        finally:
            path.unlink()

    def test_mipro_toml_with_adaptive_pool_no_level(self) -> None:
        """Test MIPRO TOML config with adaptive_pool but no level (defaults to LOW)."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"

[prompt_learning.policy]
model = "gpt-4o-mini"
provider = "openai"
inference_mode = "synth_hosted"

[prompt_learning.mipro]
num_iterations = 20
online_pool = [0, 1, 2, 3, 4]

[prompt_learning.mipro.adaptive_pool]
anchor_size = 40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            config = PromptLearningConfig.from_path(path)
            assert config.mipro is not None
            assert config.mipro.adaptive_pool is not None
            assert config.mipro.adaptive_pool.level == AdaptiveCurriculumLevel.LOW
            assert config.mipro.adaptive_pool.anchor_size == 40
        finally:
            path.unlink()


class TestErrorHandling:
    """Test error handling for adaptive pool and proxy models configs."""

    def test_invalid_adaptive_pool_level(self) -> None:
        """Test that invalid adaptive pool level raises error."""
        with pytest.raises((ValueError, KeyError)):
            resolve_adaptive_pool_config(level="INVALID")

    def test_invalid_proxy_models_missing_required_fields(self) -> None:
        """Test that proxy_models with missing required fields raises error."""
        config_data = {
            "proxy_models": {
                "hi_provider": "groq",
                # Missing hi_model, lo_provider, lo_model
            },
        }
        with pytest.raises(Exception):  # Pydantic ValidationError
            ProxyModelsConfig.model_validate(config_data["proxy_models"])

    def test_gepa_adaptive_pool_invalid_level(self) -> None:
        """Test that GEPA config with invalid adaptive_pool level raises error."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "level": "INVALID",
            },
        }
        with pytest.raises((ValueError, KeyError)):
            GEPAConfig.from_mapping(config_data)


class TestEdgeCases:
    """Test edge cases for adaptive pool and proxy models configs."""

    def test_adaptive_pool_with_empty_dict(self) -> None:
        """Test adaptive_pool with empty dict (should default to LOW)."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {},
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.LOW

    def test_adaptive_pool_with_none_dev_pool_size(self) -> None:
        """Test adaptive_pool with None dev_pool_size."""
        config = resolve_adaptive_pool_config(
            level="LOW",
            dev_pool_size=None,
        )
        assert config.pool_init_size == 150  # Uses LOW default

    def test_adaptive_pool_with_zero_dev_pool_size(self) -> None:
        """Test adaptive_pool with zero dev_pool_size."""
        # This should work but pool_init_size might be None or 0
        config = resolve_adaptive_pool_config(
            level="LOW",
            dev_pool_size=0,
        )
        # pool_init_size should be capped at 0
        assert config.pool_init_size == 0

    def test_proxy_models_with_optional_fields(self) -> None:
        """Test proxy_models with optional fields."""
        config_data = {
            "hi_provider": "groq",
            "hi_model": "openai/gpt-oss-120b",
            "lo_provider": "groq",
            "lo_model": "openai/gpt-oss-20b",
            "n_min_hi": 5,
            "r2_thresh": 0.3,
        }
        proxy_config = ProxyModelsConfig.model_validate(config_data)
        assert proxy_config.n_min_hi == 5
        assert proxy_config.r2_thresh == 0.3

    def test_gepa_config_with_both_adaptive_pool_and_proxy_models(self) -> None:
        """Test GEPA config with both adaptive_pool and proxy_models."""
        config_data = {
            "evaluation": {
                "seeds": list(range(100)),
            },
            "adaptive_pool": {
                "level": "MODERATE",
            },
            "proxy_models": {
                "hi_provider": "groq",
                "hi_model": "openai/gpt-oss-120b",
                "lo_provider": "groq",
                "lo_model": "openai/gpt-oss-20b",
            },
        }
        config = GEPAConfig.from_mapping(config_data)
        assert config.adaptive_pool is not None
        assert config.adaptive_pool.level == AdaptiveCurriculumLevel.MODERATE
        assert config.proxy_models is not None
        assert config.proxy_models.hi_model == "openai/gpt-oss-120b"

    def test_mipro_config_with_both_adaptive_pool_and_proxy_models(self) -> None:
        """Test MIPRO config with both adaptive_pool and proxy_models."""
        config_data = {
            "num_iterations": 20,
            "online_pool": list(range(100)),
            "adaptive_pool": {
                "level": "MODERATE",
            },
            "proxy_models": {
                "hi_provider": "groq",
                "hi_model": "openai/gpt-oss-120b",
                "lo_provider": "groq",
                "lo_model": "openai/gpt-oss-20b",
            },
        }
        pl_config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": config_data,
            },
        }
        config = PromptLearningConfig.from_mapping(pl_config_data)
        assert config.mipro is not None
        assert config.mipro.adaptive_pool is not None
        assert config.mipro.adaptive_pool.level == AdaptiveCurriculumLevel.MODERATE
        assert config.mipro.proxy_models is not None
        assert config.mipro.proxy_models.hi_model == "openai/gpt-oss-120b"

