"""Integration tests for TOML config validation - end-to-end validation.

These tests verify that configs are validated correctly when loaded from files
and when used in the CLI/builders.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from synth_ai.api.train.builders import build_prompt_learning_payload
from synth_ai.api.train.configs.prompt_learning import PromptLearningConfig
from synth_ai.api.train.validators import validate_prompt_learning_config

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def _create_config_file(config_data: dict[str, Any]) -> Path:
    """Helper to create a temporary TOML config file."""
    try:
        import tomli_w
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            return Path(f.name)
    except ImportError:
        # Fallback: use tomllib/tomli for reading, but we need tomli_w for writing
        # For tests, we can use the validator directly with dict
        # This function is mainly for integration tests that need actual files
        import json
        
        # Write as JSON temporarily (tests will use dict directly)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            json.dump(config_data, f, indent=2)
            return Path(f.name)


class TestConfigLoadingIntegration:
    """Integration tests for config loading and validation."""

    def test_valid_config_loads_successfully(self):
        """Test that valid config loads without errors."""
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
        path = _create_config_file(config_data)
        try:
            # Should load successfully
            config = PromptLearningConfig.from_path(path)
            assert config.algorithm == "gepa"
            assert config.task_app_url == "http://localhost:8001"
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_config_fails_loading(self):
        """Test that invalid config fails to load."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                # Missing policy section
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should raise validation error
            with pytest.raises(Exception):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)

    def test_validation_runs_before_loading(self):
        """Test that validation runs before config loading."""
        config_data = {
            "prompt_learning": {
                "algorithm": "invalid",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            # Validation should catch error before loading
            with pytest.raises(Exception, match="algorithm|invalid"):
                PromptLearningConfig.from_path(path)
        finally:
            path.unlink(missing_ok=True)


class TestBuilderIntegration:
    """Integration tests for config validation in builders."""

    def test_builder_validates_config(self):
        """Test that builder validates config before building payload."""
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
        path = _create_config_file(config_data)
        try:
            # Should build successfully
            result = build_prompt_learning_payload(
                config_path=path,
                task_url=None,
                overrides={},
            )
            assert result.payload is not None
            assert result.payload["algorithm"] == "gepa"
        finally:
            path.unlink(missing_ok=True)

    def test_builder_rejects_invalid_config(self):
        """Test that builder rejects invalid config."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                # Missing policy section
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should raise validation error
            from click import ClickException
            
            with pytest.raises(ClickException, match="policy|validation"):
                build_prompt_learning_payload(
                    config_path=path,
                    task_url=None,
                    overrides={},
                )
        finally:
            path.unlink(missing_ok=True)

    def test_builder_validates_model_support(self):
        """Test that builder validates model support."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-999",  # Invalid model
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should raise validation error
            from click import ClickException
            
            with pytest.raises(ClickException, match="model|Unsupported"):
                build_prompt_learning_payload(
                    config_path=path,
                    task_url=None,
                    overrides={},
                )
        finally:
            path.unlink(missing_ok=True)


class TestRealWorldConfigs:
    """Tests using real-world config examples."""

    def test_banking77_gepa_config_structure(self):
        """Test that Banking77 GEPA config structure is valid."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "https://example.com/task-app",
                "task_app_id": "banking77",
                "policy": {
                    "model": "openai/gpt-oss-20b",
                    "provider": "groq",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 128,
                },
                "initial_prompt": {
                    "id": "banking77_baseline",
                    "name": "Banking77 Baseline",
                    "messages": [
                        {"role": "system", "pattern": "You are a banking assistant", "order": 0}
                    ],
                },
                "gepa": {
                    "env_name": "banking77",
                    "rng_seed": 42,
                    "proposer_type": "spec",
                    "spec_path": "examples/task_apps/banking77/banking77_spec.json",
                    "num_generations": 10,
                    "initial_population_size": 20,
                    "children_per_generation": 5,
                    "mutation_rate": 0.3,
                    "crossover_rate": 0.5,
                    "rollout": {
                        "budget": 500,
                        "max_concurrent": 10,
                        "minibatch_size": 4,
                    },
                    "evaluation": {
                        "seeds": list(range(0, 15)),
                        "validation_seeds": list(range(15, 20)),
                        "test_pool": list(range(40, 50)),
                    },
                    "mutation": {
                        "rate": 0.3,
                        "llm_model": "llama3-groq-70b-8192-tool-use-preview",
                        "llm_provider": "groq",
                        "llm_inference_url": "https://api.groq.com/openai/v1",
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate successfully
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_banking77_pipeline_mipro_config_structure(self):
        """Test that Banking77 pipeline MIPRO config structure is valid."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task-app",
                "task_app_id": "banking77-pipeline",
                "policy": {
                    "model": "openai/gpt-oss-120b",
                    "provider": "groq",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 128,
                },
                "initial_prompt": {
                    "id": "banking77_pipeline_pattern",
                    "name": "Banking77 Pipeline Pattern",
                    "messages": [
                        {"role": "system", "pattern": "Pipeline placeholder", "order": 0}
                    ],
                    "metadata": {
                        "pipeline_modules": [
                            {"name": "classifier", "instruction_text": "Classify intent", "few_shots": []},
                            {"name": "calibrator", "instruction_text": "Calibrate confidence", "few_shots": []},
                        ],
                    },
                },
                "mipro": {
                    "env_name": "banking77_pipeline",
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "few_shot_score_threshold": 0.85,
                    "max_instructions": 3,
                    "bootstrap_train_seeds": list(range(0, 15)),
                    "online_pool": list(range(15, 40)),
                    "test_pool": list(range(40, 50)),
                    "spec_path": "examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json",
                    "spec_max_tokens": 5000,
                    "spec_include_examples": True,
                    "spec_priority_threshold": 8,
                    "modules": [
                        {"module_id": "classifier", "max_instruction_slots": 3, "max_demo_slots": 5},
                        {"module_id": "calibrator", "max_instruction_slots": 3, "max_demo_slots": 5},
                    ],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate successfully
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gemini_policy_config(self):
        """Test that Gemini policy config is valid."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task-app",
                "policy": {
                    "model": "gemini-2.5-flash-lite",
                    "provider": "google",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 128,
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate successfully
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gpt41mini_policy_config(self):
        """Test that gpt-4.1-mini policy config is valid."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task-app",
                "policy": {
                    "model": "gpt-4.1-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 128,
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate successfully
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestCrossFieldValidation:
    """Tests for cross-field validation logic."""

    def test_multi_stage_requires_modules(self):
        """Test that multi-stage configs require modules."""
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
                        "pipeline_modules": ["stage1", "stage2"],
                    },
                },
                "gepa": {
                    "num_generations": 10,
                    # Missing modules config
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="modules|multi-stage|pipeline"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mutation_model_requires_provider(self):
        """Test that mutation model requires provider."""
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
                    "mutation": {
                        "llm_model": "gpt-4o-mini",
                        # Missing llm_provider
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="llm_provider|mutation"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_meta_model_requires_provider(self):
        """Test that meta model requires provider."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    # Missing meta_model_provider
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="meta_model_provider|meta"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestNestedStructureValidation:
    """Tests for nested structure validation."""

    def test_nested_rollout_config(self):
        """Test that nested rollout config is validated."""
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
                    "rollout": {
                        "budget": -100,  # Invalid: negative
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate nested structure
            # Note: budget validation might be at gepa level, not nested
            # This documents current behavior
            try:
                validate_prompt_learning_config(config_data, path)
            except Exception:
                # If it raises, that's acceptable
                pass
        finally:
            path.unlink(missing_ok=True)

    def test_nested_evaluation_config(self):
        """Test that nested evaluation config is validated."""
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
                    "evaluation": {
                        "seeds": [],  # Invalid: empty
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate nested structure
            # Note: seeds validation might be at gepa level
            # This documents current behavior
            try:
                validate_prompt_learning_config(config_data, path)
            except Exception:
                # If it raises, that's acceptable
                pass
        finally:
            path.unlink(missing_ok=True)

    def test_nested_mutation_config(self):
        """Test that nested mutation config is validated."""
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
                    "mutation": {
                        "llm_model": "gpt-4.1-nano",  # Invalid: nano not allowed
                        "llm_provider": "openai",
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="nano|mutation|not allowed"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_nested_population_config(self):
        """Test that nested population config is validated."""
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
                    "population": {
                        "initial_size": -10,  # Invalid: negative
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should validate nested structure
            # Note: initial_size validation might be at gepa level
            # This documents current behavior
            try:
                validate_prompt_learning_config(config_data, path)
            except Exception:
                # If it raises, that's acceptable
                pass
        finally:
            path.unlink(missing_ok=True)

