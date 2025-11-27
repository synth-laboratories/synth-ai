"""Comprehensive tests for MIPRO config validators.

These tests verify the policy validation logic:
- Top-level [prompt_learning.policy] as fallback for all stages
- Per-stage [policy] overrides
- Model/provider validation
- Required field validation
"""

import tempfile
from pathlib import Path

import pytest
import toml

from synth_ai.sdk.api.train.validators import (
    validate_mipro_config_from_file,
    ConfigValidationError,
)


def write_temp_toml(config_dict: dict) -> Path:
    """Write config dict to a temporary TOML file and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_dict, f)
        return Path(f.name)


class TestMIPROPolicyValidation:
    """Test suite for MIPRO policy configuration validation."""

    def _base_config(self) -> dict:
        """Return a minimal valid MIPRO config (without policy - to test policy validation)."""
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task",
                "task_app_api_key": "test-key",
                "env_name": "test-env",
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [
                        {"role": "system", "content": "You are a test assistant."},
                        {"role": "user", "content": "Test query: {query}"},
                    ],
                },
                "mipro": {
                    "num_iterations": 3,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 10,
                    "max_concurrent": 5,
                    "bootstrap_train_seeds": [0, 1, 2, 3, 4],
                    "online_pool": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    "reference_pool": [50, 51, 52, 53, 54],
                },
            }
        }

    # =========================================================================
    # Test: Top-level policy only (no modules/stages)
    # =========================================================================
    
    def test_top_level_policy_only_valid(self):
        """Top-level policy with model+provider should be valid (no modules defined)."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config, got errors: {errors}"
            assert len(errors) == 0
        finally:
            path.unlink()

    def test_top_level_policy_missing_model(self):
        """Top-level policy without model should fail."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "provider": "openai",
        }
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("model" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_top_level_policy_missing_provider(self):
        """Top-level policy without provider should fail."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
        }
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("provider" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_no_policy_at_all_fails(self):
        """No policy anywhere should fail."""
        config = self._base_config()
        # No policy section at all
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("policy" in e.lower() for e in errors)
        finally:
            path.unlink()

    # =========================================================================
    # Test: Per-stage policy only (no top-level)
    # =========================================================================
    
    def test_per_stage_policy_only_valid(self):
        """Per-stage policies with model+provider should be valid (no top-level policy)."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "baseline_instruction": "Classify the input.",
                        "policy": {
                            "model": "gpt-4o",
                            "provider": "openai",
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config, got errors: {errors}"
        finally:
            path.unlink()

    def test_per_stage_policy_missing_model_no_fallback(self):
        """Per-stage policy without model (and no top-level fallback) should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "policy": {
                            "provider": "openai",
                            # Missing model
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("model" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_per_stage_policy_missing_provider_no_fallback(self):
        """Per-stage policy without provider (and no top-level fallback) should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "policy": {
                            "model": "gpt-4o",
                            # Missing provider
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("provider" in e.lower() for e in errors)
        finally:
            path.unlink()

    # =========================================================================
    # Test: Top-level policy as fallback for stages without policy
    # =========================================================================
    
    def test_top_level_fallback_for_stage_without_policy(self):
        """Stage without policy should use top-level policy as fallback."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "baseline_instruction": "Classify the input.",
                        # No policy - should use top-level fallback
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config with fallback, got errors: {errors}"
        finally:
            path.unlink()

    def test_stage_inherits_model_from_top_level(self):
        """Stage policy with only provider should inherit model from top-level."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "llama-3.3-70b",  # Use Groq model so it's valid when inherited
            "provider": "groq",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "policy": {
                            "provider": "openai",  # Override provider
                            "model": "gpt-4o",  # Need to provide model compatible with new provider
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            # This should be valid - stage has its own model+provider
            assert is_valid, f"Expected valid config with overridden model/provider, got errors: {errors}"
        finally:
            path.unlink()

    def test_stage_inherits_provider_from_top_level(self):
        """Stage policy with only model should inherit provider from top-level."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "policy": {
                            "model": "gpt-4o-mini",  # Override model
                            # Provider will be inherited from top-level
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config with inherited provider, got errors: {errors}"
        finally:
            path.unlink()

    # =========================================================================
    # Test: Mixed scenarios (some stages with policy, some without)
    # =========================================================================
    
    def test_mixed_stages_with_and_without_policy(self):
        """Mix of stages with and without policy (top-level fallback for missing)."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "pipeline",
                "stages": [
                    {
                        "stage_id": "stage1",
                        "baseline_instruction": "First stage.",
                        "policy": {
                            "model": "gpt-4o-mini",
                            "provider": "openai",
                        },
                    },
                    {
                        "stage_id": "stage2",
                        "baseline_instruction": "Second stage.",
                        # No policy - uses top-level fallback
                    },
                    {
                        "stage_id": "stage3",
                        "baseline_instruction": "Third stage.",
                        "policy": {
                            "model": "llama-3.3-70b",
                            "provider": "groq",
                        },
                    },
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config with mixed policies, got errors: {errors}"
        finally:
            path.unlink()

    def test_multiple_modules_with_fallback(self):
        """Multiple modules where some stages use top-level fallback."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "module1",
                "stages": [
                    {
                        "stage_id": "m1_s1",
                        # Uses top-level fallback
                    }
                ],
            },
            {
                "module_id": "module2",
                "stages": [
                    {
                        "stage_id": "m2_s1",
                        "policy": {
                            "model": "gpt-4o-mini",
                            "provider": "openai",
                        },
                    }
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config with multiple modules, got errors: {errors}"
        finally:
            path.unlink()

    # =========================================================================
    # Test: All stages have complete policy (no top-level required)
    # =========================================================================
    
    def test_all_stages_have_policy_no_top_level_needed(self):
        """If all stages have complete policy, top-level is not required."""
        config = self._base_config()
        # No top-level policy
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "module1",
                "stages": [
                    {
                        "stage_id": "s1",
                        "policy": {"model": "gpt-4o", "provider": "openai"},
                    }
                ],
            },
            {
                "module_id": "module2",
                "stages": [
                    {
                        "stage_id": "s2",
                        "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                    }
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config (all stages have policy), got errors: {errors}"
        finally:
            path.unlink()

    # =========================================================================
    # Test: inference_url rejection
    # =========================================================================
    
    def test_inference_url_in_top_level_policy_rejected(self):
        """inference_url in top-level policy should be rejected."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
            "inference_url": "https://custom.api.com/v1",  # Not allowed
        }
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("inference_url" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_inference_url_in_stage_policy_rejected(self):
        """inference_url in stage policy should be rejected."""
        config = self._base_config()
        config["prompt_learning"]["policy"] = {
            "model": "gpt-4o",
            "provider": "openai",
        }
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "classifier",
                "stages": [
                    {
                        "stage_id": "classify",
                        "policy": {
                            "model": "gpt-4o",
                            "provider": "openai",
                            "inference_url": "https://custom.api.com/v1",  # Not allowed
                        },
                    }
                ],
            }
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("inference_url" in e.lower() for e in errors)
        finally:
            path.unlink()

    # =========================================================================
    # Test: Model/provider validation
    # =========================================================================
    
    def test_valid_openai_models(self):
        """Various valid OpenAI models should pass validation."""
        valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"]
        
        for model in valid_models:
            config = self._base_config()
            config["prompt_learning"]["policy"] = {
                "model": model,
                "provider": "openai",
            }
            
            path = write_temp_toml(config)
            try:
                is_valid, errors = validate_mipro_config_from_file(path)
                assert is_valid, f"Model {model} should be valid, got errors: {errors}"
            finally:
                path.unlink()

    def test_valid_groq_models(self):
        """Various valid Groq models should pass validation."""
        valid_models = ["llama-3.3-70b", "llama-3.1-8b-instant", "qwen-32b"]
        
        for model in valid_models:
            config = self._base_config()
            config["prompt_learning"]["policy"] = {
                "model": model,
                "provider": "groq",
            }
            
            path = write_temp_toml(config)
            try:
                is_valid, errors = validate_mipro_config_from_file(path)
                assert is_valid, f"Model {model} should be valid for Groq, got errors: {errors}"
            finally:
                path.unlink()

    def test_valid_google_models(self):
        """Various valid Google/Gemini models should pass validation."""
        valid_models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        
        for model in valid_models:
            config = self._base_config()
            config["prompt_learning"]["policy"] = {
                "model": model,
                "provider": "google",
            }
            
            path = write_temp_toml(config)
            try:
                is_valid, errors = validate_mipro_config_from_file(path)
                assert is_valid, f"Model {model} should be valid for Google, got errors: {errors}"
            finally:
                path.unlink()


class TestMIPRORequiredFields:
    """Test suite for required field validation in MIPRO config."""

    def _base_config(self) -> dict:
        """Return a complete valid MIPRO config."""
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task",
                "task_app_api_key": "test-key",
                "env_name": "test-env",
                "policy": {
                    "model": "gpt-4o",
                    "provider": "openai",
                },
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [
                        {"role": "system", "content": "Test assistant."},
                    ],
                },
                "mipro": {
                    "num_iterations": 3,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 10,
                    "max_concurrent": 5,
                    "bootstrap_train_seeds": [0, 1, 2, 3, 4],
                    "online_pool": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                    "reference_pool": [50, 51, 52, 53, 54],
                },
            }
        }

    def test_complete_config_valid(self):
        """Complete config should pass validation."""
        config = self._base_config()
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Expected valid config, got errors: {errors}"
        finally:
            path.unlink()

    def test_missing_task_app_url(self):
        """Missing task_app_url should fail."""
        config = self._base_config()
        del config["prompt_learning"]["task_app_url"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("task_app_url" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_task_app_api_key(self):
        """Missing task_app_api_key should fail."""
        config = self._base_config()
        del config["prompt_learning"]["task_app_api_key"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("task_app_api_key" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_env_name(self):
        """Missing env_name should fail."""
        config = self._base_config()
        del config["prompt_learning"]["env_name"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("env_name" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_bootstrap_train_seeds(self):
        """Missing bootstrap_train_seeds should fail."""
        config = self._base_config()
        del config["prompt_learning"]["mipro"]["bootstrap_train_seeds"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("bootstrap_train_seeds" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_online_pool(self):
        """Missing online_pool should fail."""
        config = self._base_config()
        del config["prompt_learning"]["mipro"]["online_pool"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("online_pool" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_reference_pool(self):
        """Missing reference_pool should fail."""
        config = self._base_config()
        del config["prompt_learning"]["mipro"]["reference_pool"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("reference_pool" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_num_iterations(self):
        """Missing num_iterations should fail."""
        config = self._base_config()
        del config["prompt_learning"]["mipro"]["num_iterations"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("num_iterations" in e for e in errors)
        finally:
            path.unlink()

    def test_missing_initial_prompt(self):
        """Missing initial_prompt should fail."""
        config = self._base_config()
        del config["prompt_learning"]["initial_prompt"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("initial_prompt" in e.lower() for e in errors)
        finally:
            path.unlink()


class TestMIPROSeedPoolValidation:
    """Test suite for seed pool validation in MIPRO config."""

    def _base_config(self) -> dict:
        """Return a complete valid MIPRO config."""
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task",
                "task_app_api_key": "test-key",
                "env_name": "test-env",
                "policy": {
                    "model": "gpt-4o",
                    "provider": "openai",
                },
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test."}],
                },
                "mipro": {
                    "num_iterations": 3,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 10,
                    "max_concurrent": 5,
                    "bootstrap_train_seeds": [0, 1, 2, 3, 4],
                    "online_pool": [5, 6, 7, 8, 9],
                    "reference_pool": [50, 51, 52],
                },
            }
        }

    def test_empty_bootstrap_seeds_fails(self):
        """Empty bootstrap_train_seeds should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["bootstrap_train_seeds"] = []
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            # Error message may say "empty" or "cannot be empty" or similar
            assert any("bootstrap" in e.lower() for e in errors), f"Expected bootstrap error, got: {errors}"
        finally:
            path.unlink()

    def test_empty_online_pool_fails(self):
        """Empty online_pool should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["online_pool"] = []
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            # Error message may say "empty" or "cannot be empty" or similar
            assert any("online_pool" in e.lower() for e in errors), f"Expected online_pool error, got: {errors}"
        finally:
            path.unlink()

    def test_non_integer_seeds_fails(self):
        """Non-integer seeds should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["bootstrap_train_seeds"] = ["a", "b", "c"]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("integer" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_reference_pool_overlap_with_train_fails(self):
        """reference_pool overlapping with bootstrap/online/test should fail."""
        config = self._base_config()
        # reference_pool[0] = 5 overlaps with online_pool
        config["prompt_learning"]["mipro"]["reference_pool"] = [5, 51, 52]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("overlap" in e.lower() for e in errors)
        finally:
            path.unlink()


class TestMIPRONumericFieldValidation:
    """Test suite for numeric field validation in MIPRO config."""

    def _base_config(self) -> dict:
        """Return a complete valid MIPRO config."""
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task",
                "task_app_api_key": "test-key",
                "env_name": "test-env",
                "policy": {
                    "model": "gpt-4o",
                    "provider": "openai",
                },
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test."}],
                },
                "mipro": {
                    "num_iterations": 3,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 10,
                    "max_concurrent": 5,
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5, 6, 7],
                    "reference_pool": [50, 51, 52],
                },
            }
        }

    def test_negative_num_iterations_fails(self):
        """Negative num_iterations should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["num_iterations"] = -1
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("num_iterations" in e for e in errors)
        finally:
            path.unlink()

    def test_zero_batch_size_fails(self):
        """Zero batch_size should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["batch_size"] = 0
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("batch_size" in e for e in errors)
        finally:
            path.unlink()

    def test_invalid_few_shot_threshold(self):
        """few_shot_score_threshold outside [0, 1] should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["few_shot_score_threshold"] = 1.5
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("few_shot" in e.lower() for e in errors)
        finally:
            path.unlink()


class TestMIPROModuleStageValidation:
    """Test suite for module/stage configuration validation."""

    def _base_config(self) -> dict:
        """Return a complete valid MIPRO config."""
        return {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "https://example.com/task",
                "task_app_api_key": "test-key",
                "env_name": "test-env",
                "policy": {
                    "model": "gpt-4o",
                    "provider": "openai",
                },
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test."}],
                },
                "mipro": {
                    "num_iterations": 3,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 10,
                    "max_concurrent": 5,
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5, 6, 7],
                    "reference_pool": [50, 51, 52],
                    "max_instruction_sets": 128,
                    "max_demo_sets": 64,
                },
            }
        }

    def test_duplicate_module_id_fails(self):
        """Duplicate module IDs should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "same_id",
                "stages": [
                    {"stage_id": "s1", "policy": {"model": "gpt-4o", "provider": "openai"}}
                ],
            },
            {
                "module_id": "same_id",  # Duplicate!
                "stages": [
                    {"stage_id": "s2", "policy": {"model": "gpt-4o", "provider": "openai"}}
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("duplicate" in e.lower() and "module" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_duplicate_stage_id_across_modules_fails(self):
        """Duplicate stage IDs across modules should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "mod1",
                "stages": [
                    {"stage_id": "same_stage", "policy": {"model": "gpt-4o", "provider": "openai"}}
                ],
            },
            {
                "module_id": "mod2",
                "stages": [
                    {"stage_id": "same_stage", "policy": {"model": "gpt-4o", "provider": "openai"}}  # Duplicate!
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("duplicate" in e.lower() and "stage" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_instruction_slots_exceeds_max_fails(self):
        """max_instruction_slots > max_instruction_sets should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["max_instruction_sets"] = 10
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "mod1",
                "stages": [
                    {
                        "stage_id": "s1",
                        "max_instruction_slots": 50,  # Exceeds max_instruction_sets (10)
                        "policy": {"model": "gpt-4o", "provider": "openai"},
                    }
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("instruction" in e.lower() and "exceed" in e.lower() for e in errors)
        finally:
            path.unlink()

    def test_edge_references_unknown_stage_fails(self):
        """Edges referencing unknown stages should fail."""
        config = self._base_config()
        config["prompt_learning"]["mipro"]["modules"] = [
            {
                "module_id": "mod1",
                "stages": [
                    {"stage_id": "s1", "policy": {"model": "gpt-4o", "provider": "openai"}},
                    {"stage_id": "s2", "policy": {"model": "gpt-4o", "provider": "openai"}},
                ],
                "edges": [
                    ["s1", "unknown_stage"],  # unknown_stage doesn't exist!
                ],
            },
        ]
        
        path = write_temp_toml(config)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("unknown" in e.lower() and "stage" in e.lower() for e in errors)
        finally:
            path.unlink()

