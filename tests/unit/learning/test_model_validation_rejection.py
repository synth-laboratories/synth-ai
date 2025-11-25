"""Comprehensive unit tests for model validation in prompt learning configs.

These tests ensure that unsupported models are rejected BEFORE sending requests to the backend.
"""
import tempfile
from pathlib import Path

import pytest

from synth_ai.sdk.api.train.validators import (
    ConfigValidationError,
    validate_prompt_learning_config,
)


class TestModelValidationRejection:
    """Test that unsupported models are rejected for prompt learning."""

    def test_reject_unsupported_openai_models(self) -> None:
        """Test that unsupported OpenAI models are rejected."""
        unsupported_models = [
            "gpt-3.5-turbo",  # Not in supported list
            "gpt-4",  # Not in supported list
            "gpt-4-turbo",  # Not in supported list
            "gpt-3",  # Not in supported list
            "claude-3-opus",  # Wrong provider
            "claude-3-sonnet",  # Wrong provider
            "text-davinci-003",  # Legacy model
        ]
        
        for model in unsupported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
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
                with pytest.raises(Exception, match="Unsupported OpenAI model"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_gpt_5_pro(self) -> None:
        """Test that gpt-5-pro is explicitly rejected (too expensive)."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-5-pro",
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
            with pytest.raises(Exception, match="gpt-5-pro.*too expensive"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_reject_unsupported_groq_models(self) -> None:
        """Test that unsupported Groq models are rejected."""
        unsupported_models = [
            "llama-2-70b",  # Not in supported list
            "mixtral-8x7b",  # Not in supported list
            "gemma-7b",  # Not in supported list
            "gpt-4o-mini",  # Wrong provider (OpenAI model)
            "claude-3-opus",  # Wrong provider
        ]
        
        for model in unsupported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
                        "provider": "groq",
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
                with pytest.raises(Exception, match="Unsupported Groq model"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_unsupported_google_models(self) -> None:
        """Test that unsupported Google/Gemini models are rejected."""
        unsupported_models = [
            "gemini-pro",  # Not in supported list
            "gemini-1.5-pro",  # Not in supported list
            "gemini-1.5-flash",  # Not in supported list
            "gemini-ultra",  # Not in supported list
            "gpt-4o-mini",  # Wrong provider (OpenAI model)
            "claude-3-opus",  # Wrong provider
        ]
        
        for model in unsupported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
                        "provider": "google",
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
                with pytest.raises(Exception, match="Unsupported Google"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_unsupported_providers(self) -> None:
        """Test that unsupported providers are rejected."""
        unsupported_providers = [
            "anthropic",
            "cohere",
            "mistral",
            "together",
            "huggingface",
            "custom",
        ]
        
        for provider in unsupported_providers:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": "some-model",
                        "provider": provider,
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
                with pytest.raises(Exception, match="Unsupported provider"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_unsupported_mipro_meta_models(self) -> None:
        """Test that unsupported meta models are rejected for MIPRO."""
        unsupported_meta_models = [
            ("gpt-3.5-turbo", "openai"),
            ("gpt-4", "openai"),
            ("claude-3-opus", "anthropic"),
            ("llama-2-70b", "groq"),
            ("gemini-pro", "google"),
        ]
        
        for model, provider in unsupported_meta_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "mipro",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": "gpt-4o-mini",  # Valid policy model
                        "provider": "openai",
                        "inference_mode": "synth_hosted",
                    },
                    "mipro": {
                        "num_iterations": 5,
                        "num_evaluations_per_iteration": 2,
                        "batch_size": 6,
                        "max_concurrent": 16,
                        "meta_model": model,
                        "meta_model_provider": provider,
                    },
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5],
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(Exception, match="Unsupported"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_unsupported_gepa_mutation_models(self) -> None:
        """Test that unsupported mutation models are rejected for GEPA."""
        unsupported_mutation_models = [
            ("gpt-3.5-turbo", "openai"),
            ("gpt-4", "openai"),
            ("claude-3-opus", "anthropic"),
            ("llama-2-70b", "groq"),
            ("gemini-pro", "google"),
        ]
        
        for model, provider in unsupported_mutation_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": "gpt-4o-mini",  # Valid policy model
                        "provider": "openai",
                        "inference_mode": "synth_hosted",
                    },
                    "gepa": {
                        "num_generations": 10,
                        "mutation": {
                            "llm_model": model,
                            "llm_provider": provider,
                        },
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(Exception, match="Unsupported"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)


class TestSupportedModelsAccepted:
    """Test that supported models are accepted."""

    def test_accept_supported_openai_models(self) -> None:
        """Test that supported OpenAI models are accepted."""
        supported_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ]
        
        for model in supported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
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

    def test_accept_supported_groq_models(self) -> None:
        """Test that supported Groq models are accepted."""
        supported_models = [
            "gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b",
            "llama-3.3-70b-versatile",
            "qwen-32b",
            "qwen3-32b",
            "groq/qwen3-32b",
        ]
        
        for model in supported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
                        "provider": "groq",
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

    def test_accept_supported_google_models(self) -> None:
        """Test that supported Google/Gemini models are accepted."""
        supported_models = [
            "gemini-2.5-pro",
            "gemini-2.5-pro-gt200k",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        
        for model in supported_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,
                        "provider": "google",
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

    def test_accept_model_with_provider_prefix(self) -> None:
        """Test that models with provider prefix are accepted."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "openai/gpt-4o-mini",  # With prefix
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
            # Should not raise - prefix is stripped during validation
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestNanoModelRestrictions:
    """Test that nano models are rejected for proposal/mutation models but allowed for policy models."""

    def test_reject_nano_model_for_gepa_mutation(self) -> None:
        """Test that nano models are rejected for GEPA mutation models."""
        nano_models = [
            "gpt-4.1-nano",
            "gpt-5-nano",
            "openai/gpt-4.1-nano",
            "openai/gpt-5-nano",
        ]
        
        for model in nano_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": "gpt-4o-mini",  # Valid policy model
                        "provider": "openai",
                        "inference_mode": "synth_hosted",
                    },
                    "gepa": {
                        "num_generations": 10,
                        "mutation": {
                            "llm_model": model,
                            "llm_provider": "openai",
                        },
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(Exception, match="Nano models.*NOT allowed for proposal/mutation"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_reject_nano_model_for_mipro_meta_model(self) -> None:
        """Test that nano models are rejected for MIPRO meta models."""
        nano_models = [
            "gpt-4.1-nano",
            "gpt-5-nano",
            "openai/gpt-4.1-nano",
            "openai/gpt-5-nano",
        ]
        
        for model in nano_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "mipro",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": "gpt-4o-mini",  # Valid policy model
                        "provider": "openai",
                        "inference_mode": "synth_hosted",
                    },
                    "mipro": {
                        "num_iterations": 5,
                        "num_evaluations_per_iteration": 2,
                        "batch_size": 6,
                        "max_concurrent": 16,
                        "meta_model": model,
                        "meta_model_provider": "openai",
                    },
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5],
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(Exception, match="Nano models.*NOT allowed for proposal/mutation"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_allow_nano_model_for_policy(self) -> None:
        """Test that nano models ARE allowed for policy models."""
        nano_models = [
            "gpt-4.1-nano",
            "gpt-5-nano",
            "openai/gpt-4.1-nano",
            "openai/gpt-5-nano",
        ]
        
        for model in nano_models:
            config_data = {
                "prompt_learning": {
                    "algorithm": "gepa",
                    "task_app_url": "http://localhost:8001",
                    "policy": {
                        "model": model,  # Nano model as policy - should be allowed
                        "provider": "openai",
                        "inference_mode": "synth_hosted",
                    },
                    "gepa": {
                        "num_generations": 10,
                        "mutation": {
                            "llm_model": "gpt-4o-mini",  # Non-nano mutation model
                            "llm_provider": "openai",
                        },
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                # Should not raise - nano models are allowed for policy
                validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

    def test_allow_nano_model_for_mipro_policy(self) -> None:
        """Test that nano models ARE allowed for MIPRO policy models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4.1-nano",  # Nano model as policy - should be allowed
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",  # Non-nano meta model
                    "meta_model_provider": "openai",
                },
                "bootstrap_train_seeds": [0, 1, 2],
                "online_pool": [3, 4, 5],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise - nano models are allowed for policy
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestModelValidationEdgeCases:
    """Test edge cases for model validation."""

    def test_reject_empty_model(self) -> None:
        """Test that empty model name is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "",
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
            # Empty model is caught as "Missing required field" since empty string is falsy
            with pytest.raises(Exception, match="Missing required field.*model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_reject_missing_model(self) -> None:
        """Test that missing model is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
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
            with pytest.raises(Exception, match="model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_reject_missing_provider(self) -> None:
        """Test that missing provider is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
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
            with pytest.raises(Exception, match="provider"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_case_insensitive_model_validation(self) -> None:
        """Test that model validation is case-insensitive."""
        # Uppercase model name should still be accepted
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "GPT-4O-MINI",  # Uppercase
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
            # Should not raise - validation is case-insensitive
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_case_insensitive_provider_validation(self) -> None:
        """Test that provider validation is case-insensitive."""
        # Uppercase provider should still be accepted
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "OPENAI",  # Uppercase
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
            # Should not raise - validation is case-insensitive
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

