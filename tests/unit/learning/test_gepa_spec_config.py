"""Unit tests for GEPA spec mode config validation in SDK."""
import tempfile
from pathlib import Path

import click
import pytest

from synth_ai.sdk.api.train.validators import validate_prompt_learning_config


class TestGEPASpecConfigValidation:
    """Test GEPA spec mode config validation in SDK."""

    def test_accept_spec_mode_with_spec_path(self) -> None:
        """Test that spec mode is accepted when spec_path is provided."""
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
                    "proposer_type": "spec",
                    "spec_path": "examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json",
                    "spec_max_tokens": 5000,
                    "spec_include_examples": True,
                    "spec_priority_threshold": 8,
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
    
    def test_accept_dspy_mode_default(self) -> None:
        """Test that dspy mode (default) is accepted."""
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
                    "proposer_type": "dspy",  # Default
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
    
    def test_spec_mode_requires_spec_path(self) -> None:
        """Spec proposer_type must include a spec_path."""
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
                    "proposer_type": "spec",
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            with pytest.raises(click.ClickException, match="spec_path"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)
    
    def test_reject_invalid_proposer_type(self) -> None:
        """Test that invalid proposer_type is rejected."""
        invalid_types = ["invalid", "synth", "custom", ""]
        
        for proposer_type in invalid_types:
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
                        "proposer_type": proposer_type,
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(click.ClickException, match="Invalid proposer_type"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)
    
    def test_spec_fields_optional(self) -> None:
        """Test that spec fields are optional."""
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
                    "proposer_type": "spec",
                    "spec_path": "path/to/spec.json",
                    # spec_max_tokens, spec_include_examples, spec_priority_threshold are optional
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
    
    def test_proposer_type_requires_exact_case(self) -> None:
        """Validator currently expects exact lowercase values."""
        valid = ["spec", "dspy"]
        invalid = ["SPEC", "Spec", "DSPY", "Dspy"]

        for proposer_type in valid:
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
                        "proposer_type": proposer_type,
                        **(
                            {"spec_path": "examples/task_apps/banking77/spec.json"}
                            if proposer_type == "spec"
                            else {}
                        ),
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

        for proposer_type in invalid:
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
                        "proposer_type": proposer_type,
                    },
                }
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                path = Path(f.name)
            try:
                with pytest.raises(click.ClickException, match="Invalid proposer_type"):
                    validate_prompt_learning_config(config_data, path)
            finally:
                path.unlink(missing_ok=True)

