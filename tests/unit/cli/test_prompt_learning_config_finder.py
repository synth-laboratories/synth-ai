"""Unit tests for prompt learning config type detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from synth_ai.api.train.config_finder import _infer_config_type, discover_configs


class TestPromptLearningConfigTypeDetection:
    """Test automatic detection of prompt_learning config type."""

    def test_detect_mipro_config_from_dict(self) -> None:
        """Test detecting MIPRO config from dictionary data."""
        data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "mipro": {
                    "num_candidates": 5,
                    "num_iterations": 3,
                },
            }
        }
        config_type = _infer_config_type(data)
        assert config_type == "prompt_learning"

    def test_detect_gepa_config_from_dict(self) -> None:
        """Test detecting GEPA config from dictionary data."""
        data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "gepa": {
                    "population_size": 10,
                    "num_generations": 5,
                    "mutation_rate": 0.2,
                },
            }
        }
        config_type = _infer_config_type(data)
        assert config_type == "prompt_learning"

    def test_detect_prompt_learning_top_level_algorithm(self) -> None:
        """Test detecting prompt_learning when algorithm is at top level."""
        data = {
            "algorithm": "mipro",
            "task_app_url": "http://localhost:8001",
            "mipro": {
                "num_candidates": 5,
                "num_iterations": 3,
            },
        }
        config_type = _infer_config_type(data)
        assert config_type == "prompt_learning"

    def test_detect_rl_config_not_prompt_learning(self) -> None:
        """Test that RL config is not detected as prompt_learning."""
        data = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
            },
        }
        config_type = _infer_config_type(data)
        assert config_type == "rl"

    def test_detect_sft_config_not_prompt_learning(self) -> None:
        """Test that SFT config is not detected as prompt_learning."""
        data = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                "variety": "lora",
            },
            "job": {
                "model": "Qwen/Qwen3-0.6B",
                "data": "dataset.jsonl",
            },
        }
        config_type = _infer_config_type(data)
        assert config_type == "sft"

    def test_discover_mipro_config_from_toml(self) -> None:
        """Test discovering MIPRO config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.mipro]
num_iterations = 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            candidates = discover_configs([str(path)], requested_type=None)
            assert len(candidates) == 1
            assert candidates[0].train_type == "prompt_learning"
            # Compare resolved paths to handle symlinks (e.g., /var vs /private/var on macOS)
            assert candidates[0].path.resolve() == path.resolve()
        finally:
            path.unlink()

    def test_discover_gepa_config_from_toml(self) -> None:
        """Test discovering GEPA config from TOML file."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.policy]
model = "gpt-4o-mini"
temperature = 0.7

[prompt_learning.gepa]
num_generations = 5
mutation_rate = 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            candidates = discover_configs([str(path)], requested_type=None)
            assert len(candidates) == 1
            assert candidates[0].train_type == "prompt_learning"
            # Compare resolved paths to handle symlinks (e.g., /var vs /private/var on macOS)
            assert candidates[0].path.resolve() == path.resolve()
        finally:
            path.unlink()

    def test_discover_prompt_learning_with_type_filter(self) -> None:
        """Test discovering prompt_learning configs with type filter."""
        toml_content = """
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"

[prompt_learning.mipro]
num_iterations = 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        try:
            # Should find it when filtering for prompt_learning
            candidates = discover_configs([str(path)], requested_type="prompt_learning")
            assert len(candidates) == 1
            assert candidates[0].train_type == "prompt_learning"

            # When explicitly passing a config path, discover_configs includes it
            # regardless of requested_type. Type filtering only applies to auto-discovery.
            # So we just verify the train_type is correct.
            candidates_all = discover_configs([str(path)], requested_type="rl")
            # It will still include the config but train_type will be prompt_learning
            assert len(candidates_all) == 1
            assert candidates_all[0].train_type == "prompt_learning"
        finally:
            path.unlink()

    def test_priority_prompt_learning_over_rl_sft(self) -> None:
        """Test that prompt_learning has highest priority in detection."""
        # This tests that [prompt_learning] section is checked before RL/SFT
        data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
            },
            "algorithm": {
                "type": "offline",  # This would look like SFT
                "method": "sft",
            },
        }
        config_type = _infer_config_type(data)
        assert config_type == "prompt_learning"

    def test_invalid_prompt_learning_algorithm_ignored(self) -> None:
        """Test that invalid algorithm value doesn't trigger prompt_learning detection."""
        data = {
            "algorithm": "unknown_algorithm",
            "task_app_url": "http://localhost:8001",
        }
        config_type = _infer_config_type(data)
        # Should fall back to some other detection logic, not "prompt_learning"
        assert config_type != "prompt_learning"

    def test_case_insensitive_algorithm_detection(self) -> None:
        """Test that algorithm detection is case-insensitive."""
        data = {
            "prompt_learning": {
                "algorithm": "MIPRO",  # Uppercase
                "task_app_url": "http://localhost:8001",
            }
        }
        config_type = _infer_config_type(data)
        assert config_type == "prompt_learning"

        data2 = {
            "algorithm": "GePa",  # Mixed case
            "task_app_url": "http://localhost:8001",
        }
        config_type2 = _infer_config_type(data2)
        assert config_type2 == "prompt_learning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

