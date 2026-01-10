"""Unit tests for train TOML validation logic."""

import pytest
from synth_ai.sdk.api.train.validation import (
    InvalidRLConfigError,
    InvalidSFTConfigError,
    MissingAlgorithmError,
    MissingComputeError,
    MissingDatasetError,
    MissingModelError,
    UnsupportedAlgorithmError,
    validate_rl_config,
    validate_sft_config,
)


class TestSFTValidation:
    """Test SFT TOML validation."""

    def test_valid_sft_fft_config(self) -> None:
        """Test validation of a valid FFT SFT config."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                "variety": "fft",
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "path/to/dataset.jsonl",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 4,
                "nodes": 1,
            },
            "training": {
                "mode": "full_finetune",
                "use_qlora": False,
            },
            "hyperparameters": {
                "n_epochs": 1,
                "train_kind": "fft",
                "per_device_batch": 1,
                "gradient_accumulation_steps": 1,
                "sequence_length": 1024,
                "learning_rate": 5e-6,
                "warmup_ratio": 0.03,
                "global_batch": 4,
            },
        }

        result = validate_sft_config(config)
        assert result is not None
        assert result["job"]["model"] == "Qwen/Qwen3-4B"
        assert result["compute"]["gpu_type"] == "H100"

    def test_valid_sft_lora_config(self) -> None:
        """Test validation of a valid LoRA SFT config."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                "variety": "lora",
            },
            "job": {
                "model": "Qwen/Qwen3-0.6B",
                "data_path": "path/to/dataset.jsonl",
            },
            "compute": {
                "gpu_type": "A100",
                "gpu_count": 1,
                "nodes": 1,
            },
            "training": {
                "mode": "lora",
                "use_qlora": False,
            },
            "hyperparameters": {
                "n_epochs": 1,
                "train_kind": "peft",
            },
        }

        result = validate_sft_config(config)
        assert result is not None
        assert result["training"]["mode"] == "lora"

    def test_sft_missing_algorithm_section(self) -> None:
        """Test that missing [algorithm] section raises error."""
        config = {
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "dataset.jsonl",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(MissingAlgorithmError):
            validate_sft_config(config)

    def test_sft_missing_job_section(self) -> None:
        """Test that missing [job] section raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(InvalidSFTConfigError):
            validate_sft_config(config)

    def test_sft_missing_model(self) -> None:
        """Test that missing model raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
            },
            "job": {
                "data": "dataset.jsonl",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(MissingModelError):
            validate_sft_config(config)

    def test_sft_missing_dataset(self) -> None:
        """Test that missing dataset raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(MissingDatasetError):
            validate_sft_config(config)

    def test_sft_missing_compute_section(self) -> None:
        """Test that missing [compute] section raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                "variety": "fft",
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "dataset.jsonl",
            },
        }

        with pytest.raises(MissingComputeError):
            validate_sft_config(config)

    def test_sft_missing_gpu_type(self) -> None:
        """Test that missing gpu_type raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                "variety": "fft",
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "dataset.jsonl",
            },
            "compute": {
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(MissingComputeError):
            validate_sft_config(config)

    def test_sft_wrong_algorithm_type(self) -> None:
        """Test that wrong algorithm type raises error."""
        config = {
            "algorithm": {
                "type": "online",  # Wrong for SFT
                "method": "sft",
                "variety": "fft",
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "dataset.jsonl",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(UnsupportedAlgorithmError):
            validate_sft_config(config)

    def test_sft_missing_variety(self) -> None:
        """Test that missing variety raises error."""
        config = {
            "algorithm": {
                "type": "offline",
                "method": "sft",
                # Missing variety
            },
            "job": {
                "model": "Qwen/Qwen3-4B",
                "data": "dataset.jsonl",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 1,
                "nodes": 1,
            },
        }

        with pytest.raises(MissingAlgorithmError):
            validate_sft_config(config)


class TestRLValidation:
    """Test RL TOML validation."""

    def test_valid_rl_full_config(self) -> None:
        """Test validation of a valid full RL config."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test-full",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "type": "single_node_split",
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 1,
                "max_turns": 1,
                "batch_size": 1,
                "group_size": 2,
                "learning_rate": 5e-6,
            },
            "evaluation": {
                "instances": 2,
                "every_n_iters": 1,
                "seeds": [0, 1],
            },
        }

        result = validate_rl_config(config)
        assert result is not None
        assert result["model"]["trainer_mode"] == "full"
        assert result["rollout"]["env_name"] == "math"

    def test_valid_rl_lora_config(self) -> None:
        """Test validation of a valid LoRA RL config."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "policy": {  # NEW: unified policy
                "model_name": "Qwen/Qwen3-1.7B",
                "trainer_mode": "lora",
                "label": "test-lora",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
                "topology": {  # NEW: nested topology
                    "type": "single_node_split",
                    "gpus_for_vllm": 1,
                    "gpus_for_training": 1,
                },
            },
            "rollout": {
                "env_name": "crafter",
                "policy_name": "crafter-react",
                "max_turns": 10,
                "episodes_per_batch": 16,
                "max_concurrent_rollouts": 4,
            },
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 10,
                "max_turns": 10,
                "batch_size": 4,
                "group_size": 4,
                "learning_rate": 5e-5,
                "lora": {  # NEW: nested under training
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": ["all-linear"],
                },
            },
            "evaluation": {
                "instances": 2,
                "every_n_iters": 1,
                "seeds": [0, 1],
            },
        }

        result = validate_rl_config(config)
        assert result is not None
        assert result["policy"]["trainer_mode"] == "lora"
        # LoRA config is now nested under training
        assert "training" in result
        assert "lora" in result["training"]
        assert result["training"]["lora"]["r"] == 16

    def test_rl_missing_algorithm_section(self) -> None:
        """Test that missing [algorithm] section raises error."""
        config = {
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
        }

        with pytest.raises(MissingAlgorithmError):
            validate_rl_config(config)

    def test_rl_missing_model_section(self) -> None:
        """Test that missing [model] section raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
        }

        with pytest.raises(MissingModelError):
            validate_rl_config(config)

    def test_rl_missing_model_base_and_source(self) -> None:
        """Test that missing both base and source raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(MissingModelError):
            validate_rl_config(config)

    def test_rl_missing_trainer_mode(self) -> None:
        """Test that missing trainer_mode raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(InvalidRLConfigError):
            validate_rl_config(config)

    def test_rl_missing_compute_section(self) -> None:
        """Test that missing [compute] section raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
        }

        with pytest.raises(MissingComputeError):
            validate_rl_config(config)

    def test_rl_missing_rollout_section(self) -> None:
        """Test that missing [rollout] section raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
        }

        with pytest.raises(InvalidRLConfigError):
            validate_rl_config(config)

    def test_rl_missing_topology_section(self) -> None:
        """Test that missing [topology] section raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(InvalidRLConfigError):
            validate_rl_config(config)

    def test_rl_wrong_algorithm_type(self) -> None:
        """Test that wrong algorithm type raises error."""
        config = {
            "algorithm": {
                "type": "offline",  # Wrong for RL
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(UnsupportedAlgorithmError):
            validate_rl_config(config)

    def test_rl_missing_variety(self) -> None:
        """Test that missing variety raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                # Missing variety
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(MissingAlgorithmError):
            validate_rl_config(config)

    def test_rl_missing_model_label(self) -> None:
        """Test that missing model label raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                # Missing label
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
        }

        with pytest.raises(InvalidRLConfigError):
            validate_rl_config(config)

    def test_rl_missing_training_required_fields(self) -> None:
        """Test that missing required training fields raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
            "training": {
                "num_epochs": 1,
                # Missing other required fields
            },
        }

        with pytest.raises(InvalidRLConfigError) as exc_info:
            validate_rl_config(config)
        assert "iterations_per_epoch" in exc_info.value.detail

    def test_rl_missing_evaluation_required_fields(self) -> None:
        """Test that missing required evaluation fields raises error."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 1,
                "max_turns": 1,
                "batch_size": 1,
                "group_size": 2,
                "learning_rate": 5e-6,
            },
            "evaluation": {
                "instances": 2,
                # Missing every_n_iters and seeds
            },
        }

        with pytest.raises(InvalidRLConfigError) as exc_info:
            validate_rl_config(config)
        assert "every_n_iters" in exc_info.value.detail

    def test_rl_auto_injects_services_and_reference(self) -> None:
        """Test that services and reference sections are auto-injected if missing."""
        config = {
            "algorithm": {
                "type": "online",
                "method": "policy_gradient",
                "variety": "gspo",
            },
            "model": {
                "base": "Qwen/Qwen3-1.7B",
                "trainer_mode": "full",
                "label": "test",
            },
            "compute": {
                "gpu_type": "H100",
                "gpu_count": 2,
            },
            "topology": {
                "gpus_for_vllm": 1,
                "gpus_for_training": 1,
            },
            "rollout": {
                "env_name": "math",
                "policy_name": "math",
                "max_turns": 1,
                "episodes_per_batch": 2,
                "max_concurrent_rollouts": 2,
            },
            "training": {
                "num_epochs": 1,
                "iterations_per_epoch": 1,
                "max_turns": 1,
                "batch_size": 1,
                "group_size": 2,
                "learning_rate": 5e-6,
            },
            "evaluation": {
                "instances": 2,
                "every_n_iters": 1,
                "seeds": [0, 1],
            },
        }

        result = validate_rl_config(config)
        assert result is not None
        # Services should be injected
        assert "services" in result
        # Reference is now under compute.topology.reference_placement (migrated)
        assert "compute" in result
        assert "topology" in result["compute"]
        assert result["compute"]["topology"]["reference_placement"] == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
