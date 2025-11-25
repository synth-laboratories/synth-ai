from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from synth_ai.sdk.api.models.supported import UnsupportedModelError
from synth_ai.sdk.learning.sft.config import (
    SFTTrainingHyperparameters,
    prepare_sft_job_payload,
)


def test_prepare_sft_job_payload_happy_path():
    payload = prepare_sft_job_payload(
        model="Qwen/Qwen3-0.6B",
        training_file="file-123",
        hyperparameters={"n_epochs": 2, "learning_rate": 1e-5},
        metadata={"tags": ["demo"]},
        training_type="sft_offline",
    )
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["training_file_id"] == "file-123"
    assert payload["hyperparameters"]["n_epochs"] == 2
    assert payload["metadata"] == {"tags": ["demo"]}
    assert payload["training_type"] == "sft_offline"


def test_prepare_sft_job_payload_requires_steps():
    with pytest.raises(ValueError):
        prepare_sft_job_payload(
            model="Qwen/Qwen3-0.6B",
            training_file="file-123",
            hyperparameters={"learning_rate": 1e-5},
        )


def test_prepare_sft_job_payload_requires_training_file_by_default():
    with pytest.raises(ValueError):
        prepare_sft_job_payload(
            model="Qwen/Qwen3-0.6B",
            training_file=None,
            hyperparameters={"n_epochs": 1},
        )


def test_prepare_sft_job_payload_includes_none_when_requested():
    payload = prepare_sft_job_payload(
        model="Qwen/Qwen3-0.6B",
        training_file=None,
        hyperparameters={"n_epochs": 1},
        require_training_file=False,
        include_training_file_when_none=True,
    )
    assert "training_file_id" in payload
    assert payload["training_file_id"] is None


def test_training_hyperparameters_from_mapping_validates_known_fields():
    hp = SFTTrainingHyperparameters.from_mapping(
        {
            "n_epochs": "3",
            "batch_size": 8,
            "learning_rate": "1e-5",
            "warmup_ratio": 0.1,
            "parallelism": {"tp": 2},
        }
    )
    data = hp.to_dict()
    assert data["n_epochs"] == 3
    assert data["batch_size"] == 8
    assert data["learning_rate"] == pytest.approx(1e-5)
    assert data["warmup_ratio"] == pytest.approx(0.1)
    assert data["parallelism"] == {"tp": 2}


def test_training_hyperparameters_rejects_invalid_warmup_ratio():
    with pytest.raises(ValueError):
        SFTTrainingHyperparameters.from_mapping(
            {"n_epochs": 1, "warmup_ratio": 1.5}
        )


def test_prepare_accepts_dataclass_instance():
    hp = SFTTrainingHyperparameters.from_mapping({"n_epochs": 1})
    payload = prepare_sft_job_payload(
        model="Qwen/Qwen3-0.6B",
        training_file="file-123",
        hyperparameters=hp,
    )
    assert payload["hyperparameters"]["n_epochs"] == 1


def test_prepare_rejects_unsupported_model():
    with pytest.raises(UnsupportedModelError):
        prepare_sft_job_payload(
            model="Unknown/Model",
            training_file="file-123",
            hyperparameters={"n_epochs": 1},
        )
