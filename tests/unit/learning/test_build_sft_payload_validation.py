from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from synth_ai.sdk.api.train.builders import build_sft_payload
from synth_ai.sdk.api.train.utils import TrainError


@pytest.mark.fast
def test_build_sft_payload_rejects_legacy_train_schema(tmp_path: Path) -> None:
    # Obfuscated config path and legacy schema using [train] and [data].file
    # Our builder expects [job].data and [training]/[hyperparameters] blocks.
    toml_text = textwrap.dedent(
        """
        [backend]
        base_url = "https://example.invalid/api"
        api_key_env = "SYNTH_API_KEY"

        [train]
        training_type = "sft_offline"
        train_kind = "peft"
        model = "Qwen/Qwen3-0.6B"
        epochs = 1
        batch_size = 2
        learning_rate = 2e-5
        thinking_budget = 256

        [data]
        file = "/obfuscated/path/to/data.jsonl"

        [compute]
        gpu_type = "H100"
        gpu_count = 1
        nodes = 1
        """
    ).strip()

    cfg_path = tmp_path / "ft_qlora_prod_obfuscated.toml"
    cfg_path.write_text(toml_text + "\n", encoding="utf-8")

    with pytest.raises(TrainError):
        # Should fail because dataset is not in [job].data or --dataset override
        build_sft_payload(config_path=cfg_path, dataset_override=None, allow_experimental=None)

