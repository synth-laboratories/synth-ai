from __future__ import annotations

import textwrap
from pathlib import Path

from synth_ai.sdk.optimization.internal.configs.prompt_learning import load_toml
from synth_ai.sdk.optimization.internal.validators import validate_prompt_learning_config


def _gepa_config_dict() -> dict:
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": "http://example.com",
            "policy": {
                "inference_mode": "synth_hosted",
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
            "gepa": {
                "evaluation": {
                    "train_seeds": [0],
                    "val_seeds": [1],
                }
            },
        }
    }


def _mipro_config_dict() -> dict:
    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_url": "http://example.com",
            "env_name": "demo",
            "bootstrap_train_seeds": [0],
            "online_pool": [1, 2],
            "policy": {
                "inference_mode": "synth_hosted",
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
            "mipro": {
                "env_name": "demo",
                "bootstrap_train_seeds": [0],
                "online_pool": [1, 2],
            },
        }
    }


def test_validate_prompt_learning_config_gepa_dict() -> None:
    validate_prompt_learning_config(_gepa_config_dict(), Path("<memory>"))


def test_validate_prompt_learning_config_mipro_dict() -> None:
    validate_prompt_learning_config(_mipro_config_dict(), Path("<memory>"))


def test_validate_prompt_learning_config_gepa_toml(tmp_path: Path) -> None:
    path = tmp_path / "gepa.toml"
    path.write_text(
        textwrap.dedent(
            """
            [prompt_learning]
            algorithm = "gepa"
            task_app_url = "http://example.com"

            [prompt_learning.policy]
            inference_mode = "synth_hosted"
            provider = "openai"
            model = "gpt-4o-mini"

            [prompt_learning.gepa.evaluation]
            train_seeds = [0]
            val_seeds = [1]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    data = load_toml(path)
    validate_prompt_learning_config(data, path)


def test_validate_prompt_learning_config_mipro_toml(tmp_path: Path) -> None:
    path = tmp_path / "mipro.toml"
    path.write_text(
        textwrap.dedent(
            """
            [prompt_learning]
            algorithm = "mipro"
            task_app_url = "http://example.com"
            env_name = "demo"
            bootstrap_train_seeds = [0]
            online_pool = [1, 2]

            [prompt_learning.policy]
            inference_mode = "synth_hosted"
            provider = "openai"
            model = "gpt-4o-mini"

            [prompt_learning.mipro]
            env_name = "demo"
            bootstrap_train_seeds = [0]
            online_pool = [1, 2]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    data = load_toml(path)
    validate_prompt_learning_config(data, path)
