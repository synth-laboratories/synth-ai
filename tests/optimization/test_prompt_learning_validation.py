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
                    "train_seeds": list(range(70)),
                    "val_seeds": list(range(70, 80)),
                },
                "archive": {
                    "pareto_set_size": 10,
                },
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
            train_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
            val_seeds = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

            [prompt_learning.gepa.archive]
            pareto_set_size = 10
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
