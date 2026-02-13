from __future__ import annotations

import textwrap
from pathlib import Path

from synth_ai.sdk.optimization.internal.configs.prompt_learning import load_toml
from synth_ai.sdk.optimization.internal.validators import (
    _filter_known_gepa_alias_warnings,
    validate_prompt_learning_config,
)


def _gepa_config_dict() -> dict:
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "container_url": "http://example.com",
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
            "container_url": "http://example.com",
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


def test_validate_prompt_learning_config_gepa_proposer_backend_rlm() -> None:
    """proposer_backend='rlm' and 'agent' are accepted; rust_backend dispatches."""
    config = _gepa_config_dict()
    config["prompt_learning"]["gepa"]["proposer_backend"] = "rlm"
    config["prompt_learning"]["gepa"]["proposer"] = {"rlm": {"graph_id": "g_123"}}
    validate_prompt_learning_config(config, Path("<memory>"))


def test_validate_prompt_learning_config_gepa_proposer_backend_agent() -> None:
    """proposer_backend='agent' is accepted; rust_backend dispatches."""
    config = _gepa_config_dict()
    config["prompt_learning"]["gepa"]["proposer_backend"] = "agent"
    config["prompt_learning"]["gepa"]["proposer"] = {
        "agent": {"agent_provider": "codex"},
    }
    validate_prompt_learning_config(config, Path("<memory>"))


def test_validate_prompt_learning_config_mipro_dict() -> None:
    validate_prompt_learning_config(_mipro_config_dict(), Path("<memory>"))


def test_validate_prompt_learning_config_gepa_toml(tmp_path: Path) -> None:
    path = tmp_path / "gepa.toml"
    path.write_text(
        textwrap.dedent(
            """
            [prompt_learning]
            algorithm = "gepa"
            container_url = "http://example.com"

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
            container_url = "http://example.com"
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


def test_validate_prompt_learning_config_mipro_text_dreamer_toml(tmp_path: Path) -> None:
    path = tmp_path / "mipro_text_dreamer.toml"
    path.write_text(
        textwrap.dedent(
            """
            [prompt_learning]
            algorithm = "mipro"
            container_url = "http://example.com"

            [prompt_learning.policy]
            inference_mode = "synth_hosted"
            provider = "openai"
            model = "gpt-4o-mini"

            [prompt_learning.mipro]
            env_name = "crafter_container"
            bootstrap_train_seeds = [0]
            online_pool = [1, 2]

            [prompt_learning.mipro.ontology]
            reads = true

            [prompt_learning.mipro.text_dreamer]
            enabled = true
            mode = "observation_only"
            world_model_mode = "ontology_plus_wm"
            runtime_backend = "rhodes"
            max_pending_jobs_per_system = 2
            max_replay_rollouts = 4
            observation_trigger_every_rollouts = 1
            observation_log_window = 25
            shadow_rollouts = 2
            shadow_max_turns = 2
            shadow_timeout_seconds = 45
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    data = load_toml(path)
    validate_prompt_learning_config(data, path)


def test_filter_known_alias_warnings_includes_mipro_compatibility_keys() -> None:
    warnings = [
        "Unknown field 'proposer' in [prompt_learning.mipro]. This field will be ignored.",
        "Unknown field 'val_seeds' in [prompt_learning.mipro]. This field will be ignored.",
        "Unknown field 'mode' in [prompt_learning.mipro]. This field will be ignored.",
        "Unknown field 'custom_field' in [prompt_learning.mipro]. This field will be ignored.",
    ]
    filtered = _filter_known_gepa_alias_warnings(warnings)
    assert filtered == [
        "Unknown field 'custom_field' in [prompt_learning.mipro]. This field will be ignored."
    ]
