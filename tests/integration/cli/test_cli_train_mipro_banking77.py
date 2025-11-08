"""Tests for the Banking77 MIPRO CLI configs.

These cover the new multi-step pipeline configs to make sure the CLI builder
emits the expected module metadata without having to hit the live backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_ai.api.train.builders import build_prompt_learning_payload


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_CONFIG = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_pipeline_mipro_local.toml"
PIPELINE_TEST_CONFIG = REPO_ROOT / "examples" / "blog_posts" / "mipro" / "configs" / "banking77_pipeline_mipro_test.toml"


@pytest.fixture(autouse=True)
def _env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the API keys expected by the builder."""
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "env-key")
    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")


def _build_payload(config_path: Path):
    return build_prompt_learning_payload(
        config_path=config_path,
        task_url=None,
        overrides={"backend": "http://localhost:8000"},
    )


def test_pipeline_builder_includes_module_metadata() -> None:
    result = _build_payload(PIPELINE_CONFIG)
    config_body = result.payload["config_body"]

    prompt_cfg = config_body["prompt_learning"]
    assert result.task_url == "http://127.0.0.1:8112"
    assert prompt_cfg["task_app_api_key"] == "env-key"

    modules = prompt_cfg["initial_prompt"]["metadata"]["pipeline_modules"]
    module_names = [module["name"] for module in modules]
    assert module_names == ["classifier", "calibrator"]
    assert all("instruction_text" in module for module in modules)
    assert all("few_shots" in module for module in modules)


def test_pipeline_builder_includes_module_limits() -> None:
    result = _build_payload(PIPELINE_CONFIG)
    config_body = result.payload["config_body"]

    limits = {
        module["module_id"]: module
        for module in config_body["prompt_learning"]["mipro"]["modules"]
    }

    assert limits["classifier"]["max_instruction_slots"] == 3
    assert limits["classifier"]["max_demo_slots"] == 5
    assert limits["calibrator"]["max_instruction_slots"] == 3
    assert limits["calibrator"]["max_demo_slots"] == 5


def test_pipeline_test_config_uses_reduced_limits() -> None:
    result = _build_payload(PIPELINE_TEST_CONFIG)
    config_body = result.payload["config_body"]

    limits = {
        module["module_id"]: module
        for module in config_body["prompt_learning"]["mipro"]["modules"]
    }

    assert limits["classifier"]["max_instruction_slots"] == 2
    assert limits["calibrator"]["max_instruction_slots"] == 2
    assert limits["classifier"]["max_demo_slots"] == 4
    assert limits["calibrator"]["max_demo_slots"] == 4

    seeds = config_body["prompt_learning"]["mipro"]
    assert seeds["bootstrap_train_seeds"] == [0, 1, 2, 3]
    assert seeds["online_pool"] == [4, 5, 6, 7]
    assert seeds["test_pool"] == [8, 9, 10, 11]
