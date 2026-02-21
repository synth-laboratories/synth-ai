from __future__ import annotations

import sys
import types

import pytest

# builders.py imports click for ClickException, but this unit test only exercises
# prompt-learning payload override wiring. Provide a light stub when click is
# unavailable in minimal test environments.
if "click" not in sys.modules:  # pragma: no cover - environment-specific fallback
    click_stub = types.ModuleType("click")

    class _ClickException(Exception):
        pass

    click_stub.ClickException = _ClickException
    sys.modules["click"] = click_stub

from synth_ai.sdk.optimization.internal import prompt_learning as pl
from synth_ai.sdk.optimization.internal.builders import PromptLearningBuildResult


@pytest.fixture(autouse=True)
def _stub_rust_prompt_learning_job(monkeypatch: pytest.MonkeyPatch) -> None:
    # PromptLearningJob constructor requires synth_ai_py.PromptLearningJob to exist.
    # These tests only exercise payload override wiring, so a minimal stub is enough.
    rust_stub = types.SimpleNamespace(PromptLearningJob=object)
    monkeypatch.setattr(pl, "synth_ai_py", rust_stub)


def _config_dict(*, mode: str | None = None) -> dict:
    payload: dict = {
        "prompt_learning": {
            "container_url": "http://127.0.0.1:8102",
        }
    }
    if mode is not None:
        payload["prompt_learning"]["rollout_health_check_mode"] = mode
    return payload


def _base_config(*, config_dict: dict, overrides: dict | None = None) -> pl.PromptLearningJobConfig:
    return pl.PromptLearningJobConfig(
        config_dict=config_dict,
        backend_url="http://localhost:8000",
        api_key="sk_test",
        container_api_key="env_key_local",
        overrides=overrides or {},
    )


def test_skip_health_check_defaults_backend_rollout_mode_to_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_builder(*, raw_config, task_url, overrides, allow_experimental, source_label):  # type: ignore[no-untyped-def]
        captured["overrides"] = dict(overrides)
        return PromptLearningBuildResult(payload={"ok": True}, task_url="http://127.0.0.1:8102")

    monkeypatch.setattr(pl, "build_prompt_learning_payload_from_mapping", _fake_builder)

    job = pl.PromptLearningJob(
        _base_config(config_dict=_config_dict()),
        skip_health_check=True,
    )
    _ = job._build_payload()

    assert captured["overrides"]["prompt_learning.rollout_health_check_mode"] == "warn"


def test_declared_rollout_mode_is_respected_when_skip_health_check_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_builder(*, raw_config, task_url, overrides, allow_experimental, source_label):  # type: ignore[no-untyped-def]
        captured["overrides"] = dict(overrides)
        return PromptLearningBuildResult(payload={"ok": True}, task_url="http://127.0.0.1:8102")

    monkeypatch.setattr(pl, "build_prompt_learning_payload_from_mapping", _fake_builder)

    job = pl.PromptLearningJob(
        _base_config(config_dict=_config_dict(mode="off")),
        skip_health_check=True,
    )
    _ = job._build_payload()

    assert "prompt_learning.rollout_health_check_mode" not in captured["overrides"]


def test_override_rollout_mode_alias_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_builder(*, raw_config, task_url, overrides, allow_experimental, source_label):  # type: ignore[no-untyped-def]
        captured["overrides"] = dict(overrides)
        return PromptLearningBuildResult(payload={"ok": True}, task_url="http://127.0.0.1:8102")

    monkeypatch.setattr(pl, "build_prompt_learning_payload_from_mapping", _fake_builder)

    job = pl.PromptLearningJob(
        _base_config(
            config_dict=_config_dict(),
            overrides={"prompt_learning.backend_rollout_health_check_mode": "best_effort"},
        ),
        skip_health_check=False,
    )
    _ = job._build_payload()

    assert captured["overrides"]["prompt_learning.rollout_health_check_mode"] == "warn"


def test_invalid_declared_rollout_mode_fails_early() -> None:
    job = pl.PromptLearningJob(
        _base_config(config_dict=_config_dict(mode="sometimes")),
        skip_health_check=False,
    )
    with pytest.raises(ValueError) as exc:
        _ = job._build_payload()
    assert "Invalid rollout_health_check_mode" in str(exc.value)
