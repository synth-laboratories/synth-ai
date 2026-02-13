"""Regression: PromptLearningJob backend_url should respect env set after import."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_from_config_uses_env_backend_url_set_after_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Simulate a common dev workflow:
    # 1. import synth_ai modules (BACKEND_URL_BASE computed)
    # 2. later load `.env` / export SYNTH_BACKEND_URL
    #
    # Historically, PromptLearningJob.from_config() would still use the stale
    # import-time BACKEND_URL_BASE and could point at production unexpectedly.
    monkeypatch.delenv("SYNTH_BACKEND_URL", raising=False)

    import synth_ai.sdk.optimization.internal.prompt_learning as pl

    importlib.reload(pl)

    monkeypatch.setenv("SYNTH_BACKEND_URL", "http://localhost:8000")

    def _fake_ensure_localapi_auth(*, backend_base: str | None = None, synth_api_key: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
        return "env_test_key"

    monkeypatch.setattr(pl, "ensure_localapi_auth", _fake_ensure_localapi_auth)

    cfg_path = tmp_path / "gepa.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[prompt_learning]",
                'algorithm = "gepa"',
                'task_app_url = "http://127.0.0.1:8017"',
                'env_name = "engine_bench"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    job = pl.PromptLearningJob.from_config(
        config_path=cfg_path,
        api_key="sk_test",
    )

    assert job.config.backend_url == "http://localhost:8000"


@pytest.mark.unit
def test_from_dict_uses_env_backend_url_set_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYNTH_BACKEND_URL", raising=False)

    import synth_ai.sdk.optimization.internal.prompt_learning as pl

    importlib.reload(pl)

    monkeypatch.setenv("SYNTH_BACKEND_URL", "http://localhost:8000")

    def _fake_ensure_localapi_auth(*, backend_base: str | None = None, synth_api_key: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
        return "env_test_key"

    monkeypatch.setattr(pl, "ensure_localapi_auth", _fake_ensure_localapi_auth)

    job = pl.PromptLearningJob.from_dict(
        config_dict={"prompt_learning": {"task_app_url": "http://127.0.0.1:8017"}},
        api_key="sk_test",
    )

    assert job.config.backend_url == "http://localhost:8000"
