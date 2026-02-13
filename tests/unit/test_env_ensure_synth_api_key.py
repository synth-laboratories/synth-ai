from __future__ import annotations

import os

import pytest


@pytest.mark.unit
def test_ensure_synth_api_key_defaults_to_local_dev_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.core.utils import env as env_utils

    monkeypatch.delenv("SYNTH_API_KEY", raising=False)

    called: dict[str, object] = {}

    def _fake_mint_demo_api_key(*, backend_url=None, **kwargs):  # type: ignore[no-untyped-def]
        called["backend_url"] = backend_url
        return "sk_demo_should_not_be_used"

    monkeypatch.setattr(env_utils, "mint_demo_api_key", _fake_mint_demo_api_key)

    key = env_utils.ensure_synth_api_key(backend_url="http://localhost:8000")

    assert key == env_utils.LOCAL_DEV_SYNTH_API_KEY
    assert os.environ.get("SYNTH_API_KEY") == env_utils.LOCAL_DEV_SYNTH_API_KEY
    assert called == {}


@pytest.mark.unit
def test_ensure_synth_api_key_mints_for_remote_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.core.utils import env as env_utils

    monkeypatch.delenv("SYNTH_API_KEY", raising=False)

    def _fake_mint_demo_api_key(*, backend_url=None, **kwargs):  # type: ignore[no-untyped-def]
        assert backend_url == "https://example.com"
        return "sk_demo_123"

    monkeypatch.setattr(env_utils, "mint_demo_api_key", _fake_mint_demo_api_key)

    key = env_utils.ensure_synth_api_key(backend_url="https://example.com")
    assert key == "sk_demo_123"
    assert os.environ.get("SYNTH_API_KEY") == "sk_demo_123"
