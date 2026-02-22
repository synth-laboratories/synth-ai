from __future__ import annotations

import sys
from typing import Any

import pytest

from synth_ai.sdk.container import auth as auth_mod


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> Any:
        return self._payload


def _install_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    class _HttpxStub:
        @staticmethod
        def get(*args: Any, **kwargs: Any) -> _FakeResponse:
            del args, kwargs
            return response

    monkeypatch.setitem(sys.modules, "httpx", _HttpxStub())


@pytest.mark.unit
def test_fetch_backend_env_key_picks_newest_created_at(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        payload={
            "credentials": [
                {"name": "ENVIRONMENT_API_KEY", "plaintext": "older_key", "created_at": "2026-02-20T00:00:00Z"},
                {"name": "ENVIRONMENT_API_KEY", "plaintext": "new_key", "created_at": "2026-02-21T00:00:00Z"},
            ]
        },
    )
    _install_httpx(monkeypatch, response)

    value = auth_mod._fetch_backend_env_key("https://example.com", "sk_test")

    assert value == "new_key"


@pytest.mark.unit
def test_fetch_backend_env_key_ignores_non_environment_records(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        payload={
            "credentials": [
                {"name": "SOME_OTHER_KEY", "plaintext": "wrong", "created_at": "2026-02-22T00:00:00Z"},
                {"name": "ENVIRONMENT_API_KEY", "plaintext": "right", "created_at": "2026-02-20T00:00:00Z"},
            ]
        },
    )
    _install_httpx(monkeypatch, response)

    value = auth_mod._fetch_backend_env_key("https://example.com", "sk_test")

    assert value == "right"


@pytest.mark.unit
def test_fetch_backend_env_key_falls_back_to_first_valid_without_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status_code=200,
        payload={"credentials": [{"plaintext": "first"}, {"plaintext": "second"}]},
    )
    _install_httpx(monkeypatch, response)

    value = auth_mod._fetch_backend_env_key("https://example.com", "sk_test")

    assert value == "first"


@pytest.mark.unit
def test_fetch_backend_env_key_returns_none_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(status_code=404, payload={})
    _install_httpx(monkeypatch, response)

    value = auth_mod._fetch_backend_env_key("https://example.com", "sk_test")

    assert value is None
