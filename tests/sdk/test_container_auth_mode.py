from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from synth_ai.sdk.container._impl import auth


class _Request:
    def __init__(self, path: str, headers: dict[str, str] | None = None, host: str = "127.0.0.1"):
        self.url = SimpleNamespace(path=path)
        self.headers = headers or {}
        self.client = SimpleNamespace(host=host)


@pytest.fixture(autouse=True)
def _reset_auth_adoption_metrics() -> None:
    auth._reset_container_auth_adoption_metrics_for_tests()


def _assert_http_error(exc: Exception, status_code: int, code: str) -> None:
    assert getattr(exc, "status_code", None) == status_code
    detail = getattr(exc, "detail", {})
    assert isinstance(detail, dict)
    err = detail.get("error", {})
    assert isinstance(err, dict)
    assert err.get("code") == code


def test_optional_local_rollout_allows_loopback_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "optional_local")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    auth.require_api_key_dependency(_Request(path="/rollout", headers={}, host="127.0.0.1"))


def test_health_always_allows_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    auth.require_api_key_dependency(_Request(path="/health", headers={}, host="10.0.0.8"))


def test_optional_local_relay_marked_denies_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "optional_local")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    req = _Request(path="/rollout", headers={"x-synth-relay": "1"}, host="127.0.0.1")
    with pytest.raises(Exception) as err:
        auth.require_api_key_dependency(req)
    _assert_http_error(err.value, 401, "auth_missing")


def test_required_mode_denies_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    with pytest.raises(Exception) as err:
        auth.require_api_key_dependency(_Request(path="/rollout", headers={}, host="127.0.0.1"))
    _assert_http_error(err.value, 401, "auth_missing")


def test_required_mode_ignores_legacy_key_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    with pytest.raises(Exception) as err:
        auth.require_api_key_dependency(
            _Request(
                path="/rollout",
                headers={"authorization": "Bearer legacy-token"},
                host="10.0.0.7",
            )
        )
    _assert_http_error(err.value, 401, "auth_missing")


def test_token_header_uses_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    calls: list[tuple[str, str | None]] = []

    def _verify(header_value: str, required_scope: str | None = None) -> None:
        calls.append((header_value, required_scope))

    monkeypatch.setattr(
        auth,
        "synth_ai_py",
        SimpleNamespace(container_verify_paseto_header=_verify),
    )
    req = _Request(
        path="/task_info",
        headers={"x-synth-container-authorization": "Bearer v4.public.test"},
        host="10.0.0.5",
    )

    auth.require_api_key_dependency(req)
    assert calls == [("Bearer v4.public.test", "task_info")]


def test_required_mode_allows_loopback_relay_when_verifier_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    req = _Request(
        path="/rollout",
        headers={
            "x-synth-container-authorization": "Bearer v4.public.test",
            "authorization": "Bearer worker-token",
            "x-synth-relay": "1",
        },
        host="127.0.0.1",
    )
    auth.require_api_key_dependency(req)


def test_required_mode_denies_loopback_relay_without_worker_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    req = _Request(
        path="/rollout",
        headers={
            "x-synth-container-authorization": "Bearer v4.public.test",
            "x-synth-relay": "1",
        },
        host="127.0.0.1",
    )
    with pytest.raises(Exception) as err:
        auth.require_api_key_dependency(req)
    _assert_http_error(err.value, 401, "auth_missing")


def test_invalid_token_fails_even_in_optional_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "optional_local")

    def _verify(_header_value: str, required_scope: str | None = None) -> None:
        raise ValueError(f"bad token for scope={required_scope}")

    monkeypatch.setattr(
        auth,
        "synth_ai_py",
        SimpleNamespace(container_verify_paseto_header=_verify),
    )
    req = _Request(
        path="/rollout",
        headers={"x-synth-container-authorization": "Bearer v4.public.test"},
        host="127.0.0.1",
    )
    with pytest.raises(Exception) as err:
        auth.require_api_key_dependency(req)
    _assert_http_error(err.value, 401, "auth_invalid")


def test_disabled_mode_bypasses_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "disabled")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    auth.require_api_key_dependency(_Request(path="/rollout", headers={}, host="203.0.113.5"))


def test_missing_token_records_none_adoption_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")
    monkeypatch.setattr(auth, "synth_ai_py", None)

    with pytest.raises(HTTPException):
        auth.require_api_key_dependency(
            _Request(
                path="/rollout",
                headers={"authorization": "Bearer legacy-token"},
                host="10.0.0.7",
            )
        )

    metrics = auth.container_auth_adoption_metrics()
    assert metrics == {
        "total": 1,
        "paseto": 0,
        "none": 1,
        "paseto_pct": 0.0,
        "none_pct": 100.0,
    }


def test_paseto_records_adoption_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_AUTH_MODE", "required")

    def _verify(_header_value: str, required_scope: str | None = None) -> None:
        assert required_scope == "task_info"

    monkeypatch.setattr(
        auth,
        "synth_ai_py",
        SimpleNamespace(container_verify_paseto_header=_verify),
    )

    auth.require_api_key_dependency(
        _Request(
            path="/task_info",
            headers={"x-synth-container-authorization": "Bearer v4.public.test"},
            host="10.0.0.5",
        )
    )

    metrics = auth.container_auth_adoption_metrics()
    assert metrics == {
        "total": 1,
        "paseto": 1,
        "none": 0,
        "paseto_pct": 100.0,
        "none_pct": 0.0,
    }
