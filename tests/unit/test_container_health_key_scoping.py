from __future__ import annotations

from synth_ai.sdk.optimization.internal import container_app


class _FakeResp:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_harbor_status_probe_does_not_try_synth_api_key(
    monkeypatch,
) -> None:
    seen_keys: list[str] = []

    def _fake_http_get(url: str, headers: dict[str, str], timeout: float):
        if url.endswith("/status"):
            seen_keys.append(headers.get("X-API-Key", ""))
            return _FakeResp(401, {})
        return _FakeResp(500, {})

    monkeypatch.setenv("SYNTH_API_KEY", "sk_synth_only")
    monkeypatch.setattr(container_app, "http_get", _fake_http_get)

    health = container_app.check_container_health(
        "http://example.com/api/harbor/deployments/dep-1",
        api_key="sk_env_only",
        timeout=0.1,
        max_retries=1,
    )

    assert health.ok is False
    assert seen_keys == ["sk_env_only"]


def test_container_deployment_probe_does_not_override_api_key_with_synth_api_key(
    monkeypatch,
) -> None:
    seen_headers: list[dict[str, str]] = []

    class _FakeHttpxResponse:
        def __init__(self, status_code: int, payload: object) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = ""

        def json(self):
            return self._payload

    def _fake_httpx_get(url: str, headers: dict[str, str], timeout: float):
        seen_headers.append(dict(headers))
        return _FakeHttpxResponse(
            200,
            [{"deployment_id": "dep-1", "status": "ready"}],
        )

    import httpx

    monkeypatch.setenv("SYNTH_API_KEY", "sk_synth_only")
    monkeypatch.setattr(httpx, "get", _fake_httpx_get)

    health = container_app.check_container_health(
        "https://example.com/api/container/deployments/dep-1",
        api_key="sk_env_only",
        timeout=0.1,
        max_retries=1,
    )

    assert health.ok is True
    assert seen_headers
    assert seen_headers[0].get("X-API-Key") == "sk_env_only"


def test_https_ip_strict_never_forces_verify_false(monkeypatch) -> None:
    verify_values: list[object] = []

    class _FakeReqResp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def json(self):
            if self.status_code == 200:
                return {"status": "ready"}
            return {}

    def _fake_requests_get(url: str, headers: dict[str, str], timeout: float, **kwargs):
        verify_values.append(kwargs.get("verify"))
        if url.endswith("/health") or url.endswith("/task_info"):
            return _FakeReqResp(200)
        return _FakeReqResp(404)

    monkeypatch.setattr(container_app, "_curl_with_resolve", lambda *a, **k: None)
    monkeypatch.setattr(
        container_app,
        "_resolve_url_to_ip",
        lambda url: (url.replace("example.com", "1.2.3.4"), "example.com"),
    )
    monkeypatch.setattr(container_app.requests, "get", _fake_requests_get)

    health = container_app.check_container_health(
        "https://example.com",
        api_key="sk_env_only",
        timeout=0.1,
        max_retries=1,
    )

    assert health.ok is True
    assert verify_values
    assert False not in verify_values
