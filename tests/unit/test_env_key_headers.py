from __future__ import annotations

import types

import os
import builtins


def test_task_client_sends_all_env_keys(monkeypatch):
    # Arrange
    from synth_ai.task.client import TaskAppClient

    monkeypatch.setenv("ENVIRONMENT_API_KEY_ALIASES", "k2, k3 , k2")

    client = TaskAppClient(base_url="http://example", api_key="k1")

    # Act
    headers = client._headers()  # type: ignore[attr-defined]

    # Assert
    # Primary key
    assert headers.get("X-API-Key") == "k1"
    # Authorization bearer mirrors primary
    assert headers.get("Authorization") == "Bearer k1"

    # All keys are present (deduped, CSV)
    x_api_keys = headers.get("X-API-Keys")
    assert isinstance(x_api_keys, str)
    parts = [p.strip() for p in x_api_keys.split(",") if p.strip()]
    assert parts == ["k1", "k2", "k3"]


def test_check_task_app_health_sends_all_env_keys(monkeypatch):
    # Arrange
    from synth_ai.api.train.task_app import check_task_app_health

    sent_headers: dict[str, str] = {}

    def fake_http_get(url: str, *, headers: dict[str, str] | None = None, timeout: float = 10.0):
        nonlocal sent_headers
        sent_headers = dict(headers or {})

        # Minimal response shim
        resp = types.SimpleNamespace()
        setattr(resp, "status_code", 200)
        setattr(resp, "json", lambda: {"ok": True})
        return resp

    # Patch http_get used inside the module
    import synth_ai.api.train.task_app as task_app_mod
    monkeypatch.setenv("ENVIRONMENT_API_KEY_ALIASES", "ak2, ak3")
    monkeypatch.setattr(task_app_mod, "http_get", fake_http_get)

    # Act
    result = check_task_app_health("http://task.app", api_key="ak1")

    # Assert function completes and headers include all keys
    assert result.ok is True
    assert sent_headers.get("X-API-Key") == "ak1"
    # Authorization bearer mirrors primary
    assert sent_headers.get("Authorization") == "Bearer ak1"
    # All keys present in CSV
    csv = sent_headers.get("X-API-Keys")
    assert isinstance(csv, str)
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    assert parts == ["ak1", "ak2", "ak3"]


