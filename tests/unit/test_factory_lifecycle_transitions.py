"""Light SDK coverage for FactoryLifecycle transition routes.

These mocks pin the public client contract (path + body + typed response /
typed 409 refusal). They do not exercise the backend reducer.
"""

from __future__ import annotations

import pytest

from synth_ai.managed_research import (
    FactoryLifecycleState,
    FactoryTransitionResponse,
    SmrApiError,
    SmrControlClient,
    SmrStructuredDenialError,
)


def _factory_wire(*, status: str) -> dict[str, object]:
    return {
        "factory_id": "fac_1",
        "org_id": "org_1",
        "name": "demo",
        "kind": "customer",
        "status": status,
        "created_at": "2026-07-20T00:00:00Z",
        "updated_at": "2026-07-20T00:00:00Z",
    }


def _transition_wire(*, command: str, status: str, decision: str = "applied") -> dict[str, object]:
    return {
        "factory": _factory_wire(status=status),
        "command": command,
        "decision": decision,
        "decision_detail": None,
        "to_status": status,
        "effects": ["ScheduleWake"] if command in {"start", "resume"} else [],
        "woken_efforts": 0,
    }


@pytest.mark.parametrize(
    ("method_name", "path_suffix", "to_status"),
    [
        ("start", "start", "active"),
        ("pause", "pause", "paused"),
        ("resume", "resume", "active"),
        ("archive", "archive", "archived"),
    ],
)
def test_factories_transition_posts_named_route(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    path_suffix: str,
    to_status: str,
) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, object]]] = []

    def _request(method: str, path: str, **kwargs):
        calls.append((method, path, dict(kwargs.get("json_body") or {})))
        return _transition_wire(command=path_suffix, status=to_status)

    monkeypatch.setattr(client, "_request_json", _request)

    result = getattr(client.factories, method_name)(
        "fac_1", reason="operator", dry_run=False
    )
    assert isinstance(result, FactoryTransitionResponse)
    assert result.command == path_suffix
    assert result.to_status == FactoryLifecycleState(to_status)
    assert result.factory.status == FactoryLifecycleState(to_status)
    assert calls == [
        (
            "POST",
            f"/smr/factories/fac_1/{path_suffix}",
            {"dry_run": False, "reason": "operator"},
        )
    ]
    client.close()


def test_factories_start_dry_run_is_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    bodies: list[dict[str, object]] = []

    def _request(method: str, path: str, **kwargs):
        bodies.append(dict(kwargs.get("json_body") or {}))
        return _transition_wire(command="start", status="active", decision="preview_applied")

    monkeypatch.setattr(client, "_request_json", _request)
    result = client.factories.start("fac_1", dry_run=True)
    assert result.decision == "preview_applied"
    assert bodies == [{"dry_run": True}]
    client.close()


def test_factories_transition_refusal_surfaces_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")

    def _request(method: str, path: str, **_kwargs):
        raise SmrStructuredDenialError(
            "factory lifecycle transition refused",
            status_code=409,
            detail={
                "error_code": "factory_lifecycle_transition_refused",
                "reason": "archive_requires_terminal_children",
            },
        )

    monkeypatch.setattr(client, "_request_json", _request)
    with pytest.raises(SmrApiError) as exc_info:
        client.factories.archive("fac_1")
    err = exc_info.value
    assert err.status_code == 409
    detail = getattr(err, "detail", None) or getattr(err, "body", None) or {}
    assert detail.get("error_code") == "factory_lifecycle_transition_refused"
    assert detail.get("reason") == "archive_requires_terminal_children"
    client.close()


def test_configured_is_in_lifecycle_vocabulary() -> None:
    assert FactoryLifecycleState.CONFIGURED.value == "configured"
    assert {item.value for item in FactoryLifecycleState} == {
        "configured",
        "active",
        "paused",
        "archived",
    }
