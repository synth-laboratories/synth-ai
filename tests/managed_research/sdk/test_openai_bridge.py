from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from synth_ai.managed_research.sdk.client import (
    OPENAI_TRANSPORT_MODE_AUTO,
    OPENAI_TRANSPORT_MODE_BACKEND_BFF,
    OPENAI_TRANSPORT_MODE_DIRECT_HP,
    SmrControlClient,
)


@dataclass
class _FakeSynthClient:
    openai_agents_sdk: Any
    managed_agents: Any
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_openai_transport_mode_validation() -> None:
    client = SmrControlClient(
        api_key="test-key",
        backend_base="http://localhost:8000",
        openai_transport_mode="AUTO",
    )
    assert client.openai_transport_mode == OPENAI_TRANSPORT_MODE_AUTO
    client.close()

    with pytest.raises(ValueError, match="openai_transport_mode must be one of"):
        SmrControlClient(
            api_key="test-key",
            backend_base="http://localhost:8000",
            openai_transport_mode="invalid-mode",
        )


def test_openai_bridge_is_lazy_and_reuses_single_synth_client(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[dict[str, Any]] = []
    openai_sdk = object()
    anthropic_sdk = object()

    def fake_build(
        *,
        openai_transport_mode: str,
        openai_organization: str | None,
        openai_project: str | None,
        openai_request_id: str | None,
    ) -> _FakeSynthClient:
        calls.append(
            {
                "mode": openai_transport_mode,
                "org": openai_organization,
                "project": openai_project,
                "request_id": openai_request_id,
            }
        )
        return _FakeSynthClient(openai_agents_sdk=openai_sdk, managed_agents=anthropic_sdk)

    monkeypatch.setattr(client, "_build_synth_client", fake_build)

    assert client.openai_agents_sdk is openai_sdk
    assert client.openai_agents_sdk is openai_sdk
    assert client.managed_agents is anthropic_sdk
    assert len(calls) == 1
    assert calls[0]["mode"] == OPENAI_TRANSPORT_MODE_AUTO
    client.close()


def test_openai_agents_sdk_client_override_builds_scoped_client(monkeypatch) -> None:
    client = SmrControlClient(
        api_key="test-key",
        backend_base="http://localhost:8000",
        openai_transport_mode=OPENAI_TRANSPORT_MODE_AUTO,
    )
    created: list[dict[str, Any]] = []

    def fake_build(
        *,
        openai_transport_mode: str,
        openai_organization: str | None,
        openai_project: str | None,
        openai_request_id: str | None,
    ) -> _FakeSynthClient:
        sdk = object()
        created.append(
            {
                "mode": openai_transport_mode,
                "org": openai_organization,
                "project": openai_project,
                "request_id": openai_request_id,
                "sdk": sdk,
            }
        )
        return _FakeSynthClient(openai_agents_sdk=sdk, managed_agents=object())

    monkeypatch.setattr(client, "_build_synth_client", fake_build)

    default_sdk = client.openai_agents_sdk_client()
    override_sdk = client.openai_agents_sdk_client(
        transport_mode=OPENAI_TRANSPORT_MODE_DIRECT_HP,
        openai_organization="org_test",
        openai_project="proj_test",
        request_id="req_test",
    )

    assert default_sdk is created[0]["sdk"]
    assert override_sdk is created[1]["sdk"]
    assert created[0]["mode"] == OPENAI_TRANSPORT_MODE_AUTO
    assert created[1]["mode"] == OPENAI_TRANSPORT_MODE_DIRECT_HP
    assert created[1]["org"] == "org_test"
    assert created[1]["project"] == "proj_test"
    assert created[1]["request_id"] == "req_test"
    client.close()


def test_close_closes_lazy_synth_client(monkeypatch) -> None:
    client = SmrControlClient(
        api_key="test-key",
        backend_base="http://localhost:8000",
        openai_transport_mode=OPENAI_TRANSPORT_MODE_BACKEND_BFF,
    )
    synth_client = _FakeSynthClient(openai_agents_sdk=object(), managed_agents=object())

    def fake_build(**_: Any) -> _FakeSynthClient:
        return synth_client

    monkeypatch.setattr(client, "_build_synth_client", fake_build)
    _ = client.openai_agents_sdk
    assert client._synth_client is synth_client
    client.close()
    assert synth_client.closed is True
    assert client._synth_client is None


def test_missing_synth_ai_dependency_error_message(monkeypatch) -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    original_import = builtins.__import__

    synth_modules = [
        name for name in sys.modules if name == "synth_ai" or name.startswith("synth_ai.")
    ]
    for name in synth_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "synth_ai":
            raise ImportError("No module named synth_ai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="synth-ai is required"):
        client._build_synth_client(
            openai_transport_mode=OPENAI_TRANSPORT_MODE_AUTO,
            openai_organization=None,
            openai_project=None,
            openai_request_id=None,
        )
    client.close()
