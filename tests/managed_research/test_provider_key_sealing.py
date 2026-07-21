"""Provider-key uploads seal plaintext locally before crossing the API boundary."""

from __future__ import annotations

import base64
from typing import Any

import pytest
from nacl.public import PrivateKey, SealedBox
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _client() -> ManagedResearchClient:
    return ManagedResearchClient(
        api_key="sk-test",
        backend_base="https://api.example.test",
    )


def test_set_provider_key_seals_plaintext_before_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = PrivateKey.generate()
    public_key_b64 = base64.b64encode(bytes(private_key.public_key)).decode("ascii")
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    client = _client()

    def _request_json(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((method, path, json_body))
        if method == "GET":
            return {
                "alg": "libsodium.sealedbox.v1",
                "public_key": public_key_b64,
            }
        return {"provider": "openai", "configured": True}

    monkeypatch.setattr(client, "_request_json", _request_json)
    try:
        client.set_provider_key(
            "project-1",
            provider="openai",
            api_key="provider-secret",
        )
    finally:
        client.close()

    assert calls[0][:2] == ("GET", "/api/v1/crypto/public-key")
    assert calls[1][:2] == ("POST", "/smr/projects/project-1/provider_keys")
    posted = calls[1][2]
    assert posted is not None
    assert "api_key" not in posted
    ciphertext = base64.b64decode(posted["encrypted_key_b64"])
    plaintext = SealedBox(private_key).decrypt(ciphertext).decode("utf-8")
    assert plaintext == "provider-secret"


def test_set_provider_key_rejects_unsupported_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    calls: list[tuple[str, str]] = []

    def _request_json(method: str, path: str, **_kwargs: Any) -> dict[str, str]:
        calls.append((method, path))
        return {"alg": "rsa.v1", "public_key": "unused"}

    monkeypatch.setattr(client, "_request_json", _request_json)
    try:
        with pytest.raises(ValueError, match="unsupported backend provider-key"):
            client.set_provider_key(
                "project-1",
                provider="openai",
                api_key="provider-secret",
            )
    finally:
        client.close()

    assert calls == [("GET", "/api/v1/crypto/public-key")]


def test_set_provider_key_rejects_plaintext_and_ciphertext_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    calls: list[tuple[str, str]] = []

    def _request_json(method: str, path: str, **_kwargs: Any) -> dict[str, str]:
        calls.append((method, path))
        return {}

    monkeypatch.setattr(client, "_request_json", _request_json)
    try:
        with pytest.raises(ValueError, match="api_key or encrypted_key_b64"):
            client.set_provider_key(
                "project-1",
                provider="openai",
                api_key="provider-secret",
                encrypted_key_b64="ciphertext",
            )
    finally:
        client.close()

    assert calls == []
