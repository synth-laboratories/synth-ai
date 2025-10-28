from __future__ import annotations

import base64
import os
import types
from typing import Any

import pytest


class FakeResponse:
    def __init__(self, status_code: int, payload: Any | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


def test_encrypt_for_backend_valid_base64_return() -> None:
    from synth_ai.learning.rl import env_keys

    pub_bytes = b"\x01" * 32
    pub_b64 = base64.b64encode(pub_bytes).decode("ascii")

    ct_b64 = env_keys.encrypt_for_backend(pub_b64, "secret-value")
    # Should be base64-decodable and non-empty
    decoded = base64.b64decode(ct_b64, validate=True)
    assert isinstance(ct_b64, str) and decoded and len(decoded) > 0


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "not-base64!!",
        base64.b64encode(b"\x02" * 31).decode("ascii"),  # wrong length
    ],
)
def test_encrypt_for_backend_rejects_bad_pubkey(bad: str) -> None:
    from synth_ai.learning.rl import env_keys

    with pytest.raises(Exception):
        env_keys.encrypt_for_backend(bad, "x")


def test_setup_environment_api_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.learning.rl import env_keys

    # Prepare deterministic public key and ciphertext
    pub_b = b"\x03" * 32
    pub_b64 = base64.b64encode(pub_b).decode("ascii")
    token = "tok_abc123"
    sealed = base64.b64encode(b"sealed:" + token.encode("utf-8")).decode("ascii")

    # Stub requests.get and requests.post
    requests_mod = types.SimpleNamespace()
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, headers: dict[str, str], timeout: float) -> FakeResponse:
        assert url.endswith("/api/v1/crypto/public-key")
        return FakeResponse(200, {"alg": "libsodium.sealedbox.v1", "public_key": pub_b64})

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: float) -> FakeResponse:
        assert url.endswith("/api/v1/env-keys")
        calls.append((url, json))
        return FakeResponse(201, {"id": "cred-id", "name": json.get("name"), "updated_at": "2025-01-01T00:00:00Z"})

    requests_mod.get = fake_get  # type: ignore[attr-defined]
    requests_mod.post = fake_post  # type: ignore[attr-defined]

    monkeypatch.setenv("ENVIRONMENT_API_KEY", token)
    monkeypatch.setattr(env_keys, "requests", requests_mod)
    monkeypatch.setattr(env_keys, "encrypt_for_backend", lambda pk, s: sealed)

    out = env_keys.setup_environment_api_key("https://example.test", "sk_test_123")
    assert out.get("stored") is True
    assert calls, "no POST performed"
    name = calls[0][1].get("name")
    ct_b64 = calls[0][1].get("ciphertext_b64")
    assert name == "ENVIRONMENT_API_KEY"
    # Should pass through our deterministic value
    assert ct_b64 == sealed
    # And be base64-decodable
    _ = base64.b64decode(ct_b64, validate=True)


def test_setup_environment_api_key_rejects_bad_alg(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.learning.rl import env_keys

    bad_alg_resp = FakeResponse(200, {"alg": "unknown", "public_key": base64.b64encode(b"\x04" * 32).decode("ascii")})

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return bad_alg_resp

    monkeypatch.setattr(env_keys, "requests", types.SimpleNamespace(get=fake_get, post=None))
    with pytest.raises(RuntimeError):
        env_keys.setup_environment_api_key("https://example.test", "sk", token="x")


def test_setup_environment_api_key_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.learning.rl import env_keys

    monkeypatch.delenv("ENVIRONMENT_API_KEY", raising=False)
    with pytest.raises(ValueError):
        env_keys.setup_environment_api_key("https://example.test", "sk", token=None)



