from __future__ import annotations

import base64
import json

import pytest
import responses
from nacl.public import PrivateKey, SealedBox

from synth_ai.rl import MAX_ENVIRONMENT_API_KEY_BYTES
from synth_ai.rl import env_keys as env_keys_module
from synth_ai.rl.env_keys import encrypt_for_backend, setup_environment_api_key


def _pubkey_b64(sk: PrivateKey) -> str:
    return base64.b64encode(sk.public_key.encode()).decode("ascii")


@pytest.mark.unit
def test_encrypt_for_backend_roundtrip() -> None:
    sk = PrivateKey.generate()
    token = "secret-token"

    ct1 = encrypt_for_backend(_pubkey_b64(sk), token)
    ct2 = encrypt_for_backend(_pubkey_b64(sk), token)

    assert ct1 != ct2  # sealed boxes are randomized

    sealed_bytes = base64.b64decode(ct1)
    recovered = SealedBox(sk).decrypt(sealed_bytes).decode("utf-8")
    assert recovered == token


@responses.activate
@pytest.mark.unit
def test_setup_environment_api_key_uploads_encrypted_token(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = "https://example.test"
    api_key = "synth-api-key"
    token = "env-token"
    sk = PrivateKey.generate()

    responses.add(
        responses.GET,
        f"{backend}/api/v1/crypto/public-key",
        json={"alg": "libsodium.sealedbox.v1", "public_key": _pubkey_b64(sk)},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{backend}/api/v1/env-keys",
        json={"id": "cred-123", "name": "ENVIRONMENT_API_KEY", "updated_at": "2025-01-02T03:04:05Z"},
        status=201,
    )

    # Ensure auto-minting is not invoked when the caller provides a token.
    def _unexpected() -> str:
        raise AssertionError("should not mint")

    monkeypatch.setattr(env_keys_module, "mint_environment_api_key", _unexpected)

    result = setup_environment_api_key(backend, api_key, token=token)

    assert result == {
        "stored": True,
        "token": token,
        "id": "cred-123",
        "name": "ENVIRONMENT_API_KEY",
        "updated_at": "2025-01-02T03:04:05Z",
    }

    assert len(responses.calls) == 2
    get_call, post_call = responses.calls

    assert get_call.request.headers.get("Authorization") == f"Bearer {api_key}"
    assert post_call.request.headers.get("Authorization") == f"Bearer {api_key}"
    assert post_call.request.headers.get("Content-Type") == "application/json"

    body = json.loads(post_call.request.body)
    assert body["name"] == "ENVIRONMENT_API_KEY"

    sealed = base64.b64decode(body["ciphertext_b64"])
    recovered = SealedBox(sk).decrypt(sealed).decode("utf-8")
    assert recovered == token


@responses.activate
@pytest.mark.unit
def test_setup_environment_api_key_auto_mints_token(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = "https://example.test"
    api_key = "synth-api-key"
    minted = "minted-token"
    sk = PrivateKey.generate()

    monkeypatch.setattr(env_keys_module, "mint_environment_api_key", lambda: minted)

    responses.add(
        responses.GET,
        f"{backend}/api/v1/crypto/public-key",
        json={"alg": "libsodium.sealedbox.v1", "public_key": _pubkey_b64(sk)},
        status=200,
    )
    responses.add(
        responses.POST,
        f"{backend}/api/v1/env-keys",
        json={"id": "cred-xyz", "name": "ENVIRONMENT_API_KEY", "updated_at": "2025-03-04T05:06:07Z"},
        status=201,
    )

    result = setup_environment_api_key(backend, api_key)

    assert result["stored"] is True
    assert result["token"] == minted

    sealed = base64.b64decode(json.loads(responses.calls[1].request.body)["ciphertext_b64"])
    recovered = SealedBox(sk).decrypt(sealed).decode("utf-8")
    assert recovered == minted


@pytest.mark.unit
def test_setup_environment_api_key_validates_token_size() -> None:
    too_large = "a" * (MAX_ENVIRONMENT_API_KEY_BYTES + 1)
    with pytest.raises(ValueError):
        setup_environment_api_key("https://example.test", "api", token=too_large)


@responses.activate
@pytest.mark.unit
def test_setup_environment_api_key_rejects_unknown_algorithm() -> None:
    backend = "https://example.test"
    api_key = "synth-api-key"
    token = "token"

    responses.add(
        responses.GET,
        f"{backend}/api/v1/crypto/public-key",
        json={"alg": "different", "public_key": base64.b64encode(b"\x00" * 32).decode("ascii")},
        status=200,
    )

    with pytest.raises(RuntimeError, match="unsupported sealed box algorithm"):
        setup_environment_api_key(backend, api_key, token=token)
