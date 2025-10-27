import base64
import os
import string
from typing import Any

import httpx
import nacl.public  # type: ignore
import pytest
from synth_ai._utils.task_app_env import preflight_env_key
from synth_ai.learning.rl.secrets import mint_environment_api_key


def test_mint_environment_api_key_is_raw_hex(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_token_hex(n: int) -> str:
        captured["nbytes"] = n
        return "ab" * n

    monkeypatch.setattr("secrets.token_hex", fake_token_hex)
    minted = mint_environment_api_key()

    assert captured["nbytes"] == 32
    assert len(minted) == 64
    assert not minted.startswith("sk_")
    assert all(ch in string.hexdigits for ch in minted)
    # Parsing via fromhex ensures the value is valid hexadecimal and even-length.
    assert bytes.fromhex(minted)


def test_preflight_mints_and_uploads_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    minted_value = "f0" * 32  # 64 hex chars; matches secrets.token_hex(32) output length.
    mint_calls: list[str] = []

    def fake_mint() -> str:
        mint_calls.append("called")
        return minted_value

    monkeypatch.setattr("synth_ai.learning.rl.secrets.mint_environment_api_key", fake_mint)
    monkeypatch.delenv("ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.delenv("DEV_ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.setenv("SYNTH_API_KEY", "sk_test_123")
    monkeypatch.setenv("BACKEND_BASE_URL", "https://example.test/api")

    recorded_config: dict[str, str] = {}

    def fake_load_user_config() -> dict[str, str]:
        return recorded_config.copy()

    def fake_update_user_config(updates: dict[str, str]) -> dict[str, str]:
        recorded_config.update({k: str(v) for k, v in updates.items()})
        return recorded_config.copy()

    monkeypatch.setattr("synth_ai._utils.user_config.load_user_config", fake_load_user_config)
    monkeypatch.setattr("synth_ai._utils.task_app_env.load_user_config", fake_load_user_config)
    monkeypatch.setattr("synth_ai._utils.task_app_state.load_user_config", fake_load_user_config)
    monkeypatch.setattr("synth_ai._utils.user_config.update_user_config", fake_update_user_config)
    monkeypatch.setattr("synth_ai._utils.task_app_env.update_user_config", fake_update_user_config)
    monkeypatch.setattr("synth_ai._utils.task_app_state.update_user_config", fake_update_user_config)
    monkeypatch.setattr("synth_ai._utils.task_app_state.update_task_app_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr("synth_ai._utils.task_app_env.load_user_env", lambda override=False: {})

    public_key_bytes = b"\x01" * 32
    public_key_b64 = base64.b64encode(public_key_bytes).decode("ascii")

    class FakePublicKey:
        def __init__(self, key_material: bytes) -> None:
            assert key_material == public_key_bytes
            self.key_material = key_material

    class FakeSealedBox:
        last_plaintext: bytes | None = None

        def __init__(self, pk: FakePublicKey) -> None:
            self.pk = pk

        def encrypt(self, payload: bytes) -> bytes:
            FakeSealedBox.last_plaintext = payload
            return b"sealed:" + payload

    class FakeResponse:
        def __init__(self, status_code: int, payload: Any | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text

        def json(self) -> Any:
            return self._payload

    class FakeClient:
        instances: list["FakeClient"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.requests: list[tuple[str, str, dict[str, Any]]] = []

        def __enter__(self) -> "FakeClient":
            FakeClient.instances.append(self)
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def close(self) -> None:
            return

        def get(self, url: str, **kwargs) -> FakeResponse:
            self.requests.append(("GET", url, kwargs))
            if url.endswith("/v1/crypto/public-key"):
                return FakeResponse(200, {"public_key": public_key_b64, "alg": "libsodium.sealedbox.v1"})
            if url.endswith("/v1/env-keys/verify"):
                return FakeResponse(200, {"present": True})
            return FakeResponse(404, {})

        def post(self, url: str, **kwargs) -> FakeResponse:
            self.requests.append(("POST", url, kwargs))
            return FakeResponse(201, {"id": "env-key-id", "name": "ENVIRONMENT_API_KEY"}, text="")

    FakeClient.instances.clear()
    FakeSealedBox.last_plaintext = None

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(nacl.public, "PublicKey", FakePublicKey)
    monkeypatch.setattr(nacl.public, "SealedBox", FakeSealedBox)

    preflight_env_key(crash_on_failure=True)

    assert mint_calls == ["called"]
    assert os.environ["ENVIRONMENT_API_KEY"] == minted_value
    assert os.environ["DEV_ENVIRONMENT_API_KEY"] == minted_value
    assert recorded_config.get("ENVIRONMENT_API_KEY") == minted_value
    assert recorded_config.get("DEV_ENVIRONMENT_API_KEY") == minted_value

    assert FakeSealedBox.last_plaintext == minted_value.encode("utf-8")

    assert FakeClient.instances, "Expected HTTP client usage during preflight"
    upload_client = FakeClient.instances[-1]
    get_calls = [entry for entry in upload_client.requests if entry[0] == "GET"]
    assert any(
        entry[1].endswith("/v1/crypto/public-key") for entry in get_calls
    ), "Expected public key fetch via GET"
    post_calls = [entry for entry in upload_client.requests if entry[0] == "POST"]
    assert post_calls, "Expected an upload POST request"
    payload = post_calls[0][2]["json"]
    expected_ciphertext = base64.b64encode(b"sealed:" + minted_value.encode("utf-8")).decode("ascii")
    assert payload["name"] == "ENVIRONMENT_API_KEY"
    assert payload["ciphertext_b64"] == expected_ciphertext

    verify_calls = [entry for entry in upload_client.requests if entry[0] == "GET" and entry[1].endswith("/v1/env-keys/verify")]
    assert verify_calls, "Expected verification request after upload"
