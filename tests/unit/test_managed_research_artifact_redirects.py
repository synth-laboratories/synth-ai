from __future__ import annotations

from dataclasses import dataclass

from synth_ai.sdk.managed_research import SmrControlClient


@dataclass
class _FakeResponse:
    status_code: int
    content: bytes
    text: str = ""


def test_get_artifact_content_bytes_follows_redirects_by_default() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://example.test")
    captured: dict[str, object] = {}

    def _fake_get(path: str, *, params, follow_redirects: bool):  # type: ignore[no-untyped-def]
        captured["path"] = path
        captured["params"] = params
        captured["follow_redirects"] = follow_redirects
        return _FakeResponse(status_code=200, content=b"artifact-bytes")

    client._client.get = _fake_get  # type: ignore[assignment]
    try:
        payload = client.get_artifact_content_bytes("artifact-1")
    finally:
        client.close()

    assert payload == b"artifact-bytes"
    assert captured["path"] == "/smr/artifacts/artifact-1/content"
    assert captured["params"] == {"disposition": "inline"}
    assert captured["follow_redirects"] is True
