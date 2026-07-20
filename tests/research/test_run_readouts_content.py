"""Tests for run-readout content adapters against a stubbed client (no network)."""

from __future__ import annotations

import base64

import pytest
from synth_ai.research.run_readouts import (
    ResearchRunArtifactsContentAPI,
    ResearchRunWorkProductsContentAPI,
)


class _StubWorkProductsAPI:
    def content(self, work_product_id: str, *, as_text: bool = True) -> str | bytes:
        raw = b"work product body"
        return raw.decode("utf-8") if as_text else raw


class _StubClient:
    def __init__(self, envelope: dict[str, object]) -> None:
        self.envelope = envelope
        self.calls: list[str] = []
        self.work_products = _StubWorkProductsAPI()

    def get_artifact_content(self, artifact_id: str, *, disposition: str = "inline") -> dict:
        self.calls.append(artifact_id)
        return dict(self.envelope)


class _StubHandle:
    def __init__(self, client: _StubClient) -> None:
        self._client = client
        self.run_id = "run_test"
        self.project_id = "proj_test"


def _artifacts_api(envelope: dict[str, object]) -> ResearchRunArtifactsContentAPI:
    return ResearchRunArtifactsContentAPI(_StubHandle(_StubClient(envelope)))


def test_artifact_content_utf8_as_text() -> None:
    api = _artifacts_api({"content": "hello", "encoding": "utf-8"})
    assert api.get("art_1") == "hello"


def test_artifact_content_utf8_as_bytes() -> None:
    api = _artifacts_api({"content": "hello", "encoding": "utf-8"})
    assert api.get("art_1", as_text=False) == b"hello"


def test_artifact_content_base64_as_text() -> None:
    encoded = base64.b64encode(b"payload").decode("ascii")
    api = _artifacts_api({"content": encoded, "encoding": "base64"})
    assert api.get("art_1") == "payload"


def test_artifact_content_base64_as_bytes() -> None:
    encoded = base64.b64encode(b"\x00\x01binary").decode("ascii")
    api = _artifacts_api({"content": encoded, "encoding": "base64"})
    assert api.get("art_1", as_text=False) == b"\x00\x01binary"


def test_artifact_content_unknown_encoding_raises() -> None:
    api = _artifacts_api({"content": "x", "encoding": "zstd"})
    with pytest.raises(ValueError, match="unsupported encoding 'zstd'"):
        api.get("art_1")


def test_work_products_content_passthrough() -> None:
    handle = _StubHandle(_StubClient({}))
    api = ResearchRunWorkProductsContentAPI(handle)
    assert api.get("wp_1") == "work product body"
    assert api.get("wp_1", as_text=False) == b"work product body"
