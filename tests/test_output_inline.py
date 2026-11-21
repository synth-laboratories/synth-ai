"""swarms_run_output_content: the McpOutputContent representation over MCP (contract 1.31.0)."""

import base64
import hashlib
import json

import httpx
import pytest
import yaml

from synth_ai import AsyncSynthClient
from synth_ai._generated.policy import MCP_OUTPUT_CONTENT_MAX_BYTES
from synth_ai._transport import RetryPolicy
from synth_ai.errors import DataIntegrityError, OutputContentTooLargeError
from synth_ai.mcp.server import call_tool

ARGS = {
    "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
    "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
    "revision_id": "rev_a",
}


def api(content: bytes, media_type: str) -> AsyncSynthClient:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=content, headers={"Content-Type": media_type})
        )
    )
    return AsyncSynthClient(
        "fixture",
        base_url="https://example.test",
        http_client=http,
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0, jitter_seconds=0),
    )


@pytest.mark.parametrize(
    ("content", "media_type", "encoding"),
    [
        ("héllo ✓\n".encode(), "text/plain; charset=utf-8", "utf8"),
        (b'{"a":1}', "application/json", "utf8"),
        (bytes(range(256)) * 3, "application/octet-stream", "base64"),
        (b"f\xff\xfeo", "text/plain", "base64"),  # invalid UTF-8 is never text
        (b"", "text/plain", "utf8"),
    ],
)
async def test_mcp_tool_round_trips_bytes(content: bytes, media_type: str, encoding: str) -> None:
    digest = hashlib.sha256(content).hexdigest()
    async with api(content, media_type) as client:
        raw = await call_tool(
            client, "swarms_run_output_content", {**ARGS, "digest_sha256": digest}
        )
    out = json.loads(raw)
    assert out["encoding"] == encoding
    assert out["media_type"] == media_type
    assert out["byte_size"] == len(content)
    assert out["sha256"] == digest
    decoded = out["content"].encode() if encoding == "utf8" else base64.b64decode(out["content"])
    assert decoded == content


async def test_over_cap_is_refused_with_the_download() -> None:
    content = b"\0" * (MCP_OUTPUT_CONTENT_MAX_BYTES + 1)
    async with api(content, "application/octet-stream") as client:
        with pytest.raises(OutputContentTooLargeError) as refusal:
            await client.swarms.output_inline(
                "run_a", "out_a", revision_id="rev_a", digest_sha256="0" * 64
            )
    problem = refusal.value.problem.model_dump()
    assert refusal.value.code == "output_content_too_large_for_mcp"
    assert problem["byte_size"] == len(content)
    assert problem["max_bytes"] == MCP_OUTPUT_CONTENT_MAX_BYTES
    assert problem["download_operation"] == "getRunOutputContent"


async def test_digest_mismatch_is_refused() -> None:
    async with api(b"abc", "text/plain") as client:
        with pytest.raises(DataIntegrityError):
            await client.swarms.output_inline(
                "run_a", "out_a", revision_id="rev_a", digest_sha256="0" * 64
            )


def test_cap_is_the_contract_maximum() -> None:
    document = yaml.safe_load(open("contracts/synth-v1.openapi.yaml").read())
    schema = document["components"]["schemas"]["McpOutputContent"]
    assert schema["properties"]["byte_size"]["maximum"] == MCP_OUTPUT_CONTENT_MAX_BYTES
