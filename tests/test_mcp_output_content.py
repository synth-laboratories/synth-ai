import base64
import hashlib
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai.errors import ResponseContractError
from synth_ai.mcp.output_content import OUTPUT_BYTES_MAX
from synth_ai.mcp.server import call_tool


@pytest.mark.parametrize(
    "body",
    [
        b"# Result\nUnicode: \xcf\x80\n",
        b"\x00\xff\x80",
        b"",
        b'{"json":true}',
        b"x" * OUTPUT_BYTES_MAX,
    ],
)
async def test_mcp_output_round_trips_exact_bytes_and_pinned_reference(body):
    digest = hashlib.sha256(body).hexdigest()

    def handle(request):
        assert request.headers["Authorization"] == "Bearer fixture"
        assert dict(request.url.params) == {"revision_id": "revision", "digest_sha256": digest}
        return httpx.Response(
            200, content=body, headers={"Content-Type": "application/octet-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            result = json.loads(
                await call_tool(
                    client,
                    "swarms_run_output_content",
                    {
                        "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
                        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
                        "revision_id": "revision",
                        "digest_sha256": digest,
                    },
                )
            )
    assert result == {
        "encoding": "base64",
        "data": base64.b64encode(body).decode("ascii"),
        "content_type": "application/octet-stream",
        "revision_id": "revision",
        "digest_sha256": digest,
    }


@pytest.mark.parametrize("failure", ["digest", "size"])
async def test_invalid_or_oversized_content_is_a_typed_mcp_error(failure):
    body = b"x" * (OUTPUT_BYTES_MAX + 1) if failure == "size" else b"actual"
    digest = hashlib.sha256(body if failure == "size" else b"expected").hexdigest()

    def handle(request):
        return httpx.Response(
            200, content=body, headers={"Content-Type": "application/octet-stream"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(ResponseContractError) as error:
                await call_tool(
                    client,
                    "swarms_run_output_content",
                    {
                        "run_id": "run_AAAAAAAAAAAAAAAAAAAAAA",
                        "output_id": "out_AAAAAAAAAAAAAAAAAAAAAA",
                        "revision_id": "revision",
                        "digest_sha256": digest,
                    },
                )
    assert error.value.problem.code == (
        "mcp_response_too_large" if failure == "size" else "mcp_output_content_invalid"
    )
