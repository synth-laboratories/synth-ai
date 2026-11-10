import hashlib
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai.errors import InvalidRequestError, UploadFormatUnsupportedError
from synth_ai.mcp.server import call_tool


async def test_upload_finalize_and_launch_keep_identity_and_custody(contract, tmp_path):
    upload_id = "upl_AAAAAAAAAAAAAAAAAAAAAA"
    content = b"# Task\nReturn done.\n"
    filename = tmp_path / "task.md"
    filename.write_bytes(content)
    manifest = dict(
        contract["paths"]["/v1/files"]["post"]["responses"]["201"]["content"]["application/json"][
            "example"
        ]
    )
    manifest.update(size_bytes=len(content), digest_sha256=hashlib.sha256(content).hexdigest())
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "PUT":
            assert request.url.host == "storage.example.test"
            assert "Authorization" not in request.headers
            assert "Cookie" not in request.headers
            assert request.content == content
            return httpx.Response(200)
        assert request.headers["Authorization"] == "Bearer fixture"
        if request.url.path.endswith("/uploads"):
            body = json.loads(request.content)
            assert body["media_type"] == "text/markdown"
            assert body["digest_sha256"] == manifest["digest_sha256"]
            return httpx.Response(
                201,
                json={
                    "upload_id": upload_id,
                    "state": "pending",
                    "file": manifest,
                    "put": {
                        "url": "https://storage.example.test/object?signature=opaque",
                        "headers": {
                            "content-type": "text/markdown",
                            "content-length": str(len(content)),
                        },
                        "expires_at": "2026-11-06T00:00:00Z",
                    },
                },
            )
        if request.url.path.endswith("/finalize"):
            assert request.headers["Idempotency-Key"] == "upload-one"
            return httpx.Response(
                200, json={"upload_id": upload_id, "state": "committed", "file": manifest}
            )
        assert request.url.path == "/v1/runs"
        assert json.loads(request.content)["inputs"]["files"] == [
            {key: manifest[key] for key in ("file_id", "revision_id", "digest_sha256")}
        ]
        snapshot = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
        return httpx.Response(202, json=snapshot)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        headers={"Authorization": "must-not-leak", "Cookie": "must-not-leak"},
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            file = await client.swarms.upload_file(str(filename), idempotency_key="upload-one")
            reference = {
                key: getattr(file, key) for key in ("file_id", "revision_id", "digest_sha256")
            }
            await client.swarms.launch("Task", budget="1.00", inputs={"files": [reference]})
    assert len(calls) == 4


async def test_mcp_upload_text_uses_public_operation_and_formats(contract):
    file = contract["paths"]["/v1/files"]["post"]["responses"]["201"]["content"][
        "application/json"
    ]["example"]

    def handle(request):
        assert request.url.path == "/v1/files"
        assert json.loads(request.content) == {
            "name": "task.md",
            "media_type": "text/markdown",
            "content": "# Task",
        }
        return httpx.Response(201, json=file)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            result = await call_tool(
                client,
                "swarms_upload_file",
                {
                    "idempotency_key": "text-one",
                    "body": {"name": "task.md", "media_type": "text/markdown", "content": "# Task"},
                },
            )
            assert json.loads(result)["digest_sha256"] == file["digest_sha256"]
            with pytest.raises(UploadFormatUnsupportedError) as caught:
                await client.swarms.upload_file("task.pdf")
            assert caught.value.code == "upload_format_unsupported"
            assert caught.value.status == 422
            assert caught.value.supported_formats == (
                "text/plain",
                "text/markdown",
                "application/json",
            )
            with pytest.raises(UploadFormatUnsupportedError):
                await client.swarms.upload_text("x", name="a.pdf", media_type="application/pdf")
            with pytest.raises(UploadFormatUnsupportedError):
                await client.swarms.prepare_upload(
                    name="a.pdf", media_type="application/pdf", size_bytes=1, digest_sha256="0" * 64
                )
            with pytest.raises(UploadFormatUnsupportedError) as tool:
                await call_tool(
                    client,
                    "swarms_upload_file",
                    {
                        "idempotency_key": "pdf",
                        "body": {"name": "a.pdf", "media_type": "application/pdf", "content": "x"},
                    },
                )
            assert tool.value.problem.code == "upload_format_unsupported"
            with pytest.raises(InvalidRequestError) as bad:
                await call_tool(
                    client,
                    "swarms_upload_file",
                    {
                        "idempotency_key": "",
                        "body": {"name": "a.md", "media_type": "text/markdown", "content": "x"},
                    },
                )
            assert bad.value.problem.code == "tool_arguments_invalid"
