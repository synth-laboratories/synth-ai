import asyncio
import hashlib
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._directory_source import DirectorySnapshot
from synth_ai.errors import DirectoryUploadCancelledError, DirectoryUploadError


def manifest(contract, name, content):
    result = dict(
        contract["paths"]["/v1/files"]["post"]["responses"]["201"]["content"]["application/json"][
            "example"
        ]
    )
    result.update(
        name=name,
        media_type="text/markdown" if name.endswith(".md") else "application/octet-stream",
        size_bytes=len(content),
        digest_sha256=hashlib.sha256(content).hexdigest(),
    )
    return result


async def test_relative_binary_upload_snapshot_and_stable_per_file_keys(contract, tmp_path):
    (tmp_path / "nested").mkdir()
    contents = {"a.md": b"# Task", "nested/policy.bin": b"\0\xff\x80"}
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
    prepared, uploaded, keys = {}, [], {}

    def handle(request):
        if request.method == "PUT":
            assert "Authorization" not in request.headers
            assert "Cookie" not in request.headers
            name = "a.md" if request.url.path == "/a" else "nested/policy.bin"
            assert request.content == contents[name]
            uploaded.append(name)
            return httpx.Response(200)
        key = request.headers["Idempotency-Key"]
        if request.url.path == "/v1/uploads":
            body = json.loads(request.content)
            name = body["name"]
            assert body["digest_sha256"] == hashlib.sha256(contents[name]).hexdigest()
            assert body["size_bytes"] == len(contents[name])
            assert body["media_type"] == manifest(contract, name, contents[name])["media_type"]
            if name in keys:
                assert keys[name] == key
            keys[name] = key
            committed = name in prepared
            file = manifest(contract, name, contents[name])
            prepared[name] = file
            # Both originals can change after the snapshot without changing submitted bytes.
            for path in contents:
                (tmp_path / path).write_bytes(b"changed")
            response = {
                "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                "state": "committed" if committed else "pending",
                "file": file,
            }
            if not committed:
                response["put"] = {
                    "url": "https://storage.example.test/" + ("a" if name == "a.md" else "b"),
                    "headers": {},
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            return httpx.Response(201, json=response)
        assert request.url.path.endswith("/finalize")
        name = next(name for name, value in keys.items() if value == key)
        return httpx.Response(
            200,
            json={
                "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                "state": "committed",
                "file": prepared[name],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            first = await client.swarms.upload_directory(str(tmp_path), idempotency_key="batch-one")
            for name, data in contents.items():
                (tmp_path / name).write_bytes(data)
            second = await client.swarms.upload_directory(
                str(tmp_path), idempotency_key="batch-one"
            )
    assert [file.name for file in first] == ["a.md", "nested/policy.bin"]
    assert first == second
    assert uploaded == ["a.md", "nested/policy.bin"]
    assert len(set(keys.values())) == 2


async def test_directory_failure_retains_committed_files_and_identity(contract, tmp_path):
    (tmp_path / "a.md").write_bytes(b"done")
    (tmp_path / "b.bin").write_bytes(b"pending")

    def handle(request):
        body = json.loads(request.content)
        if body["name"] == "a.md":
            return httpx.Response(
                201,
                json={
                    "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                    "state": "committed",
                    "file": manifest(contract, "a.md", b"done"),
                },
            )
        return httpx.Response(503, json={"message": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http, retries=0
        ) as client:
            with pytest.raises(DirectoryUploadError) as caught:
                await client.swarms.upload_directory(
                    str(tmp_path), idempotency_key="retained-batch"
                )
    error = caught.value
    assert error.name == "b.bin"
    assert [file.name for file in error.completed_files] == ["a.md"]
    assert error.idempotency_key == "retained-batch"
    assert (
        error.file_idempotency_key
        == hashlib.sha256(b"synth-directory.v1\0retained-batch\0b.bin").hexdigest()
    )
    assert error.__cause__ is not None


async def test_entire_directory_is_validated_before_api_effects(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"valid")
    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"not-authorized-by-directory")
    (tmp_path / "z-link").symlink_to(outside)
    calls = []

    def handle(request):
        calls.append(request)
        pytest.fail("invalid directory must not submit an upload")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            with pytest.raises(ValueError):
                await client.swarms.upload_directory(str(tmp_path), idempotency_key="invalid")
    assert calls == []


async def test_cancellation_during_api_call_closes_snapshot(tmp_path, monkeypatch):
    (tmp_path / "task.md").write_bytes(b"task")
    snapshots = []
    original = DirectorySnapshot.prepare

    def prepare(path):
        snapshot = original(path)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(DirectorySnapshot, "prepare", prepare)
    ready = asyncio.Event()

    async def handle(request):
        assert request.url.path == "/v1/uploads"
        ready.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            task = asyncio.create_task(
                client.swarms.upload_directory(str(tmp_path), idempotency_key="cancelled-batch")
            )
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert isinstance(caught.value, DirectoryUploadCancelledError)
            assert caught.value.idempotency_key == "cancelled-batch"
            assert caught.value.name == "task.md"
            assert caught.value.completed_files == ()
    assert snapshots[0].stream.closed


async def test_cancellation_retains_previously_committed_file(contract, tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\0\xff")
    (tmp_path / "b.bin").write_bytes(b"pending")

    async def handle(request):
        body = json.loads(request.content)
        if body["name"] == "b.bin":
            raise asyncio.CancelledError()
        return httpx.Response(
            201,
            json={
                "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                "state": "committed",
                "file": manifest(contract, "a.bin", b"\0\xff"),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            with pytest.raises(DirectoryUploadCancelledError) as caught:
                await client.swarms.upload_directory(str(tmp_path), idempotency_key="cancel-batch")
    assert [file.name for file in caught.value.completed_files] == ["a.bin"]
    assert caught.value.name == "b.bin"
    assert caught.value.idempotency_key == "cancel-batch"
    assert not isinstance(caught.value, Exception)


async def test_repeated_cancel_during_snapshot_close_retains_partial_custody(
    contract, tmp_path, monkeypatch
):
    import threading
    from dataclasses import replace

    (tmp_path / "a.bin").write_bytes(b"done")
    (tmp_path / "b.bin").write_bytes(b"pending")
    closing, release, pending = threading.Event(), threading.Event(), asyncio.Event()
    original = DirectorySnapshot.prepare

    class SlowClose:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            closing.set()
            release.wait(timeout=5)
            self.stream.close()

    def prepare(path):
        snapshot = original(path)
        return replace(snapshot, stream=SlowClose(snapshot.stream))

    monkeypatch.setattr(DirectorySnapshot, "prepare", prepare)

    async def handle(request):
        if json.loads(request.content)["name"] == "a.bin":
            return httpx.Response(
                201,
                json={
                    "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                    "state": "committed",
                    "file": manifest(contract, "a.bin", b"done"),
                },
            )
        pending.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http
        ) as client:
            task = asyncio.create_task(
                client.swarms.upload_directory(str(tmp_path), idempotency_key="double-cancel")
            )
            await asyncio.wait_for(pending.wait(), timeout=5)
            task.cancel()
            assert await asyncio.to_thread(closing.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(DirectoryUploadCancelledError) as caught:
                await task
    assert [file.name for file in caught.value.completed_files] == ["a.bin"]
    assert caught.value.name == "b.bin"
    assert caught.value.idempotency_key == "double-cancel"
