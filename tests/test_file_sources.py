import asyncio
import hashlib
import threading

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._file_source import CHUNK_BYTES, FileIO, FileSource, open_source
from synth_ai.errors import DataIntegrityError, UploadTransferError


async def test_source_streams_bounded_chunks_and_detects_replacement(tmp_path):
    path = tmp_path / "task.txt"
    content = b"a" * (CHUNK_BYTES * 3 + 1)
    path.write_bytes(content)
    io = FileIO()
    async with open_source(path, io) as source:
        chunks = [chunk async for chunk in source.chunks(io)]
        assert max(map(len, chunks)) <= CHUNK_BYTES
        assert b"".join(chunks) == content
        assert source.digest_sha256 == hashlib.sha256(content).hexdigest()
        replacement = tmp_path / "replacement"
        replacement.write_bytes(content)
        replacement.replace(path)
        with pytest.raises(DataIntegrityError):
            source.verify()
    assert source.stream.closed


async def test_cancellation_joins_disk_worker_before_returning():
    io = FileIO()
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait(2)
        return 1

    task = asyncio.create_task(io.run(work))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_source_changed_after_prepare_never_finalizes_upload(contract, tmp_path, monkeypatch):
    path = tmp_path / "task.txt"
    path.write_bytes(b"original")
    sources = []
    original = FileSource.prepare

    def prepare(path):
        source = original(path)
        sources.append(source)
        return source

    monkeypatch.setattr(FileSource, "prepare", prepare)
    file = dict(
        contract["paths"]["/v1/files"]["post"]["responses"]["201"]["content"]["application/json"][
            "example"
        ]
    )
    file.update(size_bytes=8, digest_sha256=hashlib.sha256(b"original").hexdigest())
    calls = []

    def handle(request):
        calls.append(request.url.path)
        path.write_bytes(b"different")
        return httpx.Response(
            201,
            json={
                "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                "state": "pending",
                "file": file,
                "put": {
                    "url": "https://storage.example.test/object",
                    "headers": {},
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "key", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(UploadTransferError) as caught:
                await api.swarms.upload_file(str(path), idempotency_key="same-upload")
    assert calls == ["/v1/uploads"]
    assert caught.value.idempotency_key == "same-upload"
    assert isinstance(caught.value.__cause__, DataIntegrityError)
    assert sources[0].stream.closed


async def test_cancellation_closes_source_and_does_not_finalize(contract, tmp_path, monkeypatch):
    path = tmp_path / "task.txt"
    path.write_bytes(b"original")
    sources = []
    original = FileSource.prepare

    def prepare(path):
        source = original(path)
        sources.append(source)
        return source

    monkeypatch.setattr(FileSource, "prepare", prepare)
    ready = asyncio.Event()

    async def handle(request):
        ready.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "key", base_url="https://example.test", http_client=http
        ) as api:
            task = asyncio.create_task(api.swarms.upload_file(str(path)))
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert len(sources) == 1 and sources[0].stream.closed
