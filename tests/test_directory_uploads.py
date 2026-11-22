import asyncio
import hashlib
import json
from base64 import urlsafe_b64encode
from collections import Counter
from uuid import UUID

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._directory_source import DirectorySnapshot
from synth_ai.errors import DataIntegrityError, DirectoryUploadCancelledError, DirectoryUploadError


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


@pytest.mark.parametrize(
    "fault", ["lost-finalize-reply", "missing-object", "identity-mismatch", "concurrent"]
)
async def test_directory_replay_refreshes_original_upload_before_put(contract, tmp_path, fault):
    contents = {"a.bin": b"first"}
    if fault != "concurrent":
        contents.update({"b.bin": b"\x00\xff", "c.bin": b"last"})
    files = {}
    ids = {}
    for index, (name, content) in enumerate(contents.items(), 1):
        (tmp_path / name).write_bytes(content)
        fragment = urlsafe_b64encode(UUID(int=index).bytes).decode().rstrip("=")
        ids[name] = "upl_" + fragment
        files[name] = {
            **manifest(contract, name, content),
            "file_id": "fil_" + fragment,
            "revision_id": "rev_" + fragment,
            "created_at": "2026-11-18T10:59:21.987464Z",
        }
    names = {upload_id: name for name, upload_id in ids.items()}
    prepares, objects, commits = {}, set(), set()
    put_attempts, reads = Counter(), Counter()
    inject_fault = True
    first_put_waiting = asyncio.Event()
    second_put_entered = asyncio.Event()

    def upload(name, *, committed, fresh_grant=False):
        body = {
            "upload_id": ids[name],
            "state": "committed" if committed else "pending",
            "file": files[name],
        }
        if not committed:
            body["put"] = {
                "url": f"https://storage.example.test/{name}?fresh={int(fresh_grant)}",
                "headers": {"if-none-match": "*"},
                "expires_at": "2099-01-01T00:00:00Z",
            }
        return body

    async def handle(request):
        nonlocal inject_fault
        if request.method == "PUT":
            name = request.url.path.removeprefix("/")
            put_attempts[name] += 1
            assert request.content == contents[name]
            assert request.headers["If-None-Match"] == "*"
            if name == "b.bin" and fault == "missing-object" and inject_fault:
                inject_fault = False
                raise httpx.ConnectError("transfer did not reach storage", request=request)
            if name == "a.bin" and fault == "concurrent":
                if put_attempts[name] == 1:
                    first_put_waiting.set()
                    await second_put_entered.wait()
                else:
                    second_put_entered.set()
            if name in objects:
                return httpx.Response(412)
            if reads[name]:
                assert request.url.params["fresh"] == "1"
            objects.add(name)
            return httpx.Response(200)
        if request.method == "GET":
            name = names[request.url.path.rsplit("/", 1)[1]]
            reads[name] += 1
            current = upload(name, committed=name in commits, fresh_grant=True)
            if fault == "identity-mismatch":
                current["file"] = files["c.bin"]
            return httpx.Response(200, json=current)
        if request.url.path == "/v1/uploads":
            name = json.loads(request.content)["name"]
            key = request.headers["Idempotency-Key"]
            replay = name in prepares
            assert prepares.setdefault(name, key) == key
            # Core replays the original 201, including its pending state and first grant.
            headers = {"Idempotent-Replayed": "true"} if replay else {}
            receipt = upload(name, committed=False)
            receipt["file"] = dict(files[name], created_at="2026-11-18T10:59:21.987464269Z")
            return httpx.Response(201, headers=headers, json=receipt)
        name = names[request.url.path.split("/")[-2]]
        assert request.headers["Idempotency-Key"] == prepares[name]
        assert name in objects
        commits.add(name)
        if (
            name == "b.bin"
            and fault in {"lost-finalize-reply", "identity-mismatch"}
            and inject_fault
        ):
            inject_fault = False
            return httpx.Response(
                409,
                json={
                    "code": "fixture_lost_finalize_reply",
                    "detail": "Owned post-commit response fault",
                    "status": 409,
                },
            )
        return httpx.Response(200, json=upload(name, committed=True))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://api.example.test", http_client=http, retries=0
        ) as client:

            async def upload_directory():
                return await client.swarms.upload_directory(str(tmp_path), idempotency_key="batch")

            if fault == "concurrent":
                async with asyncio.timeout(3), asyncio.TaskGroup() as tasks:
                    first = tasks.create_task(upload_directory())
                    await first_put_waiting.wait()
                    second = tasks.create_task(upload_directory())
                assert first.result() == second.result()
                assert [file.model_dump(mode="json") for file in first.result()] == list(
                    files.values()
                )
                assert put_attempts == {"a.bin": 2}
                assert commits == objects == set(contents)
                return

            with pytest.raises(DirectoryUploadError) as caught:
                await upload_directory()
            error = caught.value
            assert [file.file_id for file in error.completed_files] == [files["a.bin"]["file_id"]]
            assert error.name == "b.bin"
            assert error.file_idempotency_key == prepares["b.bin"]
            if fault == "identity-mismatch":
                puts_before_replay = put_attempts.copy()
                with pytest.raises(DirectoryUploadError) as mismatch:
                    await upload_directory()
                assert isinstance(mismatch.value.__cause__, DataIntegrityError)
                assert puts_before_replay == put_attempts
                return
            recovered = await upload_directory()
    assert [file.model_dump(mode="json") for file in recovered] == list(files.values())
    assert put_attempts == {"a.bin": 1, "b.bin": 1 + (fault == "missing-object"), "c.bin": 1}
    assert reads == {"a.bin": 1, "b.bin": 1}
    assert commits == objects == set(contents)


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
