"""Public repository clients negotiate, bind exact identities and own cleanup."""

import asyncio
import hashlib
import json
import threading
import time

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts import core as operations
from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.errors import ContractError, TransportError
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.core.commit_digest import CommitDigestMismatch
from synth_ai.sdk.core.repo_archive import ArchiveTransfer
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from synth_ai.types.refs import Ref
from tests.http_fixtures import capabilities_body
from tests.test_repository_archive_decode import fixture_tar

pytestmark = pytest.mark.unit
META = {
    "contract": "core.v1",
    "request_id": "core-server",
    "operation_id": None,
    "observed_at": "2026-10-19T20:00:00Z",
}
RAW, DIGEST = fixture_tar(
    [
        ("nested/file", "100644", b"verified\n"),
        ("executable", "100755", b"#!/bin/sh\n"),
        ("link", "120000", b"nested/file"),
        ("submodule", "160000", b"a" * 40),
    ]
)
REF = Ref.commit("repo-test", "b" * 40, DIGEST)


class Server:
    def __init__(self, variant="ok"):
        self.variant = variant
        self.requests = []
        self.responses = []

    def __call__(self, request):
        if request.url.path.endswith("/capabilities"):
            body = capabilities_body()
            body["contracts"].append(
                {"contract": "core.v1", "schema_sha256": schema_sha256("core.v1")}
            )
            body["contracts"].append(
                {"contract": "swarms.v1", "schema_sha256": schema_sha256("swarms.v1")}
            )
            body["operations"].append(
                {
                    "operation_id": "swarms.get",
                    "contract": "swarms.v1",
                    "state": "enabled",
                    "surfaces": ["http"],
                    "reason": None,
                }
            )
            body["operations"].extend(
                {
                    "operation_id": op.id,
                    "contract": "core.v1",
                    "state": "enabled",
                    "surfaces": ["http"],
                    "reason": None,
                }
                for op in operations.ALL_OPERATIONS
            )
            body["contracts"].sort(key=lambda item: item["contract"])
            body["operations"].sort(key=lambda item: item["operation_id"])
            snapshot = {
                key: value
                for key, value in body.items()
                if key not in {"metadata", "capability_sha256"}
            }
            body["capability_sha256"] = snapshot_sha256(
                CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
            )
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=body)
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer fixture"
        if request.url.path.endswith("/swarms/runs/run-1"):
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "swarms.v1"},
                json={
                    "metadata": {**META, "contract": "swarms.v1"},
                    "run_id": "run-1",
                    "project_id": "project-test",
                    "ref": Ref.run(
                        "run-1", ("e" if self.variant == "run_changed" else "d") * 64
                    ).model_dump(),
                    "state": "completed",
                    "model": "gpt-6-luna",
                    "provider": "openai",
                    "result_available": True,
                    "execution_closed": True,
                    "accounting_final": True,
                    "cleanup_complete": True,
                },
            )
        if request.url.path.endswith("/archive"):
            headers = {
                "X-Synth-Contract": "core.v1",
                "X-Synth-Archive-Format": "synth.git-raw-tar.v1",
                "X-Synth-Repo-Id": REF.record_id,
                "X-Synth-Commit-SHA": REF.revision,
                "X-Synth-Commit-Manifest-SHA256": REF.digest_sha256,
                "X-Synth-Content-SHA256": hashlib.sha256(RAW).hexdigest(),
                "X-Synth-Archive-Entries": "4",
                "X-Synth-Archive-Bytes": str(len(RAW)),
                "Content-Type": "application/x-tar",
                "X-Request-Id": "archive-server",
            }
            raw = RAW
            if self.variant == "hash":
                raw = RAW.replace(b"verified", b"tampered")
            elif self.variant == "tree":
                raw = RAW.replace(b"verified", b"tampered")
                headers["X-Synth-Content-SHA256"] = hashlib.sha256(raw).hexdigest()
            elif self.variant == "identity":
                headers["X-Synth-Repo-Id"] = "another-repo"
            elif self.variant == "short":
                raw = RAW[:-512]
            elif self.variant == "infra":
                response = httpx.Response(
                    503, headers={"Content-Encoding": "gzip"}, stream=NeverRead()
                )
                self.responses.append(response)
                return response
            elif self.variant == "occupied":
                self.destination.mkdir()
            assert request.url.params["digest_sha256"] == REF.digest_sha256
            assert "If-Match" not in request.headers
            assert request.headers["Accept"] == "application/x-tar"
            response = httpx.Response(200, headers=headers, stream=httpx.ByteStream(raw))
            self.responses.append(response)
            return response
        if request.url.path.endswith("/refs"):
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "core.v1"},
                json={
                    "metadata": META,
                    "items": [
                        {
                            "name": "refs/heads/workers/a",
                            "commit_sha": REF.revision,
                            "ref": REF.model_dump(),
                        }
                    ],
                    "next_cursor": None,
                    "complete": True,
                },
            )
        if "/commits/" in request.url.path:
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "core.v1"},
                json={
                    "metadata": META,
                    "project_id": "project-test",
                    "repo_id": REF.record_id,
                    "ref": REF.model_dump(),
                    "sha": REF.revision,
                    "tree_sha": "c" * 40,
                    "parent_shas": [],
                    "author_name": "Fixture",
                    "author_email": "fixture@example.test",
                    "authored_at": "2026-10-19T19:00:00Z",
                    "summary": "fixture",
                    "manifest_digest_sha256": DIGEST,
                    "entries": 4,
                    "bytes": 40,
                },
            )
        if request.url.path.endswith("/diff"):
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "core.v1"},
                json={
                    "metadata": META,
                    "project_id": "project-test",
                    "repo_id": REF.record_id,
                    "base": REF.model_dump(),
                    "head": REF.model_dump(),
                    "files": [],
                    "patch": "",
                    "truncated": False,
                },
            )
        if request.url.path.endswith("/clone-tokens"):
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "core.v1"},
                json={
                    "metadata": {**META, "operation_id": "mint-1"},
                    "project_id": "project-test",
                    "repo_id": REF.record_id,
                    "clone_url": "https://git.example.test/project.git",
                    "auth_header": "Bearer secret-fixture-capability",
                    "expires_at": "2026-10-19T20:05:00Z",
                    "scope": "project",
                },
            )
        if request.url.path.endswith("/links"):
            endpoint = Ref.run("run-1", "d" * 64)
            if self.variant == "link_changed":
                endpoint = Ref.run("run-1", "e" * 64)
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "core.v1"},
                json={
                    "metadata": META,
                    "items": [
                        {
                            "link_id": "link-1",
                            "project_id": "project-test",
                            "subject": endpoint.model_dump(),
                            "relation": "produced",
                            "object": REF.model_dump(),
                            "producer": {"authority": "swarms", "subject_id": "run-1"},
                            "created_at": "2026-10-19T20:00:00Z",
                        }
                    ],
                    "next_cursor": None,
                    "complete": True,
                },
            )
        return httpx.Response(
            200,
            headers={"X-Synth-Contract": "core.v1"},
            json={
                "metadata": META,
                "project_id": "project-test",
                "repo_id": REF.record_id,
                "default_branch": "main",
                "ref": REF.model_dump(),
            },
        )


class NeverRead(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self):
        raise AssertionError("Infrastructure bodies must be classified before decompression")
        yield b""

    async def __aiter__(self):
        raise AssertionError("Infrastructure bodies must be classified before decompression")
        yield b""


async def invoke(async_mode, server, destination):
    if async_mode:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(server), headers={"If-Match": "borrowed"}
        ) as network:
            async with AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client:
                repo = await client.repos.get("project-test")
                assert (await repo.refs(prefix="refs/heads/workers/")).items[0].ref == REF
                result = await repo.archive(REF, destination)
                assert not client._transport._streams and not client._transport._responses
            assert not network.is_closed
            return result
    with httpx.Client(
        transport=httpx.MockTransport(server), headers={"If-Match": "borrowed"}
    ) as network:
        with SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client:
            repo = client.repos.get("project-test")
            assert repo.refs(prefix="refs/heads/workers/").items[0].ref == REF
            result = repo.archive(REF, destination)
            assert not client._transport._streams and not client._transport._responses
        assert not network.is_closed
        return result


@pytest.mark.parametrize("async_mode", [False, True])
def test_public_archive_negotiation_exact_tree_and_no_overwrite(tmp_path, async_mode):
    server = Server()
    destination = tmp_path / "checkout"
    receipt = asyncio.run(invoke(async_mode, server, destination))
    assert receipt.ref == REF and receipt.entries == 4 and receipt.byte_size == len(RAW)
    assert (
        receipt.destination == destination
        and receipt.content_sha256 == hashlib.sha256(RAW).hexdigest()
    )
    assert (destination / "link").read_bytes() == b"verified\n"
    assert (destination / "submodule").is_dir()
    assert not list(tmp_path.glob(".synth-repo-*"))
    assert all(response.is_closed for response in server.responses)
    assert len([r for r in server.requests if r.url.path.endswith("/archive")]) == 1


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "variant,kind",
    [
        ("hash", ContractError),
        ("tree", CommitDigestMismatch),
        ("identity", ContractError),
        ("short", ContractError),
        ("infra", TransportError),
        ("occupied", FileExistsError),
    ],
)
def test_archive_failure_cleans_all_private_work_and_never_publishes(
    tmp_path, async_mode, variant, kind
):
    server = Server(variant)
    server.destination = destination = tmp_path / "checkout"
    with pytest.raises(kind) as caught:
        asyncio.run(invoke(async_mode, server, destination))
    assert destination.exists() == (variant == "occupied")
    assert not list(tmp_path.glob(".synth-repo-*"))
    assert all(response.is_closed for response in server.responses)
    if isinstance(caught.value, ContractError):
        assert (
            caught.value.operation == "repos.archive"
            and caught.value.request_id == "archive-server"
        )
    assert len([r for r in server.requests if r.url.path.endswith("/archive")]) == 1


def test_existing_destination_refuses_before_any_descriptor_or_archive_request(tmp_path):
    server = Server()
    destination = tmp_path / "checkout"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        asyncio.run(invoke(False, server, destination))
    assert not any(
        "/commits/" in r.url.path or r.url.path.endswith("/archive") for r in server.requests
    )


def test_async_cancel_joins_extraction_and_cleans_scratch(tmp_path, monkeypatch):
    started, stopped = threading.Event(), threading.Event()
    original = ArchiveTransfer.extract

    def slow(self, archive):
        started.set()
        try:
            while not self.stop.wait(0.005):
                self.remaining()
            self.remaining()
            return original(self, archive)
        finally:
            stopped.set()

    monkeypatch.setattr(ArchiveTransfer, "extract", slow)

    async def run():
        task = asyncio.create_task(invoke(True, Server(), tmp_path / "checkout"))
        deadline = time.monotonic() + 2
        while not started.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.001)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(run())
    assert not (tmp_path / "checkout").exists() and not list(tmp_path.glob(".synth-repo-*"))


@pytest.mark.parametrize("async_mode", [False, True])
def test_repository_metadata_diff_clone_mutation_and_exact_links(async_mode):
    server = Server()

    async def run():
        if async_mode:
            async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as network:
                async with AsyncSynthClient(
                    api_key="fixture",
                    base_url="https://backend.test",
                    options=ClientOptions(async_http_client=network),
                ) as client:
                    repo = await client.repos.get("project-test")
                    assert (await repo.commit(REF.revision)).ref == REF
                    assert (await repo.diff(REF, REF)).head == REF
                    token = await repo.clone_url(
                        mutation=MutationOptions(idempotency_key="mint-key")
                    )
                    links = await client.links.list(
                        "project-test", subject=Ref.run("run-1", "d" * 64), relation="produced"
                    )
        else:
            with httpx.Client(transport=httpx.MockTransport(server)) as network:
                with SynthClient(
                    api_key="fixture",
                    base_url="https://backend.test",
                    options=ClientOptions(http_client=network),
                ) as client:
                    repo = client.repos.get("project-test")
                    assert repo.commit(REF.revision).ref == REF
                    assert repo.diff(REF, REF).head == REF
                    token = repo.clone_url(mutation=MutationOptions(idempotency_key="mint-key"))
                    links = client.links.list(
                        "project-test", subject=Ref.run("run-1", "d" * 64), relation="produced"
                    )
        assert token.auth_header.startswith("Bearer ") and "secret-fixture" not in repr(token)
        assert links.items[0].object == REF

    asyncio.run(run())
    mint = next(request for request in server.requests if request.method == "POST")
    assert mint.headers["Idempotency-Key"] == "mint-key"
    assert json.loads(mint.content) == {"ttl_seconds": 300}
    link_query = server.requests[-1].url.params
    assert (
        link_query["subject"] == "swarms:run:run-1:1"
        and link_query["subject_digest_sha256"] == "d" * 64
    )


def test_links_refuses_digest_substitution_and_bounded_queries_before_network():
    server = Server("link_changed")
    with httpx.Client(transport=httpx.MockTransport(server)) as network:
        with SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client:
            with pytest.raises(ContractError):
                client.links.list(
                    "project-test", subject=Ref.run("run-1", "d" * 64), relation="produced"
                )
            count = len(server.requests)
            for options in ({}, {"subject": REF, "object": REF}, {"subject": REF, "limit": 201}):
                with pytest.raises(ValueError):
                    client.links.list("project-test", **options)
            assert len(server.requests) == count


def test_sync_client_close_waits_for_extraction_before_returning(tmp_path, monkeypatch):
    started, stopped = threading.Event(), threading.Event()
    errors = []

    def slow(self, archive):
        started.set()
        try:
            while not self.stop.wait(0.005):
                self.remaining()
            self.remaining()
        finally:
            stopped.set()

    monkeypatch.setattr(ArchiveTransfer, "extract", slow)
    with httpx.Client(transport=httpx.MockTransport(Server())) as network:
        client = SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        )
        repo = client.repos.get("project-test")

        def read():
            try:
                repo.archive(REF, tmp_path / "checkout")
            except Exception as error:
                errors.append(error)

        reader = threading.Thread(target=read)
        reader.start()
        assert started.wait(2)
        client.close()
        assert stopped.is_set()
        reader.join(2)
        assert not reader.is_alive() and len(errors) == 1 and isinstance(errors[0], TransportError)
        assert not client._transport._streams and not client._transport._responses
    assert not (tmp_path / "checkout").exists() and not list(tmp_path.glob(".synth-repo-*"))


@pytest.mark.parametrize("async_mode", [False, True])
def test_run_get_retains_exact_launch_ref_and_provenance_refuses_identity_change(async_mode):
    server = Server()

    async def run():
        if async_mode:
            async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as network:
                async with AsyncSynthClient(
                    api_key="fixture",
                    base_url="https://backend.test",
                    options=ClientOptions(async_http_client=network),
                ) as client:
                    handle = await client.swarms.get("run-1")
                    assert handle.ref == Ref.run("run-1", "d" * 64)
                    assert (await handle.produced()).items[0].object == REF
                    server.variant = "run_changed"
                    with pytest.raises(ContractError):
                        await handle.status()
        else:
            with httpx.Client(transport=httpx.MockTransport(server)) as network:
                with SynthClient(
                    api_key="fixture",
                    base_url="https://backend.test",
                    options=ClientOptions(http_client=network),
                ) as client:
                    handle = client.swarms.get("run-1")
                    assert handle.ref == Ref.run("run-1", "d" * 64)
                    assert handle.produced().items[0].object == REF
                    server.variant = "run_changed"
                    with pytest.raises(ContractError):
                        handle.status()
        assert handle.ref == Ref.run("run-1", "d" * 64)

    asyncio.run(run())
