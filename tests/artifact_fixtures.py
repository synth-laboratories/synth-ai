"""Offline Artifact/Swarms backend fixtures shared by the SDK-surface tests."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json

import httpx
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.types.refs import Ref
from tests.test_repository_client import REF, Server

BASE = "https://backend.test"
RUN_REF = Ref.run("run-1", "d" * 64)
API = "/api/v1"


def meta(contract="core.v1", operation_id=None):
    return {
        "contract": contract,
        "request_id": "fixture-server",
        "operation_id": operation_id,
        "observed_at": "2026-10-19T20:00:00Z",
    }


def artifact_body(
    artifact_id="art-1",
    revision=1,
    path="report.md",
    content=b"hello artifact\n",
    media="text/markdown",
    labels=None,
    extra=None,
):
    obj = {
        "logical_path": path,
        "digest_sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "media_type": media,
    }
    unsigned = {
        "schema_version": "synth.artifact-platform.v1",
        "manifest_schema_version": "1",
        "publication_id": "pub-1",
        "collection_id": "col-1",
        "revision": revision,
        "objects": [obj],
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "metadata": meta(),
        "artifact_id": artifact_id,
        "project_id": "project-test",
        "kind": "report",
        "labels": labels or {},
        "labels_revision": 1,
        "ref": Ref.artifact(artifact_id, str(revision), digest).model_dump(),
        "manifest": {**unsigned, "manifest_digest": digest},
        "aliases": [],
        "aliases_complete": True,
        "scan": "accepted",
        "created_at": "2026-10-19T19:00:00Z",
        **(extra or {}),
    }


def snapshot_body(**changes):
    body = {
        "metadata": meta("swarms.v1"),
        "run_id": "run-1",
        "project_id": "project-test",
        "ref": RUN_REF.model_dump(),
        "state": "completed",
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": True,
        "execution_closed": True,
        "accounting_final": True,
        "cleanup_complete": True,
    }
    body.update(changes)
    return body


def objective(identifier="oeq-1", answer="The answer.\n"):
    return {
        "id": identifier,
        "kind": "oeq",
        "title": "Question",
        "statement": "What is it?",
        "required": True,
        "criteria": ["be right"],
        "definition_revision": 1,
        "state": "verified",
        "answer": {"revision": 1, "text": answer, "sources": [], "by": "worker-1"},
        "future_field": {"ignored": True},
    }


def link(ref, relation="produced"):
    return {
        "link_id": "link-" + ref.record_id,
        "project_id": "project-test",
        "subject": RUN_REF.model_dump(),
        "relation": relation,
        "object": ref.model_dump(),
        "producer": {"authority": "swarms", "subject_id": "run-1"},
        "created_at": "2026-10-19T20:00:00Z",
    }


def page(items, contract="core.v1", **extra):
    return {
        "metadata": meta(contract),
        "items": items,
        "next_cursor": None,
        "complete": True,
        **extra,
    }


def ok(body, contract="core.v1"):
    return httpx.Response(200, headers={"X-Synth-Contract": contract}, json=body)


class Fake(Server):
    """Routes the new surfaces; everything else falls through to the repository server."""

    def __init__(self):
        super().__init__()
        self.snapshot = snapshot_body()
        self.links = []
        self.refs_pages = None
        self.content = b"hello artifact\n"
        self.artifact_revisions = {}
        self.upload_state = "received"
        self.label_calls = []
        self.run_list_items = [snapshot_body()]
        self.bodies = []

    def artifact(self, artifact_id="art-1", revision=1):
        return artifact_body(artifact_id, revision, content=self.content)

    def __call__(self, request):
        path = request.url.path.removeprefix(API)
        if request.content:
            self.bodies.append((request.method, path, json.loads(request.content)))
        if path == "/capabilities":
            return super().__call__(request)
        self.requests.append(request)
        if path == "/swarms/runs/run-1" and request.method == "GET":
            return ok(self.snapshot, "swarms.v1")
        if path == "/swarms/runs" and request.method == "GET":
            return ok(page(self.run_list_items, "swarms.v1"), "swarms.v1")
        if path == "/swarms/runs" and request.method == "POST":
            return ok(self.snapshot, "swarms.v1")
        if path == "/swarms/runs/run-1/labels":
            body = json.loads(request.content)
            self.label_calls.append(body)
            labels = {**self.snapshot.get("labels", {}), **body["set"]}
            for key in body["remove"]:
                labels.pop(key, None)
            self.snapshot = {
                **self.snapshot,
                "labels": labels,
                "labels_revision": body["expected_revision"] + 1,
            }
            return ok(self.snapshot, "swarms.v1")
        if path == "/links":
            return ok(page([link(ref) for ref in self.links]))
        if path.endswith("/refs") and self.refs_pages is not None:
            index = int(request.url.params.get("cursor", "0") or 0)
            items = self.refs_pages[index]
            more = index + 1 < len(self.refs_pages)
            body = page(items)
            body["complete"], body["next_cursor"] = not more, str(index + 1) if more else None
            return ok(body)
        if path == "/projects/project-test/artifacts" and request.method == "GET":
            return ok(page([self.artifact()]))
        if path == "/projects/project-test/artifacts/uploads":
            return ok(self.upload_status(self.upload_state))
        if path == "/artifacts/uploads/up-1":
            return ok(self.upload_status(self.upload_state))
        if path == "/artifacts/uploads/up-1/finalize":
            return ok(
                {
                    "metadata": meta(operation_id="finalize-1"),
                    "upload_id": "up-1",
                    "artifact": self.artifact(),
                    "state": "finalized",
                }
            )
        if path.endswith("/labels"):
            self.label_calls.append(json.loads(request.content))
            return ok(self.artifact())
        if "/aliases/" in path:
            ref = Ref.model_validate(self.artifact()["ref"])
            return ok(
                {
                    "metadata": meta(),
                    "project_id": "project-test",
                    "name": path.rsplit("/", 1)[1],
                    "ref": ref.model_dump(),
                    "revision": 1,
                }
            )
        if path.endswith("/content"):
            artifact = self.artifact(path.split("/")[4], int(path.split("/")[6]))
            return httpx.Response(
                200,
                stream=httpx.ByteStream(self.content),
                headers={
                    "X-Synth-Contract": "core.v1",
                    "X-Synth-Artifact-Id": artifact["artifact_id"],
                    "X-Synth-Artifact-Revision": artifact["ref"]["revision"],
                    "X-Synth-Artifact-Manifest-SHA256": artifact["ref"]["digest_sha256"],
                    "X-Synth-Content-SHA256": hashlib.sha256(self.content).hexdigest(),
                    "ETag": '"' + hashlib.sha256(self.content).hexdigest() + '"',
                    "Content-Type": "text/markdown",
                    "Content-Length": str(len(self.content)),
                },
            )
        if path.startswith("/projects/project-test/artifacts/") and request.method == "GET":
            return ok(self.artifact(path.rsplit("/", 1)[1]))
        self.requests.pop()
        return super().__call__(request)

    def upload_status(self, state):
        manifest = self.artifact()["manifest"]["objects"][0]
        return {
            "metadata": meta(),
            "upload_id": "up-1",
            "project_id": "project-test",
            "artifact_id": "art-1",
            "revision": 1,
            "state": state,
            "manifest": {
                "filename": "report.md",
                "byte_size": manifest["size_bytes"],
                "digest_sha256": manifest["digest_sha256"],
                "media_type": "text/markdown",
            },
            "kind": "file",
            "transfer_url": "https://backend.test/put" if state == "prepared" else None,
            "expires_at": "2099-01-01T00:00:00Z" if state == "prepared" else None,
        }


async def wait(value):
    return await value if inspect.isawaitable(value) else value


def drive(async_mode, handler, steps):
    """Run async `steps(client, wait)` against sync or async clients on a mock network."""

    async def main():
        if async_mode:
            async with AsyncSynthClient(api_key="fixture", base_url=BASE) as client:
                await client._transport.client.aclose()
                client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

                async def negotiated(*args):
                    return None

                client._transport._negotiate = negotiated
                return await steps(client, wait)
        with SynthClient(api_key="fixture", base_url=BASE) as client:
            client._transport.client.close()
            client._transport.client = httpx.Client(transport=httpx.MockTransport(handler))
            client._transport._negotiate = lambda *args: None
            return await steps(client, wait)

    return asyncio.run(main())


__all__ = ["REF", "Fake", "artifact_body", "drive", "link", "objective", "snapshot_body", "wait"]
