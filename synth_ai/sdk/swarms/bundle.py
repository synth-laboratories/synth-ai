"""Atomic, digest-verified export of a terminal run's produced resources."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from pydantic import Field, StrictInt

from synth_ai.core.http.blocking import CancellationScope, CLEANUP_SECONDS
from synth_ai.core.http.lifetime import await_cleanup
from synth_ai.errors import ContractError, RequestTimeoutError, ResultNotReadyError, TransportError
from synth_ai.options import ReadOptions
from synth_ai.sdk.core.artifact_download import ArtifactDownloadOptions
from synth_ai.sdk.core.artifacts import Artifacts, AsyncArtifacts
from synth_ai.sdk.core.atomic import publish_directory
from synth_ai.sdk.core.repo_archive import archive_destination, remove_owned_tree
from synth_ai.sdk.core.repos import Repositories, AsyncRepositories
from synth_ai.sdk.core.types import RepositoryArchiveOptions
from synth_ai.types.base import Model
from synth_ai.types.refs import Ref

TERMINAL = frozenset({"succeeded", "completed", "failed", "cancelled", "partial"})


class RunDownloadOptions(Model):
    timeout_seconds: float = Field(default=900, gt=0, le=900)
    maximum_bytes: StrictInt = Field(default=512 * 1024 * 1024, ge=1, le=2 * 1024 * 1024 * 1024)
    maximum_artifacts: StrictInt = Field(default=64, ge=1, le=64)
    maximum_links: StrictInt = Field(default=10000, ge=1, le=10000)


class RunDownloadReceipt(Model):
    run: Ref
    destination: Path
    final_commit: Ref | None = None
    artifacts: tuple[Ref, ...]
    byte_size: StrictInt = Field(ge=0)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BundleState:
    def __init__(self, destination, options):
        self.destination = archive_destination(destination)
        self.options = options
        self.deadline = time.monotonic() + options.timeout_seconds
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.scratch = None
        self.used = 0

    def remaining(self):
        if self.stop.is_set():
            raise TransportError(
                "Run download was closed", code="run_download_cancelled", stage="run_download"
            )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError(
                "Run download deadline expired", code="run_download_timeout", stage="run_download"
            )
        return remaining

    def read(self):
        return ReadOptions(timeout_seconds=self.remaining(), read_retry_count=0)

    def bytes_left(self):
        self.remaining()
        remaining = self.options.maximum_bytes - self.used
        if remaining < 1:
            raise ContractError(
                "Run download exceeded its total byte bound",
                code="run_download_bound",
                stage="run_download",
            )
        return remaining

    def prepare(self):
        self.remaining()
        self.scratch = Path(tempfile.mkdtemp(prefix=".synth-run-", dir=self.destination.parent))
        return self.scratch

    def publish(self, run, commit, artifacts, links, snapshot):
        import hashlib

        from synth_ai.sdk.core.archive_decode import safe_path

        objectives = {
            "objectives": [item.model_dump(mode="json") for item in snapshot.objectives],
            "guidance_coverage": [
                item.model_dump(mode="json") for item in snapshot.guidance_coverage
            ],
        }
        objective_bytes = (
            json.dumps(objectives, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        with (self.scratch / "objectives.json").open("xb") as output:
            output.write(objective_bytes)
            output.flush()
            os.fsync(output.fileno())
        for objective in snapshot.objectives:
            if objective.answer is None:
                continue
            parts = safe_path(objective.id + ".md")
            if len(parts) != 1:
                raise ContractError(
                    "Objective answer filename is not a single component",
                    code="objective_path_invalid",
                )
            target = self.scratch / "answers" / parts[0]
            target.parent.mkdir(exist_ok=True)
            with target.open("xb") as output:
                output.write(objective.answer.text.encode())
                output.flush()
                os.fsync(output.fileno())
        document = {
            "schema_version": "synth.run-bundle.v1",
            "run": run.model_dump(mode="json"),
            "produced": [link.model_dump(mode="json") for link in links],
            "artifacts": [reference.model_dump(mode="json") for reference in artifacts],
        }
        if commit is not None:
            document["final_commit"] = commit.model_dump(mode="json")
        raw = (json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
        with (self.scratch / "MANIFEST.json").open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        receipt = RunDownloadReceipt(
            run=run,
            destination=self.destination,
            final_commit=commit,
            artifacts=tuple(artifacts),
            byte_size=self.used,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
        with self.lock:
            self.remaining()
            publish_directory(self.scratch, self.destination)
            self.scratch = None
        return receipt

    def cleanup(self):
        if self.scratch is not None:
            remove_owned_tree(self.scratch)
            self.scratch = None


class SyncBundle(BundleState):
    def __init__(self, *args):
        super().__init__(*args)
        self.scope = CancellationScope()
        self.reader = threading.get_ident()

    def close(self):
        with self.lock:
            self.stop.set()
        self.scope.cancel()
        if threading.get_ident() != self.reader and not self.finished.wait(4 * CLEANUP_SECONDS):
            raise TransportError(
                "Run download cleanup timed out", code="run_download_cleanup_timeout"
            )


class AsyncBundle(BundleState):
    def __init__(self, *args):
        super().__init__(*args)
        self.owner = asyncio.current_task()

    async def aclose(self):
        with self.lock:
            self.stop.set()
        if (
            self.owner is not None
            and self.owner != asyncio.current_task()
            and not self.owner.done()
        ):
            self.owner.cancel()

            async def joined():
                try:
                    await self.owner
                except asyncio.CancelledError:
                    pass

            await await_cleanup(asyncio.create_task(joined()))


def collect(page, links, seen, options):
    if len(links) + len(page.items) > options.maximum_links:
        raise ContractError("Run provenance exceeded the selected bound", code="run_links_bound")
    links.extend(page.items)
    if page.complete:
        return None
    cursor = page.next_cursor
    if cursor is None or cursor in seen:
        raise ContractError(
            "Run provenance cursor did not advance", code="run_links_cursor_invalid"
        )
    seen.add(cursor)
    return cursor


def selected_artifacts(links, options):
    selected = {}
    for link in links:
        ref = link.object
        if ref.authority == "artifacts" and ref.kind == "artifact":
            key = (ref.record_id, ref.revision)
            if key in selected and selected[key] != ref:
                raise ContractError(
                    "Run produced conflicting artifact identities", code="run_artifact_conflict"
                )
            selected[key] = ref
    if len(selected) > options.maximum_artifacts:
        raise ContractError("Run artifacts exceeded the selected bound", code="run_artifacts_bound")
    return tuple(selected[key] for key in sorted(selected))


def select_commit(page, run, links):
    name = "refs/heads/worker/run-" + run.run_id
    tips = [item.ref for item in page.items if item.name == name]
    if len(tips) != 1:
        raise ResultNotReadyError(
            "Final run branch is not available", code="run_final_commit_unavailable"
        )
    tip = tips[0]
    if not any(link.object == tip for link in links):
        raise ResultNotReadyError(
            "Final run commit is awaiting provenance reconciliation",
            code="run_final_commit_pending",
        )
    return tip


def download(run, destination, options=None):
    state = SyncBundle(destination, options or RunDownloadOptions())
    transport = run._transport
    transport._register_stream(state)
    try:
        with state.scope.activate():
            snapshot = run.status(request=state.read())
            if snapshot.state not in TERMINAL:
                raise ResultNotReadyError(
                    "Run download requires a terminal run", code="run_not_terminal"
                )
            links, seen, cursor = [], set(), None
            while True:
                cursor = collect(
                    run.produced(cursor=cursor, request=state.read()), links, seen, state.options
                )
                if cursor is None:
                    break
            artifacts = selected_artifacts(links, state.options)
            repos = None
            commit = None
            if any(
                link.object.authority == "git" and link.object.kind == "commit" for link in links
            ):
                repos = Repositories(transport).get(snapshot.project_id, request=state.read())
                commit = select_commit(
                    repos.refs(
                        prefix="refs/heads/worker/run-" + run.run_id,
                        limit=1,
                        request=state.read(),
                    ),
                    run,
                    links,
                )
            root = state.prepare()
            if commit is not None:
                receipt = repos.archive(
                    commit,
                    root / "code",
                    options=RepositoryArchiveOptions(
                        timeout_seconds=state.remaining(),
                        maximum_bytes=min(512 * 1024 * 1024, state.bytes_left()),
                    ),
                )
                state.used += receipt.byte_size
            client = Artifacts(transport)
            for ref in artifacts:
                # Revisions occupy distinct directories; a run may intentionally
                # produce more than one immutable version of the same artifact.
                target = root / "artifacts" / ref.record_id / ref.revision
                target.mkdir(parents=True)
                descriptor = client.get(snapshot.project_id, ref, request=state.read())
                item = descriptor.manifest.objects[0]
                from synth_ai.sdk.core.archive_decode import safe_path

                path = target.joinpath(*safe_path(item.logical_path))
                path.parent.mkdir(parents=True, exist_ok=True)
                receipt = client.download(
                    snapshot.project_id,
                    ref,
                    path,
                    options=ArtifactDownloadOptions(
                        timeout_seconds=state.remaining(),
                        maximum_bytes=min(64 * 1024 * 1024, state.bytes_left()),
                    ),
                )
                state.used += receipt.byte_size
            if commit is not None:
                current = select_commit(
                    repos.refs(
                        prefix="refs/heads/worker/run-" + run.run_id,
                        limit=1,
                        request=state.read(),
                    ),
                    run,
                    links,
                )
                if current != commit:
                    raise ContractError(
                        "Run branch changed during download", code="run_commit_changed"
                    )
            final_snapshot = run.status(request=state.read())
            return state.publish(run.ref, commit, artifacts, links, final_snapshot)
    finally:
        try:
            state.cleanup()
        finally:
            state.finished.set()
            transport._unregister_stream(state)


async def download_async(run, destination, options=None):
    state = AsyncBundle(destination, options or RunDownloadOptions())
    transport = run._transport
    transport._register_stream(state)
    try:
        snapshot = await run.status(request=state.read())
        if snapshot.state not in TERMINAL:
            raise ResultNotReadyError(
                "Run download requires a terminal run", code="run_not_terminal"
            )
        links, seen, cursor = [], set(), None
        while True:
            cursor = collect(
                await run.produced(cursor=cursor, request=state.read()), links, seen, state.options
            )
            if cursor is None:
                break
        artifacts = selected_artifacts(links, state.options)
        repos = None
        commit = None
        if any(link.object.authority == "git" and link.object.kind == "commit" for link in links):
            repos = await AsyncRepositories(transport).get(
                snapshot.project_id, request=state.read()
            )
            commit = select_commit(
                await repos.refs(
                    prefix="refs/heads/worker/run-" + run.run_id,
                    limit=1,
                    request=state.read(),
                ),
                run,
                links,
            )
        root = state.prepare()
        if commit is not None:
            receipt = await repos.archive(
                commit,
                root / "code",
                options=RepositoryArchiveOptions(
                    timeout_seconds=state.remaining(),
                    maximum_bytes=min(512 * 1024 * 1024, state.bytes_left()),
                ),
            )
            state.used += receipt.byte_size
        client = AsyncArtifacts(transport)
        for ref in artifacts:
            target = root / "artifacts" / ref.record_id / ref.revision
            target.mkdir(parents=True)
            descriptor = await client.get(snapshot.project_id, ref, request=state.read())
            item = descriptor.manifest.objects[0]
            from synth_ai.sdk.core.archive_decode import safe_path

            path = target.joinpath(*safe_path(item.logical_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt = await client.download(
                snapshot.project_id,
                ref,
                path,
                options=ArtifactDownloadOptions(
                    timeout_seconds=state.remaining(),
                    maximum_bytes=min(64 * 1024 * 1024, state.bytes_left()),
                ),
            )
            state.used += receipt.byte_size
        if commit is not None:
            current = select_commit(
                await repos.refs(
                    prefix="refs/heads/worker/run-" + run.run_id,
                    limit=1,
                    request=state.read(),
                ),
                run,
                links,
            )
            if current != commit:
                raise ContractError("Run branch changed during download", code="run_commit_changed")
        final_snapshot = await run.status(request=state.read())
        return state.publish(run.ref, commit, artifacts, links, final_snapshot)
    finally:
        try:
            state.cleanup()
        finally:
            state.finished.set()
            transport._unregister_stream(state)
