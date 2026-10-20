"""One bounded archive exchange, verified tree, exclusive directory publication."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import httpx

from synth_ai.contracts.core import REPOSITORY_ARCHIVE
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.blocking import CLEANUP_SECONDS, CancellationScope
from synth_ai.core.http.cleanup import handle_cleanup_failure
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.http.lifetime import await_cleanup
from synth_ai.core.http.response import annotate_response_error, classify_exchange, decode_response
from synth_ai.errors import ContractError, RequestTimeoutError, SynthError, TransportError
from synth_ai.types.base import ClientTiming, Model, ResponseModel
from synth_ai.types.refs import Ref

from .archive_decode import CHUNK_BYTES, extract_verified, invalid
from .atomic import publish_directory
from .types import RepositoryArchiveOptions, RepositoryArchiveReceipt


def archive_destination(destination: Path) -> Path:
    """Reject occupied destinations before negotiation or descriptor lookup."""
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError("Download destination already exists", str(destination))
    if not destination.parent.is_dir():
        raise FileNotFoundError("Download destination parent must exist", str(destination.parent))
    return destination


def remove_owned_tree(root: Path) -> None:
    """Iterative cleanup handles valid deep trees without following symlinks."""
    pending = [(root, False)]
    while pending:
        path, visited = pending.pop()
        if not stat.S_ISDIR(path.lstat().st_mode):
            path.unlink()
        elif visited:
            path.rmdir()
        else:
            pending.append((path, True))
            with os.scandir(path) as children:
                pending.extend((Path(child.path), False) for child in children)


class ArchiveTransfer:
    operation = REPOSITORY_ARCHIVE
    accept = "application/x-tar"
    request_digest: str | None = None

    def query(self) -> dict[str, str | int]:
        return {"commit": self.reference.revision, "digest_sha256": self.reference.digest_sha256}

    def __init__(
        self, reference: Ref, destination: Path, options: RepositoryArchiveOptions, deadline: float
    ) -> None:
        self.reference, self.destination, self.options = reference, destination, options
        self.deadline = deadline
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.response: httpx.Response | None = None
        self.scratch: Path | None = None
        self.size, self.entries = 0, 0
        self.digest = ""
        self.received = 0
        self.hasher = hashlib.sha256()
        self.publication_lock = threading.Lock()

    def remaining(self) -> float:
        if self.stop.is_set():
            raise TransportError(
                "Repository archive observation was closed",
                code="archive_cancelled",
                stage="archive_download",
            )
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError(
                "Repository archive deadline expired",
                code="archive_timeout",
                stage="archive_download",
            )
        return remaining

    def headers(self, response: httpx.Response) -> None:
        self.remaining()
        decision = classify_exchange(
            response,
            None,
            contract="core.v1",
            mutation=False,
            identity=None,
            attempt=1,
            attempts=1,
            operation=REPOSITORY_ARCHIVE.id,
        )
        if decision.error is not None:
            raise decision.error
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise invalid("Compressed repository archives are outside the byte contract")
        if response.status_code != 200:
            return
        expected = {
            "X-Synth-Contract": "core.v1",
            "X-Synth-Archive-Format": "synth.git-raw-tar.v1",
            "X-Synth-Repo-Id": self.reference.record_id,
            "X-Synth-Commit-SHA": self.reference.revision,
            "X-Synth-Commit-Manifest-SHA256": self.reference.digest_sha256,
        }
        if any(response.headers.get(key) != value for key, value in expected.items()):
            raise invalid(
                "Repository archive response changed the selected identity",
                "archive_identity_mismatch",
            )
        media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media != "application/x-tar":
            raise invalid("Repository archive has an unsupported media type")
        self.digest = response.headers.get("X-Synth-Content-SHA256", "")
        if re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise invalid("Repository archive lacks a canonical byte digest")
        self.size = self._integer(response, "X-Synth-Archive-Bytes", self.options.maximum_bytes)
        self.entries = self._integer(
            response, "X-Synth-Archive-Entries", self.options.maximum_files
        )
        length = response.headers.get("Content-Length")
        if length is not None and (
            re.fullmatch(r"[0-9]{1,20}", length) is None or int(length) != self.size
        ):
            raise invalid("Repository archive content length differs from its receipt")

    @staticmethod
    def _integer(response: httpx.Response, name: str, maximum: int) -> int:
        value = response.headers.get(name, "")
        if re.fullmatch(r"[0-9]{1,20}", value) is None or int(value) > maximum:
            raise invalid(
                "Repository archive receipt exceeds the selected bounds", "archive_limit_exceeded"
            )
        return int(value)

    def consume(self, chunk: bytes) -> None:
        self.remaining()
        self.received += len(chunk)
        if self.received > self.size:
            raise invalid("Repository archive exceeded its declared byte count")
        self.hasher.update(chunk)

    def finish_bytes(self) -> None:
        self.remaining()
        if self.received != self.size or self.hasher.hexdigest() != self.digest:
            raise invalid(
                "Repository archive byte count or digest verification failed",
                "archive_bytes_mismatch",
            )

    def prepare(self) -> Path:
        self.remaining()
        self.scratch = Path(tempfile.mkdtemp(prefix=".synth-repo-", dir=self.destination.parent))
        return self.scratch / "archive.tar"

    def extract(self, archive: Path) -> Path:
        self.remaining()
        assert self.scratch is not None
        tree = self.scratch / "tree"
        tree.mkdir(mode=0o700)
        with archive.open("rb") as source:
            extract_verified(
                source,
                tree,
                self.reference.digest_sha256,
                self.entries,
                self.options,
                self.check,
            )
        return tree

    def check(self) -> None:
        self.remaining()

    def publish(self, tree: Path) -> Model:
        with self.publication_lock:
            self.remaining()
            receipt = RepositoryArchiveReceipt(
                ref=self.reference,
                destination=self.destination,
                byte_size=self.size,
                content_sha256=self.digest,
                entries=self.entries,
            )
            publish_directory(tree, self.destination)
        return receipt

    def cleanup(self) -> None:
        if self.scratch is not None:
            remove_owned_tree(self.scratch)
            self.scratch = None

    def annotate(self, error: SynthError) -> None:
        # An earlier capability refusal already identifies its own operation.
        if error.operation is not None and error.operation != self.operation.id:
            return
        error.operation = self.operation.id
        if self.response is not None:
            error.status = self.response.status_code
            request_id = self.response.headers.get("X-Request-Id", "")
            if re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", request_id):
                error.request_id = request_id
            error.client_request_id = self.response.request.headers.get("X-Request-Id")
            annotate_response_error(self.response, error)


def error_response(response: httpx.Response, body: bytes, operation=REPOSITORY_ARCHIVE) -> None:
    decode_response(
        response,
        body,
        ResponseModel,
        ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
        None,
        False,
        "core.v1",
        operation=operation.id,
    )
    raise ContractError(
        "Byte download returned an unexpected status",
        code="download_status_invalid",
        operation=operation.id,
    )


class SyncArchive(ArchiveTransfer):
    def __init__(self, transport: HttpTransport, *args) -> None:
        super().__init__(*args)
        self.transport = transport
        self.scope = CancellationScope()
        self.reader = threading.get_ident()

    def close(self) -> None:
        with self.publication_lock:
            self.stop.set()
        self.scope.cancel()
        if threading.get_ident() != self.reader and not self.finished.wait(4 * CLEANUP_SECONDS):
            raise TransportError(
                "Repository archive cleanup did not complete",
                code="archive_cleanup_timeout",
                stage="archive_cleanup",
            )


def _download_owned(
    transport: HttpTransport,
    path: str,
    state: SyncArchive,
) -> Model:
    transport._register_stream(state)
    primary: BaseException | None = None
    try:
        with state.scope.activate():
            response = transport._open_stream(
                state.operation,
                path=path,
                query=state.query(),
                remaining_seconds=state.remaining(),
                digest=state.request_digest,
                accept=state.accept,
            )
            state.response = response
            state.headers(response)
            if response.status_code != 200:
                body = bytearray()
                for chunk in response.iter_raw(CHUNK_BYTES):
                    state.remaining()
                    if len(body) + len(chunk) > min(65536, transport.maximum_response_bytes):
                        raise ContractError(
                            "Byte error response exceeded its ceiling",
                            code="download_error_too_large",
                        )
                    body.extend(chunk)
                error_response(response, bytes(body), state.operation)
            archive = state.prepare()
            with archive.open("xb") as output:
                for chunk in response.iter_raw(CHUNK_BYTES):
                    state.consume(chunk)
                    output.write(chunk)
                state.finish_bytes()
                output.flush()
                os.fsync(output.fileno())
            return state.publish(state.extract(archive))
    except (httpx.TransportError, TimeoutError) as cause:
        primary = (
            RequestTimeoutError("Repository archive timed out")
            if isinstance(cause, (httpx.TimeoutException, TimeoutError))
            else TransportError("Repository archive exchange failed; no automatic retry")
        )
        state.annotate(primary)
        raise primary from cause
    except BaseException as error:
        primary = error
        if isinstance(error, SynthError):
            state.annotate(error)
        raise
    finally:
        try:
            with state.scope.cleanup():
                try:
                    if state.response is not None:
                        transport._release_response(state.response)
                finally:
                    state.cleanup()
        except Exception as cause:
            handle_cleanup_failure(cause, primary, mutation=False, identity=None)
        finally:
            transport._unregister_stream(state)
            state.finished.set()


class AsyncArchive(ArchiveTransfer):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.reader = asyncio.current_task()
        self.done = asyncio.Event()

    async def aclose(self) -> None:
        self.stop.set()
        if self.reader is not None and self.reader is not asyncio.current_task():
            self.reader.cancel()
            await self.done.wait()

    async def extract_owned(self, archive: Path) -> Path:
        worker = asyncio.create_task(asyncio.to_thread(self.extract, archive))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError as cancellation:
            self.stop.set()
            primary = cancellation

            # Never detach a filesystem worker on cancellation. It checks the
            # stop event between every bounded read, member and directory.
            async def join() -> None:
                try:
                    await worker
                except Exception as cause:
                    if not isinstance(cause, TransportError) or cause.code != "archive_cancelled":
                        handle_cleanup_failure(cause, primary, mutation=False, identity=None)

            await await_cleanup(asyncio.create_task(join()))
            raise


async def _download_owned_async(
    transport: AsyncHttpTransport,
    path: str,
    state: AsyncArchive,
) -> Model:
    transport._register_stream(state)
    primary: BaseException | None = None
    try:
        async with asyncio.timeout(state.remaining()):
            response = await transport._open_stream(
                state.operation,
                path=path,
                query=state.query(),
                remaining_seconds=state.remaining(),
                digest=state.request_digest,
                accept=state.accept,
            )
            state.response = response
            state.headers(response)
            if response.status_code != 200:
                body = bytearray()
                async for chunk in response.aiter_raw(CHUNK_BYTES):
                    state.remaining()
                    if len(body) + len(chunk) > min(65536, transport.maximum_response_bytes):
                        raise ContractError(
                            "Byte error response exceeded its ceiling",
                            code="download_error_too_large",
                        )
                    body.extend(chunk)
                error_response(response, bytes(body), state.operation)
            archive = state.prepare()
            with archive.open("xb") as output:
                async for chunk in response.aiter_raw(CHUNK_BYTES):
                    state.consume(chunk)
                    output.write(chunk)
                state.finish_bytes()
                output.flush()
                os.fsync(output.fileno())
            tree = await state.extract_owned(archive)
            return state.publish(tree)
    except (httpx.TransportError, TimeoutError) as cause:
        primary = (
            RequestTimeoutError("Repository archive timed out")
            if isinstance(cause, (httpx.TimeoutException, TimeoutError))
            else TransportError("Repository archive exchange failed; no automatic retry")
        )
        state.annotate(primary)
        raise primary from cause
    except BaseException as error:
        primary = error
        if isinstance(error, SynthError):
            state.annotate(error)
        raise
    finally:

        async def cleanup() -> None:
            try:
                if state.response is not None:
                    await transport._release_response(state.response)
            finally:
                await asyncio.to_thread(state.cleanup)

        try:
            await await_cleanup(asyncio.create_task(cleanup()))
        except (Exception, asyncio.CancelledError) as cause:
            handle_cleanup_failure(cause, primary, mutation=False, identity=None)
        finally:
            transport._unregister_stream(state)
            state.finished.set()
            state.done.set()


def download_archive(
    transport: HttpTransport,
    path: str,
    reference: Ref,
    destination: Path,
    options: RepositoryArchiveOptions,
    *,
    deadline: float,
) -> RepositoryArchiveReceipt:
    state = SyncArchive(transport, reference, archive_destination(destination), options, deadline)
    return cast(RepositoryArchiveReceipt, _download_owned(transport, path, state))


async def download_archive_async(
    transport: AsyncHttpTransport,
    path: str,
    reference: Ref,
    destination: Path,
    options: RepositoryArchiveOptions,
    *,
    deadline: float,
) -> RepositoryArchiveReceipt:
    state = AsyncArchive(reference, archive_destination(destination), options, deadline)
    return cast(RepositoryArchiveReceipt, await _download_owned_async(transport, path, state))
