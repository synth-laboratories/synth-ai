"""Scoped repository handles; every read rechecks current server authority."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import TypeAdapter

from synth_ai.contracts import core as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.refs import Ref

from .types import (
    CloneToken,
    CloneTokenRequest,
    GitSHA,
    RepositoryArchiveOptions,
    RepositoryArchiveReceipt,
    RepositoryCommit,
    RepositoryDiff,
    RepositoryRef,
    RepositorySnapshot,
    commit_ref,
)


def selected_commit(value: Ref | str, repo_id: str) -> str:
    return (
        commit_ref(value, repo_id).revision
        if isinstance(value, Ref)
        else TypeAdapter(GitSHA).validate_python(value)
    )


def selected_refs(limit, cursor, prefix):
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Repository refs limit must be in 1..=50 for bounded digest lookup")
    # A Git ref name may occupy4096 UTF-8 bytes; its bound cursor includes a
    # base64 envelope. Other SDK pagination retains its ordinary2048byte bound.
    query = page_parameters(limit, None)
    if cursor is not None:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 8192:
            raise ValueError("Repository refs cursor exceeds its8192character bound")
        query["cursor"] = cursor
    if prefix is not None:
        if (
            not isinstance(prefix, str)
            or not prefix.startswith("refs/")
            or len(prefix.encode()) > 4096
        ):
            raise ValueError("Repository ref prefix must be a bounded refs/ prefix")
        query["prefix"] = prefix
    return query


def selected_diff(base, head, repo_id, max_bytes, max_files):
    if type(max_bytes) is not int or not 1 <= max_bytes <= 8 * 1024 * 1024:
        raise ValueError("Diff max_bytes must be in 1..=8388608")
    if type(max_files) is not int or not 1 <= max_files <= 2000:
        raise ValueError("Diff max_files must be in 1..=2000")
    query = {
        "base": selected_commit(base, repo_id),
        "head": selected_commit(head, repo_id),
        "max_bytes": max_bytes,
        "max_files": max_files,
    }
    if isinstance(base, Ref):
        query["base_digest_sha256"] = base.digest_sha256
    if isinstance(head, Ref):
        query["head_digest_sha256"] = head.digest_sha256
    return query


def checked_diff(result, snapshot, base, head, query):
    return verified(
        result,
        result.project_id == snapshot.project_id
        and result.repo_id == snapshot.repo_id
        and result.base.revision == query["base"]
        and result.head.revision == query["head"]
        and (not isinstance(base, Ref) or result.base == base)
        and (not isinstance(head, Ref) or result.head == head)
        and len(result.files) <= query["max_files"]
        and (result.patch is None or len(result.patch.encode()) <= query["max_bytes"]),
        "Diff changed selected commit identities, project or bounds",
    )


def checked_clone(result, snapshot, ttl):
    lifetime = (result.expires_at - result.metadata.observed_at).total_seconds()
    return verified(
        result,
        result.project_id == snapshot.project_id
        and result.repo_id == snapshot.repo_id
        and 0 < lifetime <= ttl,
        "Clone token changed selected project, repository or lifetime",
    )


class Repository:
    def __init__(self, transport: HttpTransport, snapshot: RepositorySnapshot) -> None:
        self._transport, self.snapshot = transport, snapshot
        self.project_id, self.repo_id = snapshot.project_id, snapshot.repo_id
        self.default_branch, self.ref = snapshot.default_branch, snapshot.ref
        self._path = f"/projects/{identifier(self.project_id)}/repo"

    def refs(
        self,
        *,
        prefix: str | None = None,
        limit: int = 50,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[RepositoryRef]:
        return self._transport.request(
            operations.REPOSITORY_REFS,
            Page[RepositoryRef],
            path=self._path + "/refs",
            query=selected_refs(limit, cursor, prefix),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(
                    item.ref.record_id == self.repo_id
                    and (prefix is None or item.name.startswith(prefix))
                    for item in result.items
                ),
                "Repository refs changed selected identity, prefix or bound",
            ),
        )

    def commit(self, sha: str, *, request: ReadOptions | None = None) -> RepositoryCommit:
        sha = selected_commit(sha, self.repo_id)
        return self._transport.request(
            operations.REPOSITORY_COMMIT,
            RepositoryCommit,
            path=self._path + "/commits/" + sha,
            options=request,
            validate=lambda result: verified(
                result,
                result.project_id == self.project_id
                and result.repo_id == self.repo_id
                and result.sha == sha,
                "Commit changed selected identity",
            ),
        )

    def diff(
        self,
        base: Ref | str,
        head: Ref | str,
        *,
        max_bytes: int = 1024 * 1024,
        max_files: int = 200,
        request: ReadOptions | None = None,
    ) -> RepositoryDiff:
        query = selected_diff(base, head, self.repo_id, max_bytes, max_files)
        return self._transport.request(
            operations.REPOSITORY_DIFF,
            RepositoryDiff,
            path=self._path + "/diff",
            query=query,
            options=request,
            validate=lambda result: checked_diff(result, self.snapshot, base, head, query),
        )

    def clone_url(self, *, ttl: int = 300, mutation: MutationOptions | None = None) -> CloneToken:
        body = CloneTokenRequest(ttl_seconds=ttl)
        return self._transport.request(
            operations.CLONE_TOKEN,
            CloneToken,
            path=self._path + "/clone-tokens",
            body=body,
            options=mutation,
            validate=lambda result: checked_clone(result, self.snapshot, ttl),
        )

    def archive(
        self,
        commit: Ref | str,
        destination: Path,
        *,
        options: RepositoryArchiveOptions | None = None,
    ) -> RepositoryArchiveReceipt:
        from .repo_archive import archive_destination, download_archive

        selected = options or RepositoryArchiveOptions()
        deadline = time.monotonic() + selected.timeout_seconds
        destination = archive_destination(destination)
        reference = (
            commit
            if isinstance(commit, Ref)
            else self.commit(
                commit,
                request=ReadOptions(timeout_seconds=selected.timeout_seconds, read_retry_count=0),
            ).ref
        )
        commit_ref(reference, self.repo_id)
        return download_archive(
            self._transport,
            self._path + "/archive",
            reference,
            destination,
            selected,
            deadline=deadline,
        )


class Repositories:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, project_id: str, *, request: ReadOptions | None = None) -> Repository:
        project_id = identifier(project_id)
        snapshot = self._transport.request(
            operations.GET_REPOSITORY,
            RepositorySnapshot,
            path=f"/projects/{project_id}/repo",
            options=request,
            validate=lambda result: verified(
                result, result.project_id == project_id, "Repository project changed"
            ),
        )
        return Repository(self._transport, snapshot)


class AsyncRepository:
    def __init__(self, transport: AsyncHttpTransport, snapshot: RepositorySnapshot) -> None:
        self._transport, self.snapshot = transport, snapshot
        self.project_id, self.repo_id = snapshot.project_id, snapshot.repo_id
        self.default_branch, self.ref = snapshot.default_branch, snapshot.ref
        self._path = f"/projects/{identifier(self.project_id)}/repo"

    async def refs(
        self,
        *,
        prefix: str | None = None,
        limit: int = 50,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[RepositoryRef]:
        return await self._transport.request(
            operations.REPOSITORY_REFS,
            Page[RepositoryRef],
            path=self._path + "/refs",
            query=selected_refs(limit, cursor, prefix),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(
                    item.ref.record_id == self.repo_id
                    and (prefix is None or item.name.startswith(prefix))
                    for item in result.items
                ),
                "Repository refs changed selected identity, prefix or bound",
            ),
        )

    async def commit(self, sha: str, *, request: ReadOptions | None = None) -> RepositoryCommit:
        sha = selected_commit(sha, self.repo_id)
        return await self._transport.request(
            operations.REPOSITORY_COMMIT,
            RepositoryCommit,
            path=self._path + "/commits/" + sha,
            options=request,
            validate=lambda result: verified(
                result,
                result.project_id == self.project_id
                and result.repo_id == self.repo_id
                and result.sha == sha,
                "Commit changed selected identity",
            ),
        )

    async def diff(
        self,
        base: Ref | str,
        head: Ref | str,
        *,
        max_bytes: int = 1024 * 1024,
        max_files: int = 200,
        request: ReadOptions | None = None,
    ) -> RepositoryDiff:
        query = selected_diff(base, head, self.repo_id, max_bytes, max_files)
        return await self._transport.request(
            operations.REPOSITORY_DIFF,
            RepositoryDiff,
            path=self._path + "/diff",
            query=query,
            options=request,
            validate=lambda result: checked_diff(result, self.snapshot, base, head, query),
        )

    async def clone_url(
        self, *, ttl: int = 300, mutation: MutationOptions | None = None
    ) -> CloneToken:
        body = CloneTokenRequest(ttl_seconds=ttl)
        return await self._transport.request(
            operations.CLONE_TOKEN,
            CloneToken,
            path=self._path + "/clone-tokens",
            body=body,
            options=mutation,
            validate=lambda result: checked_clone(result, self.snapshot, ttl),
        )

    async def archive(
        self,
        commit: Ref | str,
        destination: Path,
        *,
        options: RepositoryArchiveOptions | None = None,
    ) -> RepositoryArchiveReceipt:
        from .repo_archive import archive_destination, download_archive_async

        selected = options or RepositoryArchiveOptions()
        deadline = time.monotonic() + selected.timeout_seconds
        destination = archive_destination(destination)
        reference = (
            commit
            if isinstance(commit, Ref)
            else (
                await self.commit(
                    commit,
                    request=ReadOptions(
                        timeout_seconds=selected.timeout_seconds, read_retry_count=0
                    ),
                )
            ).ref
        )
        commit_ref(reference, self.repo_id)
        return await download_archive_async(
            self._transport,
            self._path + "/archive",
            reference,
            destination,
            selected,
            deadline=deadline,
        )


class AsyncRepositories:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, project_id: str, *, request: ReadOptions | None = None) -> AsyncRepository:
        project_id = identifier(project_id)
        snapshot = await self._transport.request(
            operations.GET_REPOSITORY,
            RepositorySnapshot,
            path=f"/projects/{project_id}/repo",
            options=request,
            validate=lambda result: verified(
                result, result.project_id == project_id, "Repository project changed"
            ),
        )
        return AsyncRepository(self._transport, snapshot)
