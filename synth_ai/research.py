"""Typed client contracts for Interactive Artifacts, Experiments, and Research Logs.

Implements the SDK client surface for SYN-5973 (Hosted Interactive Artifacts) and
SYN-5974 (Multi-Experiment & Notebook Research Logs) over the generated OpenAPI
1.43.0 operations with strict frozen models and 1:1 async/sync parity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ._generated.models import (
    HostedArtifact,
    HostedArtifactKind,
    HostedArtifactList,
    HostedArtifactOrigin,
    HostedArtifactRevision,
    HostedArtifactRevisionByteState,
    HostedArtifactShareGrant,
    HostedArtifactShareGrantScope,
    McpOutputContent,
    ResearchExactReference,
    ResearchExactReferenceSource,
    ResearchExperimentList,
    ResearchExperimentRecord,
    ResearchLogEntry,
    ResearchLogEntryAuthorKind,
    ResearchLogEntryCategory,
    ResearchLogEntryList,
    ResearchLogEntrySyncState,
    ResearchLogExport,
    ResearchLogSummary,
    ResearchTrialCharge,
    ResearchTrialMeasurement,
    ResearchTrialRecord,
    ResearchTrialScore,
)
from ._inline_content import inline_content
from ._validation import verify_sha256
from .errors import DataIntegrityError, SynthError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

    from ._sync import (
        ArtifactsClient as ArtifactsClient,
    )
    from ._sync import (
        ExperimentsClient as ExperimentsClient,
    )
    from ._sync import (
        ResearchLogsClient as ResearchLogsClient,
    )
    from ._transport import Transport

ArtifactKind = HostedArtifactKind
ArtifactOrigin = HostedArtifactOrigin
ArtifactByteState = HostedArtifactRevisionByteState
ArtifactShareScope = HostedArtifactShareGrantScope
AuthorKind = ResearchLogEntryAuthorKind
LogCategory = ResearchLogEntryCategory
LogSyncState = ResearchLogEntrySyncState
ReferenceSource = ResearchExactReferenceSource
ExactReference = ResearchExactReference
ArtifactRevision = HostedArtifactRevision
ArtifactShareGrant = HostedArtifactShareGrant
ArtifactList = HostedArtifactList
TrialMeasurement = ResearchTrialMeasurement
TrialScore = ResearchTrialScore
TrialCharge = ResearchTrialCharge
TrialRecord = ResearchTrialRecord
ExperimentRecord = ResearchExperimentRecord
ExperimentList = ResearchExperimentList
ResearchLogPage = ResearchLogEntryList

__all__ = [
    "ArtifactByteState",
    "ArtifactKind",
    "ArtifactList",
    "ArtifactOrigin",
    "ArtifactRevision",
    "ArtifactShareGrant",
    "ArtifactShareScope",
    "ArtifactsClient",
    "AsyncArtifacts",
    "AsyncExperiments",
    "AsyncResearchLogs",
    "AuthorKind",
    "ExactReference",
    "ExperimentList",
    "ExperimentRecord",
    "ExperimentsClient",
    "HostedArtifact",
    "LogCategory",
    "LogSyncState",
    "ReferenceSource",
    "ResearchLogEntry",
    "ResearchLogExport",
    "ResearchLogPage",
    "ResearchLogSummary",
    "ResearchLogsClient",
    "TrialCharge",
    "TrialMeasurement",
    "TrialOutcome",
    "TrialRecord",
    "TrialScore",
    "TrialState",
    "TrialVerification",
]


class TrialOutcome(StrEnum):
    """Evaluation outcome for a single trial score."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class TrialVerification(StrEnum):
    """Independent verification state for a trial score."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class TrialState(StrEnum):
    """Lifecycle state of an experiment trial."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AsyncArtifacts:
    """Async client for Hosted Interactive Artifacts (`SYN-5973`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_artifact(
        self,
        notebook_id: str,
        title: str,
        kind: ArtifactKind | str,
        content: str,
        summary: str,
        *,
        origin: ArtifactOrigin | str = ArtifactOrigin.AGENT,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Create a versioned interactive artifact (`v1`) for a notebook or run."""
        payload: dict[str, JsonValue] = {
            "notebook_id": notebook_id,
            "title": title,
            "kind": ArtifactKind(kind).value,
            "origin": ArtifactOrigin(origin).value,
            "content": content,
            "summary": summary,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return await self._transport.request(
            "createArtifact",
            HostedArtifact,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def update_artifact(
        self,
        artifact_id: str,
        content: str,
        summary: str,
        *,
        author_kind: AuthorKind | str | None = None,
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Append a new immutable revision (`v2`, `v3`, ...) to an existing artifact."""
        payload: dict[str, JsonValue] = {"content": content, "summary": summary}
        if author_kind is not None:
            payload["author_kind"] = AuthorKind(author_kind).value
        return await self._transport.request(
            "updateArtifact",
            HostedArtifact,
            path={"artifact_id": artifact_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def list_artifacts(
        self,
        notebook_id: str,
        *,
        run_id: str | None = None,
        kind: ArtifactKind | str | None = None,
        origin: ArtifactOrigin | str | None = None,
        q: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ArtifactList:
        """Read one page of a notebook's hosted artifacts, newest first.

        Filter by run, kind, origin or `q` (case-insensitive literal text; `%` and `_`
        match literally). Pass the page's `next_cursor` as `cursor` while `has_more`.
        """
        return await self._transport.request(
            "listNotebookArtifacts",
            HostedArtifactList,
            path={"notebook_id": notebook_id},
            query={
                "run_id": run_id,
                "kind": ArtifactKind(kind).value if kind is not None else None,
                "origin": ArtifactOrigin(origin).value if origin is not None else None,
                "q": q,
                "limit": limit,
                "cursor": cursor,
            },
        )

    async def get_artifact(self, artifact_id: str) -> HostedArtifact:
        """Fetch an interactive artifact and its full revision history."""
        return await self._transport.request(
            "getArtifact",
            HostedArtifact,
            path={"artifact_id": artifact_id},
        )

    def _verified_artifact_bytes(self, raw: object, digest_sha256: str | None) -> bytes:
        if not isinstance(raw, bytes):
            raise SynthError("Unexpected response model for getArtifactContent")
        if digest_sha256 is not None:
            verify_sha256(raw, digest_sha256, label="artifact")
        return raw

    async def get_artifact_content(
        self,
        artifact_id: str,
        *,
        revision_id: str | None = None,
        digest_sha256: str | None = None,
    ) -> str:
        """Fetch the verified UTF-8 source payload for an artifact revision."""
        raw = self._verified_artifact_bytes(
            await self._transport.invoke(
                "getArtifactContent",
                path={"artifact_id": artifact_id},
                query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            ),
            digest_sha256,
        )
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise DataIntegrityError("artifact content is not valid UTF-8") from error

    async def get_artifact_content_inline(
        self,
        artifact_id: str,
        *,
        revision_id: str | None = None,
        digest_sha256: str | None = None,
    ) -> McpOutputContent:
        """Fetch an artifact revision payload as the contract's McpOutputContent."""
        raw = self._verified_artifact_bytes(
            await self._transport.invoke(
                "getArtifactContent",
                path={"artifact_id": artifact_id},
                query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            ),
            digest_sha256,
        )
        return inline_content(raw, "text/plain; charset=utf-8")

    async def share_artifact(
        self,
        artifact_id: str,
        *,
        scope: ArtifactShareScope | str = ArtifactShareScope.ORG,
        idempotency_key: str | None = None,
    ) -> ArtifactShareGrant:
        """Mint or update an org/link-scoped share grant for an artifact."""
        payload: dict[str, JsonValue] = {"scope": ArtifactShareScope(scope).value}
        return await self._transport.request(
            "shareArtifact",
            HostedArtifactShareGrant,
            path={"artifact_id": artifact_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def revoke_artifact_share(self, artifact_id: str) -> ArtifactShareGrant:
        """Revoke the active share grant for an artifact."""
        return await self._transport.request(
            "revokeArtifactShare",
            HostedArtifactShareGrant,
            path={"artifact_id": artifact_id},
        )


class AsyncExperiments:
    """Async client for Multi-Experiment tracking (`SYN-5974`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_experiment(
        self,
        notebook_id: str,
        run_id: str,
        title: str,
        specification: str,
        arms: Sequence[str],
        *,
        hypothesis: str | None = None,
        pinned_inputs: Sequence[str] = (),
        stopping_criteria: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExperimentRecord:
        """Create a versioned experiment plan with named comparison arms."""
        payload: dict[str, JsonValue] = {
            "notebook_id": notebook_id,
            "run_id": run_id,
            "title": title,
            "specification": specification,
            "arms": list(arms),
            "pinned_inputs": list(pinned_inputs),
        }
        if hypothesis is not None:
            payload["hypothesis"] = hypothesis
        if stopping_criteria is not None:
            payload["stopping_criteria"] = stopping_criteria
        return await self._transport.request(
            "createExperiment",
            ResearchExperimentRecord,
            path={"notebook_id": notebook_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def revise_experiment(
        self,
        experiment_id: str,
        *,
        title: str | None = None,
        specification: str | None = None,
        hypothesis: str | None = None,
        arms: Sequence[str] | None = None,
        stopping_criteria: str | None = None,
        idempotency_key: str | None = None,
    ) -> ExperimentRecord:
        """Create a new immutable revision of an existing experiment plan."""
        payload: dict[str, JsonValue] = {}
        if title is not None:
            payload["title"] = title
        if specification is not None:
            payload["specification"] = specification
        if hypothesis is not None:
            payload["hypothesis"] = hypothesis
        if arms is not None:
            payload["arms"] = list(arms)
        if stopping_criteria is not None:
            payload["stopping_criteria"] = stopping_criteria
        return await self._transport.request(
            "reviseExperiment",
            ResearchExperimentRecord,
            path={"experiment_id": experiment_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Read an experiment record and its associated trial scores."""
        return await self._transport.request(
            "getExperiment",
            ResearchExperimentRecord,
            path={"experiment_id": experiment_id},
        )

    async def list_experiments(
        self,
        notebook_id: str,
        *,
        run_id: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ExperimentList:
        """Read one page of a notebook's experiments, newest first.

        Filter by run or `q` (case-insensitive literal match on title, specification or
        hypothesis; `%` and `_` match literally). Pass `next_cursor` as `cursor` while `has_more`.
        """
        return await self._transport.request(
            "listNotebookExperiments",
            ResearchExperimentList,
            path={"notebook_id": notebook_id},
            query={"run_id": run_id, "q": q, "limit": limit, "cursor": cursor},
        )


class AsyncResearchLogs:
    """Async client for append-only Notebook & Run Research Logs (`SYN-5974`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_log(
        self,
        notebook_id: str,
        *,
        title: str,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> ResearchLogSummary:
        """Initialize or register a notebook research log."""
        payload: dict[str, JsonValue] = {"notebook_id": notebook_id, "title": title}
        if summary is not None:
            payload["summary"] = summary
        return await self._transport.request(
            "initResearchLog",
            ResearchLogSummary,
            path={"notebook_id": notebook_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def append_entry(
        self,
        notebook_id: str,
        category: LogCategory | str,
        text: str,
        *,
        run_id: str | None = None,
        author_kind: AuthorKind | str = AuthorKind.LOCAL_AGENT,
        author_label: str | None = None,
        correction_of: str | None = None,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append a structured research-log entry with optional exact citations."""
        payload: dict[str, JsonValue] = {
            "notebook_id": notebook_id,
            "category": LogCategory(category).value,
            "text": text,
            "author_kind": AuthorKind(author_kind).value,
            "sources": [source.model_dump(mode="json") for source in sources],
            "artifact_ids": list(artifact_ids),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        if author_label is not None:
            payload["author_label"] = author_label
        if correction_of is not None:
            payload["correction_of"] = correction_of
        return await self._transport.request(
            "appendResearchLogEntry",
            ResearchLogEntry,
            path={"notebook_id": notebook_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def correct_entry(
        self,
        notebook_id: str,
        entry_id: str,
        text: str,
        *,
        run_id: str | None = None,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append an explicit correction referencing a prior research-log entry."""
        payload: dict[str, JsonValue] = {
            "text": text,
            "sources": [source.model_dump(mode="json") for source in sources],
            "artifact_ids": list(artifact_ids),
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return await self._transport.request(
            "correctResearchLogEntry",
            ResearchLogEntry,
            path={"notebook_id": notebook_id, "entry_id": entry_id},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def search_entries(
        self,
        notebook_id: str,
        *,
        query: str | None = None,
        category: LogCategory | str | None = None,
        run_id: str | None = None,
        author_kind: AuthorKind | str | None = None,
        q: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ResearchLogPage:
        """Read one page of research-log entries in log order, filtered by text, category, run or author.

        `q` is an alias of `query`; when both are set they must be equal. Text matches
        literally (`%` and `_` included). Pass `next_cursor` as `cursor` while `has_more`.
        """
        return await self._transport.request(
            "searchResearchLog",
            ResearchLogEntryList,
            path={"notebook_id": notebook_id},
            query={
                "query": query,
                "q": q,
                "category": LogCategory(category).value if category is not None else None,
                "run_id": run_id,
                "author_kind": AuthorKind(author_kind).value if author_kind is not None else None,
                "limit": limit,
                "cursor": cursor,
            },
        )

    async def export_log(
        self, notebook_id: str, *, run_id: str | None = None, fmt: str = "markdown"
    ) -> ResearchLogExport:
        """Export a notebook or run research log in markdown or JSON format."""
        if fmt not in {"markdown", "json"}:
            raise ValueError("fmt must be 'markdown' or 'json'")
        return await self._transport.request(
            "exportResearchLog",
            ResearchLogExport,
            path={"notebook_id": notebook_id},
            query={"run_id": run_id, "format": fmt},
        )


def __getattr__(name: str) -> type[ArtifactsClient | ExperimentsClient | ResearchLogsClient]:
    """Resolve generated synchronous research clients lazily to avoid import cycles."""
    if name == "ArtifactsClient":
        from ._sync import ArtifactsClient

        return ArtifactsClient
    if name == "ExperimentsClient":
        from ._sync import ExperimentsClient

        return ExperimentsClient
    if name == "ResearchLogsClient":
        from ._sync import ResearchLogsClient

        return ResearchLogsClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
