"""Typed client contracts for Interactive Artifacts, Experiments, and Research Logs.

Implements the SDK client surface for SYN-5973 (Hosted Interactive Artifacts) and
SYN-5974 (Multi-Experiment & Project Research Logs) with strict frozen models and
1:1 async/sync parity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue

from ._generated.models import Money  # noqa: TC001 -- evaluated by Pydantic model construction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._blocking import Runner
    from ._transport import Transport

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


class _StrictRecord(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class ReferenceSource(StrEnum):
    """Authority backing an exact citation reference."""

    RESEARCH = "research"
    ARTIFACT = "artifact"
    RUN = "run"


class ExactReference(_StrictRecord):
    """Pinned citation with exact revision and SHA-256 digest."""

    source: ReferenceSource
    kind: str
    record_id: str
    revision: str
    digest_sha256: str


class ArtifactKind(StrEnum):
    """Supported interactive artifact content types."""

    HTML = "html"
    REACT = "react"
    SVG = "svg"
    MERMAID = "mermaid"
    MARKDOWN = "markdown"
    CHART = "chart"


class ArtifactOrigin(StrEnum):
    """Producer class for an interactive artifact."""

    AGENT = "agent"
    HUMAN = "human"


class ArtifactByteState(StrEnum):
    """Custody state of an artifact revision payload."""

    AVAILABLE = "available"
    MISSING_BYTES = "missing_bytes"
    ACCESS_DENIED = "access_denied"
    PAYLOAD_TOO_LARGE = "payload_too_large"


class ArtifactShareScope(StrEnum):
    """Sharing visibility scope for an interactive artifact."""

    ORG = "org"
    LINK = "link"


class AuthorKind(StrEnum):
    """Author identity class for research notes and artifact revisions."""

    HUMAN = "human"
    LOCAL_AGENT = "local_agent"
    SWARM_ACTOR = "swarm_actor"


class LogCategory(StrEnum):
    """Structured research-log entry categories."""

    HYPOTHESIS = "hypothesis"
    ATTEMPT = "attempt"
    OBSERVATION = "observation"
    FAILURE = "failure"
    DECISION = "decision"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    RATIONALE = "rationale"


class LogSyncState(StrEnum):
    """Synchronization and evidence custody status of a log entry."""

    SYNCED = "synced"
    LOCAL_PENDING = "local_pending"
    MISSING_EVIDENCE = "missing_evidence"


class ArtifactRevision(_StrictRecord):
    """Immutable versioned revision of a hosted interactive artifact."""

    revision: int
    revision_id: str
    digest_sha256: str
    byte_size: int
    media_type: str
    created_at: str
    author_kind: AuthorKind
    summary: str
    byte_state: ArtifactByteState
    provenance: ExactReference


class ArtifactShareGrant(_StrictRecord):
    """Org or link-scoped sharing grant for an interactive artifact."""

    share_id: str
    scope: ArtifactShareScope
    url: str
    revoked: bool
    created_at: str


class HostedArtifact(_StrictRecord):
    """Hosted interactive artifact with ordered revision history."""

    artifact_id: str
    run_id: str | None = None
    project_id: str
    title: str
    kind: ArtifactKind
    origin: ArtifactOrigin
    current_revision: int
    revisions: tuple[ArtifactRevision, ...]
    share: ArtifactShareGrant | None = None
    pinned: bool = False


class ArtifactList(_StrictRecord):
    """Page of interactive artifacts scoped to a project or run."""

    items: tuple[HostedArtifact, ...]


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


class TrialMeasurement(_StrictRecord):
    """Typed scalar measurement with unit and optional uncertainty."""

    name: str
    value: float | None = None
    unit: str
    uncertainty: float | None = None


class TrialScore(_StrictRecord):
    """Verified or unverified scorer receipt attached to a trial."""

    record: ExactReference
    outcome: TrialOutcome
    verification: TrialVerification
    scorer_receipt: ExactReference | None = None
    measurements: tuple[TrialMeasurement, ...] = ()
    missing_evidence: tuple[str, ...] = ()


class TrialCharge(_StrictRecord):
    """Monetary cost attributed to a single experiment trial."""

    amount: Money
    final: bool


class TrialRecord(_StrictRecord):
    """One executed arm/trial inside an experiment."""

    trial_id: str
    experiment_id: str
    arm: str
    parent_trial_id: str | None = None
    depth: int
    state: TrialState
    charge: TrialCharge | None = None
    uplift_per_dollar: float | None = None
    scores: tuple[TrialScore, ...] = ()
    deviations: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    note_ids: tuple[str, ...] = ()


class ExperimentRecord(_StrictRecord):
    """Versioned experiment specification and observed trial records."""

    experiment_id: str
    project_id: str
    run_id: str
    revision: int
    title: str
    specification: str
    hypothesis: str | None = None
    arms: tuple[str, ...]
    pinned_inputs: tuple[str, ...] = ()
    stopping_criteria: str | None = None
    provenance: ExactReference
    trials: tuple[TrialRecord, ...] = ()


class ExperimentList(_StrictRecord):
    """List of project or run experiments."""

    items: tuple[ExperimentRecord, ...]


class ResearchLogSummary(_StrictRecord):
    """Metadata header for a project research log."""

    project_id: str
    title: str
    summary: str | None = None
    entry_count: int = 0
    created_at: str


class ResearchLogEntry(_StrictRecord):
    """Append-only research log entry with exact provenance citations."""

    entry_id: str
    project_id: str
    run_id: str | None = None
    category: LogCategory
    author_kind: AuthorKind
    author_label: str
    text: str
    occurred_at: str
    recorded_at: str
    sync_state: LogSyncState
    correction_of: str | None = None
    sources: tuple[ExactReference, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()


class ResearchLogPage(_StrictRecord):
    """Ordered slice of research log entries."""

    items: tuple[ResearchLogEntry, ...]


class ResearchLogExport(_StrictRecord):
    """Rendered export of a project or run research log."""

    project_id: str
    run_id: str | None = None
    format: str
    content: str
    entry_count: int


def _enc(value: str) -> str:
    return quote(value, safe="")


class AsyncArtifacts:
    """Async client for Hosted Interactive Artifacts (`SYN-5973`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_artifact(
        self,
        project_id: str,
        title: str,
        kind: ArtifactKind | str,
        content: str,
        summary: str,
        *,
        origin: ArtifactOrigin | str = ArtifactOrigin.AGENT,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Create a versioned interactive artifact (`v1`) for a project or run."""
        payload: dict[str, JsonValue] = {
            "project_id": project_id,
            "title": title,
            "kind": ArtifactKind(kind).value,
            "origin": ArtifactOrigin(origin).value,
            "content": content,
            "summary": summary,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        return await self._transport.request_json(
            "POST",
            "/v1/artifacts",
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
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Append a new immutable revision (`v2`, `v3`, ...) to an existing artifact."""
        payload: dict[str, JsonValue] = {"content": content, "summary": summary}
        return await self._transport.request_json(
            "POST",
            f"/v1/artifacts/{_enc(artifact_id)}/revisions",
            HostedArtifact,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def list_artifacts(self, project_id: str, *, run_id: str | None = None) -> ArtifactList:
        """List hosted interactive artifacts for a project, optionally filtered by run."""
        return await self._transport.request_json(
            "GET",
            f"/v1/projects/{_enc(project_id)}/artifacts",
            ArtifactList,
            query={"run_id": run_id},
        )

    async def get_artifact(self, artifact_id: str) -> HostedArtifact:
        """Fetch an interactive artifact and its full revision history."""
        return await self._transport.request_json(
            "GET",
            f"/v1/artifacts/{_enc(artifact_id)}",
            HostedArtifact,
        )

    async def get_artifact_content(
        self, artifact_id: str, *, revision_id: str, digest_sha256: str
    ) -> str:
        """Fetch the verified UTF-8 source payload for an exact artifact revision."""
        return await self._transport.request_text(
            "GET",
            f"/v1/artifacts/{_enc(artifact_id)}/content",
            query={"revision_id": revision_id, "digest_sha256": digest_sha256},
        )

    async def share_artifact(
        self,
        artifact_id: str,
        *,
        scope: ArtifactShareScope | str = ArtifactShareScope.ORG,
        idempotency_key: str | None = None,
    ) -> ArtifactShareGrant:
        """Mint or update an org/link-scoped share grant for an artifact."""
        payload: dict[str, JsonValue] = {"scope": ArtifactShareScope(scope).value}
        return await self._transport.request_json(
            "POST",
            f"/v1/artifacts/{_enc(artifact_id)}/share",
            ArtifactShareGrant,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def revoke_artifact_share(self, artifact_id: str) -> ArtifactShareGrant:
        """Revoke the active share grant for an artifact."""
        return await self._transport.request_json(
            "DELETE",
            f"/v1/artifacts/{_enc(artifact_id)}/share",
            ArtifactShareGrant,
        )


class AsyncExperiments:
    """Async client for Multi-Experiment tracking (`SYN-5974`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_experiment(
        self,
        project_id: str,
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
            "project_id": project_id,
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
        return await self._transport.request_json(
            "POST",
            f"/v1/projects/{_enc(project_id)}/experiments",
            ExperimentRecord,
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
        return await self._transport.request_json(
            "POST",
            f"/v1/experiments/{_enc(experiment_id)}/revisions",
            ExperimentRecord,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Read an experiment record and its associated trial scores."""
        return await self._transport.request_json(
            "GET",
            f"/v1/experiments/{_enc(experiment_id)}",
            ExperimentRecord,
        )

    async def list_experiments(
        self, project_id: str, *, run_id: str | None = None
    ) -> ExperimentList:
        """List experiments in a project, optionally filtered to one run."""
        return await self._transport.request_json(
            "GET",
            f"/v1/projects/{_enc(project_id)}/experiments",
            ExperimentList,
            query={"run_id": run_id},
        )


class AsyncResearchLogs:
    """Async client for append-only Project & Run Research Logs (`SYN-5974`)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_log(
        self,
        project_id: str,
        *,
        title: str,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> ResearchLogSummary:
        """Initialize or register a project research log."""
        payload: dict[str, JsonValue] = {"project_id": project_id, "title": title}
        if summary is not None:
            payload["summary"] = summary
        return await self._transport.request_json(
            "POST",
            f"/v1/projects/{_enc(project_id)}/research-log/init",
            ResearchLogSummary,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def append_entry(
        self,
        project_id: str,
        category: LogCategory | str,
        text: str,
        *,
        run_id: str | None = None,
        author_kind: AuthorKind | str = AuthorKind.LOCAL_AGENT,
        author_label: str | None = None,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append a structured research-log entry with optional exact citations."""
        payload: dict[str, JsonValue] = {
            "project_id": project_id,
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
        return await self._transport.request_json(
            "POST",
            f"/v1/projects/{_enc(project_id)}/research-log",
            ResearchLogEntry,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def correct_entry(
        self,
        project_id: str,
        entry_id: str,
        text: str,
        *,
        category: LogCategory | str = LogCategory.CORRECTION,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append an explicit correction referencing a prior research-log entry."""
        payload: dict[str, JsonValue] = {
            "project_id": project_id,
            "category": LogCategory(category).value,
            "text": text,
            "correction_of": entry_id,
            "sources": [source.model_dump(mode="json") for source in sources],
            "artifact_ids": list(artifact_ids),
        }
        return await self._transport.request_json(
            "POST",
            f"/v1/projects/{_enc(project_id)}/research-log/{_enc(entry_id)}/corrections",
            ResearchLogEntry,
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def search_entries(
        self,
        project_id: str,
        *,
        query: str | None = None,
        category: LogCategory | str | None = None,
        run_id: str | None = None,
        author_kind: AuthorKind | str | None = None,
    ) -> ResearchLogPage:
        """Search project research-log entries by text, category, run, or author class."""
        return await self._transport.request_json(
            "GET",
            f"/v1/projects/{_enc(project_id)}/research-log",
            ResearchLogPage,
            query={
                "query": query,
                "category": LogCategory(category).value if category is not None else None,
                "run_id": run_id,
                "author_kind": AuthorKind(author_kind).value if author_kind is not None else None,
            },
        )

    async def export_log(
        self, project_id: str, *, run_id: str | None = None, fmt: str = "markdown"
    ) -> ResearchLogExport:
        """Export a project or run research log in markdown or JSON format."""
        return await self._transport.request_json(
            "GET",
            f"/v1/projects/{_enc(project_id)}/research-log/export",
            ResearchLogExport,
            query={"run_id": run_id, "format": fmt},
        )


class ArtifactsClient:
    """Synchronous delegate for Hosted Interactive Artifacts (`SYN-5973`)."""

    def __init__(self, runner: Runner, inner: AsyncArtifacts) -> None:
        self._runner = runner
        self._inner = inner

    def create_artifact(
        self,
        project_id: str,
        title: str,
        kind: ArtifactKind | str,
        content: str,
        summary: str,
        *,
        origin: ArtifactOrigin | str = ArtifactOrigin.AGENT,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Create a versioned interactive artifact (`v1`) for a project or run."""
        return self._runner.call(
            self._inner.create_artifact(
                project_id,
                title,
                kind,
                content,
                summary,
                origin=origin,
                run_id=run_id,
                idempotency_key=idempotency_key,
            )
        )

    def update_artifact(
        self,
        artifact_id: str,
        content: str,
        summary: str,
        *,
        idempotency_key: str | None = None,
    ) -> HostedArtifact:
        """Append a new immutable revision (`v2`, `v3`, ...) to an existing artifact."""
        return self._runner.call(
            self._inner.update_artifact(
                artifact_id, content, summary, idempotency_key=idempotency_key
            )
        )

    def list_artifacts(self, project_id: str, *, run_id: str | None = None) -> ArtifactList:
        """List hosted interactive artifacts for a project, optionally filtered by run."""
        return self._runner.call(self._inner.list_artifacts(project_id, run_id=run_id))

    def get_artifact(self, artifact_id: str) -> HostedArtifact:
        """Fetch an interactive artifact and its full revision history."""
        return self._runner.call(self._inner.get_artifact(artifact_id))

    def get_artifact_content(
        self, artifact_id: str, *, revision_id: str, digest_sha256: str
    ) -> str:
        """Fetch the verified UTF-8 source payload for an exact artifact revision."""
        return self._runner.call(
            self._inner.get_artifact_content(
                artifact_id, revision_id=revision_id, digest_sha256=digest_sha256
            )
        )

    def share_artifact(
        self,
        artifact_id: str,
        *,
        scope: ArtifactShareScope | str = ArtifactShareScope.ORG,
        idempotency_key: str | None = None,
    ) -> ArtifactShareGrant:
        """Mint or update an org/link-scoped share grant for an artifact."""
        return self._runner.call(
            self._inner.share_artifact(artifact_id, scope=scope, idempotency_key=idempotency_key)
        )

    def revoke_artifact_share(self, artifact_id: str) -> ArtifactShareGrant:
        """Revoke the active share grant for an artifact."""
        return self._runner.call(self._inner.revoke_artifact_share(artifact_id))


class ExperimentsClient:
    """Synchronous delegate for Multi-Experiment tracking (`SYN-5974`)."""

    def __init__(self, runner: Runner, inner: AsyncExperiments) -> None:
        self._runner = runner
        self._inner = inner

    def create_experiment(
        self,
        project_id: str,
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
        return self._runner.call(
            self._inner.create_experiment(
                project_id,
                run_id,
                title,
                specification,
                arms,
                hypothesis=hypothesis,
                pinned_inputs=pinned_inputs,
                stopping_criteria=stopping_criteria,
                idempotency_key=idempotency_key,
            )
        )

    def revise_experiment(
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
        return self._runner.call(
            self._inner.revise_experiment(
                experiment_id,
                title=title,
                specification=specification,
                hypothesis=hypothesis,
                arms=arms,
                stopping_criteria=stopping_criteria,
                idempotency_key=idempotency_key,
            )
        )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Read an experiment record and its associated trial scores."""
        return self._runner.call(self._inner.get_experiment(experiment_id))

    def list_experiments(self, project_id: str, *, run_id: str | None = None) -> ExperimentList:
        """List experiments in a project, optionally filtered to one run."""
        return self._runner.call(self._inner.list_experiments(project_id, run_id=run_id))


class ResearchLogsClient:
    """Synchronous delegate for append-only Project & Run Research Logs (`SYN-5974`)."""

    def __init__(self, runner: Runner, inner: AsyncResearchLogs) -> None:
        self._runner = runner
        self._inner = inner

    def create_log(
        self,
        project_id: str,
        *,
        title: str,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> ResearchLogSummary:
        """Initialize or register a project research log."""
        return self._runner.call(
            self._inner.create_log(
                project_id, title=title, summary=summary, idempotency_key=idempotency_key
            )
        )

    def append_entry(
        self,
        project_id: str,
        category: LogCategory | str,
        text: str,
        *,
        run_id: str | None = None,
        author_kind: AuthorKind | str = AuthorKind.LOCAL_AGENT,
        author_label: str | None = None,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append a structured research-log entry with optional exact citations."""
        return self._runner.call(
            self._inner.append_entry(
                project_id,
                category,
                text,
                run_id=run_id,
                author_kind=author_kind,
                author_label=author_label,
                sources=sources,
                artifact_ids=artifact_ids,
                idempotency_key=idempotency_key,
            )
        )

    def correct_entry(
        self,
        project_id: str,
        entry_id: str,
        text: str,
        *,
        category: LogCategory | str = LogCategory.CORRECTION,
        sources: Sequence[ExactReference] = (),
        artifact_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> ResearchLogEntry:
        """Append an explicit correction referencing a prior research-log entry."""
        return self._runner.call(
            self._inner.correct_entry(
                project_id,
                entry_id,
                text,
                category=category,
                sources=sources,
                artifact_ids=artifact_ids,
                idempotency_key=idempotency_key,
            )
        )

    def search_entries(
        self,
        project_id: str,
        *,
        query: str | None = None,
        category: LogCategory | str | None = None,
        run_id: str | None = None,
        author_kind: AuthorKind | str | None = None,
    ) -> ResearchLogPage:
        """Search project research-log entries by text, category, run, or author class."""
        return self._runner.call(
            self._inner.search_entries(
                project_id,
                query=query,
                category=category,
                run_id=run_id,
                author_kind=author_kind,
            )
        )

    def export_log(
        self, project_id: str, *, run_id: str | None = None, fmt: str = "markdown"
    ) -> ResearchLogExport:
        """Export a project or run research log in markdown or JSON format."""
        return self._runner.call(self._inner.export_log(project_id, run_id=run_id, fmt=fmt))
