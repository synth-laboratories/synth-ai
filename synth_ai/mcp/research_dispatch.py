"""MCP dispatch for Track B Artifacts, Experiments, Research Logs, and Run Review."""

from collections.abc import Mapping

from pydantic import BaseModel, JsonValue

from synth_ai._generated.models import (
    AppendResearchLogEntryRequest,
    CorrectResearchLogEntryRequest,
    CreateHostedArtifactRequest,
    CreateResearchExperimentRequest,
    InitResearchLogRequest,
    ReviseResearchExperimentRequest,
    ShareHostedArtifactRequest,
    UpdateHostedArtifactRequest,
)
from synth_ai.aclient import AsyncSynthClient

from ._arguments import optional, required
from .artifactory_dispatch import dispatch_artifactory

__all__ = ["dispatch_research"]


async def dispatch_research(
    client: AsyncSynthClient, operation: str, arguments: Mapping[str, JsonValue]
) -> BaseModel | None:
    match operation:
        case "createArtifact":
            body = required(arguments, "body", CreateHostedArtifactRequest)
            return await client.artifacts.create_artifact(
                body.notebook_id,
                body.title,
                body.kind.value,
                body.content,
                body.summary,
                origin=body.origin.value if body.origin is not None else "agent",
                run_id=body.run_id,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getArtifact":
            return await client.artifacts.get_artifact(required(arguments, "artifact_id", str))
        case "updateArtifact":
            body = required(arguments, "body", UpdateHostedArtifactRequest)
            return await client.artifacts.update_artifact(
                required(arguments, "artifact_id", str),
                body.content,
                body.summary,
                author_kind=body.author_kind.value if body.author_kind is not None else None,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "getArtifactContent":
            return await client.artifacts.get_artifact_content_inline(
                required(arguments, "artifact_id", str),
                revision_id=optional(arguments, "revision_id", str),
                digest_sha256=optional(arguments, "digest_sha256", str),
            )
        case "shareArtifact":
            body = required(arguments, "body", ShareHostedArtifactRequest)
            return await client.artifacts.share_artifact(
                required(arguments, "artifact_id", str),
                scope=body.scope.value,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "revokeArtifactShare":
            return await client.artifacts.revoke_artifact_share(
                required(arguments, "artifact_id", str)
            )
        case "listNotebookArtifacts":
            return await client.artifacts.list_artifacts(
                required(arguments, "notebook_id", str),
                run_id=optional(arguments, "run_id", str),
            )
        case "createExperiment":
            body = required(arguments, "body", CreateResearchExperimentRequest)
            return await client.experiments.create_experiment(
                required(arguments, "notebook_id", str),
                body.run_id,
                body.title,
                body.specification,
                body.arms,
                hypothesis=body.hypothesis,
                pinned_inputs=body.pinned_inputs or (),
                stopping_criteria=body.stopping_criteria,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "listNotebookExperiments":
            return await client.experiments.list_experiments(
                required(arguments, "notebook_id", str),
                run_id=optional(arguments, "run_id", str),
            )
        case "getExperiment":
            return await client.experiments.get_experiment(
                required(arguments, "experiment_id", str)
            )
        case "reviseExperiment":
            body = required(arguments, "body", ReviseResearchExperimentRequest)
            return await client.experiments.revise_experiment(
                required(arguments, "experiment_id", str),
                title=body.title,
                specification=body.specification,
                hypothesis=body.hypothesis,
                arms=body.arms,
                stopping_criteria=body.stopping_criteria,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "initResearchLog":
            body = required(arguments, "body", InitResearchLogRequest)
            return await client.research_logs.create_log(
                required(arguments, "notebook_id", str),
                title=body.title,
                summary=body.summary,
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "appendResearchLogEntry":
            body = required(arguments, "body", AppendResearchLogEntryRequest)
            return await client.research_logs.append_entry(
                required(arguments, "notebook_id", str),
                body.category.value,
                body.text,
                run_id=body.run_id,
                author_kind=body.author_kind.value
                if body.author_kind is not None
                else "local_agent",
                author_label=body.author_label,
                correction_of=body.correction_of,
                sources=body.sources or (),
                artifact_ids=body.artifact_ids or (),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "searchResearchLog":
            return await client.research_logs.search_entries(
                required(arguments, "notebook_id", str),
                query=optional(arguments, "query", str),
                category=optional(arguments, "category", str),
                run_id=optional(arguments, "run_id", str),
                author_kind=optional(arguments, "author_kind", str),
            )
        case "correctResearchLogEntry":
            body = required(arguments, "body", CorrectResearchLogEntryRequest)
            return await client.research_logs.correct_entry(
                required(arguments, "notebook_id", str),
                required(arguments, "entry_id", str),
                body.text,
                run_id=body.run_id,
                sources=body.sources or (),
                artifact_ids=body.artifact_ids or (),
                idempotency_key=optional(arguments, "idempotency_key", str),
            )
        case "exportResearchLog":
            return await client.research_logs.export_log(
                required(arguments, "notebook_id", str),
                run_id=optional(arguments, "run_id", str),
                fmt=optional(arguments, "format", str) or "markdown",
            )
        case "getRunReview":
            return await client.swarms.review(required(arguments, "run_id", str))
        case _:
            return await dispatch_artifactory(client, operation, arguments)
