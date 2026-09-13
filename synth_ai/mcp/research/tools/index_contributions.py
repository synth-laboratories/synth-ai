"""Opt-in deliberate Contribution lifecycle tools; no file discovery or publication.

See sibling docs/drafts/synth-index-agent-integration-2026-09-12.md.
Each invocation performs exactly one canonical SDK operation. Transfer of bytes,
qualification, public release, rewards and status reads are not implemented here.
"""

from pydantic import Field
from synth_ai.mcp.research.registry import WRITE_SCOPES, JSONDict, ToolDefinition
from synth_ai.sdk.index.contracts import ContributionReference, IndexContract
from synth_ai.sdk.index.contributions import (
    ContributionDraft,
    ContributionUploadPrepared,
    ContributionUploadSpec,
)
from synth_ai.sdk.index.submission import ContributionSubmitSpec

from .index import IndexClientFactory

INDEX_CONTRIBUTION_TOOL_NAMES = frozenset(
    {
        "research_index_contribution_create",
        "research_index_contribution_prepare_upload",
        "research_index_contribution_finalize",
        "research_index_contribution_submit",
    }
)


class ContributionCreateRequest(IndexContract):
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.-]+$")


class ContributionPrepareRequest(IndexContract):
    draft: ContributionDraft
    upload: ContributionUploadSpec


class ContributionFinalizeRequest(IndexContract):
    draft: ContributionDraft
    prepared: ContributionUploadPrepared


class ContributionSubmitRequest(IndexContract):
    reference: ContributionReference
    submission: ContributionSubmitSpec


def build_index_contribution_tools(client_factory: IndexClientFactory) -> list[ToolDefinition]:
    """Build write-scoped tools without discovering files, credentials or clients.

    See sibling docs/drafts/synth-index-agent-integration-2026-09-12.md.
    The host must explicitly opt in and enforce declared scopes; backend auth is
    still authoritative. Do not infer write permission from retrieval enablement.
    """

    def create(arguments: JSONDict) -> JSONDict:
        request = ContributionCreateRequest.model_validate(arguments)
        with client_factory() as client:
            return client.contributions.create(idempotency_key=request.idempotency_key).model_dump(
                mode="json"
            )

    def prepare_upload(arguments: JSONDict) -> JSONDict:
        request = ContributionPrepareRequest.model_validate(arguments)
        with client_factory() as client:
            return client.contributions.prepare_upload(request.draft, request.upload).model_dump(
                mode="json"
            )

    def finalize(arguments: JSONDict) -> JSONDict:
        request = ContributionFinalizeRequest.model_validate(arguments)
        with client_factory() as client:
            return client.contributions.finalize(request.draft, request.prepared).model_dump(
                mode="json"
            )

    def submit(arguments: JSONDict) -> JSONDict:
        request = ContributionSubmitRequest.model_validate(arguments)
        with client_factory() as client:
            return client.contributions.submit(request.reference, request.submission).model_dump(
                mode="json"
            )

    return [
        ToolDefinition(
            name="research_index_contribution_create",
            description="Create a private Contribution draft only when the user requests contribution. Supply and preserve an explicit idempotency key for retries. Does not select files, upload, submit, publish or award credits.",
            input_schema=ContributionCreateRequest.model_json_schema(),
            handler=create,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_index_contribution_prepare_upload",
            description="Prepare signed upload instructions for an explicitly supplied Contribution package and server-issued private draft. No files are read or transferred. Treat signed URLs as sensitive and transient. Preserve the publication ID for retries; this does not finalize, submit or publish.",
            input_schema=ContributionPrepareRequest.model_json_schema(),
            handler=prepare_upload,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_index_contribution_finalize",
            description="Finalize (seal) one prepared Contribution publication after separately transferring the declared bytes. The backend verifies uploaded objects. This does not submit for review, change audience, publish or award credits.",
            input_schema=ContributionFinalizeRequest.model_json_schema(),
            handler=finalize,
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="research_index_contribution_submit",
            description="Deliberately submit one exact Contribution revision and finalized publication for qualification. Reuse the same publication ID after uncertain failures. Submission is not approval or public release and does not award credits.",
            input_schema=ContributionSubmitRequest.model_json_schema(),
            handler=submit,
            required_scopes=WRITE_SCOPES,
        ),
    ]
