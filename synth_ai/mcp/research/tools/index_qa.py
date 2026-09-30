"""Local Index MCP tools for private QA cases and Contribution lifecycle.

Each tool maps to exactly one SDK operation, carries one least-privilege scope tag
(read / intake / review / coordinate / qa:read / publish; none implies another;
legacy index:write means intake + account only) and reports the
exact resulting case version, revision or status. QA acceptance is private
coordination: it is not certified science, public release, or a reward. Hosted
Index routes for QA are rollout-gated and may be absent; that surfaces as a typed
error, never as a local fallback. Bytes come only from the server for an assigned
case; nothing here reads the local disk.
"""

import base64
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Annotated
from uuid import UUID

from pydantic import Field, StrictInt
from synth_ai.core.errors import (
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
)
from synth_ai.mcp.research.registry import (
    INDEX_INTAKE_SCOPES,
    JSONDict,
    ToolDefinition,
)
from synth_ai.sdk.index.client import IndexAPI
from synth_ai.sdk.index.contracts import Identifier, IndexContract
from synth_ai.sdk.index.lifecycle import PublicationSpec, RevisionCreateSpec, WithdrawalSpec
from synth_ai.sdk.index.qa import (
    AcceptAssignmentSpec,
    AdjudicationSpec,
    AppealSpec,
    AssignmentSpec,
    CaseAction,
    CaseEventSpec,
    CreateCaseSpec,
    EscalationSpec,
    InternalNoteSpec,
)
from synth_ai.sdk.index.qa_checks import RecordCheckSpec
from synth_ai.sdk.index.qa_preflight import RunPreflightSpec
from synth_ai.sdk.index.qa_reviews import RecordReviewSpec
from synth_ai.sdk.index.scopes import required_scopes as operation_scopes

QaClientFactory = Callable[[], AbstractContextManager[object]]

QA_READ_TOOL_NAMES: tuple[str, ...] = (
    "index_qa_case",
    "index_qa_events",
    "index_qa_assignments",
    "index_qa_checks",
    "index_qa_reviews",
    "index_qa_package",
    "index_qa_asset",
)
QA_MUTATING_TOOL_NAMES: tuple[str, ...] = (
    "index_qa_case_create",
    "index_qa_contributor_event",
    "index_contribution_revise",
    "index_contribution_withdraw",
    "index_qa_assignment_accept",
    "index_qa_reviewer_event",
    "index_qa_check_record",
    "index_qa_preflight",
    "index_qa_secret_scan",
    "index_qa_review_record",
    "index_qa_invite_reviewer",
    "index_qa_assignment_revoke",
    "index_qa_adjudicate",
    "index_qa_appeal",
    "index_qa_escalate",
    "index_qa_internal_note",
    "index_contribution_publish",
)

_ASSET_MAX_BYTES = 4 * 1024 * 1024
_KEY = Field(min_length=1, max_length=119, pattern=r"^[a-zA-Z0-9_.-]+$")
_Text = Annotated[str, Field(min_length=1, max_length=4096)]
_Version = Annotated[StrictInt, Field(ge=0)]

CONTRIBUTOR_ACTIONS = (CaseAction.MESSAGE, CaseAction.RESPOND)
REVIEWER_ACTIONS = (
    CaseAction.MESSAGE,
    CaseAction.REQUEST_CHANGES,
    CaseAction.APPROVE,
    CaseAction.REJECT,
)


class IndexToolError(SynthError):
    """Typed local refusal with a stable code; never a silent fallback."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: SynthErrorCategory = SynthErrorCategory.VALIDATION,
    ) -> None:
        failure = SynthFailure(
            code=SynthErrorCode(code),
            category=category,
            operation=None,
            request_id=None,
            correlation_id=None,
            retry=RetryDirective(retryable=False),
            status=None,
            detail=message,
        )
        super().__init__(message, failure=failure)


class CaseRequest(IndexContract):
    case_id: UUID


class CasePageRequest(CaseRequest):
    after: Annotated[StrictInt, Field(ge=0)] = 0


class AssetRequest(CaseRequest):
    asset_id: Identifier


class CaseCreateRequest(IndexContract):
    spec: CreateCaseSpec


class ActionRequest(CaseRequest):
    action: CaseAction
    expected_version: _Version = Field(
        description="Case version you last read; a stale version is refused, never merged."
    )
    message: _Text
    idempotency_key: str = _KEY


class FencedRequest(CaseRequest):
    """Names the exact case version, sealed manifest digest and rubric version."""

    expected_version: _Version
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rubric_version: Identifier
    message: _Text
    idempotency_key: str = _KEY


class ReviseRequest(IndexContract):
    contribution_id: Identifier
    spec: RevisionCreateSpec
    idempotency_key: str = _KEY


class WithdrawRequest(IndexContract):
    contribution_id: Identifier
    spec: WithdrawalSpec
    idempotency_key: str = _KEY


class PublishRequest(IndexContract):
    contribution_id: Identifier
    spec: PublicationSpec
    idempotency_key: str = _KEY


class AcceptRequest(IndexContract):
    assignment_id: UUID
    spec: AcceptAssignmentSpec


class RevokeRequest(IndexContract):
    assignment_id: UUID


class InviteRequest(CaseRequest):
    spec: AssignmentSpec


class CheckRecordRequest(CaseRequest):
    spec: RecordCheckSpec
    idempotency_key: str = _KEY


class PreflightRequest(CaseRequest):
    spec: RunPreflightSpec


class ReviewRecordRequest(CaseRequest):
    spec: RecordReviewSpec
    idempotency_key: str = _KEY


def _require_keyed(client: object) -> IndexAPI:
    if not isinstance(client, IndexAPI):
        raise IndexToolError(
            "index_credential_required",
            "This Index tool needs an authorized API credential; no anonymous fallback exists.",
            category=SynthErrorCategory.AUTHENTICATION,
        )
    return client


def _json(model: object) -> JSONDict:
    return model.model_dump(mode="json")  # type: ignore[attr-defined]


def build_qa_tools(client_factory: QaClientFactory) -> list[ToolDefinition]:
    def call(operation: Callable[[IndexAPI], object]) -> object:
        with client_factory() as client:
            return operation(_require_keyed(client))

    def case(arguments: JSONDict) -> JSONDict:
        request = CaseRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.case(request.case_id)))

    def events(arguments: JSONDict) -> JSONDict:
        request = CasePageRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.events(request.case_id, after=request.after)))

    def assignments(_arguments: JSONDict) -> JSONDict:
        listed = call(lambda c: c.qa.assignments())
        return {"assignments": [_json(item) for item in listed]}  # type: ignore[union-attr]

    def checks(arguments: JSONDict) -> JSONDict:
        request = CasePageRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.checks(request.case_id, after=request.after)))

    def reviews(arguments: JSONDict) -> JSONDict:
        request = CasePageRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.reviews(request.case_id, after=request.after)))

    def package(arguments: JSONDict) -> JSONDict:
        request = CaseRequest.model_validate(arguments)

        def read(c: IndexAPI) -> object:
            return c.qa.package(c.qa.case(request.case_id))

        return _json(call(read))

    def asset(arguments: JSONDict) -> JSONDict:
        request = AssetRequest.model_validate(arguments)

        def read(c: IndexAPI) -> JSONDict:
            sealed = c.qa.case(request.case_id)
            declared = {a.asset_id: a for a in c.qa.package(sealed).assets}.get(request.asset_id)
            if declared is None:
                raise IndexToolError(
                    "index_qa_asset_undeclared", "Asset is not declared by the sealed package."
                )
            size = declared.object.size_bytes
            if size > _ASSET_MAX_BYTES:
                raise IndexToolError(
                    "index_qa_asset_too_large",
                    f"Asset is {size} bytes; the tool bound is {_ASSET_MAX_BYTES}.",
                )
            body = c.qa.asset(
                request.case_id,
                request.asset_id,
                digest_sha256=declared.object.digest_sha256,
                size_bytes=size,
            )
            return {
                "case_id": str(request.case_id),
                "asset_id": request.asset_id,
                "size_bytes": size,
                "digest_sha256": declared.object.digest_sha256,
                "content_base64": base64.b64encode(body).decode("ascii"),
            }

        return call(read)  # type: ignore[return-value]

    def case_create(arguments: JSONDict) -> JSONDict:
        request = CaseCreateRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.create_case(request.spec)))

    def event_for(allowed: tuple[CaseAction, ...]) -> Callable[[JSONDict], JSONDict]:
        def handler(arguments: JSONDict) -> JSONDict:
            request = ActionRequest.model_validate(arguments)
            if request.action not in allowed:
                raise IndexToolError(
                    "index_qa_action_not_permitted_by_tool",
                    f"{request.action.value} is not available through this tool; allowed: "
                    + ", ".join(action.value for action in allowed),
                    category=SynthErrorCategory.AUTHORIZATION,
                )
            spec = CaseEventSpec(
                expected_version=request.expected_version,
                action=request.action,
                message=request.message,
            )
            return _json(
                call(
                    lambda c: c.qa.append_event(
                        request.case_id, spec, idempotency_key=request.idempotency_key
                    )
                )
            )

        return handler

    def fenced(helper: str, spec_type: type) -> Callable[[JSONDict], JSONDict]:
        def handler(arguments: JSONDict) -> JSONDict:
            request = FencedRequest.model_validate(arguments)
            spec = spec_type(
                expected_version=request.expected_version,
                manifest_digest=request.manifest_digest,
                rubric_version=request.rubric_version,
                message=request.message,
            )
            return _json(
                call(
                    lambda c: getattr(c.qa, helper)(
                        request.case_id, spec, idempotency_key=request.idempotency_key
                    )
                )
            )

        return handler

    def revise(arguments: JSONDict) -> JSONDict:
        request = ReviseRequest.model_validate(arguments)
        return _json(
            call(
                lambda c: c.contributions.revisions.create(
                    request.contribution_id,
                    request.spec,
                    idempotency_key=request.idempotency_key,
                )
            )
        )

    def withdraw(arguments: JSONDict) -> JSONDict:
        request = WithdrawRequest.model_validate(arguments)
        return _json(
            call(
                lambda c: c.contributions.withdraw(
                    request.contribution_id, request.spec, idempotency_key=request.idempotency_key
                )
            )
        )

    def publish(arguments: JSONDict) -> JSONDict:
        request = PublishRequest.model_validate(arguments)
        return _json(
            call(
                lambda c: c.contributions.publish(
                    request.contribution_id, request.spec, idempotency_key=request.idempotency_key
                )
            )
        )

    def accept(arguments: JSONDict) -> JSONDict:
        request = AcceptRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.accept_assignment(request.assignment_id, request.spec)))

    def revoke(arguments: JSONDict) -> JSONDict:
        request = RevokeRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.revoke_assignment(request.assignment_id)))

    def invite(arguments: JSONDict) -> JSONDict:
        request = InviteRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.invite_reviewer(request.case_id, request.spec)))

    def check_record(arguments: JSONDict) -> JSONDict:
        request = CheckRecordRequest.model_validate(arguments)
        return _json(
            call(
                lambda c: c.qa.record_check(
                    request.case_id, request.spec, idempotency_key=request.idempotency_key
                )
            )
        )

    def preflight(arguments: JSONDict) -> JSONDict:
        request = PreflightRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.preflight(request.case_id, request.spec)))

    def secret_scan(arguments: JSONDict) -> JSONDict:
        request = PreflightRequest.model_validate(arguments)
        return _json(call(lambda c: c.qa.secret_scan(request.case_id, request.spec)))

    def review_record(arguments: JSONDict) -> JSONDict:
        request = ReviewRecordRequest.model_validate(arguments)
        return _json(
            call(
                lambda c: c.qa.record_review(
                    request.case_id, request.spec, idempotency_key=request.idempotency_key
                )
            )
        )

    fence_note = (
        " Requires expected_version, the case's sealed manifest_digest and rubric_version "
        "(a stale or mismatched fence is refused, never merged) and an idempotency key."
    )
    private_note = (
        " QA state is private coordination: it is not certified science, public release "
        "or a reward. Reports the exact resulting case version."
    )

    def tool(name, description, schema, handler, scopes, *, any_of_scopes=()) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            input_schema=schema.model_json_schema(),
            handler=handler,
            required_scopes=scopes if any_of_scopes else (),
            any_of_scopes=any_of_scopes or scopes,
        )

    write = INDEX_INTAKE_SCOPES
    # Tags are the SDK class table (sdk/index/scopes.py), asserted equal to the
    # backend table. Any-of, least privilege first; the backend still decides
    # role, assignment and ownership per case.
    case_get = operation_scopes("index.qa.cases.get")
    events_list = operation_scopes("index.qa.events.list")
    assignments_list = operation_scopes("index.qa.assignments.list")
    checks_list = operation_scopes("index.qa.checks.list")
    reviews_list = operation_scopes("index.qa.reviews.list")
    qa_read = operation_scopes("index.qa.package.retrieve")
    case_create_scopes = operation_scopes("index.qa.cases.create")
    events_create = operation_scopes("index.qa.events.create")
    appeal_scopes = operation_scopes("index.qa.appeals.create")
    escalate_scopes = operation_scopes("index.qa.escalations.create")
    note_scopes = operation_scopes("index.qa.notes.create")
    adjudicate_scopes = operation_scopes("index.qa.adjudications.create")
    accept_scopes = operation_scopes("index.qa.assignments.accept")
    invite_scopes = operation_scopes("index.qa.assignments.create")
    revoke_scopes = operation_scopes("index.qa.assignments.revoke")
    check_record_scopes = operation_scopes("index.qa.checks.record")
    preflight_scopes = operation_scopes("index.qa.checks.preflight")
    secret_scan_scopes = operation_scopes("index.qa.checks.secret_scan")
    review_record_scopes = operation_scopes("index.qa.reviews.record")
    publish_scopes = operation_scopes("index.contributions.publication.create")
    withdraw_scopes = operation_scopes("index.contributions.withdrawal.create")

    class NoArguments(IndexContract):
        pass

    return [
        tool(
            "index_qa_case",
            "Read one QA case: state, version, exact revision and manifest digest (404 when hidden)."
            + private_note,
            CaseRequest,
            case,
            case_get,
        ),
        tool(
            "index_qa_events",
            "Page the visible QA conversation (messages, findings, decisions) after a sequence cursor; reuse next_after.",
            CasePageRequest,
            events,
            events_list,
        ),
        tool(
            "index_qa_assignments",
            "List your reviewer invitations (bounded); accept one with index_qa_assignment_accept.",
            NoArguments,
            assignments,
            assignments_list,
        ),
        tool(
            "index_qa_checks",
            "Page recorded QA check attempts and findings for a case.",
            CasePageRequest,
            checks,
            checks_list,
        ),
        tool(
            "index_qa_reviews",
            "Page recorded reviewer recommendations for a case.",
            CasePageRequest,
            reviews,
            reviews_list,
        ),
        tool(
            "index_qa_package",
            "Read the exact sealed package of the case revision you are assigned to review. Refused for unassigned, expired or revoked assignments. Also reads case metadata, so the token needs a case-read scope (intake, review or coordinate) besides index:qa:read.",
            CaseRequest,
            package,
            qa_read,
            any_of_scopes=case_get,
        ),
        tool(
            "index_qa_asset",
            f"Read one declared asset of the assigned package, at most {_ASSET_MAX_BYTES} bytes, verified against the sealed size and SHA256 before return (base64).",
            AssetRequest,
            asset,
            qa_read,
            any_of_scopes=case_get,
        ),
        tool(
            "index_qa_case_create",
            "Open a QA case for one exact submitted revision and manifest digest. Does not review, approve or publish."
            + private_note,
            CaseCreateRequest,
            case_create,
            case_create_scopes,
        ),
        tool(
            "index_qa_contributor_event",
            "As the contributor, message or respond to requested changes. Appeals use index_qa_appeal. Requires expected_version and an idempotency key; reuse the key after an uncertain response."
            + private_note,
            ActionRequest,
            event_for(CONTRIBUTOR_ACTIONS),
            events_create,
        ),
        tool(
            "index_contribution_revise",
            "Repair after review: open a private child revision draft of a parent revision you own. Upload and submit it separately; the parent stays immutable.",
            ReviseRequest,
            revise,
            write,
        ),
        tool(
            "index_contribution_withdraw",
            "Withdraw your Contribution from Search and new reads. Prior downloads cannot be recalled. Reports the resulting generation and status.",
            WithdrawRequest,
            withdraw,
            withdraw_scopes,
        ),
        tool(
            "index_qa_assignment_accept",
            "Accept your reviewer assignment, declaring conflict-freedom and provenance (human, agent_assisted or agent). A false declaration is recorded, not hidden.",
            AcceptRequest,
            accept,
            accept_scopes,
        ),
        tool(
            "index_qa_reviewer_event",
            "As an assigned reviewer, message, request changes, approve or reject (escalation uses index_qa_escalate). Approval is private QA acceptance only, not publication.",
            ActionRequest,
            event_for(REVIEWER_ACTIONS),
            events_create,
        ),
        tool(
            "index_qa_check_record",
            "Record one QA check attempt with typed findings (idempotent).",
            CheckRecordRequest,
            check_record,
            check_record_scopes,
        ),
        tool(
            "index_qa_preflight",
            "Run the server-side preflight checks for a case run.",
            PreflightRequest,
            preflight,
            preflight_scopes,
        ),
        tool(
            "index_qa_secret_scan",
            "Run the server-side secret scan for a case run.",
            PreflightRequest,
            secret_scan,
            secret_scan_scopes,
        ),
        tool(
            "index_qa_review_record",
            "Record a criterion-by-criterion review recommendation for the exact case revision (idempotent). A reviewer cannot self-approve their own Contribution.",
            ReviewRecordRequest,
            review_record,
            review_record_scopes,
        ),
        tool(
            "index_qa_invite_reviewer",
            "Coordinator: invite a named reviewer (user and org) with an expiry.",
            InviteRequest,
            invite,
            invite_scopes,
        ),
        tool(
            "index_qa_assignment_revoke",
            "Coordinator: revoke a reviewer assignment; revocation applies to in-flight reads.",
            RevokeRequest,
            revoke,
            revoke_scopes,
        ),
        tool(
            "index_qa_adjudicate",
            "Coordinator: independent decision on an escalated or appealed case; the only outcome is reopening fresh independent review (never approval)."
            + fence_note
            + private_note,
            FencedRequest,
            fenced("adjudicate", AdjudicationSpec),
            adjudicate_scopes,
        ),
        tool(
            "index_qa_appeal",
            "Contributor: appeal a rejection or private acceptance for independent adjudication. An appeal is not an approval."
            + fence_note
            + private_note,
            FencedRequest,
            fenced("appeal", AppealSpec),
            appeal_scopes,
        ),
        tool(
            "index_qa_escalate",
            "Contributor or reviewer: escalate the case to a coordinator."
            + fence_note
            + private_note,
            FencedRequest,
            fenced("escalate", EscalationSpec),
            escalate_scopes,
        ),
        tool(
            "index_qa_internal_note",
            "Reviewer/coordinator: add an internal note that is never delivered to the contributor."
            + fence_note
            + private_note,
            FencedRequest,
            fenced("add_internal_note", InternalNoteSpec),
            note_scopes,
        ),
        tool(
            "index_contribution_publish",
            "Publisher: publish an independently approved revision to its sealed audience. Requires a separate publisher grant plus rights and consent; QA acceptance or index:write never grants it. Reports the resulting generation, current revision and status.",
            PublishRequest,
            publish,
            publish_scopes,
        ),
    ]
