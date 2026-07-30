"""Accepted-decision-only public Delivery contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class _DeliveryContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    @classmethod
    def from_wire(cls, value: object):
        return cls.model_validate(value)

    def to_wire(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class DeliveryState(StrEnum):
    QUEUED = "queued"
    ADMITTING = "admitting"
    PLACING = "placing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_CI = "waiting_for_ci"
    REQUIRES_ATTENTION = "requires_attention"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in {
            DeliveryState.REQUIRES_ATTENTION,
            DeliveryState.SUCCEEDED,
            DeliveryState.FAILED,
            DeliveryState.CANCELED,
        }


class PullRequestState(StrEnum):
    OPEN = "open"
    MERGED = "merged"
    CLOSED = "closed"


class DeliveryAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    READY = "ready"
    MERGED = "merged"
    CLOSED = "closed"


class Delivery(_DeliveryContract):
    delivery_id: Identifier = Field(validation_alias="session_id")
    task_id: Identifier
    objective: str = Field(min_length=1)
    actor_id: Identifier
    originating_run_id: Identifier
    decision_id: Identifier
    repository_registration_id: Identifier
    revision_admission_id: Identifier
    state: DeliveryState
    current_workspace_revision_id: str | None = None
    current_environment_lease_id: str | None = None
    current_head_sha: GitSha | None = None
    codex_thread_id: str | None = None
    created_at: datetime
    updated_at: datetime
    last_event_cursor: int = Field(ge=0)


class PullRequestBinding(_DeliveryContract):
    contract_version: Literal["internal-code-factory.github-delivery.v1"]
    pull_request_binding_id: Identifier
    session_id: Identifier
    repository_registration_id: Identifier
    github_repository_id: int = Field(ge=1)
    pr_number: int = Field(ge=1)
    pr_node_id: str = Field(min_length=1)
    url: AnyHttpUrl
    base_ref: str = Field(min_length=1)
    head_ref: str = Field(min_length=1)
    draft: bool
    state: PullRequestState
    delivery_actor_id: Identifier
    credential_reference_id: Identifier
    policy_version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    current_delivery_revision_id: Identifier


class DeliveryRevision(_DeliveryContract):
    delivery_revision_id: Identifier
    pull_request_binding_id: Identifier
    request_id: Identifier
    idempotency_key: Identifier
    workspace_revision_id: Identifier
    base_sha: GitSha
    head_sha: GitSha
    evidence_digest: Sha256Digest
    action: DeliveryAction
    provider_observed_head_sha: GitSha
    provider_updated_at: datetime
    created_at: datetime


class GitHubObservation(_DeliveryContract):
    github_observation_id: Identifier
    pull_request_binding_id: Identifier
    session_id: Identifier
    observed_head_sha: GitSha
    draft: bool
    state: PullRequestState
    observed_at: datetime
    payload_digest: Sha256Digest


class DeliveryReceipt(_DeliveryContract):
    contract_version: Literal["internal-code-factory.milestone2.v1"]
    receipt_version: Literal["internal-code-factory.milestone2-receipt.v2"]
    event_contract_version: Literal["internal-code-factory.event.v2"]
    receipt_id: Identifier
    receipt_digest: Sha256Digest
    task_id: Identifier
    originating_run_id: Identifier
    decision_id: Identifier
    delivery: Delivery = Field(validation_alias="session")
    actor_id: Identifier
    pull_request_binding: PullRequestBinding
    delivery_revisions: list[DeliveryRevision] = Field(min_length=1)
    github_observations: list[GitHubObservation] = Field(min_length=1)
    foreign_resources_touched: Literal[False]
    fixture_terminal_action: None
    emitted_at: datetime

    @model_validator(mode="after")
    def validate_exact_open_draft(self) -> DeliveryReceipt:
        delivery = self.delivery
        binding = self.pull_request_binding
        if delivery.state is not DeliveryState.SUCCEEDED:
            raise ValueError("Delivery authority receipt requires a succeeded Delivery")
        if (
            self.task_id != delivery.task_id
            or self.originating_run_id != delivery.originating_run_id
            or self.decision_id != delivery.decision_id
            or self.actor_id != delivery.actor_id
            or binding.session_id != delivery.delivery_id
            or binding.repository_registration_id != delivery.repository_registration_id
            or binding.delivery_actor_id != delivery.actor_id
        ):
            raise ValueError("Delivery authority identities drifted")
        if not binding.draft or binding.state is not PullRequestState.OPEN:
            raise ValueError("public Delivery must remain an open draft PR")
        revisions = {
            revision.delivery_revision_id: revision for revision in self.delivery_revisions
        }
        if len(revisions) != len(self.delivery_revisions):
            raise ValueError("Delivery receipt contains duplicate revision IDs")
        if any(
            revision.pull_request_binding_id != binding.pull_request_binding_id
            for revision in self.delivery_revisions
        ):
            raise ValueError("Delivery revision crossed its PR binding")
        if any(
            revision.action in {DeliveryAction.READY, DeliveryAction.MERGED, DeliveryAction.CLOSED}
            for revision in self.delivery_revisions
        ):
            raise ValueError("SDK cannot accept ready/merge/close evidence")
        current = revisions.get(binding.current_delivery_revision_id)
        if (
            current is None
            or delivery.current_head_sha is None
            or delivery.current_workspace_revision_id is None
            or current.workspace_revision_id != delivery.current_workspace_revision_id
            or current.head_sha != delivery.current_head_sha
            or current.provider_observed_head_sha != current.head_sha
        ):
            raise ValueError("Delivery current revision does not bind its exact head")
        if any(
            observation.pull_request_binding_id != binding.pull_request_binding_id
            or observation.session_id != delivery.delivery_id
            for observation in self.github_observations
        ):
            raise ValueError("GitHub observation crossed its Delivery binding")
        latest = max(self.github_observations, key=lambda item: item.observed_at)
        if (
            latest.observed_head_sha != current.head_sha
            or not latest.draft
            or latest.state is not PullRequestState.OPEN
        ):
            raise ValueError("latest GitHub observation is not the exact open draft head")
        return self


class EnsureDraftDeliveryRequest(_DeliveryContract):
    decision_id: Identifier
    idempotency_key: Identifier


class DraftDeliveryAuthorityResponse(_DeliveryContract):
    delivery: Delivery
    pull_request_binding: PullRequestBinding
    authority_receipt: DeliveryReceipt

    @model_validator(mode="after")
    def validate_authority_join(self) -> DraftDeliveryAuthorityResponse:
        if self.authority_receipt.delivery != self.delivery:
            raise ValueError("Delivery response does not match its authority receipt")
        if self.authority_receipt.pull_request_binding != self.pull_request_binding:
            raise ValueError("PR binding does not match its authority receipt")
        return self


__all__ = [
    "Delivery",
    "DeliveryAction",
    "DeliveryReceipt",
    "DeliveryRevision",
    "DeliveryState",
    "DraftDeliveryAuthorityResponse",
    "EnsureDraftDeliveryRequest",
    "GitHubObservation",
    "PullRequestBinding",
    "PullRequestState",
]
