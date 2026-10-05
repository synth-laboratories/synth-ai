"""Typed contracts for Intern identity, Task grants and project Sublinear Task reads.

Backend remains the contract authority. Each model mirrors one backend wire
shape exactly (no SDK-side defaults or widening):

* identities -- ``backend packages/intern/identity_api.py``
  (``intern.identity_provision_request.v1`` / ``..._receipt.v1`` /
  ``intern.identity_selection.v1`` / ``intern.identity_catalog_page.v1``);
* Task grants -- ``backend packages/intern/task_grants.py`` inputs and the
  ``synth.sublinear-task-grant.v1`` document returned by
  ``public.intern_task_grant_declare`` / ``_revoke`` /
  ``backend_context_bind_grant_declare`` (alembic 20261012/20261014/20261018);
* project Sublinear Tasks -- ``backend app/api/v1/managed_research/collaboration.py``
  over ``services/sublinear/types.py`` ``SublinearIssue`` / ``SublinearComment``.

All models forbid unknown fields and validated with the same closed
vocabularies and ordering rules the backend enforces, so an invalid request
fails locally before any HTTP effect. Responses forbid unknown fields too: an
unknown field is a contract change, not something to ignore.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

_PRINTABLE_ID = r"^[!-~]+$"
_LOWER_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_wire(cls, value: object) -> Self:
        return cls.model_validate(value)


def _canonical_uuid(value: str, *, code: str, allow_nil: bool = True) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(code) from error
    if str(parsed) != value or (not allow_nil and parsed.int == 0):
        raise ValueError(code)
    return value


# -- Identities -------------------------------------------------------------


class InternIdentityProvisionKind(StrEnum):
    ENSURE_DEFAULT = "ensure_default"
    CREATE_ADDITIONAL = "create_additional"


class InternIdentityProvisionRequest(_Contract):
    """``POST /smr/research-intern/identities``; ``command_id`` is the idempotency key."""

    schema_version: Literal["intern.identity_provision_request.v1"] = (
        "intern.identity_provision_request.v1"
    )
    command_id: str = Field(min_length=1, max_length=256, pattern=_PRINTABLE_ID)
    kind: InternIdentityProvisionKind
    display_name: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def _storage_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("intern_provisioning_display_name_invalid")
        return value


class InternIdentity(_Contract):
    org_id: str
    research_intern_id: str
    is_default: StrictBool

    @field_validator("org_id", "research_intern_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        return _canonical_uuid(value, code="intern_identity_uuid_invalid")


class InternIdentityProvisionReceipt(_Contract):
    schema_version: Literal["intern.identity_provision_receipt.v1"]
    org_id: str
    command_id: str
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity: InternIdentity


class InternIdentitySelection(_Contract):
    schema_version: Literal["intern.identity_selection.v1"]
    identity: InternIdentity


class InternIdentityCatalogPage(_Contract):
    schema_version: Literal["intern.identity_catalog_page.v1"]
    org_id: str
    identities: tuple[InternIdentity, ...] = Field(max_length=32)
    after_intern_id: str | None


# -- Task grants ------------------------------------------------------------


class InternTaskOperation(StrEnum):
    """Closed ``TASK_OPERATIONS`` vocabulary (backend packages/intern/task_grants.py)."""

    COMMAND_CREATE = "task.command.create"
    COMMAND_SET_DEPENDENCIES = "task.command.set_dependencies"
    COMMAND_SET_PRIORITY = "task.command.set_priority"
    COMMAND_ADOPT = "task.command.adopt"
    COMMAND_ASSIGN = "task.command.assign"
    COMMAND_OPEN_WAIT = "task.command.open_wait"
    COMMAND_CANCEL_WAIT = "task.command.cancel_wait"
    COMMAND_REQUEST_EFFORT_TASK = "task.command.request_effort_task"
    COMMAND_RETAIN_ATTEMPT = "task.command.retain_attempt"
    COMMAND_UPDATE_ATTEMPT = "task.command.update_attempt"
    WAIT_TEMPORAL_TIMER = "task.wait.temporal_timer"
    RECEIPT_READ = "task.receipt.read"
    HISTORY_READ = "task.history.read"
    CATALOG_READ = "task.catalog.read"


def _sorted_unique(values: list[str], code: str) -> list[str]:
    if values != sorted(set(values)):
        raise ValueError(code)
    return values


def _canonical_projects(values: list[str]) -> list[str]:
    _sorted_unique(values, "intern_task_grant_projects_invalid")
    for value in values:
        _canonical_uuid(value, code="intern_task_grant_identity_invalid", allow_nil=False)
    return values


class InternTaskGrantDeclaration(_Contract):
    """``POST /smr/research-intern/task-grants``. Exact retry returns the retained grant."""

    schema_version: Literal["intern.task_grant_declaration.v1"] = "intern.task_grant_declaration.v1"
    operation_id: str = Field(min_length=1, max_length=128, pattern=_PRINTABLE_ID)
    intern_id: str
    smr_project_ids: list[str] = Field(min_length=1, max_length=16)
    task_operations: list[str] = Field(min_length=1, max_length=16)
    policy_revision: str = Field(min_length=1, max_length=128, pattern=_PRINTABLE_ID)
    expires_unix_ms: StrictInt = Field(gt=0)

    @field_validator("intern_id")
    @classmethod
    def _intern(cls, value: str) -> str:
        return _canonical_uuid(value, code="intern_task_grant_identity_invalid", allow_nil=False)

    @field_validator("smr_project_ids")
    @classmethod
    def _projects(cls, values: list[str]) -> list[str]:
        return _canonical_projects(values)

    @field_validator("task_operations")
    @classmethod
    def _operations(cls, values: list[str]) -> list[str]:
        allowed = {op.value for op in InternTaskOperation}
        if any(value not in allowed for value in values):
            raise ValueError("intern_task_grant_operations_invalid")
        return _sorted_unique(values, "intern_task_grant_operations_invalid")


class BackendContextBindGrantDeclaration(_Contract):
    """``POST /smr/research-intern/backend-context-bind-grants`` (no operation list)."""

    schema_version: Literal["backend.context_bind_grant_declaration.v1"] = (
        "backend.context_bind_grant_declaration.v1"
    )
    operation_id: str = Field(min_length=1, max_length=128, pattern=_PRINTABLE_ID)
    intern_id: str
    smr_project_ids: list[str] = Field(min_length=1, max_length=16)
    policy_revision: str = Field(min_length=1, max_length=128, pattern=_PRINTABLE_ID)
    expires_unix_ms: StrictInt = Field(gt=0)

    @field_validator("intern_id")
    @classmethod
    def _intern(cls, value: str) -> str:
        return _canonical_uuid(value, code="intern_task_grant_identity_invalid", allow_nil=False)

    @field_validator("smr_project_ids")
    @classmethod
    def _projects(cls, values: list[str]) -> list[str]:
        return _canonical_projects(values)


class InternTaskGrantRevocation(_Contract):
    """``POST /smr/research-intern/task-grants/{grant_id}/revoke``.

    ``expected_revocation_epoch`` is the epoch read from the active grant; an
    exact retry after a committed revoke is accepted (backend compares to
    ``epoch + 1``), any other epoch refuses ``grant_stale``.
    """

    schema_version: Literal["intern.task_grant_revocation.v1"] = "intern.task_grant_revocation.v1"
    expected_revocation_epoch: StrictInt = Field(ge=0)


class InternTaskGrantStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class SublinearCatalogProjects(_Contract):
    kind: Literal["selected"]
    project_ids: tuple[str, ...]
    include_direct: StrictBool


class InternTaskGrant(_Contract):
    """``synth.sublinear-task-grant.v1`` document, plus the served status/epoch."""

    schema_: Literal["synth.sublinear-task-grant.v1"] = Field(alias="schema")
    grant_id: str = Field(pattern=r"^grt_[0-9a-f]{32}$")
    subject: str
    generation: StrictInt = Field(ge=1)
    organization_id: str
    intern_id: str
    policy_revision: str
    revocation_epoch: StrictInt = Field(ge=0)
    status: InternTaskGrantStatus
    task_operations: tuple[str, ...]
    catalog_projects: SublinearCatalogProjects
    expires_unix_ms: StrictInt = Field(gt=0)
    grant_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @field_validator("organization_id", "intern_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        if not _LOWER_UUID.match(value):
            raise ValueError("intern_task_grant_identity_invalid")
        return value

    @field_validator("expires_unix_ms", mode="before")
    @classmethod
    def _numeric_expiry(cls, value: object) -> object:
        # jsonb numeric: an integral value decoded as float is still integral.
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @property
    def active(self) -> bool:
        return self.status is InternTaskGrantStatus.ACTIVE


# -- Project Sublinear Tasks ------------------------------------------------


class SublinearWorkflowState(_Contract):
    id: str
    name: str
    type: str | None = None


class SublinearTask(_Contract):
    """``dataclasses.asdict(SublinearIssue)`` with the backend-resolved ``url``."""

    id: str
    identifier: str
    title: str
    url: str
    description: str | None = None
    state: SublinearWorkflowState | None = None
    project_id: str | None = None
    team_id: str | None = None


class SublinearTaskComment(_Contract):
    id: str
    body: str
    url: str


class ProjectSublinearTask(_Contract):
    """``GET /smr/projects/{project_id}/sublinear/tasks/{task_id}``."""

    project_id: str
    task: SublinearTask


class ProjectSublinearTaskList(_Contract):
    """``GET /smr/projects/{project_id}/sublinear/tasks``."""

    project_id: str
    sublinear_project_id: str
    tasks: tuple[SublinearTask, ...]


class ProjectSublinearTaskComments(_Contract):
    """``GET /smr/projects/{project_id}/sublinear/tasks/{task_id}/comments``."""

    project_id: str
    task_id: str
    comments: tuple[SublinearTaskComment, ...]


__all__ = [
    "BackendContextBindGrantDeclaration",
    "InternIdentity",
    "InternIdentityCatalogPage",
    "InternIdentityProvisionKind",
    "InternIdentityProvisionReceipt",
    "InternIdentityProvisionRequest",
    "InternIdentitySelection",
    "InternTaskGrant",
    "InternTaskGrantDeclaration",
    "InternTaskGrantRevocation",
    "InternTaskGrantStatus",
    "InternTaskOperation",
    "ProjectSublinearTask",
    "ProjectSublinearTaskComments",
    "ProjectSublinearTaskList",
    "SublinearCatalogProjects",
    "SublinearTask",
    "SublinearTaskComment",
    "SublinearWorkflowState",
]
