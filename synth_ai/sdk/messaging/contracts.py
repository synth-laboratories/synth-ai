"""Typed messaging responses; backend specifications/workshop-messaging-api.md."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessagingModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class Principal(MessagingModel):
    kind: Literal["human", "intern_sync", "intern_async", "actor", "system"]
    id: str
    org_id: str


class Scope(MessagingModel):
    kind: Literal["org", "factory", "effort", "project", "sync_session", "async_runtime"]
    id: str


class Thread(MessagingModel):
    thread_id: UUID
    org_id: str
    scope: Scope
    title: str | None
    idempotency_key: str | None = None
    created_at: datetime


class Message(MessagingModel):
    message_id: UUID
    thread_id: UUID
    seq: int = Field(ge=1)
    kind: Literal["ask", "answer", "steer", "notice", "actor_runtime", "blocker", "handoff_ping"]
    body: str
    payload: Any
    sender: Principal
    idempotency_key: str | None
    correlation_id: str | None
    parent_message_id: UUID | None = None
    causation_id: str | None = None
    created_at: datetime


class HistorySkip(MessagingModel):
    after_seq: int = Field(ge=0)
    through_seq: int = Field(ge=0)
    reason: str


class HistoryPage(MessagingModel):
    thread_id: UUID
    requested_after_seq: int = Field(ge=0)
    history_after_seq: int = Field(ge=0)
    effective_after_seq: int = Field(ge=0)
    skipped: HistorySkip | None
    messages: tuple[Message, ...]
    next_after_seq: int = Field(ge=0)
    has_more: bool


class Enrollment(MessagingModel):
    enrollment_id: UUID
    org_id: str
    owner: Principal
    device_id: str
    session_id: str
    label: str | None
    principal: Principal
    incarnation: int = Field(ge=1)
    revoked_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class Grant(MessagingModel):
    grant_id: UUID
    org_id: str
    thread_id: UUID
    enrollment_id: UUID
    principal: Principal
    operations: tuple[Literal["read", "publish"], ...]
    history_after_seq: int = Field(ge=0)
    expires_at: datetime
    incarnation: int = Field(ge=1)
    generation: int = Field(ge=0)
    status: Literal["active", "revoked"]
    state: Literal["active", "revoked", "expired"]
    granted_by: Principal
    created_at: datetime
    updated_at: datetime
