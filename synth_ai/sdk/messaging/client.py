"""Typed authenticated messaging client. See README.md and backend messaging spec.

No automatic retries, direct MQ connections, or credential issuance. Publication
is acceptance only; device execution and delivery must be observed independently.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID

from pydantic import TypeAdapter

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

from .contracts import Enrollment, Grant, HistoryPage, Message, Thread

MessageKind = Literal[
    "ask", "answer", "steer", "notice", "actor_runtime", "blocker", "handoff_ping"
]


def _identity(value: str | UUID) -> str:
    return str(UUID(str(value)))


def _unwrap(value: Any, key: str) -> Any:
    if not isinstance(value, dict) or key not in value:
        raise ValueError(f"messaging response missing {key}")
    return value[key]


class ThreadsAPI:
    """Backend-owned threads through the normal Synth credential."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(self, *, title: str | None = None, idempotency_key: str) -> Thread:
        """create threads using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST",
            "/api/v1/mq/threads",
            json_body={"title": title, "idempotency_key": idempotency_key},
        )
        return TypeAdapter(Thread).validate_python(payload)

    def get(self, thread_id: str | UUID) -> Thread:
        """get threads using the backend-owned contract."""
        payload = self._transport.request_json("GET", f"/api/v1/mq/threads/{_identity(thread_id)}")
        return TypeAdapter(Thread).validate_python(payload)

    def publish(
        self,
        thread_id: str | UUID,
        *,
        kind: MessageKind,
        body: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        parent_message_id: str | UUID | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        """publish threads using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST",
            f"/api/v1/mq/threads/{_identity(thread_id)}/messages",
            json_body={
                "kind": kind,
                "body": body,
                "idempotency_key": idempotency_key,
                "payload": payload if payload is not None else {},
                "parent_message_id": _identity(parent_message_id)
                if parent_message_id is not None
                else None,
                "correlation_id": correlation_id,
            },
        )
        return TypeAdapter(Message).validate_python(payload)

    def history(self, thread_id: str | UUID, *, after_seq: int = 0, limit: int = 50) -> HistoryPage:
        """history threads using the backend-owned contract."""
        payload = self._transport.request_json(
            "GET",
            f"/api/v1/mq/threads/{_identity(thread_id)}/history",
            params={"after_seq": after_seq, "limit": limit},
        )
        return TypeAdapter(HistoryPage).validate_python(payload)


class EnrollmentsAPI:
    """Backend-owned enrollments through the normal Synth credential."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> tuple[Enrollment, ...]:
        """list enrollments using the backend-owned contract."""
        payload = self._transport.request_json("GET", "/api/v1/mq/enrollments")
        return TypeAdapter(tuple[Enrollment, ...]).validate_python(_unwrap(payload, "enrollments"))

    def get(self, enrollment_id: str | UUID) -> Enrollment:
        """get enrollments using the backend-owned contract."""
        payload = self._transport.request_json(
            "GET", f"/api/v1/mq/enrollments/{_identity(enrollment_id)}"
        )
        return TypeAdapter(Enrollment).validate_python(_unwrap(payload, "enrollment"))

    def revoke(self, enrollment_id: str | UUID) -> Enrollment:
        """revoke enrollments using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST", f"/api/v1/mq/enrollments/{_identity(enrollment_id)}/revoke", json_body={}
        )
        return TypeAdapter(Enrollment).validate_python(_unwrap(payload, "enrollment"))


class GrantsAPI:
    """Backend-owned grants through the normal Synth credential."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(
        self,
        *,
        thread_id: str | UUID,
        enrollment_id: str | UUID,
        operations: Sequence[Literal["read", "publish"]],
        ttl_seconds: int,
        history_after_seq: int | None = None,
    ) -> Grant:
        """create grants using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST",
            "/api/v1/mq/grants",
            json_body={
                "thread_id": _identity(thread_id),
                "enrollment_id": _identity(enrollment_id),
                "operations": list(operations),
                "ttl_seconds": ttl_seconds,
                "history_after_seq": history_after_seq,
            },
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    def list(
        self, *, thread_id: str | UUID | None = None, enrollment_id: str | UUID | None = None
    ) -> tuple[Grant, ...]:
        """list grants using the backend-owned contract."""
        payload = self._transport.request_json(
            "GET",
            "/api/v1/mq/grants",
            params={
                key: _identity(value)
                for key, value in {"thread_id": thread_id, "enrollment_id": enrollment_id}.items()
                if value is not None
            },
        )
        return TypeAdapter(tuple[Grant, ...]).validate_python(_unwrap(payload, "grants"))

    def get(self, grant_id: str | UUID) -> Grant:
        """get grants using the backend-owned contract."""
        payload = self._transport.request_json("GET", f"/api/v1/mq/grants/{_identity(grant_id)}")
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    def revoke(self, grant_id: str | UUID) -> Grant:
        """revoke grants using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST", f"/api/v1/mq/grants/{_identity(grant_id)}/revoke", json_body={}
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    def restore(self, grant_id: str | UUID) -> Grant:
        """restore grants using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST", f"/api/v1/mq/grants/{_identity(grant_id)}/restore", json_body={}
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    def renew(self, grant_id: str | UUID, *, ttl_seconds: int) -> Grant:
        """renew grants using the backend-owned contract."""
        payload = self._transport.request_json(
            "POST",
            f"/api/v1/mq/grants/{_identity(grant_id)}/renew",
            json_body={"ttl_seconds": ttl_seconds},
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))


class MessagingClient:
    """Typed messaging front door; no MQ administrative credentials."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        credential = resolve_api_credential(api_key)
        self._transport = HttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self._transport.client.follow_redirects = False
        self.threads = ThreadsAPI(self._transport)
        self.enrollments = EnrollmentsAPI(self._transport)
        self.grants = GrantsAPI(self._transport)

    def close(self) -> None:
        """Close the messaging HTTP transport."""
        self._transport.close()


class AsyncThreadsAPI:
    """Backend-owned threads through the normal Synth credential."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(self, *, title: str | None = None, idempotency_key: str) -> Thread:
        """create threads using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST",
            "/api/v1/mq/threads",
            json_body={"title": title, "idempotency_key": idempotency_key},
        )
        return TypeAdapter(Thread).validate_python(payload)

    async def get(self, thread_id: str | UUID) -> Thread:
        """get threads using the backend-owned contract."""
        payload = await self._transport.request_json(
            "GET", f"/api/v1/mq/threads/{_identity(thread_id)}"
        )
        return TypeAdapter(Thread).validate_python(payload)

    async def publish(
        self,
        thread_id: str | UUID,
        *,
        kind: MessageKind,
        body: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        parent_message_id: str | UUID | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        """publish threads using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST",
            f"/api/v1/mq/threads/{_identity(thread_id)}/messages",
            json_body={
                "kind": kind,
                "body": body,
                "idempotency_key": idempotency_key,
                "payload": payload if payload is not None else {},
                "parent_message_id": _identity(parent_message_id)
                if parent_message_id is not None
                else None,
                "correlation_id": correlation_id,
            },
        )
        return TypeAdapter(Message).validate_python(payload)

    async def history(
        self, thread_id: str | UUID, *, after_seq: int = 0, limit: int = 50
    ) -> HistoryPage:
        """history threads using the backend-owned contract."""
        payload = await self._transport.request_json(
            "GET",
            f"/api/v1/mq/threads/{_identity(thread_id)}/history",
            params={"after_seq": after_seq, "limit": limit},
        )
        return TypeAdapter(HistoryPage).validate_python(payload)


class AsyncEnrollmentsAPI:
    """Backend-owned enrollments through the normal Synth credential."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self) -> tuple[Enrollment, ...]:
        """list enrollments using the backend-owned contract."""
        payload = await self._transport.request_json("GET", "/api/v1/mq/enrollments")
        return TypeAdapter(tuple[Enrollment, ...]).validate_python(_unwrap(payload, "enrollments"))

    async def get(self, enrollment_id: str | UUID) -> Enrollment:
        """get enrollments using the backend-owned contract."""
        payload = await self._transport.request_json(
            "GET", f"/api/v1/mq/enrollments/{_identity(enrollment_id)}"
        )
        return TypeAdapter(Enrollment).validate_python(_unwrap(payload, "enrollment"))

    async def revoke(self, enrollment_id: str | UUID) -> Enrollment:
        """revoke enrollments using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST", f"/api/v1/mq/enrollments/{_identity(enrollment_id)}/revoke", json_body={}
        )
        return TypeAdapter(Enrollment).validate_python(_unwrap(payload, "enrollment"))


class AsyncGrantsAPI:
    """Backend-owned grants through the normal Synth credential."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self,
        *,
        thread_id: str | UUID,
        enrollment_id: str | UUID,
        operations: Sequence[Literal["read", "publish"]],
        ttl_seconds: int,
        history_after_seq: int | None = None,
    ) -> Grant:
        """create grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST",
            "/api/v1/mq/grants",
            json_body={
                "thread_id": _identity(thread_id),
                "enrollment_id": _identity(enrollment_id),
                "operations": list(operations),
                "ttl_seconds": ttl_seconds,
                "history_after_seq": history_after_seq,
            },
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    async def list(
        self, *, thread_id: str | UUID | None = None, enrollment_id: str | UUID | None = None
    ) -> tuple[Grant, ...]:
        """list grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "GET",
            "/api/v1/mq/grants",
            params={
                key: _identity(value)
                for key, value in {"thread_id": thread_id, "enrollment_id": enrollment_id}.items()
                if value is not None
            },
        )
        return TypeAdapter(tuple[Grant, ...]).validate_python(_unwrap(payload, "grants"))

    async def get(self, grant_id: str | UUID) -> Grant:
        """get grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "GET", f"/api/v1/mq/grants/{_identity(grant_id)}"
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    async def revoke(self, grant_id: str | UUID) -> Grant:
        """revoke grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST", f"/api/v1/mq/grants/{_identity(grant_id)}/revoke", json_body={}
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    async def restore(self, grant_id: str | UUID) -> Grant:
        """restore grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST", f"/api/v1/mq/grants/{_identity(grant_id)}/restore", json_body={}
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))

    async def renew(self, grant_id: str | UUID, *, ttl_seconds: int) -> Grant:
        """renew grants using the backend-owned contract."""
        payload = await self._transport.request_json(
            "POST",
            f"/api/v1/mq/grants/{_identity(grant_id)}/renew",
            json_body={"ttl_seconds": ttl_seconds},
        )
        return TypeAdapter(Grant).validate_python(_unwrap(payload, "grant"))


class AsyncMessagingClient:
    """Typed messaging front door; no MQ administrative credentials."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        credential = resolve_api_credential(api_key)
        self._transport = AsyncHttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self._transport.client.follow_redirects = False
        self.threads = AsyncThreadsAPI(self._transport)
        self.enrollments = AsyncEnrollmentsAPI(self._transport)
        self.grants = AsyncGrantsAPI(self._transport)

    async def close(self) -> None:
        """Close the messaging HTTP transport."""
        await self._transport.close()
