"""Execute generated operations under one identity, payload snapshot and deadline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from ._generated.models import MODELS, Problem
from ._generated.operations import OPERATIONS
from ._retry import (
    ADMISSION_RETRY_BUDGET_SECONDS,
    admission_pending,
    admission_wait,
)
from ._validation import require_non_empty, validate_idempotency_key
from .diagnostics import (
    DiagnosticPhase,
    OperationDiagnostic,
    RetrySuppressed,
    RetrySuppressedReason,
)
from .errors import (
    ResponseContractError,
    TransportFailureError,
    UpstreamUnavailableError,
    local_problem,
    raise_for_response,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from ._retry import RetryPolicy
    from .diagnostics import DiagnosticHook


def ambiguous_response(response: httpx.Response) -> bool:
    """Only an authoritative not-accepted Problem permits a mutation retry."""
    from ._generated.models import Problem

    try:
        problem = Problem.model_validate_json(response.content)
    except (ValueError, ValidationError):
        return True
    legacy_unknown = (
        problem.model_extra is not None and problem.model_extra.get("outcome_unknown") is True
    )
    return (
        problem.status != response.status_code
        or problem.outcome != "not_accepted"
        or legacy_unknown
    )


def response_problem(response: httpx.Response) -> Problem | None:
    try:
        return Problem.model_validate_json(response.content)
    except (ValueError, ValidationError):
        return None


def retryable_response(response: httpx.Response) -> bool:
    """Only a valid server Problem with retryable=true permits overload recovery."""
    try:
        problem = Problem.model_validate_json(response.content)
    except (ValueError, ValidationError):
        # Refuse the retry; raise_for_response will surface the malformed body.
        return False
    return problem.status == response.status_code and problem.retryable is True


@dataclass(frozen=True, slots=True)
class OperationResult:
    value: BaseModel | bytes | None
    replayed: bool


@dataclass
class _Progress:
    attempts: int = 0
    request_id: str | None = None
    idempotency_key: str | None = None


class OperationExecutor:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout: float,
        retry_policy: RetryPolicy,
        headers: Callable[[bool], dict[str, str]],
        diagnostic_hook: DiagnosticHook | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.timeout = timeout
        self.retry_policy = retry_policy
        self.headers = headers
        self.diagnostic_hook = diagnostic_hook
        self.sleep = sleep
        self.clock = clock

    async def invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | list[str] | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> OperationResult:
        started = self.clock()
        progress = _Progress()
        try:
            result = await self._invoke(
                operation_id,
                path=path,
                query=query,
                body=body,
                idempotency_key=idempotency_key,
                progress=progress,
                started=started,
            )
        except BaseException:
            self._emit(operation_id, DiagnosticPhase.FAILED, progress, started)
            raise
        self._emit(operation_id, DiagnosticPhase.SUCCEEDED, progress, started)
        return result

    def _emit(
        self,
        operation: str,
        phase: DiagnosticPhase,
        progress: _Progress,
        started: float,
        retry_suppressed: RetrySuppressed | None = None,
    ) -> None:
        if self.diagnostic_hook is None:
            return

        def correlation(value: str | None) -> str | None:
            return hashlib.sha256(value.encode()).hexdigest() if value is not None else None

        event = OperationDiagnostic(
            operation,
            phase,
            progress.attempts,
            max(0.0, self.clock() - started),
            correlation(progress.request_id),
            correlation(progress.idempotency_key),
            retry_suppressed,
        )
        try:
            self.diagnostic_hook(event)
        except Exception:  # noqa: BLE001
            # A diagnostic sink is observational; it cannot change an operation.
            return

    async def _invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | list[str] | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
        progress: _Progress,
        started: float,
    ) -> OperationResult:
        op = OPERATIONS[operation_id]
        route = op["path"]
        if set(path or {}) != set(op["path_params"]):
            raise ValueError("path parameters do not match operation")
        for name, value in (path or {}).items():
            validated = require_non_empty(f"path parameter {name!r}", value)
            route = route.replace("{" + name + "}", quote(validated, safe=""))
        params = {k: v for k, v in (query or {}).items() if v is not None}
        if set(params) - set(op["query_params"]):
            raise ValueError("query parameters do not match operation")
        validated_key = validate_idempotency_key(idempotency_key)
        payload = (
            body.model_dump(mode="json", exclude_unset=True)
            if isinstance(body, BaseModel)
            else body
        )
        if op["request_model"] is not None and payload is not None:
            payload = (
                MODELS[op["request_model"]]
                .model_validate(payload)
                .model_dump(mode="json", exclude_unset=True)
            )
        # Serialize once before any await. Caller-owned nested containers cannot
        # change request semantics between retries under the same identity.
        content = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = self.headers(op["auth"])
        if op["idempotency"] in {"durable_key", "owner_key", "mq_owner_key"}:
            headers["Idempotency-Key"] = (
                validated_key if validated_key is not None else str(uuid4())
            )
        if content is not None:
            headers["Content-Type"] = "application/json"
        progress.request_id = headers["X-Request-Id"]
        progress.idempotency_key = headers.get("Idempotency-Key")
        deadline = self.clock() + self.retry_policy.deadline_seconds
        response: httpx.Response | None = None
        admission_started = self.clock()
        admission_retries = 0
        attempt = 0
        while attempt < self.retry_policy.attempts:
            progress.attempts = attempt + admission_retries + 1
            try:
                remaining = deadline - self.clock()
                async with asyncio.timeout(max(0, remaining)):
                    response = await self.client.request(
                        op["method"],
                        self.base_url + route,
                        params=params,
                        content=content,
                        follow_redirects=False,
                        headers=headers,
                        timeout=min(self.timeout, max(0, remaining)),
                    )
            except (httpx.TransportError, TimeoutError) as error:
                # A timeout after admission is unknown. Keep the same key for caller-directed retry.
                raise TransportFailureError(
                    local_problem(
                        "transport_failed",
                        operation_id,
                        unknown=op["method"] not in {"GET", "HEAD"}
                        and not isinstance(
                            error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
                        ),
                    ),
                    request_id=headers["X-Request-Id"],
                    idempotency_key=headers.get("Idempotency-Key"),
                ) from error
            if "Idempotency-Key" in headers and admission_pending(
                response_problem(response), response.status_code
            ):
                # Core refused before admission and holds no body: resend the same key.
                # These waits use their own budget, not the policy's attempts or deadline.
                wait = admission_wait(response.headers.get("Retry-After"))
                if self.clock() - admission_started + wait > ADMISSION_RETRY_BUDGET_SECONDS:
                    break
                self._emit(operation_id, DiagnosticPhase.RETRY, progress, started)
                await self.sleep(wait)
                admission_retries += 1
                deadline = self.clock() + self.retry_policy.deadline_seconds
                continue
            if (
                response.status_code not in {429, 503}
                or attempt + 1 == self.retry_policy.attempts
                or op["retry"] == "never"
                or not retryable_response(response)
                or (op["method"] not in {"GET", "HEAD"} and ambiguous_response(response))
            ):
                break
            retry_after = response.headers.get("Retry-After")
            delay = self.retry_policy.delay(retry_after, attempt)
            remaining = deadline - self.clock()
            if delay is None or delay >= remaining:
                # The caller sees only the last refusal; say why no further attempt was sent.
                self._emit(
                    operation_id,
                    DiagnosticPhase.RETRY_SUPPRESSED,
                    progress,
                    started,
                    RetrySuppressed(
                        RetrySuppressedReason.RETRY_AFTER_INVALID
                        if delay is None
                        else RetrySuppressedReason.DEADLINE_EXCEEDED,
                        response.status_code,
                        retry_after,
                        delay,
                        max(0.0, remaining),
                    ),
                )
                break
            self._emit(operation_id, DiagnosticPhase.RETRY, progress, started)
            await asyncio.sleep(delay)
            attempt += 1
        assert response is not None  # RetryPolicy guarantees at least one attempt.
        try:
            raise_for_response(response)
        except UpstreamUnavailableError as error:
            if admission_pending(error.problem, response.status_code):
                error.attempts = progress.attempts
                error.elapsed_seconds = max(0.0, self.clock() - admission_started)
            raise
        replayed = response.headers.get("Idempotent-Replayed") == "true"
        if op["binary"]:
            return OperationResult(response.content, replayed)
        if op["response_model"] is not None:
            if op["nullable_response"] and response.content.strip() == b"null":
                return OperationResult(None, replayed)
            try:
                value = MODELS[op["response_model"]].model_validate_json(response.content)
                return OperationResult(value, replayed)
            except ValidationError as error:
                raise ResponseContractError(
                    local_problem(
                        "response_contract_invalid",
                        operation_id,
                        unknown=op["method"] not in {"GET", "HEAD"},
                    ),
                    request_id=response.headers.get("x-request-id") or headers["X-Request-Id"],
                    idempotency_key=headers.get("Idempotency-Key"),
                ) from error
        return OperationResult(None, replayed)
