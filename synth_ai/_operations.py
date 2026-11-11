"""Execute generated operations under one identity, payload snapshot and deadline."""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from ._generated.models import MODELS
from ._generated.operations import OPERATIONS
from ._retry import RetryPolicy
from .diagnostics import DiagnosticHook, DiagnosticPhase, OperationDiagnostic
from .errors import ResponseContractError, TransportFailureError, local_problem, raise_for_response


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
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.timeout = timeout
        self.retry_policy = retry_policy
        self.headers = headers
        self.diagnostic_hook = diagnostic_hook

    async def invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> BaseModel | bytes | None:
        started = time.monotonic()
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
        self, operation: str, phase: DiagnosticPhase, progress: _Progress, started: float
    ) -> None:
        if self.diagnostic_hook is None:
            return

        def correlation(value: str | None) -> str | None:
            return hashlib.sha256(value.encode()).hexdigest() if value is not None else None

        event = OperationDiagnostic(
            operation,
            phase,
            progress.attempts,
            max(0.0, time.monotonic() - started),
            correlation(progress.request_id),
            correlation(progress.idempotency_key),
        )
        try:
            self.diagnostic_hook(event)
        except Exception:
            # A diagnostic sink is observational; it cannot change an operation.
            pass

    async def _invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
        progress: _Progress,
        started: float,
    ) -> BaseModel | bytes | None:
        op = OPERATIONS[operation_id]
        route = op["path"]
        if set(path or {}) != set(op["path_params"]):
            raise ValueError("path parameters do not match operation")
        for name, value in (path or {}).items():
            route = route.replace("{" + name + "}", quote(value, safe=""))
        params = {k: v for k, v in (query or {}).items() if v is not None}
        if set(params) - set(op["query_params"]):
            raise ValueError("query parameters do not match operation")
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
        if op["idempotency"] == "durable_key":
            headers["Idempotency-Key"] = idempotency_key or str(uuid4())
        if content is not None:
            headers["Content-Type"] = "application/json"
        progress.request_id = headers["X-Request-Id"]
        progress.idempotency_key = headers.get("Idempotency-Key")
        deadline = time.monotonic() + self.retry_policy.deadline_seconds
        response: httpx.Response | None = None
        for attempt in range(self.retry_policy.attempts):
            progress.attempts = attempt + 1
            try:
                remaining = deadline - time.monotonic()
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
            if (
                response.status_code not in {429, 503}
                or attempt + 1 == self.retry_policy.attempts
                or op["retry"] == "never"
                or (op["method"] not in {"GET", "HEAD"} and ambiguous_response(response))
            ):
                break
            delay = self.retry_policy.delay(response.headers.get("Retry-After"), attempt)
            if delay is None or delay >= deadline - time.monotonic():
                break
            self._emit(operation_id, DiagnosticPhase.RETRY, progress, started)
            await asyncio.sleep(delay)
        assert response is not None  # RetryPolicy guarantees at least one attempt.
        raise_for_response(response)
        if op["binary"]:
            return response.content
        if op["response_model"] is not None:
            if op["nullable_response"] and response.content.strip() == b"null":
                return None
            try:
                return MODELS[op["response_model"]].model_validate_json(response.content)
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
        return None
