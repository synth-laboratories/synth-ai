"""Execute generated operations under one identity, payload snapshot and deadline."""

import asyncio
import time
from collections.abc import Callable, Mapping
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, JsonValue, ValidationError

from ._generated.models import MODELS
from ._generated.operations import OPERATIONS
from ._retry import RetryPolicy
from .errors import ResponseContractError, TransportFailureError, local_problem, raise_for_response


def ambiguous_response(response: httpx.Response) -> bool:
    """Honor the owner's explicit uncertainty even on a retryable HTTP status."""
    try:
        body = response.json()
    except (ValueError, UnicodeError):
        return False
    return bool(
        isinstance(body, dict)
        and (body.get("outcome") == "unknown" or body.get("outcome_unknown") is True)
    )


class OperationExecutor:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout: float,
        retry_policy: RetryPolicy,
        headers: Callable[[bool], dict[str, str]],
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.timeout = timeout
        self.retry_policy = retry_policy
        self.headers = headers

    async def invoke(
        self,
        operation_id: str,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
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
        headers = self.headers(op["auth"])
        if op["idempotency"] == "durable_key":
            headers["Idempotency-Key"] = idempotency_key or str(uuid4())
        deadline = time.monotonic() + self.retry_policy.deadline_seconds
        response: httpx.Response | None = None
        for attempt in range(self.retry_policy.attempts):
            try:
                remaining = deadline - time.monotonic()
                async with asyncio.timeout(max(0, remaining)):
                    response = await self.client.request(
                        op["method"],
                        self.base_url + route,
                        params=params,
                        json=payload,
                        headers=headers,
                        timeout=min(self.timeout, max(0, remaining)),
                    )
            except (httpx.TransportError, TimeoutError) as error:
                # A timeout after admission is unknown. Keep the same key for caller-directed retry.
                raise TransportFailureError(
                    local_problem(
                        "transport_failed",
                        operation_id,
                        unknown=op["method"] not in {"GET", "HEAD"},
                    ),
                    request_id=headers["X-Request-Id"],
                    idempotency_key=headers.get("Idempotency-Key"),
                ) from error
            if (
                response.status_code not in {429, 503}
                or attempt + 1 == self.retry_policy.attempts
                or op["retry"] == "never"
                or ambiguous_response(response)
            ):
                break
            delay = self.retry_policy.delay(response.headers.get("Retry-After"), attempt)
            if delay is None or delay >= deadline - time.monotonic():
                break
            await asyncio.sleep(delay)
        assert response is not None  # RetryPolicy guarantees at least one attempt.
        raise_for_response(response)
        if op["binary"]:
            return response.content
        if op["response_model"] is not None:
            try:
                return MODELS[op["response_model"]].model_validate_json(response.content)
            except ValidationError as error:
                raise ResponseContractError(
                    local_problem(
                        "response_contract_invalid",
                        operation_id,
                        unknown=op["idempotency"] == "durable_key",
                    ),
                    request_id=response.headers.get("x-request-id") or headers["X-Request-Id"],
                    idempotency_key=headers.get("Idempotency-Key"),
                ) from error
        return None
