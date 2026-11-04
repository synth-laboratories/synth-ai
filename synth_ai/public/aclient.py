"""Async public Swarms v1 clients: the same calls as :mod:`synth_ai.public.client`."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from decimal import Decimal
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from synth_ai.auth.provider import StoredCredentialProvider

from .client import (
    ACCOUNT_PATH,
    API_KEYS_PATH,
    BILLING_PATH,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    RUNS_PATH,
    SERVICE,
    default_limits,
    idempotency,
    project_or_refuse,
    run_failure,
)
from .errors import (
    TransportFailureError,
    WaitTimeoutError,
    client_error,
    raise_for_response,
)
from .models import (
    Account as AccountModel,
)
from .models import (
    ApiKeyList,
    Balance,
    CheckoutRequest,
    CheckoutSession,
    ControlReceipt,
    ErrorDetail,
    EventPage,
    LaunchRequest,
    Limits,
    Money,
    OutputPage,
    PlanCatalog,
    RunEvent,
    RunOutput,
    RunPage,
    RunResult,
    RunSnapshot,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class AsyncPublicTransport:
    """Owns one ``httpx.AsyncClient`` and attaches the caller's credential."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        credential_provider: StoredCredentialProvider | None = None,
        session_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider = credential_provider
        self._session_token = session_token
        self._owned = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._timeout = timeout_seconds
        self.sleep = sleep
        self.clock = clock

    async def _headers(self, *, authenticated: bool, idempotency_key: str | None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if authenticated:
            if self._session_token is not None:
                headers["X-Clerk-Session-Token"] = self._session_token
            if self._api_key is not None:
                headers["Authorization"] = f"Bearer {self._api_key}"
            elif self._provider is not None and self._session_token is None:
                token = await self._provider.aresolve(self.clock() + self._timeout)
                headers["Authorization"] = f"Bearer {token}"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def send(
        self,
        method: str,
        path: str,
        *,
        stage: str,
        params: dict[str, Any] | None = None,
        body: BaseModel | None = None,
        idempotency_key: str | None = None,
        authenticated: bool = True,
    ) -> httpx.Response:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        headers = await self._headers(authenticated=authenticated, idempotency_key=idempotency_key)
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            response = await self._http.request(
                method,
                f"{self.base_url}{path}",
                params=query or None,
                content=None if body is None else body.model_dump_json(exclude_none=True),
                headers=headers,
            )
        except httpx.HTTPError as error:
            raise TransportFailureError(
                0,
                ErrorDetail(
                    code="transport_failed", service=SERVICE, stage=stage, cause=type(error).__name__
                ),
            ) from error
        raise_for_response(response, stage=stage)
        return response

    async def json(
        self, model: type[ModelT], method: str, path: str, *, stage: str, **kw: Any
    ) -> ModelT:
        response = await self.send(method, path, stage=stage, **kw)
        try:
            return model.model_validate_json(response.content)
        except ValueError as error:
            raise client_error(
                "response_contract_mismatch",
                stage,
                cause=type(error).__name__,
                status=response.status_code,
            ) from error

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()
        if self._provider is not None:
            self._provider.close()


class AsyncSwarms:
    """``client.swarms`` on :class:`AsyncSynthClient`."""

    def __init__(self, transport: AsyncPublicTransport, *, project_id: str | None = None) -> None:
        self._t = transport
        self._project_id = project_id

    async def run(
        self,
        message: str,
        budget: Decimal | int | str | Money | None,
        *,
        project_id: str | None = None,
        limits: Limits | None = None,
        timeout_seconds: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        idempotency_key: str | None = None,
        **launch_fields: Any,
    ) -> RunResult:
        """Launch a run, wait for it to close and return its outputs."""
        run = await self.launch(
            message, budget, project_id=project_id, limits=limits,
            idempotency_key=idempotency_key, **launch_fields,
        )
        closed = await self.wait(
            run.run_id, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds
        )
        if closed.state == "failed":
            raise run_failure(closed)
        return RunResult(run=closed, outputs=(await self.outputs(closed.run_id)).items)

    async def launch(
        self,
        message: str | None,
        budget: Decimal | int | str | Money | None = None,
        *,
        project_id: str | None = None,
        limits: Limits | None = None,
        idempotency_key: str | None = None,
        **launch_fields: Any,
    ) -> RunSnapshot:
        request = LaunchRequest(
            project_id=project_or_refuse(project_id or self._project_id),
            limits=limits or default_limits(budget),
            message=message,
            **launch_fields,
        )
        return await self._t.json(
            RunSnapshot, "POST", RUNS_PATH, stage="swarms.launch",
            body=request, idempotency_key=idempotency(idempotency_key),
        )

    async def get(self, run_id: str) -> RunSnapshot:
        return await self._t.json(RunSnapshot, "GET", f"{RUNS_PATH}/{run_id}", stage="swarms.get")

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> RunSnapshot:
        deadline = None if timeout_seconds is None else self._t.clock() + timeout_seconds
        while True:
            snapshot = await self.get(run_id)
            if snapshot.terminal or snapshot.execution_closed:
                return snapshot
            if deadline is not None and self._t.clock() >= deadline:
                raise client_error("wait_timeout", "swarms.wait", cause=run_id,
                                   kind=WaitTimeoutError)
            await self._t.sleep(poll_seconds)

    async def events(self, run_id: str, *, cursor: str | None = None,
                     limit: int | None = None) -> EventPage:
        """One page of events after ``cursor`` (a previous ``next_cursor``/``resume_cursor``)."""
        return await self._t.json(
            EventPage, "GET", f"{RUNS_PATH}/{run_id}/events",
            stage="swarms.events", params={"cursor": cursor, "limit": limit},
        )

    async def iter_events(
        self, run_id: str, *, cursor: str | None = None
    ) -> AsyncIterator[RunEvent]:
        """Every event currently available after ``cursor``, following ``next_cursor``."""
        while True:
            page = await self.events(run_id, cursor=cursor)
            for event in page.items:
                yield event
            if page.complete or page.next_cursor is None:
                return
            cursor = page.next_cursor

    async def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> ControlReceipt:
        return await self._t.json(
            ControlReceipt, "POST", f"{RUNS_PATH}/{run_id}/cancel",
            stage="swarms.cancel", idempotency_key=idempotency(idempotency_key),
        )

    async def outputs(self, run_id: str) -> OutputPage:
        return await self._t.json(OutputPage, "GET", f"{RUNS_PATH}/{run_id}/outputs",
                                  stage="swarms.outputs")

    async def output(self, run_id: str, output_id: str) -> RunOutput:
        return await self._t.json(RunOutput, "GET", f"{RUNS_PATH}/{run_id}/outputs/{output_id}",
                            stage="swarms.output")

    async def list(self, *, cursor: str | None = None, limit: int | None = None,
             project_id: str | None = None) -> RunPage:
        return await self._t.json(RunPage, "GET", RUNS_PATH, stage="swarms.list",
                            params={"cursor": cursor, "limit": limit, "project_id": project_id})


class AsyncAccount:
    def __init__(self, transport: AsyncPublicTransport) -> None:
        self._t = transport

    async def get(self) -> AccountModel:
        return await self._t.json(AccountModel, "GET", ACCOUNT_PATH, stage="account.get")

    async def keys(self, *, purpose: str | None = None, offset: int | None = None,
                   limit: int | None = None) -> ApiKeyList:
        return await self._t.json(
            ApiKeyList, "GET", API_KEYS_PATH, stage="account.keys",
            params={"purpose": purpose, "offset": offset, "limit": limit},
        )


class AsyncBilling:
    def __init__(self, transport: AsyncPublicTransport) -> None:
        self._t = transport

    async def balance(self) -> Balance:
        return await self._t.json(Balance, "GET", f"{BILLING_PATH}/balance",
                                  stage="billing.balance")

    async def plans(self) -> PlanCatalog:
        return await self._t.json(PlanCatalog, "GET", f"{BILLING_PATH}/plans",
                                  stage="billing.plans", authenticated=False)

    async def checkout(self, *, plan_id: str | None = None,
                       credit_cents: int | None = None) -> CheckoutSession:
        body = CheckoutRequest(plan_id=plan_id, credit_cents=credit_cents)
        return await self._t.json(CheckoutSession, "POST", f"{BILLING_PATH}/checkout",
                                  stage="billing.checkout", body=body)
