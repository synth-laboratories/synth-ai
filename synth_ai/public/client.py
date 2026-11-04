"""Public Swarms v1 clients: ``swarms``, ``account`` and ``billing``.

Paths follow ``contracts/public/swarms-v1.openapi.yaml`` in synth-backend.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from synth_ai.auth.provider import StoredCredentialProvider

from .errors import (
    RunFailedError,
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

SERVICE = "synth-ai"
# Route constants. Swarms and account keep the legacy /api/v1 prefixes; billing is
# served by the core at /v1/billing/* (coordinator correction 2026-11-04).
RUNS_PATH = "/api/v1/swarms/runs"
ACCOUNT_PATH = "/api/v1/account"
API_KEYS_PATH = "/v1/api-keys"
BILLING_PATH = "/v1/billing"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 2.0
# Contract maxima: the budget is the bound that matters for one-call runs.
DEFAULT_WALL_TIME_SECONDS = 86400
DEFAULT_MAXIMUM_TASKS = 1000
DEFAULT_MAXIMUM_PARALLEL_TASKS = 1000


class PublicTransport:
    """Owns one httpx client and attaches the caller's credential."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        credential_provider: StoredCredentialProvider | None = None,
        session_token: str | None = None,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider = credential_provider
        self._session_token = session_token
        self._owned = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout_seconds)
        self._timeout = timeout_seconds
        self.sleep = sleep
        self.clock = clock

    def _headers(self, *, authenticated: bool, idempotency_key: str | None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if authenticated:
            if self._session_token is not None:
                headers["X-Clerk-Session-Token"] = self._session_token
            if self._api_key is not None:
                headers["Authorization"] = f"Bearer {self._api_key}"
            elif self._provider is not None and self._session_token is None:
                token = self._provider.resolve(self.clock() + self._timeout)
                headers["Authorization"] = f"Bearer {token}"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def send(
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
        try:
            response = self._http.request(
                method,
                f"{self.base_url}{path}",
                params=query or None,
                content=None if body is None else body.model_dump_json(exclude_none=True),
                headers={
                    **self._headers(authenticated=authenticated, idempotency_key=idempotency_key),
                    **({} if body is None else {"Content-Type": "application/json"}),
                },
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

    def json(self, model: type[ModelT], method: str, path: str, *, stage: str, **kw: Any) -> ModelT:
        response = self.send(method, path, stage=stage, **kw)
        try:
            return model.model_validate_json(response.content)
        except ValueError as error:
            raise client_error(
                "response_contract_mismatch",
                stage,
                cause=type(error).__name__,
                status=response.status_code,
            ) from error

    def close(self) -> None:
        if self._owned:
            self._http.close()
        if self._provider is not None:
            self._provider.close()


def idempotency(idempotency_key: str | None) -> str:
    return idempotency_key or str(uuid.uuid4())


def project_or_refuse(project_id: str | None) -> str:
    if not project_id:
        raise client_error("project_id_required", "swarms.launch",
                           cause="pass project_id or set SYNTH_PROJECT_ID")
    return project_id


def run_failure(closed: RunSnapshot) -> RunFailedError:
    return RunFailedError(
        0,
        ErrorDetail(
            code=closed.failure_code or "run_failed",
            service=SERVICE,
            stage="swarms.run",
            cause=closed.run_id,
        ),
    )


def default_limits(budget: Decimal | int | str | Money | None) -> Limits:
    charge = budget if isinstance(budget, Money) or budget is None else Money.usd(budget)
    return Limits(
        maximum_charge=charge,
        wall_time_seconds=DEFAULT_WALL_TIME_SECONDS,
        maximum_tasks=DEFAULT_MAXIMUM_TASKS,
        maximum_parallel_tasks=DEFAULT_MAXIMUM_PARALLEL_TASKS,
    )


class Swarms:
    """``client.swarms``: launch, observe, control and collect Swarms runs."""

    def __init__(self, transport: PublicTransport, *, project_id: str | None = None) -> None:
        self._t = transport
        self._project_id = project_id

    def run(
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
        """Launch a run, wait for it to close and return its outputs.

        ``budget`` is the maximum charge in USD. Raises :class:`RunFailedError`
        when the run ends in state ``failed``.
        """
        run = self.launch(
            message,
            budget,
            project_id=project_id,
            limits=limits,
            idempotency_key=idempotency_key,
            **launch_fields,
        )
        closed = self.wait(run.run_id, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
        if closed.state == "failed":
            raise run_failure(closed)
        return RunResult(run=closed, outputs=self.outputs(closed.run_id).items)

    def launch(
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
        return self._t.json(
            RunSnapshot, "POST", RUNS_PATH, stage="swarms.launch",
            body=request, idempotency_key=idempotency(idempotency_key),
        )

    def get(self, run_id: str) -> RunSnapshot:
        return self._t.json(RunSnapshot, "GET", f"{RUNS_PATH}/{run_id}",
                            stage="swarms.get")

    def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> RunSnapshot:
        """Poll until the run reaches a terminal state or closes execution."""
        deadline = None if timeout_seconds is None else self._t.clock() + timeout_seconds
        while True:
            snapshot = self.get(run_id)
            if snapshot.terminal or snapshot.execution_closed:
                return snapshot
            if deadline is not None and self._t.clock() >= deadline:
                raise client_error("wait_timeout", "swarms.wait", cause=run_id,
                                   kind=WaitTimeoutError)
            self._t.sleep(poll_seconds)

    def events(self, run_id: str, *, cursor: str | None = None,
               limit: int | None = None) -> EventPage:
        """One page of events after ``cursor`` (a previous ``next_cursor``/``resume_cursor``)."""
        return self._t.json(
            EventPage, "GET", f"{RUNS_PATH}/{run_id}/events",
            stage="swarms.events", params={"cursor": cursor, "limit": limit},
        )

    def iter_events(self, run_id: str, *, cursor: str | None = None) -> Iterator[RunEvent]:
        """Every event currently available after ``cursor``, following ``next_cursor``."""
        while True:
            page = self.events(run_id, cursor=cursor)
            yield from page.items
            if page.complete or page.next_cursor is None:
                return
            cursor = page.next_cursor

    def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> ControlReceipt:
        return self._t.json(
            ControlReceipt, "POST", f"{RUNS_PATH}/{run_id}/cancel",
            stage="swarms.cancel", idempotency_key=idempotency(idempotency_key),
        )

    def outputs(self, run_id: str) -> OutputPage:
        return self._t.json(OutputPage, "GET", f"{RUNS_PATH}/{run_id}/outputs",
                            stage="swarms.outputs")

    def output(self, run_id: str, output_id: str) -> RunOutput:
        return self._t.json(RunOutput, "GET", f"{RUNS_PATH}/{run_id}/outputs/{output_id}",
                            stage="swarms.output")

    def list(self, *, cursor: str | None = None, limit: int | None = None,
             project_id: str | None = None) -> RunPage:
        return self._t.json(RunPage, "GET", RUNS_PATH, stage="swarms.list",
                            params={"cursor": cursor, "limit": limit, "project_id": project_id})


class Account:
    """``client.account``: the caller's org, principal, limits and keys."""

    def __init__(self, transport: PublicTransport) -> None:
        self._t = transport

    def get(self) -> AccountModel:
        return self._t.json(AccountModel, "GET", ACCOUNT_PATH, stage="account.get")

    def keys(self, *, purpose: str | None = None, offset: int | None = None,
             limit: int | None = None) -> ApiKeyList:
        return self._t.json(
            ApiKeyList, "GET", API_KEYS_PATH, stage="account.keys",
            params={"purpose": purpose, "offset": offset, "limit": limit},
        )



class Billing:
    """``client.billing``: wallet balance, plan catalog and hosted checkout."""

    def __init__(self, transport: PublicTransport) -> None:
        self._t = transport

    def balance(self) -> Balance:
        return self._t.json(Balance, "GET", f"{BILLING_PATH}/balance", stage="billing.balance")

    def plans(self) -> PlanCatalog:
        return self._t.json(PlanCatalog, "GET", f"{BILLING_PATH}/plans", stage="billing.plans",
                            authenticated=False)

    def checkout(self, *, plan_id: str | None = None,
                 credit_cents: int | None = None) -> CheckoutSession:
        """Hosted checkout for a plan or a custom credit amount (exactly one)."""
        body = CheckoutRequest(plan_id=plan_id, credit_cents=credit_cents)
        return self._t.json(CheckoutSession, "POST", f"{BILLING_PATH}/checkout",
                            stage="billing.checkout", body=body)
