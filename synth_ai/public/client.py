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
    ErrorDetail,
    LaunchRequest,
    Limits,
    Money,
    PlanCatalog,
    RunControlSummary,
    RunEvent,
    RunEventPage,
    RunOutputList,
    RunPage,
    RunResult,
    RunSnapshot,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

SERVICE = "synth-ai"
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


def _key(idempotency_key: str | None) -> str:
    return idempotency_key or str(uuid.uuid4())


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

    def _project(self, project_id: str | None) -> str:
        selected = project_id or self._project_id
        if not selected:
            raise client_error("project_id_required", "swarms.launch",
                               cause="pass project_id or set SYNTH_PROJECT_ID")
        return selected

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
            failure = closed.failure
            raise RunFailedError(
                0,
                ErrorDetail(
                    code=failure.code if failure else "run_failed",
                    service=SERVICE,
                    stage="swarms.run",
                    cause=closed.run_id,
                ),
            )
        return RunResult(run=closed, outputs=self.outputs(closed.run_id).outputs)

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
            project_id=self._project(project_id),
            limits=limits or default_limits(budget),
            message=message,
            **launch_fields,
        )
        return self._t.json(
            RunSnapshot, "POST", "/api/v1/swarms/runs", stage="swarms.launch",
            body=request, idempotency_key=_key(idempotency_key),
        )

    def get(self, run_id: str) -> RunSnapshot:
        return self._t.json(RunSnapshot, "GET", f"/api/v1/swarms/runs/{run_id}",
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

    def events(self, run_id: str, *, after: int | None = None,
               limit: int | None = None) -> RunEventPage:
        return self._t.json(
            RunEventPage, "GET", f"/api/v1/swarms/runs/{run_id}/events",
            stage="swarms.events", params={"after": after, "limit": limit},
        )

    def iter_events(self, run_id: str, *, after: int | None = None) -> Iterator[RunEvent]:
        """Every event currently available after ``after``, following pages."""
        cursor = after
        while True:
            page = self.events(run_id, after=cursor)
            yield from page.events
            if page.next_after is None or not page.events:
                return
            cursor = page.next_after

    def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> RunControlSummary:
        return self._t.json(
            RunControlSummary, "POST", f"/api/v1/swarms/runs/{run_id}/cancel",
            stage="swarms.cancel", idempotency_key=_key(idempotency_key),
        )

    def outputs(self, run_id: str) -> RunOutputList:
        return self._t.json(RunOutputList, "GET", f"/api/v1/swarms/runs/{run_id}/outputs",
                            stage="swarms.outputs")

    def output_content(self, run_id: str, output_id: str) -> bytes:
        return self._t.send(
            "GET", f"/api/v1/swarms/runs/{run_id}/outputs/{output_id}/content",
            stage="swarms.output_content",
        ).content

    def list(self, *, cursor: str | None = None, limit: int | None = None) -> RunPage:
        return self._t.json(RunPage, "GET", "/api/v1/swarms/runs", stage="swarms.list",
                            params={"cursor": cursor, "limit": limit})


class Account:
    """``client.account``: the caller's org, principal, limits and keys."""

    def __init__(self, transport: PublicTransport) -> None:
        self._t = transport

    def get(self) -> AccountModel:
        return self._t.json(AccountModel, "GET", "/v1/account", stage="account.get")

    def keys(self, *, purpose: str | None = None, offset: int | None = None,
             limit: int | None = None) -> ApiKeyList:
        return self._t.json(
            ApiKeyList, "GET", "/v1/api-keys", stage="account.keys",
            params={"purpose": purpose, "offset": offset, "limit": limit},
        )



class Billing:
    """``client.billing``: wallet balance, plan catalog and hosted checkout."""

    def __init__(self, transport: PublicTransport) -> None:
        self._t = transport

    def balance(self) -> Balance:
        return self._t.json(Balance, "GET", "/api/v1/account/billing/balance",
                            stage="billing.balance")

    def plans(self) -> PlanCatalog:
        return self._t.json(PlanCatalog, "GET", "/api/v1/account/billing/catalog",
                            stage="billing.plans", authenticated=False)

    def checkout(self, kind: str, *, plan_id: str | None = None,
                 amount_microcents: int | None = None) -> CheckoutSession:
        body = CheckoutRequest(kind=kind, plan_id=plan_id,  # type: ignore[arg-type]
                               amount_microcents=amount_microcents)
        return self._t.json(CheckoutSession, "POST", "/api/v1/account/billing/checkout",
                            stage="billing.checkout", body=body)
