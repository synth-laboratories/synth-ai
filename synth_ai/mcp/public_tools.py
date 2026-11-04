"""Public MCP tools on the Swarms v1 contract, executed through the public client.

These tools are not capability-negotiated: synth-backend core is the authority
and refuses with the typed error body, which the server reports verbatim.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from synth_ai.public.models import (
    Account,
    Balance,
    PlanCatalog,
    RunControlSummary,
    RunEventPage,
    RunOutputList,
    RunPage,
    RunResult,
    RunSnapshot,
)

from .contracts import JSON_OBJECT, JsonObject

if TYPE_CHECKING:
    from synth_ai.client import AsyncSynthClient

CONTRACT = "swarms-public.v1"

RequestT = TypeVar("RequestT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_Args):
    pass


class RunArgs(_Args):
    message: str = Field(min_length=1, max_length=8192)
    budget_usd: str = Field(pattern=r"^\d+(\.\d+)?$", description="Maximum charge in USD.")
    project_id: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0, le=86400)


class LaunchArgs(_Args):
    message: str = Field(min_length=1, max_length=8192)
    budget_usd: str = Field(pattern=r"^\d+(\.\d+)?$", description="Maximum charge in USD.")
    project_id: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)


class RunRef(_Args):
    run_id: str = Field(min_length=1, max_length=128)


class WaitArgs(RunRef):
    timeout_seconds: float | None = Field(default=None, gt=0, le=86400)


class EventsArgs(RunRef):
    after: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=1, le=500)


class CancelArgs(RunRef):
    idempotency_key: str | None = Field(default=None, max_length=128)


class ListArgs(_Args):
    cursor: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


@dataclass(frozen=True, slots=True)
class PublicTool(Generic[RequestT, ResultT]):
    name: str
    operation: str
    description: str
    scope: str | None
    request: type[RequestT]
    result: type[ResultT]
    execute: Callable[[AsyncSynthClient, RequestT], Awaitable[ResultT]]
    mutates: bool = False
    paid: bool = False
    keyed: bool = True
    additional_scopes: tuple[str, ...] = ()
    negotiated = False

    @property
    def idempotent(self) -> bool:
        # Mutations send an Idempotency-Key the caller can supply.
        return True

    def scope_for(self, arguments: JsonObject) -> str | None:
        return self.scope

    def input_schema(self) -> JsonObject:
        return JSON_OBJECT.validate_python(self.request.model_json_schema())

    def output_schema(self) -> JsonObject:
        return JSON_OBJECT.validate_python(self.result.model_json_schema())

    def validate_arguments(self, arguments: JsonObject) -> JsonObject:
        parsed = self.request.model_validate_json(JSON_OBJECT.dump_json(arguments))
        return JSON_OBJECT.validate_json(parsed.model_dump_json())

    def validate_result(self, result: JsonObject) -> JsonObject:
        parsed = self.result.model_validate_json(JSON_OBJECT.dump_json(result))
        return JSON_OBJECT.validate_json(parsed.model_dump_json(exclude_none=True))

    async def invoke(self, client: AsyncSynthClient, arguments: JsonObject) -> JsonObject:
        parsed = self.request.model_validate_json(JSON_OBJECT.dump_json(arguments))
        result = await self.execute(client, parsed)
        return JSON_OBJECT.validate_json(result.model_dump_json(exclude_none=True))


async def _run(c: AsyncSynthClient, a: RunArgs) -> RunResult:
    return await c.swarms.run(
        a.message, a.budget_usd, project_id=a.project_id, timeout_seconds=a.timeout_seconds
    )


async def _launch(c: AsyncSynthClient, a: LaunchArgs) -> RunSnapshot:
    return await c.swarms.launch(
        a.message, a.budget_usd, project_id=a.project_id, idempotency_key=a.idempotency_key
    )


async def _get(c: AsyncSynthClient, a: RunRef) -> RunSnapshot:
    return await c.swarms.get(a.run_id)


async def _wait(c: AsyncSynthClient, a: WaitArgs) -> RunSnapshot:
    return await c.swarms.wait(a.run_id, timeout_seconds=a.timeout_seconds)


async def _events(c: AsyncSynthClient, a: EventsArgs) -> RunEventPage:
    return await c.swarms.events(a.run_id, after=a.after, limit=a.limit)


async def _cancel(c: AsyncSynthClient, a: CancelArgs) -> RunControlSummary:
    return await c.swarms.cancel(a.run_id, idempotency_key=a.idempotency_key)


async def _outputs(c: AsyncSynthClient, a: RunRef) -> RunOutputList:
    return await c.swarms.outputs(a.run_id)


async def _list(c: AsyncSynthClient, a: ListArgs) -> RunPage:
    return await c.swarms.list(cursor=a.cursor, limit=a.limit)


async def _account(c: AsyncSynthClient, a: NoArgs) -> Account:
    return await c.account.get()


async def _balance(c: AsyncSynthClient, a: NoArgs) -> Balance:
    return await c.billing.balance()


async def _plans(c: AsyncSynthClient, a: NoArgs) -> PlanCatalog:
    return await c.billing.plans()


PUBLIC_TOOLS: tuple[PublicTool, ...] = (
    PublicTool("swarms_run", "swarms.run", "Launch a Swarms run within a USD budget, wait for it "
               "to close and return its outputs.", "swarms:run", RunArgs, RunResult, _run,
               mutates=True, paid=True),
    PublicTool("swarms_launch", "swarms.launch", "Launch a Swarms run within a USD budget.",
               "swarms:run", LaunchArgs, RunSnapshot, _launch, mutates=True, paid=True),
    PublicTool("swarms_get", "swarms.get", "Read a run snapshot.", "swarms:read", RunRef,
               RunSnapshot, _get),
    PublicTool("swarms_wait", "swarms.wait", "Wait until a run closes.", "swarms:read",
               WaitArgs, RunSnapshot, _wait),
    PublicTool("swarms_events", "swarms.events", "Read ordered run events after a sequence.",
               "swarms:read", EventsArgs, RunEventPage, _events),
    PublicTool("swarms_cancel", "swarms.cancel", "Cancel a run.", "swarms:control", CancelArgs,
               RunControlSummary, _cancel, mutates=True),
    PublicTool("swarms_outputs", "swarms.outputs", "List a run's outputs.", "swarms:read",
               RunRef, RunOutputList, _outputs),
    PublicTool("swarms_list", "swarms.list", "List runs, newest first.", "swarms:read",
               ListArgs, RunPage, _list),
    PublicTool("account_get", "account.get", "Read the caller's org, principal and limits.",
               None, NoArgs, Account, _account),
    PublicTool("billing_balance", "billing.balance", "Read the wallet balance in microcents.",
               "account:billing:read", NoArgs, Balance, _balance),
    PublicTool("billing_plans", "billing.plans", "List plans.", None, NoArgs, PlanCatalog,
               _plans, keyed=False),
)

PUBLIC_OPERATIONS = frozenset(tool.operation for tool in PUBLIC_TOOLS)


def is_negotiated(tool: object) -> bool:
    """Legacy tools are capability-negotiated; public v1 tools are not."""
    return getattr(tool, "negotiated", True)
