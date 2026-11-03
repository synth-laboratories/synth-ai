"""The account.v1 snapshot shape shared by local and qualified hosted MCP."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field

from synth_ai.errors import AuthenticationError, SynthError
from synth_ai.options import ReadOptions
from synth_ai.types.account_cloud import AccountBalance, AccountUsageSummary, CurrentPlan
from synth_ai.types.base import Model


class Unavailable(Model):
    state: Literal["unavailable"] = "unavailable"
    code: Literal["account_scope_required"] | None = None


class PlanReady(Model):
    state: Literal["ready"] = "ready"
    value: CurrentPlan


class BalanceReady(Model):
    state: Literal["ready"] = "ready"
    value: AccountBalance


class UsageReady(Model):
    state: Literal["ready"] = "ready"
    value: AccountUsageSummary


class AccountUsageSnapshot(Model):
    model_config = ConfigDict(serialize_by_alias=True)

    schema_version: Literal["synth.account-usage.v1"] = Field(
        default="synth.account-usage.v1", alias="schema"
    )
    plan: PlanReady | Unavailable
    balance: BalanceReady | Unavailable
    usage: UsageReady | Unavailable


async def account_usage_snapshot(client, _arguments) -> AccountUsageSnapshot:
    """Use the selected client's account reads; never fall back to another identity."""
    now = datetime.now(UTC)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    options = ReadOptions(timeout_seconds=8, read_retry_count=0)

    async def observe(call, ready):
        try:
            value = await asyncio.wait_for(call(), timeout=9)
            return ready(value=value)
        except AuthenticationError:
            raise
        except (SynthError, TimeoutError) as error:
            code = (
                "account_scope_required"
                if getattr(error, "code", None) == "account_scope_required"
                else None
            )
            return Unavailable(code=code)

    async def usage():
        if start == now:
            return Unavailable()
        return await observe(lambda: client.account.usage(start, now, request=options), UsageReady)

    plan, balance, current_usage = await asyncio.gather(
        observe(lambda: client.account.plan(request=options), PlanReady),
        observe(lambda: client.account.balance(request=options), BalanceReady),
        usage(),
    )
    return AccountUsageSnapshot(plan=plan, balance=balance, usage=current_usage)
