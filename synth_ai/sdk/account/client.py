"""Read-only account policy; see backend ACCOUNT_POLICY_ENGINE.md."""

from datetime import datetime, timedelta

from synth_ai.contracts.account import ALERTS, BALANCE, CONTEXT, EVENTS, LIMITS, PLAN, USAGE
from synth_ai.contracts.operations import ACCOUNT_ENTITLEMENTS
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.options import ReadOptions
from synth_ai.types.account import AccountEntitlements
from synth_ai.types.account_cloud import (
    AccountBalance,
    AccountContext,
    AccountLimits,
    AccountUsageEventPage,
    AccountUsageSummary,
    CreditAlertState,
    CurrentPlan,
    Product,
)


def _window(
    period_start: datetime, period_end: datetime, product: Product | None, actor_id: str | None
) -> dict[str, str | int]:
    if period_start.utcoffset() is None or period_end.utcoffset() is None:
        raise ValueError("Usage periods require timezone-aware timestamps")
    if not timedelta(0) < period_end - period_start <= timedelta(days=366):
        raise ValueError("Usage period must be positive and at most 366 days")
    query: dict[str, str | int] = {
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
    }
    if product is not None:
        if product not in {"index", "swarms", "monitor", "inference"}:
            raise ValueError("Unknown usage product")
        query["product"] = product
    if actor_id is not None:
        if not 1 <= len(actor_id) <= 256:
            raise ValueError("Actor filter must be bounded")
        query["actor_id"] = actor_id
    return query


def _page(query: dict[str, str | int], cursor: str | None, limit: int) -> dict[str, str | int]:
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("Event limit must be 1..200")
    query["limit"] = limit
    if cursor is not None:
        if not 1 <= len(cursor) <= 2048:
            raise ValueError("Event cursor must be bounded")
        query["cursor"] = cursor
    return query


class AccountClient:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def entitlements(self, *, request: ReadOptions | None = None) -> AccountEntitlements:
        """Read this organization's current access conditions and observed limits."""
        return self._transport.request(ACCOUNT_ENTITLEMENTS, AccountEntitlements, options=request)

    def context(self, *, request: ReadOptions | None = None) -> AccountContext:
        return self._transport.request(CONTEXT, AccountContext, options=request)

    def plan(self, *, request: ReadOptions | None = None) -> CurrentPlan:
        return self._transport.request(PLAN, CurrentPlan, options=request)

    def balance(self, *, request: ReadOptions | None = None) -> AccountBalance:
        return self._transport.request(BALANCE, AccountBalance, options=request)

    def alerts(self, *, request: ReadOptions | None = None) -> CreditAlertState:
        return self._transport.request(ALERTS, CreditAlertState, options=request)

    def limits(self, *, request: ReadOptions | None = None) -> AccountLimits:
        return self._transport.request(LIMITS, AccountLimits, options=request)

    def usage(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        product: Product | None = None,
        actor_id: str | None = None,
        request: ReadOptions | None = None,
    ) -> AccountUsageSummary:
        return self._transport.request(
            USAGE,
            AccountUsageSummary,
            query=_window(period_start, period_end, product, actor_id),
            options=request,
        )

    def events(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        product: Product | None = None,
        actor_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        request: ReadOptions | None = None,
    ) -> AccountUsageEventPage:
        return self._transport.request(
            EVENTS,
            AccountUsageEventPage,
            query=_page(_window(period_start, period_end, product, actor_id), cursor, limit),
            options=request,
        )


class AsyncAccountClient:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def entitlements(self, *, request: ReadOptions | None = None) -> AccountEntitlements:
        """Read this organization's current access conditions and observed limits."""
        return await self._transport.request(
            ACCOUNT_ENTITLEMENTS, AccountEntitlements, options=request
        )

    async def context(self, *, request: ReadOptions | None = None) -> AccountContext:
        return await self._transport.request(CONTEXT, AccountContext, options=request)

    async def plan(self, *, request: ReadOptions | None = None) -> CurrentPlan:
        return await self._transport.request(PLAN, CurrentPlan, options=request)

    async def balance(self, *, request: ReadOptions | None = None) -> AccountBalance:
        return await self._transport.request(BALANCE, AccountBalance, options=request)

    async def alerts(self, *, request: ReadOptions | None = None) -> CreditAlertState:
        return await self._transport.request(ALERTS, CreditAlertState, options=request)

    async def limits(self, *, request: ReadOptions | None = None) -> AccountLimits:
        return await self._transport.request(LIMITS, AccountLimits, options=request)

    async def usage(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        product: Product | None = None,
        actor_id: str | None = None,
        request: ReadOptions | None = None,
    ) -> AccountUsageSummary:
        return await self._transport.request(
            USAGE,
            AccountUsageSummary,
            query=_window(period_start, period_end, product, actor_id),
            options=request,
        )

    async def events(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        product: Product | None = None,
        actor_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        request: ReadOptions | None = None,
    ) -> AccountUsageEventPage:
        return await self._transport.request(
            EVENTS,
            AccountUsageEventPage,
            query=_page(_window(period_start, period_end, product, actor_id), cursor, limit),
            options=request,
        )
