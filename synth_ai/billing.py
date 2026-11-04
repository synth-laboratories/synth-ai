"""Exact decimal money and cursor-based billing reads."""

from __future__ import annotations

from typing import Any

from ._generated.models import (
    Balance,
    BillingHistoryPage,
    BillingStatementPage,
    CheckoutSession,
    PlanCatalog,
)
from ._transport import Transport


class AsyncBilling:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def balance(self) -> Balance:
        return await self._transport.request("getBalance", Balance)

    async def plans(self) -> PlanCatalog:
        return await self._transport.request("listPlans", PlanCatalog)

    async def history(
        self,
        *,
        cursor: str | None = None,
        limit: int = 20,
        cause: str | None = None,
        run_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> BillingHistoryPage:
        return await self._transport.request(
            "listBillingHistory",
            BillingHistoryPage,
            query={
                "cursor": cursor,
                "limit": limit,
                "cause": cause,
                "run_id": run_id,
                "since": since,
                "until": until,
            },
        )

    async def invoices(self, *, cursor: str | None = None, limit: int = 20) -> BillingStatementPage:
        return await self._transport.request(
            "listInvoices", BillingStatementPage, query={"cursor": cursor, "limit": limit}
        )

    async def checkout(
        self,
        *,
        plan_id: str | None = None,
        credit: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutSession:
        body: dict[str, Any] = {}
        if plan_id is not None:
            body["plan_id"] = plan_id
        if credit is not None:
            body["credit"] = credit
        return await self._transport.request(
            "createCheckout", CheckoutSession, body=body, idempotency_key=idempotency_key
        )
