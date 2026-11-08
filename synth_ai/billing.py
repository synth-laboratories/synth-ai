"""Exact decimal money and cursor-based billing reads."""

from __future__ import annotations

from ._generated.models import (
    Balance,
    BillingHistoryPage,
    BillingStatementPage,
    CheckoutAttempt,
    CheckoutRequest,
    CheckoutSession,
    Money,
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

    async def statements(
        self, *, months: int | None = None, cursor: str | None = None, limit: int = 20
    ) -> BillingStatementPage:
        return await self._transport.request(
            "listStatements",
            BillingStatementPage,
            query={"months": months, "cursor": cursor, "limit": limit},
        )

    async def checkout_status(self, checkout_id: str) -> CheckoutAttempt:
        return await self._transport.request(
            "getCheckout", CheckoutAttempt, path={"id": checkout_id}
        )

    async def checkout(
        self,
        *,
        plan_id: str | None = None,
        credit: Money | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutSession:
        body = CheckoutRequest(plan_id=plan_id, credit=credit)
        body = CheckoutRequest.model_validate(body.model_dump(exclude_none=True))
        return await self._transport.request(
            "createCheckout", CheckoutSession, body=body, idempotency_key=idempotency_key
        )
