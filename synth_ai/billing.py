"""Exact decimal money and cursor-based billing reads."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from ._generated.models import (
    AutoReload,
    AutoReloadRequest,
    Balance,
    BillingHistoryEntry,
    BillingHistoryPage,
    BillingStatement,
    BillingStatementPage,
    CheckoutAttempt,
    CheckoutRequest,
    CheckoutSession,
    Money,
    PaymentMethodSession,
    PlanCatalog,
    SetSpendAlertsRequest,
    SpendAlerts,
    Subscription,
    SubscriptionChangePreview,
    SubscriptionChangeRequest,
)
from ._pagination import paginate
from ._transport import Transport
from .errors import SynthError


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

    async def subscription(self) -> Subscription | None:
        """The org's subscription, or None when it has never had one."""
        result = await self._transport.invoke("getSubscription")
        if result is not None and not isinstance(result, Subscription):
            raise SynthError("Unexpected response model for getSubscription")
        return result

    async def cancel_subscription(self, *, idempotency_key: str | None = None) -> Subscription:
        return await self._transport.request(
            "cancelSubscription", Subscription, idempotency_key=idempotency_key
        )

    async def resume_subscription(self, *, idempotency_key: str | None = None) -> Subscription:
        return await self._transport.request(
            "resumeSubscription", Subscription, idempotency_key=idempotency_key
        )

    async def preview_subscription_change(self, plan_id: str) -> SubscriptionChangePreview:
        return await self._transport.request(
            "previewSubscriptionChange", SubscriptionChangePreview, query={"plan_id": plan_id}
        )

    async def change_subscription_plan(
        self, plan_id: str, *, idempotency_key: str | None = None
    ) -> Subscription:
        body = SubscriptionChangeRequest.model_validate({"plan_id": plan_id})
        return await self._transport.request(
            "changeSubscriptionPlan", Subscription, body=body, idempotency_key=idempotency_key
        )

    async def payment_method_session(
        self, *, idempotency_key: str | None = None
    ) -> PaymentMethodSession:
        return await self._transport.request(
            "createPaymentMethodSession", PaymentMethodSession, idempotency_key=idempotency_key
        )

    async def auto_reload(self) -> AutoReload:
        return await self._transport.request("getAutoReload", AutoReload)

    async def set_auto_reload(
        self,
        *,
        enabled: bool,
        threshold: Money | None = None,
        amount: Money | None = None,
        disclosure_version: str | None = None,
        idempotency_key: str | None = None,
    ) -> AutoReload:
        """Save auto-reload; the caller must be an org admin or owner.

        Turning it off works with an API key. Turning it on, or changing it while on,
        needs a signed-in browser session, so an API key or OAuth grant is refused
        with `browser_session_required` (403); enable it in the console.
        `disclosure_version` is the `disclosure.version` from `auto_reload()`.
        """
        fields: dict[str, bool | Money | str] = {"enabled": enabled}
        if threshold is not None:
            fields["threshold"] = threshold
        if amount is not None:
            fields["amount"] = amount
        if disclosure_version is not None:
            fields["disclosure_version"] = disclosure_version
        body = AutoReloadRequest.model_validate(fields)
        return await self._transport.request(
            "setAutoReload", AutoReload, body=body, idempotency_key=idempotency_key
        )

    async def spend_alerts(self) -> SpendAlerts:
        return await self._transport.request("getSpendAlerts", SpendAlerts)

    async def set_spend_alerts(
        self,
        *,
        monthly_amount: Money | None,
        enabled: bool | None = None,
        idempotency_key: str | None = None,
    ) -> SpendAlerts:
        """Set the monthly alert amount; None clears it. `enabled` is left unchanged if omitted."""
        fields: dict[str, bool | Money | None] = {"monthly_amount": monthly_amount}
        if enabled is not None:
            fields["enabled"] = enabled
        body = SetSpendAlertsRequest.model_validate(fields)
        return await self._transport.request(
            "setSpendAlerts", SpendAlerts, body=body, idempotency_key=idempotency_key
        )

    async def iter_history(
        self,
        *,
        cursor: str | None = None,
        limit: int = 20,
        cause: str | None = None,
        run_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> AsyncGenerator[BillingHistoryEntry, None]:
        async def fetch(next_cursor: str | None) -> BillingHistoryPage:
            return await self.history(
                cursor=next_cursor,
                limit=limit,
                cause=cause,
                run_id=run_id,
                since=since,
                until=until,
            )

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_statements(
        self, *, months: int | None = None, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[BillingStatement, None]:
        async def fetch(next_cursor: str | None) -> BillingStatementPage:
            return await self.statements(months=months, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item
