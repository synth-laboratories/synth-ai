"""Exact decimal money and cursor-based billing reads."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    CheckoutRequestPackId,
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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

__all__ = ["AsyncBilling"]


class AsyncBilling:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def balance(self) -> Balance:
        """Wallet position and funding breakdown (weekly plan allowance, prepaid credits, holds, headroom) as exact USD Money values."""
        return await self._transport.request("getBalance", Balance)

    async def plans(self) -> PlanCatalog:
        """price_state is read from Stripe by lookup key (cached 60 s; a failed lookup is not cached).
        checkout_available is true only when POST /v1/billing/checkout can succeed for that plan:
        consoles grey out checkout otherwise.
        """
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
        """Each item is one append-only ledger entry: grants (weekly included usage, purchases,
        promotions, operator credits), holds and releases, settlements, refunds, clawbacks and
        restores. `delta_microcents` moves the ledger; `held_delta_microcents` moves held value.
        Refusals: `invalid_cursor` (422), `invalid_limit` (422), `invalid_history_filter` (422, detail names the parameter).
        """
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
        """One row per month that has ledger entries. Movements are sums of entry deltas by kind:
        purchases (purchased grants), allowance (included grants), credits (promotional and operator grants),
        conversion (legacy carry-over grants), charges (settlements, negative), refunds (Stripe/operator refunds
        and clawbacks), adjustments and expirations. `closing_ledger_microcents` = opening + the month's deltas;
        the latest closing equals the balance's `ledger_microcents`. `unspent_returned_microcents` is held value
        returned to available by cancel/failure refunds (it moves held, not the ledger). Usage columns sum the
        month's charged usage lines. Refusal: `invalid_limit` (422) when `months` is outside 1..36.
        """
        return await self._transport.request(
            "listStatements",
            BillingStatementPage,
            query={"months": months, "cursor": cursor, "limit": limit},
        )

    async def checkout_status(self, checkout_id: str) -> CheckoutAttempt:
        """The attempt row is recorded before Stripe is called, and its id is the Stripe idempotency key, so a checkout
        that lost its response stays readable. `state` is `pending` until Stripe answers, then `open`, then `completed`
        or `expired` from the Stripe webhook; `failed` means Stripe refused the session. Credits appear in the balance
        only from the webhook. Refusals: `invalid_id` (422), `checkout_not_found` (404).
        """
        return await self._transport.request(
            "getCheckout", CheckoutAttempt, path={"id": checkout_id}
        )

    async def checkout(
        self,
        *,
        plan_id: str | None = None,
        pack_id: CheckoutRequestPackId | None = None,
        credit: Money | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutSession:
        """Exactly one of `plan_id` (a purchasable recurring plan: `standard.monthly`, `max.monthly`),
        `pack_id` (a credit pack: `credits.10` $10, `credits.25` $25, `credits.100` $100; priced inline)
        or `credit` (`Money` between `$5.00` and `$10000.00` USD) is set. Refusals: `checkout_request_invalid` (422), `credit_pack_unknown` (422),
        `plan_not_purchasable` (422), `malformed_request` (422), `subscription_already_active` (409, for a `plan_id` when the org holds a subscription Stripe has not ended: `active`, `past_due`, `incomplete`, `unpaid` or `paused`; one subscription per org), `config_missing` (503, names the variable), `stripe_price_missing` (503, names the lookup key),
        `stripe_checkout_failed` (502, `upstream_kind: stripe`).
        """
        body = CheckoutRequest(plan_id=plan_id, pack_id=pack_id, credit=credit)
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
        """Requires an org admin or owner in addition to `account:billing:write` (owner decision 2026-11-10; buying
        stays open to any member). An API key or OAuth grant acts with its owning user's current role, read on every
        request. Ends the plan at the end of the current period, with no refund; the plan stays active until then. A
        `past_due`, `unpaid` or `paused` subscription ends now. Already ending or already ended: answers the stored state and
        writes nothing to Stripe. The request takes no body (or `{}`). A refusal before the Stripe write (403, 404, 409)
        leaves the Idempotency-Key unused; a 5xx releases it, so the same key may retry. Refusals:
        `billing_admin_required` (403, not an org admin), `subscription_not_found` (404), `subscription_not_self_serve` (409, no Stripe subscription),
        `subscription_not_cancelable` (409, `incomplete` or `incomplete_expired`), `subscription_binding_conflict` (409),
        `idempotency_in_progress` (409, `Retry-After: 1`), `stripe_subscription_read_failed` (502),
        `stripe_subscription_cancel_failed` (502, `upstream_status` and Stripe's code as `cause`),
        `config_missing` (503).
        """
        return await self._transport.request(
            "cancelSubscription", Subscription, idempotency_key=idempotency_key
        )

    async def resume_subscription(self, *, idempotency_key: str | None = None) -> Subscription:
        """Requires an org admin or owner in addition to `account:billing:write`; an API key acts with its owning user's
        current role. Undoes a pending cancel; the plan renews at period end. The request takes no body (or `{}`).
        A refusal before the Stripe write (403, 404, 409) leaves the Idempotency-Key unused; a 5xx releases it.
        Refusals: `billing_admin_required` (403), `subscription_not_found` (404), `subscription_not_self_serve` (409), `subscription_not_resumable` (409, no
        cancel pending or the subscription already ended), `subscription_binding_conflict` (409),
        `idempotency_in_progress` (409, `Retry-After: 1`), `stripe_subscription_read_failed` (502),
        `stripe_subscription_resume_failed` (502, `upstream_status` and Stripe's code as `cause`), `config_missing` (503).
        """
        return await self._transport.request(
            "resumeSubscription", Subscription, idempotency_key=idempotency_key
        )

    async def preview_subscription_change(self, plan_id: str) -> SubscriptionChangePreview:
        """Stripe's invoice preview of the switch `changeSubscriptionPlan` would make: the new plan starts a new monthly
        period now and the unused part of the current plan is credited against it. `amount_due_now` is what the
        saved card is charged now; `credit_applied` is the unused part of the current plan; a credit larger than
        the new price stays as account credit for later invoices. Writes nothing. Any member with
        `account:billing:read` may read it. Refusals: `plan_unknown` (400), `subscription_not_found` (404),
        `subscription_cancel_pending` (409), `subscription_not_changeable` (409, `past_due`, `unpaid`, `paused` or
        `incomplete`), `subscription_plan_unchanged` (409), `subscription_not_self_serve` (409),
        `subscription_binding_conflict` (409), `stripe_subscription_read_failed` (502),
        `stripe_invoice_preview_failed` (502), `stripe_price_missing` (503), `config_missing` (503).
        """
        return await self._transport.request(
            "previewSubscriptionChange", SubscriptionChangePreview, query={"plan_id": plan_id}
        )

    async def change_subscription_plan(
        self, plan_id: str, *, idempotency_key: str | None = None
    ) -> Subscription:
        """Requires an org admin or owner in addition to `account:billing:write`; an API key acts with its owning
        user's current role. The switch takes effect now, upgrade or downgrade: the new plan starts a new monthly
        period now, the unused part of the current plan is credited against the first charge (any excess stays
        as account credit, never refunded), and the saved card is charged the rest. The switch applies only if that
        payment succeeds; otherwise the plan is unchanged. The weekly allowance restarts at the new plan's amount;
        the unused allowance of the current week is not carried over. Prepaid credit is untouched.
        A refusal before the Stripe write (400, 403, 404, 409) leaves the Idempotency-Key unused; a 5xx releases it,
        so the same key may retry. Refusals: `plan_unknown` (400), `billing_admin_required` (403),
        `subscription_change_payment_failed` (402, the payment was declined; the plan is unchanged and the
        declined switch is final: its open invoice is voided, so paying it later applies nothing),
        `subscription_not_found` (404), `subscription_cancel_pending` (409, resume first),
        `subscription_not_changeable` (409, `past_due`, `unpaid`, `paused` or `incomplete`; update the card first),
        `subscription_plan_unchanged` (409), `subscription_not_self_serve` (409),
        `subscription_binding_conflict` (409), `idempotency_in_progress` (409, `Retry-After: 1`),
        `stripe_subscription_read_failed` (502), `stripe_subscription_change_failed` (502, `upstream_status` and
        Stripe's code as `cause`), `stripe_price_missing` (503), `config_missing` (503).
        """
        body = SubscriptionChangeRequest.model_validate({"plan_id": plan_id})
        return await self._transport.request(
            "changeSubscriptionPlan", Subscription, body=body, idempotency_key=idempotency_key
        )

    async def payment_method_session(
        self, *, idempotency_key: str | None = None
    ) -> PaymentMethodSession:
        """Requires an org admin or owner in addition to `account:billing:write`. Returns a short-lived Stripe portal URL
        limited to updating payment methods and viewing invoices (a dedicated portal configuration Core creates on first
        use and finds again by its metadata `synth_purpose=payment_methods`; the account's default configuration is
        never used). The portal returns to the console's Billing page. Paying a past-due or unpaid invoice with the new
        card returns the subscription to `active` through Stripe's webhooks. The request takes no body (or `{}`).
        A refusal before the Stripe call leaves the Idempotency-Key unused; a 5xx releases it. Refusals:
        `billing_admin_required` (403), `billing_customer_not_found` (404, the org has no Stripe customer),
        `portal_configuration_unsafe` (503, Core's tagged configuration allows more than payment methods and invoices),
        `stripe_portal_configuration_read_failed`, `stripe_portal_configuration_create_failed`,
        `stripe_portal_session_failed` (502, `upstream_kind: stripe`), `config_missing` (503).
        """
        return await self._transport.request(
            "createPaymentMethodSession", PaymentMethodSession, idempotency_key=idempotency_key
        )

    async def auto_reload(self) -> AutoReload:
        """Off by default for every org. When on, a debit that leaves the org's purchased-credit balance (promotional and
        plan credit do not count) below `threshold` charges the saved card `amount` off-session: at most one charge in
        flight, at most `disclosure.limits.daily_count` charges per UTC day and at most
        `disclosure.limits.monthly_ceiling` charged per UTC calendar month. A reload a limit blocks is skipped (one
        email and the `auto_reload_capped` billing attention until the window ends; auto-reload stays on). Nothing is
        charged while the org is suspended. The credits are granted when Stripe confirms the payment. Core turns
        auto-reload off and sets `disabled_reason` (open set) when: a charge is declined or the bank asks the customer
        to authenticate (Stripe's code, for example `card_declined`, `authentication_required`); Stripe refuses the
        charge or gives no answer after its retries (`charge_failed`); a charge is disputed (`dispute`); no card is
        saved when a reload is due or the card is detached (`card_removed`; older rows say `payment_method_missing`);
        the org's last admin or owner is removed (`admin_removed`). The admins are emailed. `payment_method_on_file`
        reads the Stripe customer's saved card; `disclosure` is what an admin must be shown before turning it on, with
        the card a reload would charge now (a card change does not turn it off). Any member with
        `account:billing:read` may read it. Refusals: `stripe_customer_read_failed` (502),
        `stripe_payment_method_read_failed` (502), `config_missing` (503).
        """
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
        """Alerts fire when the org's settled spend in the current UTC month (the charges behind `getUsageSummary`)
        reaches 50, 75, 90 and 100 percent of `monthly_amount`, once per threshold per month. `monthly_amount` is
        null until the org sets one, and no alert fires then. `last_notified` is the most recent recorded alert,
        or null. No email is sent; this field and the console are the notice.
        Any member with `account:billing:read` may read it.
        """
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

    async def set_spend_alerts_request(
        self, request: SetSpendAlertsRequest, *, idempotency_key: str | None = None
    ) -> SpendAlerts:
        """Apply the canonical request, preserving whether enabled was supplied."""
        return await self._transport.request(
            "setSpendAlerts", SpendAlerts, body=request, idempotency_key=idempotency_key
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
        """Yield billing history across cursor pages, preserving the supplied filters."""

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
        """Yield billing statements across cursor pages."""

        async def fetch(next_cursor: str | None) -> BillingStatementPage:
            return await self.statements(months=months, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item
