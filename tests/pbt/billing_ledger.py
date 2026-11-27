"""Exact-integer billing wallet, Stripe webhook, auto-reload, promo & plan-switch model for SDK PBT (SYN-6050)."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from .hosted_catalog import cents_to_money_str, money_str_to_cents

__all__ = ["BillingLedgerState", "HoldRecord", "make_problem"]

_FIXED_TS = "2026-11-27T18:00:00Z"
_ORG_ID = "00000000-0000-4000-8000-000000000001"


def _money(cents: int) -> dict[str, JsonValue]:
    return {"amount": cents_to_money_str(cents), "currency": "USD"}


def make_problem(
    status: int, code: str, detail: str, *, stage: str = "public_api"
) -> dict[str, JsonValue]:
    return {
        "type": f"https://api.usesynth.ai/problems/{code}",
        "title": code.replace("_", " ").title(),
        "status": status,
        "detail": detail,
        "instance": "/v1/pbt",
        "code": code,
        "service": "synth-backend",
        "stage": stage,
        "outcome": "not_accepted",
    }


@dataclass
class HoldRecord:
    hold_id: str
    placed_cents: int
    settled_cents: int = 0
    released_cents: int = 0

    @property
    def active_cents(self) -> int:
        return self.placed_cents - self.settled_cents - self.released_cents


class BillingLedgerState:
    """Exact-cent billing ledger implementing Free ($5/day) vs PAYG ($100/day, $500/week) invariants."""

    def __init__(self) -> None:
        self._seq = 0
        self.plan: str = "free"
        self.included_remaining_cents: int = 1000
        self.included_limit_cents: int = 1000
        self.purchased_cents: int = 0
        self.promotional_cents: int = 0
        self.operator_cents: int = 0
        self.held_cents: int = 0
        self.daily_spend_cents: int = 0
        self.weekly_spend_cents: int = 0
        self.holds: dict[str, HoldRecord] = {}
        self._stripe_events: dict[str, dict[str, JsonValue]] = {}
        self._redeemed_promos: set[str] = set()
        self._promo_campaigns: dict[str, tuple[str, int]] = {
            "LAUNCH5": ("20000000-0000-4000-8000-000000000101", 500),
            "RESEARCH10": ("20000000-0000-4000-8000-000000000102", 1000),
            "SWARM20": ("20000000-0000-4000-8000-000000000103", 2000),
        }
        self.auto_reload_enabled: bool = False
        self.auto_reload_threshold_cents: int = 1000
        self.auto_reload_amount_cents: int = 2500
        self.auto_reload_disabled_reason: str | None = None
        self.auto_reload_daily_count: int = 0
        self.auto_reload_monthly_cents: int = 0
        self.subscription_status: str = "active"
        self.subscription_cancel_at_period_end: bool = False

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:022d}"

    def _next_uuid(self) -> str:
        self._seq += 1
        return str(uuid.UUID(int=0x20000000_0000_4000_8000_000000000000 + self._seq))

    @property
    def ledger_cents(self) -> int:
        return (
            self.purchased_cents
            + self.promotional_cents
            + self.operator_cents
            + self.included_remaining_cents
        )

    @property
    def available_cents(self) -> int:
        return self.ledger_cents - self.held_cents

    @property
    def payg_credits_active(self) -> bool:
        return self.plan == "free" and self.purchased_cents > 0

    @property
    def daily_cap_cents(self) -> int:
        if self.plan == "free":
            return 10000 if self.payg_credits_active else 500
        if self.plan == "standard.monthly":
            return 4000
        return 40000

    @property
    def weekly_cap_cents(self) -> int:
        if self.plan == "free":
            return 50000 if self.payg_credits_active else 0
        if self.plan == "standard.monthly":
            return 15000
        return 150000

    def deliver_stripe_webhook(
        self,
        *,
        event_id: str,
        amount_cents: int,
        payment_succeeded: bool = True,
        amount_matches_checkout: bool = True,
    ) -> tuple[int, dict[str, JsonValue], bool]:
        if event_id in self._stripe_events:
            return 200, copy.deepcopy(self._stripe_events[event_id]), True
        if (
            not payment_succeeded
            or not amount_matches_checkout
            or amount_cents < 500
            or amount_cents > 1_000_000
        ):
            return (
                409,
                make_problem(
                    409,
                    "stripe_payment_evidence_conflict",
                    "Checkout Session and Payment Intent evidence disagree.",
                    stage="billing_stripe_webhook",
                ),
                False,
            )
        self.purchased_cents += amount_cents
        receipt: dict[str, JsonValue] = {
            "event_id": event_id,
            "org_id": _ORG_ID,
            "credited_cents": amount_cents,
            "purchased_cents": self.purchased_cents,
        }
        self._stripe_events[event_id] = receipt
        return 200, copy.deepcopy(receipt), False

    def reserve_hold(self, amount_cents: int) -> tuple[bool, str | None, str | None]:
        if amount_cents <= 0:
            return False, None, "hold_amount_invalid"
        if amount_cents > self.available_cents:
            return False, None, "insufficient_balance"
        if self.daily_spend_cents + amount_cents > self.daily_cap_cents:
            return False, None, "daily_spend_cap_exceeded"
        if (
            self.weekly_cap_cents > 0
            and self.weekly_spend_cents + amount_cents > self.weekly_cap_cents
        ):
            return False, None, "weekly_spend_cap_exceeded"
        hold_id = self._next_id("hld")
        self.holds[hold_id] = HoldRecord(hold_id=hold_id, placed_cents=amount_cents)
        self.held_cents += amount_cents
        self.daily_spend_cents += amount_cents
        self.weekly_spend_cents += amount_cents
        return True, hold_id, None

    def settle_hold(self, hold_id: str, debit_cents: int) -> tuple[bool, str | None]:
        hold = self.holds.get(hold_id)
        if hold is None or hold.active_cents <= 0:
            return False, "hold_not_active"
        if debit_cents < 0 or debit_cents > hold.active_cents:
            return False, "debit_exceeds_hold"
        release_cents = hold.active_cents - debit_cents
        rem = debit_cents
        take_inc = min(self.included_remaining_cents, rem)
        rem -= take_inc
        take_promo = min(self.promotional_cents, rem)
        rem -= take_promo
        take_purch = min(self.purchased_cents, rem)
        rem -= take_purch
        if rem != 0:
            return False, "ledger_shortfall"

        next_purchased = self.purchased_cents - take_purch
        next_daily = max(0, self.daily_spend_cents - release_cents)
        next_weekly = max(0, self.weekly_spend_cents - release_cents)
        next_payg = self.plan == "free" and next_purchased > 0
        next_daily_cap = (
            (10000 if next_payg else 500) if self.plan == "free" else self.daily_cap_cents
        )
        if next_daily > next_daily_cap:
            return False, "spend_window_would_exceed_free_cap"

        self.included_remaining_cents -= take_inc
        self.promotional_cents -= take_promo
        self.purchased_cents = next_purchased
        self.held_cents -= hold.active_cents
        hold.settled_cents += debit_cents
        hold.released_cents += release_cents
        self.daily_spend_cents = next_daily
        self.weekly_spend_cents = next_weekly
        return True, None

    def trigger_auto_reload_if_due(self, *, card_declined: bool = False) -> tuple[bool, str]:
        if not self.auto_reload_enabled:
            return False, "disabled"
        if self.purchased_cents >= self.auto_reload_threshold_cents:
            return False, "above_threshold"
        if card_declined:
            self.auto_reload_enabled = False
            self.auto_reload_disabled_reason = "card_declined"
            return False, "card_declined"
        if self.auto_reload_daily_count >= 3:
            return False, "daily_cap_reached"
        if self.auto_reload_monthly_cents + self.auto_reload_amount_cents > 500_000:
            return False, "monthly_ceiling_reached"
        self.purchased_cents += self.auto_reload_amount_cents
        self.auto_reload_daily_count += 1
        self.auto_reload_monthly_cents += self.auto_reload_amount_cents
        return True, "reloaded"

    def rollover_utc_day(self) -> None:
        self.daily_spend_cents = 0
        self.auto_reload_daily_count = 0

    def balance_payload(self) -> dict[str, JsonValue]:
        used_inc = max(0, self.included_limit_cents - self.included_remaining_cents)
        return {
            "org_id": _ORG_ID,
            "plan": self.plan,
            "payg_credits_active": self.payg_credits_active,
            "ledger": _money(self.ledger_cents),
            "held": _money(self.held_cents),
            "headroom": _money(self.available_cents),
            "available": _money(self.available_cents),
            "purchased_credits": _money(self.purchased_cents),
            "promotional_credits": _money(self.promotional_cents),
            "operator_credits": _money(self.operator_cents),
            "debt": _money(0),
            "included_allowance": {
                "limit": _money(self.included_limit_cents),
                "used": _money(used_inc),
                "remaining": _money(self.included_remaining_cents),
                "reset_at": "2026-12-04T00:00:00Z",
            },
        }

    def auto_reload_payload(self) -> dict[str, JsonValue]:
        return {
            "enabled": self.auto_reload_enabled,
            "threshold": _money(self.auto_reload_threshold_cents),
            "amount": _money(self.auto_reload_amount_cents),
            "disabled_reason": self.auto_reload_disabled_reason,
            "payment_method_on_file": True,
            "disclosure": {
                "version": "2026-11-10.1",
                "template": "Charge {brand} ending {last4} {amount} below {threshold}.",
                "card": {"brand": "Visa", "last4": "4242"},
                "limits": {
                    "max_amount": _money(100_000),
                    "daily_count": 3,
                    "monthly_ceiling": _money(500_000),
                },
            },
        }

    def subscription_payload(self) -> dict[str, JsonValue]:
        return {
            "plan_id": self.plan,
            "status": self.subscription_status,
            "current_period_end": "2026-12-27T00:00:00Z",
            "cancel_at_period_end": self.subscription_cancel_at_period_end,
            "cancel_at": "2026-12-27T00:00:00Z" if self.subscription_cancel_at_period_end else None,
        }

    def put_auto_reload(self, body: Mapping[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        enabled = bool(body.get("enabled", False))
        t_raw = body.get("threshold")
        a_raw = body.get("amount")
        thresh_cents = (
            money_str_to_cents(str(t_raw["amount"]))
            if isinstance(t_raw, dict)
            else self.auto_reload_threshold_cents
        )
        amt_cents = (
            money_str_to_cents(str(a_raw["amount"]))
            if isinstance(a_raw, dict)
            else self.auto_reload_amount_cents
        )
        if enabled and (
            thresh_cents < 500
            or thresh_cents > 100_000
            or amt_cents < 1000
            or amt_cents > 100_000
            or amt_cents < thresh_cents
        ):
            return 422, make_problem(
                422, "auto_reload_bounds_invalid", "Auto-reload threshold/amount out of bounds."
            )
        if enabled:
            self.auto_reload_disabled_reason = None
        self.auto_reload_enabled = enabled
        self.auto_reload_threshold_cents = thresh_cents
        self.auto_reload_amount_cents = amt_cents
        return 200, self.auto_reload_payload()

    def change_subscription(
        self, body: Mapping[str, JsonValue]
    ) -> tuple[int, dict[str, JsonValue]]:
        plan_id = str(body.get("plan_id", ""))
        plan_allowances = {
            "free": (1000, 500, 0),
            "standard.monthly": (6000, 4000, 15000),
            "max.monthly": (30000, 40000, 150000),
        }
        if plan_id not in plan_allowances:
            return 400, make_problem(400, "plan_unknown", f"Unknown plan_id {plan_id!r}.")
        if plan_id == self.plan:
            return 409, make_problem(409, "subscription_plan_unchanged", "Already on plan.")
        if self.subscription_cancel_at_period_end:
            return 409, make_problem(409, "subscription_cancel_pending", "Cancel pending.")
        new_inc, base_daily, base_weekly = plan_allowances[plan_id]
        next_payg = plan_id == "free" and self.purchased_cents > 0
        next_daily = 10000 if next_payg else base_daily
        next_weekly = 50000 if next_payg else base_weekly
        next_ledger = self.purchased_cents + self.promotional_cents + self.operator_cents + new_inc
        if (
            next_ledger < self.held_cents
            or self.daily_spend_cents > next_daily
            or (next_weekly > 0 and self.weekly_spend_cents > next_weekly)
        ):
            return 409, make_problem(
                409,
                "plan_switch_blocked_by_active_commitments",
                "Active holds or spend windows exceed target plan caps.",
            )
        self.plan = plan_id
        self.included_limit_cents = new_inc
        self.included_remaining_cents = new_inc
        return 200, self.subscription_payload()

    def redeem_promo(self, body: Mapping[str, JsonValue]) -> tuple[int, dict[str, JsonValue]]:
        code = str(body.get("code", "")).strip()
        if code not in self._promo_campaigns:
            return 404, make_problem(404, "promo_code_invalid", f"Unknown promo {code!r}.")
        if code in self._redeemed_promos:
            return 409, make_problem(409, "promo_already_redeemed", f"Promo {code!r} redeemed.")
        campaign_id, credit_cents = self._promo_campaigns[code]
        self._redeemed_promos.add(code)
        self.promotional_cents += credit_cents
        return (
            200,
            {
                "redemption_id": self._next_uuid(),
                "campaign_id": campaign_id,
                "org_id": _ORG_ID,
                "redeemed_at": _FIXED_TS,
                "grant_ids": [self._next_uuid()],
                "override_ids": [],
                "replayed": False,
            },
        )
