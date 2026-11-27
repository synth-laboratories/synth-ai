"""SDK Hypothesis RuleBasedStateMachine suite 3: Billing wallet, webhooks, auto-reload, promos & plan switches (SYN-6050)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from pydantic import JsonValue
from typing_extensions import override

from synth_ai._generated.models import Balance, Money, PromoRedemption
from synth_ai.errors import SwarmsAPIError

from .config import SdkPbtConfig
from .mock_transport import SdkPbtHarness, money_str_to_cents
from .trace import TraceRecorder, TraceSpecKind, verify_trace_conformance

__all__ = ["test_sdk_billing_wallet_stateful_pbt"]


def _make_sdk_billing_wallet_machine(
    *,
    config: SdkPbtConfig,
    trace_dir: Path,
) -> type[RuleBasedStateMachine]:
    run_counter = {"n": 0}

    class SdkBillingWalletMachine(RuleBasedStateMachine):
        """Drives SynthClient.billing & SynthClient.account.redeem_promo verifying wallet & PAYG/Free cap invariants."""

        def __init__(self) -> None:
            super().__init__()
            run_counter["n"] += 1
            self._instance_id = run_counter["n"]
            self.harness = SdkPbtHarness(config)
            self.trace = TraceRecorder(TraceSpecKind.BILLING_WALLET)
            self.step_seq = 0
            self.delivered_events: list[tuple[str, int]] = []
            self.redeemed_promos: dict[str, tuple[str, PromoRedemption]] = {}
            self.active_holds: list[str] = []
            self.webhook_replays = 0
            self._verify_balance()
            self.trace.record_init(self._tla_state())

        def _next_key(self, prefix: str) -> str:
            self.step_seq += 1
            return f"sdk-bill-{prefix}-{self._instance_id}-{self.step_seq}"

        def _verify_balance(self) -> Balance:
            bal = self.harness.client.billing.balance()
            ledger = money_str_to_cents(bal.ledger.amount)
            held = money_str_to_cents(bal.held.amount)
            available = money_str_to_cents(bal.available.amount)
            purchased = money_str_to_cents(bal.purchased_credits.amount)
            promo = money_str_to_cents(bal.promotional_credits.amount)
            operator = money_str_to_cents(bal.operator_credits.amount)
            included_rem = (
                money_str_to_cents(bal.included_allowance.remaining.amount)
                if bal.included_allowance is not None
                else 0
            )
            assert available == ledger - held
            assert ledger == purchased + promo + operator + included_rem
            assert held >= 0 and available >= 0
            expected_payg = bal.plan == "free" and purchased > 0
            assert bal.payg_credits_active == expected_payg
            return bal

        def _tla_state(self) -> dict[str, JsonValue]:
            if self.harness.backend is not None:
                c = self.harness.backend
                return {
                    "plan": c.plan,
                    "payg_credits_active": c.payg_credits_active,
                    "ledger_cents": c.ledger_cents,
                    "held_cents": c.held_cents,
                    "available_cents": c.available_cents,
                    "purchased_cents": c.purchased_cents,
                    "promotional_cents": c.promotional_cents,
                    "included_remaining_cents": c.included_remaining_cents,
                    "daily_spend_cents": c.daily_spend_cents,
                    "weekly_spend_cents": c.weekly_spend_cents,
                    "daily_cap_cents": c.daily_cap_cents,
                    "weekly_cap_cents": c.weekly_cap_cents,
                    "auto_reload_enabled": c.auto_reload_enabled,
                    "auto_reload_daily_count": c.auto_reload_daily_count,
                    "auto_reload_monthly_cents": c.auto_reload_monthly_cents,
                    "webhook_replays": self.webhook_replays,
                }
            bal = self._verify_balance()
            payg = bal.payg_credits_active
            daily_cap = (
                (10000 if payg else 500)
                if bal.plan == "free"
                else (4000 if bal.plan == "standard.monthly" else 40000)
            )
            weekly_cap = (
                (50000 if payg else 0)
                if bal.plan == "free"
                else (15000 if bal.plan == "standard.monthly" else 150000)
            )
            return {
                "plan": bal.plan,
                "payg_credits_active": payg,
                "ledger_cents": money_str_to_cents(bal.ledger.amount),
                "held_cents": money_str_to_cents(bal.held.amount),
                "available_cents": money_str_to_cents(bal.available.amount),
                "purchased_cents": money_str_to_cents(bal.purchased_credits.amount),
                "promotional_cents": money_str_to_cents(bal.promotional_credits.amount),
                "included_remaining_cents": (
                    money_str_to_cents(bal.included_allowance.remaining.amount)
                    if bal.included_allowance is not None
                    else 0
                ),
                "daily_spend_cents": 0,
                "weekly_spend_cents": 0,
                "daily_cap_cents": daily_cap,
                "weekly_cap_cents": weekly_cap,
                "auto_reload_enabled": False,
                "auto_reload_daily_count": 0,
                "auto_reload_monthly_cents": 0,
                "webhook_replays": self.webhook_replays,
            }

        @rule(
            amount_cents=st.sampled_from([500, 1000, 2500, 5000, 10000]),
            payment_succeeded=st.booleans(),
            amount_matches=st.booleans(),
        )
        def stripe_checkout_webhook(
            self, amount_cents: int, payment_succeeded: bool, amount_matches: bool
        ) -> None:
            if self.harness.backend is None:
                return
            before_purchased = self.harness.backend.purchased_cents
            event_id = f"evt_sdk_{self._instance_id}_{len(self.delivered_events) + 1}"
            status, _body, dup = self.harness.backend.deliver_stripe_webhook(
                event_id=event_id,
                amount_cents=amount_cents,
                payment_succeeded=payment_succeeded,
                amount_matches_checkout=amount_matches,
            )
            assert not dup
            if payment_succeeded and amount_matches:
                assert status == 200
                assert self.harness.backend.purchased_cents == before_purchased + amount_cents
                self.delivered_events.append((event_id, amount_cents))
                self._verify_balance()
                self.trace.record_step(
                    "StripePurchaseCredit", self._tla_state(), params=[amount_cents]
                )
            else:
                assert status == 409
                assert self.harness.backend.purchased_cents == before_purchased
                self.trace.record_step("RefuseOperation", self._tla_state())

        @precondition(lambda self: len(self.delivered_events) > 0)
        @rule(data=st.data())
        def replay_stripe_webhook(self, data: st.DataObject) -> None:
            if self.harness.backend is None:
                return
            event_id, amount_cents = data.draw(st.sampled_from(self.delivered_events))
            before_ledger = self.harness.backend.ledger_cents
            status, _body, dup = self.harness.backend.deliver_stripe_webhook(
                event_id=event_id, amount_cents=amount_cents
            )
            assert status == 200 and dup
            assert self.harness.backend.ledger_cents == before_ledger
            self.webhook_replays += 1
            self._verify_balance()
            self.trace.record_step("StripeWebhookReplay", self._tla_state())

        @rule(code=st.sampled_from(["LAUNCH5", "RESEARCH10", "SWARM20"]), reuse_key=st.booleans())
        def redeem_promo(self, code: str, reuse_key: bool) -> None:
            prev = self.redeemed_promos.get(code)
            if reuse_key and prev is not None:
                key, orig = prev
                replayed = self.harness.client.account.redeem_promo(code, idempotency_key=key)
                assert replayed == orig
                self._verify_balance()
                self.trace.record_step("RefuseOperation", self._tla_state())
                return

            key = self._next_key(f"promo-{code}")
            if prev is None:
                try:
                    redemption = self.harness.client.account.redeem_promo(code, idempotency_key=key)
                except SwarmsAPIError as exc:
                    if exc.status not in {404, 409}:
                        raise
                    self.trace.record_step("RefuseOperation", self._tla_state())
                else:
                    self.redeemed_promos[code] = (key, redemption)
                    self._verify_balance()
                    promo_amt = {"LAUNCH5": 500, "RESEARCH10": 1000, "SWARM20": 2000}[code]
                    self.trace.record_step(
                        "RedeemPromoCredit", self._tla_state(), params=[promo_amt]
                    )
            else:
                with pytest.raises(SwarmsAPIError) as exc_info:
                    self.harness.client.account.redeem_promo(code, idempotency_key=key)
                assert exc_info.value.status == 409
                self._verify_balance()
                self.trace.record_step("RefuseOperation", self._tla_state())

        @rule(amount_cents=st.sampled_from([100, 250, 500, 1500, 6000, 12000]))
        def reserve_run_hold(self, amount_cents: int) -> None:
            if self.harness.backend is None:
                return
            ok, hold_id, reason = self.harness.backend.reserve_hold(amount_cents)
            if ok:
                assert hold_id is not None
                self.active_holds.append(hold_id)
                self._verify_balance()
                self.trace.record_step("PlaceHold", self._tla_state(), params=[amount_cents])
            else:
                assert reason in {
                    "insufficient_balance",
                    "daily_spend_cap_exceeded",
                    "weekly_spend_cap_exceeded",
                }
                self.trace.record_step("RefuseOperation", self._tla_state())

        @precondition(lambda self: len(self.active_holds) > 0)
        @rule(quarters=st.sampled_from([0, 1, 2, 4]))
        def settle_active_hold(self, quarters: int) -> None:
            if self.harness.backend is None:
                return
            hold_id = self.active_holds[0]
            hold = self.harness.backend.holds[hold_id]
            active_before = hold.active_cents
            debit = (active_before * quarters) // 4
            release = active_before - debit
            ok, _reason = self.harness.backend.settle_hold(hold_id, debit)
            if ok:
                self.active_holds.pop(0)
                assert hold.placed_cents == hold.settled_cents + hold.released_cents
                self._verify_balance()
                self.trace.record_step("SettleHold", self._tla_state(), params=[debit, release])
            else:
                self.trace.record_step("RefuseOperation", self._tla_state())

        @rule(
            enabled=st.booleans(),
            threshold_cents=st.sampled_from([500, 1000, 2500]),
            amount_cents=st.sampled_from([1000, 2500, 5000]),
        )
        def configure_auto_reload(
            self, enabled: bool, threshold_cents: int, amount_cents: int
        ) -> None:
            threshold = Money(
                amount=f"{threshold_cents // 100}.{threshold_cents % 100:02d}", currency="USD"
            )
            amount = Money(amount=f"{amount_cents // 100}.{amount_cents % 100:02d}", currency="USD")
            key = self._next_key("autoreload")
            if enabled and amount_cents < threshold_cents:
                with pytest.raises(SwarmsAPIError) as exc_info:
                    self.harness.client.billing.set_auto_reload(
                        enabled=enabled,
                        threshold=threshold,
                        amount=amount,
                        disclosure_version="2026-11-10.1",
                        idempotency_key=key,
                    )
                assert exc_info.value.status == 422
                self.trace.record_step("RefuseOperation", self._tla_state())
            else:
                saved = self.harness.client.billing.set_auto_reload(
                    enabled=enabled,
                    threshold=threshold,
                    amount=amount,
                    disclosure_version="2026-11-10.1",
                    idempotency_key=key,
                )
                assert saved.enabled == enabled
                self.trace.record_step("ConfigureAutoReload", self._tla_state(), params=[enabled])

        @rule(card_declined=st.booleans())
        def trigger_auto_reload(self, card_declined: bool) -> None:
            if self.harness.backend is None:
                return
            fired, reason = self.harness.backend.trigger_auto_reload_if_due(
                card_declined=card_declined
            )
            if fired:
                self._verify_balance()
                self.trace.record_step(
                    "TriggerAutoReload",
                    self._tla_state(),
                    params=[self.harness.backend.auto_reload_amount_cents],
                )
            elif reason == "card_declined":
                self.trace.record_step("ConfigureAutoReload", self._tla_state(), params=[False])
            else:
                self.trace.record_step("RefuseOperation", self._tla_state())

        @rule(target_plan=st.sampled_from(["free", "standard.monthly", "max.monthly"]))
        def switch_plan(self, target_plan: str) -> None:
            try:
                sub = self.harness.client.billing.change_subscription_plan(
                    target_plan, idempotency_key=self._next_key(f"plan-{target_plan}")
                )
            except SwarmsAPIError as exc:
                if exc.status not in {400, 409}:
                    raise
                self.trace.record_step("RefuseOperation", self._tla_state())
            else:
                assert sub.plan_id == target_plan
                self._verify_balance()
                inc_amt = {"free": 1000, "standard.monthly": 6000, "max.monthly": 30000}[
                    target_plan
                ]
                self.trace.record_step(
                    "SwitchPlan", self._tla_state(), params=[target_plan, inc_amt]
                )

        @rule()
        def rollover_utc_day(self) -> None:
            if self.harness.backend is None:
                return
            self.harness.backend.rollover_utc_day()
            self.trace.record_step("RolloverDay", self._tla_state())

        @invariant()
        def wallet_and_cap_invariants(self) -> None:
            bal = self._verify_balance()
            if self.harness.backend is not None:
                c = self.harness.backend
                assert c.available_cents == c.ledger_cents - c.held_cents
                assert c.daily_spend_cents <= c.daily_cap_cents
                if c.weekly_cap_cents > 0:
                    assert c.weekly_spend_cents <= c.weekly_cap_cents
                if bal.plan == "free":
                    if c.purchased_cents > 0:
                        assert (
                            c.payg_credits_active
                            and c.daily_cap_cents == 10000
                            and c.weekly_cap_cents == 50000
                        )
                    else:
                        assert (
                            (not c.payg_credits_active)
                            and c.daily_cap_cents == 500
                            and c.weekly_cap_cents == 0
                        )
                assert c.auto_reload_daily_count <= 3
                assert c.auto_reload_monthly_cents <= 500_000

        @override
        def teardown(self) -> None:
            try:
                out_path = trace_dir / f"sdk_billing_wallet_{self._instance_id:04d}.jsonl"
                self.trace.write_jsonl(out_path)
                result = verify_trace_conformance(
                    out_path, TraceSpecKind.BILLING_WALLET, spec_dir=config.spec_dir
                )
                assert result.ok
            finally:
                self.harness.close()

    return SdkBillingWalletMachine


def test_sdk_billing_wallet_stateful_pbt(sdk_pbt_config: SdkPbtConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="sdk-pbt-billing-wallet-") as tmp:
        trace_dir = sdk_pbt_config.trace_out if sdk_pbt_config.trace_out is not None else Path(tmp)
        machine_cls = _make_sdk_billing_wallet_machine(config=sdk_pbt_config, trace_dir=trace_dir)
        run_state_machine_as_test(
            machine_cls,
            settings=settings(
                max_examples=sdk_pbt_config.max_examples,
                stateful_step_count=sdk_pbt_config.stateful_step_count,
                deadline=None,
                suppress_health_check=[HealthCheck.too_slow],
            ),
        )
