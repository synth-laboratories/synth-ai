-------------------------- MODULE BillingWallet ---------------------------
EXTENDS Integers

(*
 * TLA+ specification for Billing wallet, Stripe webhook idempotency,
 * auto-reload caps, promo redemption, plan switches, and Free ($5/day) vs
 * PAYG ($100/day, $500/week) spend-window invariants (SYN-6050).
 *
 * Monetary values are exact integer cents.
 * Compatible with `everystate conform -trace <run.jsonl> BillingWallet.tla`.
 *)

VARIABLES
    plan,
    payg_credits_active,
    ledger_cents,
    held_cents,
    available_cents,
    purchased_cents,
    promotional_cents,
    included_remaining_cents,
    daily_spend_cents,
    weekly_spend_cents,
    daily_cap_cents,
    weekly_cap_cents,
    auto_reload_enabled,
    auto_reload_daily_count,
    auto_reload_monthly_cents,
    webhook_replays

vars == <<
    plan,
    payg_credits_active,
    ledger_cents,
    held_cents,
    available_cents,
    purchased_cents,
    promotional_cents,
    included_remaining_cents,
    daily_spend_cents,
    weekly_spend_cents,
    daily_cap_cents,
    weekly_cap_cents,
    auto_reload_enabled,
    auto_reload_daily_count,
    auto_reload_monthly_cents,
    webhook_replays
>>

Plans == {"free", "standard.monthly", "max.monthly"}

Min(a, b) == IF a <= b THEN a ELSE b

ExpectedDailyCap(p, payg) ==
    IF p = "free" THEN (IF payg THEN 10000 ELSE 500)
    ELSE IF p = "standard.monthly" THEN 4000
    ELSE 40000

ExpectedWeeklyCap(p, payg) ==
    IF p = "free" THEN (IF payg THEN 50000 ELSE 0)
    ELSE IF p = "standard.monthly" THEN 15000
    ELSE 150000

TypeOK ==
    /\ plan \in Plans
    /\ payg_credits_active \in BOOLEAN
    /\ ledger_cents >= 0
    /\ held_cents >= 0
    /\ available_cents >= 0
    /\ purchased_cents >= 0
    /\ promotional_cents >= 0
    /\ included_remaining_cents >= 0
    /\ daily_spend_cents >= 0
    /\ weekly_spend_cents >= 0
    /\ daily_cap_cents > 0
    /\ weekly_cap_cents >= 0
    /\ auto_reload_enabled \in BOOLEAN
    /\ auto_reload_daily_count >= 0
    /\ auto_reload_monthly_cents >= 0
    /\ webhook_replays >= 0

SafetyInvariant ==
    /\ TypeOK
    /\ available_cents = ledger_cents - held_cents
    /\ ledger_cents = purchased_cents + promotional_cents + included_remaining_cents
    /\ (payg_credits_active <=> (plan = "free" /\ purchased_cents > 0))
    /\ daily_cap_cents = ExpectedDailyCap(plan, payg_credits_active)
    /\ weekly_cap_cents = ExpectedWeeklyCap(plan, payg_credits_active)
    /\ daily_spend_cents <= daily_cap_cents
    /\ (weekly_cap_cents = 0 \/ weekly_spend_cents <= weekly_cap_cents)
    /\ auto_reload_daily_count <= 3
    /\ auto_reload_monthly_cents <= 500000

Init ==
    /\ plan = "free"
    /\ payg_credits_active = FALSE
    /\ ledger_cents = 1000
    /\ held_cents = 0
    /\ available_cents = 1000
    /\ purchased_cents = 0
    /\ promotional_cents = 0
    /\ included_remaining_cents = 1000
    /\ daily_spend_cents = 0
    /\ weekly_spend_cents = 0
    /\ daily_cap_cents = 500
    /\ weekly_cap_cents = 0
    /\ auto_reload_enabled = FALSE
    /\ auto_reload_daily_count = 0
    /\ auto_reload_monthly_cents = 0
    /\ webhook_replays = 0

StripePurchaseCredit(amt) ==
    /\ amt >= 500
    /\ amt <= 1000000
    /\ purchased_cents' = purchased_cents + amt
    /\ ledger_cents' = ledger_cents + amt
    /\ available_cents' = available_cents + amt
    /\ payg_credits_active' = (plan = "free" /\ purchased_cents' > 0)
    /\ daily_cap_cents' = ExpectedDailyCap(plan, payg_credits_active')
    /\ weekly_cap_cents' = ExpectedWeeklyCap(plan, payg_credits_active')
    /\ UNCHANGED <<plan, held_cents, promotional_cents, included_remaining_cents,
                   daily_spend_cents, weekly_spend_cents, auto_reload_enabled,
                   auto_reload_daily_count, auto_reload_monthly_cents, webhook_replays>>

StripeWebhookReplay ==
    /\ webhook_replays' = webhook_replays + 1
    /\ UNCHANGED <<plan, payg_credits_active, ledger_cents, held_cents,
                   available_cents, purchased_cents, promotional_cents,
                   included_remaining_cents, daily_spend_cents, weekly_spend_cents,
                   daily_cap_cents, weekly_cap_cents, auto_reload_enabled,
                   auto_reload_daily_count, auto_reload_monthly_cents>>

RedeemPromoCredit(amt) ==
    /\ amt > 0
    /\ promotional_cents' = promotional_cents + amt
    /\ ledger_cents' = ledger_cents + amt
    /\ available_cents' = available_cents + amt
    /\ UNCHANGED <<plan, payg_credits_active, held_cents, purchased_cents,
                   included_remaining_cents, daily_spend_cents, weekly_spend_cents,
                   daily_cap_cents, weekly_cap_cents, auto_reload_enabled,
                   auto_reload_daily_count, auto_reload_monthly_cents, webhook_replays>>

PlaceHold(amt) ==
    /\ amt > 0
    /\ amt <= available_cents
    /\ daily_spend_cents + amt <= daily_cap_cents
    /\ (weekly_cap_cents = 0 \/ weekly_spend_cents + amt <= weekly_cap_cents)
    /\ held_cents' = held_cents + amt
    /\ available_cents' = available_cents - amt
    /\ daily_spend_cents' = daily_spend_cents + amt
    /\ weekly_spend_cents' = weekly_spend_cents + amt
    /\ UNCHANGED <<plan, payg_credits_active, ledger_cents, purchased_cents,
                   promotional_cents, included_remaining_cents, daily_cap_cents,
                   weekly_cap_cents, auto_reload_enabled, auto_reload_daily_count,
                   auto_reload_monthly_cents, webhook_replays>>

SettleHold(debit, rel) ==
    /\ debit >= 0
    /\ rel >= 0
    /\ debit + rel > 0
    /\ debit + rel <= held_cents
    /\ LET take_inc == Min(included_remaining_cents, debit)
           rem1 == debit - take_inc
           take_promo == Min(promotional_cents, rem1)
           rem2 == rem1 - take_promo
           take_purch == Min(purchased_cents, rem2)
       IN
        /\ take_inc + take_promo + take_purch = debit
        /\ included_remaining_cents' = included_remaining_cents - take_inc
        /\ promotional_cents' = promotional_cents - take_promo
        /\ purchased_cents' = purchased_cents - take_purch
        /\ held_cents' = held_cents - debit - rel
        /\ ledger_cents' = ledger_cents - debit
        /\ available_cents' = ledger_cents' - held_cents'
        /\ payg_credits_active' = (plan = "free" /\ purchased_cents' > 0)
        /\ daily_cap_cents' = ExpectedDailyCap(plan, payg_credits_active')
        /\ weekly_cap_cents' = ExpectedWeeklyCap(plan, payg_credits_active')
        /\ daily_spend_cents' = IF daily_spend_cents >= rel THEN daily_spend_cents - rel ELSE 0
        /\ weekly_spend_cents' = IF weekly_spend_cents >= rel THEN weekly_spend_cents - rel ELSE 0
        /\ daily_spend_cents' <= daily_cap_cents'
        /\ (weekly_cap_cents' = 0 \/ weekly_spend_cents' <= weekly_cap_cents')
    /\ UNCHANGED <<plan, auto_reload_enabled, auto_reload_daily_count,
                   auto_reload_monthly_cents, webhook_replays>>

TriggerAutoReload(amt) ==
    /\ auto_reload_enabled
    /\ auto_reload_daily_count < 3
    /\ amt >= 1000
    /\ amt <= 100000
    /\ auto_reload_monthly_cents + amt <= 500000
    /\ purchased_cents' = purchased_cents + amt
    /\ ledger_cents' = ledger_cents + amt
    /\ available_cents' = available_cents + amt
    /\ auto_reload_daily_count' = auto_reload_daily_count + 1
    /\ auto_reload_monthly_cents' = auto_reload_monthly_cents + amt
    /\ payg_credits_active' = (plan = "free" /\ purchased_cents' > 0)
    /\ daily_cap_cents' = ExpectedDailyCap(plan, payg_credits_active')
    /\ weekly_cap_cents' = ExpectedWeeklyCap(plan, payg_credits_active')
    /\ UNCHANGED <<plan, held_cents, promotional_cents, included_remaining_cents,
                   daily_spend_cents, weekly_spend_cents, auto_reload_enabled,
                   webhook_replays>>

ConfigureAutoReload(en) ==
    /\ en \in BOOLEAN
    /\ auto_reload_enabled' = en
    /\ UNCHANGED <<plan, payg_credits_active, ledger_cents, held_cents,
                   available_cents, purchased_cents, promotional_cents,
                   included_remaining_cents, daily_spend_cents, weekly_spend_cents,
                   daily_cap_cents, weekly_cap_cents, auto_reload_daily_count,
                   auto_reload_monthly_cents, webhook_replays>>

SwitchPlan(next_plan, next_inc) ==
    /\ next_plan \in Plans
    /\ next_plan # plan
    /\ next_inc \in {1000, 6000, 30000}
    /\ plan' = next_plan
    /\ payg_credits_active' = (next_plan = "free" /\ purchased_cents > 0)
    /\ daily_cap_cents' = ExpectedDailyCap(next_plan, payg_credits_active')
    /\ weekly_cap_cents' = ExpectedWeeklyCap(next_plan, payg_credits_active')
    /\ daily_spend_cents <= daily_cap_cents'
    /\ (weekly_cap_cents' = 0 \/ weekly_spend_cents <= weekly_cap_cents')
    /\ included_remaining_cents' = next_inc
    /\ ledger_cents' = purchased_cents + promotional_cents + next_inc
    /\ ledger_cents' >= held_cents
    /\ available_cents' = ledger_cents' - held_cents
    /\ UNCHANGED <<held_cents, purchased_cents, promotional_cents,
                   daily_spend_cents, weekly_spend_cents, auto_reload_enabled,
                   auto_reload_daily_count, auto_reload_monthly_cents, webhook_replays>>

RolloverDay ==
    /\ daily_spend_cents' = 0
    /\ auto_reload_daily_count' = 0
    /\ UNCHANGED <<plan, payg_credits_active, ledger_cents, held_cents,
                   available_cents, purchased_cents, promotional_cents,
                   included_remaining_cents, weekly_spend_cents, daily_cap_cents,
                   weekly_cap_cents, auto_reload_enabled, auto_reload_monthly_cents,
                   webhook_replays>>

RefuseOperation ==
    /\ UNCHANGED vars

SettlePairs ==
    {<<0, 100>>, <<25, 75>>, <<50, 50>>, <<100, 0>>,
     <<0, 250>>, <<62, 188>>, <<125, 125>>, <<250, 0>>,
     <<0, 500>>, <<125, 375>>, <<250, 250>>, <<500, 0>>,
     <<0, 1500>>, <<375, 1125>>, <<750, 750>>, <<1500, 0>>,
     <<0, 6000>>, <<1500, 4500>>, <<3000, 3000>>, <<6000, 0>>,
     <<0, 12000>>, <<3000, 9000>>, <<6000, 6000>>, <<12000, 0>>}

Next ==
    \/ (\E amt \in {500, 1000, 2500, 5000, 10000}: StripePurchaseCredit(amt))
    \/ StripeWebhookReplay
    \/ (\E amt \in {500, 1000, 2000}: RedeemPromoCredit(amt))
    \/ (\E amt \in {100, 250, 500, 1500, 6000, 12000}: PlaceHold(amt))
    \/ (\E pair \in SettlePairs: SettleHold(pair[1], pair[2]))
    \/ (\E amt \in {1000, 2500, 5000}: TriggerAutoReload(amt))
    \/ (\E en \in BOOLEAN: ConfigureAutoReload(en))
    \/ (\E p \in Plans: \E inc \in {1000, 6000, 30000}: SwitchPlan(p, inc))
    \/ RolloverDay
    \/ RefuseOperation

Spec == Init /\ [][Next]_vars

=============================================================================
