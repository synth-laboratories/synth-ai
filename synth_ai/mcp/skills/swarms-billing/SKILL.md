---
name: swarms-billing
description: Understand Cloud Swarms billing — weekly allowance vs prepaid PAYG credits, active holds and headroom, credit pack checkouts vs subscription plan switches, and billing refusal recovery.
---

# Cloud Swarms Billing and Spend Management

Use these tools to inspect funding headroom before launching runs, purchase prepaid credits or switch plans, and resolve spend refusals.

## 1. Inspect Balance, Holds, and Usage

Call `account_billing_balance` (`account:billing:read`). Key fields:
- `plan`: current subscription plan.
- `included_allowance`: recurring weekly plan allowance included with the subscription.
- `payg_credits_active`, `purchased_credits`, `promotional_credits`, `operator_credits`: prepaid PAYG credit buckets drawn after/alongside included allowance.
- `held`: funds currently reserved for active/queued runs (released when runs settle).
- `headroom` and `available`: spendable capacity for new run reservations.
- `debt`: any unsettled balance that must be cleared.

For historical spend and settlement state:
- `account_usage_summary` (`since`, `until` — max 366 days): returns `total`, `pending`, `settling`, `funding`, `by_source`, `by_day`, `by_key`.
- `account_usage_for_run` (`run_id`): line-item usage records for one run.
- `account_billing_history` (optional `cause`, `run_id`, `since`, `until`): ledger entries.

## 2. Inspect Plans and Credit Packs

Call `account_billing_plans_list` (public — requires no OAuth scope) to inspect:
- `plans` and `switchable_plans`: subscription tiers and eligible switch targets.
- `credit_packs` and `custom_credit`: purchasable prepaid PAYG top-ups.

## 3. Credit Packs vs Subscription Plan Switches

Choose the right flow — they are distinct operations:
- **Buy prepaid PAYG credits**: call `account_billing_checkout_create` (`body: {pack_id}` or `body: {credit}`, `idempotency_key`) and poll status with `account_billing_checkout_get` (`id`).
- **Start a first subscription** (when `account_billing_subscription_get` shows no active subscription): `account_billing_checkout_create` (`body: {plan_id}`, `idempotency_key`).
- **Switch an active subscription plan**: do **not** call `account_billing_checkout_create` with `plan_id` (`subscription_already_active`). Instead:
  1. Preview proration with `account_billing_subscription_change_preview` (`plan_id`).
  2. Apply with `account_billing_subscription_change` (`body: {plan_id}`, `idempotency_key`).
  3. If a cancellation is pending (`subscription_cancel_pending`), call `account_billing_subscription_resume` first.

## 4. Handling Billing Refusals

- `run_budget_below_call_bound`: raise `limits.maximum_charge` to at least `minimum_budget_micros` and relaunch with a new `idempotency_key`.
- `insufficient_credits` or `budget_exceeded`: check `account_billing_balance` (`held` vs `available` vs `included_allowance`). Wait for active holds to settle or weekly allowance to reset, buy a credit pack, or lower `limits.maximum_charge`; retry with a new key.
- `launch_choice_unaffordable`: raise `limits.maximum_charge` or choose a lower-cost profile from `swarms_launch_choices`.
- `spend_entitlement_exceeded`: org-level spend cap reached; ask an org admin to adjust entitlements.
- `billing_admin_required` (403): mutation requires an organization admin or owner.
- `browser_session_required` (403): action (such as enabling auto-reload) must be completed in a signed-in browser session in the Console.
- `subscription_change_payment_failed` (402): payment declined and switch is final/unapplied; update the card before retrying with a new key.
