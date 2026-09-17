from synth_ai.sdk.index.search import SearchMode, SearchUsage
from synth_ai.sdk.index.usage_accounting import CustomerCharge, SearchUsageSummaryRow


def test_installed_sdk_accepts_deep_v1_settlement_fields():
    usage = SearchUsage(
        mode=SearchMode.DEEP,
        billing_scope="private",
        price_version="synth.index.deep.v1",
        amount_cents=50,
        receipt_id="search_1",
        funding_source="index_deep_beta",
        reserved_cents=100,
        released_cents=50,
        terminal_outcome="usable_partial",
    )
    assert usage.amount_cents == 50
    receipt = CustomerCharge(
        price_version="synth.index.deep.v1",
        reserved_microcents=100_000_000,
        settled_microcents=50_000_000,
        released_microcents=50_000_000,
        adjustment_microcents=-10_000_000,
        settlement_state="settled",
        funding_source="index_deep_beta",
        intent_hash="a" * 64,
        allowance_period_key="2026-09",
        terminal_outcome="usable_partial",
    )
    assert receipt.funding_source == usage.funding_source
    assert receipt.settled_microcents + receipt.adjustment_microcents == 40_000_000


def test_installed_sdk_accepts_reconciled_deep_summary_row():
    row = SearchUsageSummaryRow(
        mode=SearchMode.DEEP,
        model_identity="qualification-model",
        price_version="synth.index.deep.v1",
        funding_source="index_deep_beta",
        terminal_outcome="usable_partial",
        settlement_state="settled",
        search_count=1,
        physical_attempt_count=1,
        infrastructure_cost_usd_micros=420,
        reserved_microcents=100_000_000,
        customer_charge_microcents=50_000_000,
        released_microcents=50_000_000,
        refunded_microcents=0,
        adjustment_microcents=-10_000_000,
        pending_event_count=0,
    )
    assert row.reserved_microcents == (
        row.customer_charge_microcents + row.released_microcents
    )
    assert row.customer_charge_microcents + row.adjustment_microcents == 40_000_000
