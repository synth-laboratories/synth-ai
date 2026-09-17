from synth_ai.sdk.index.search import SearchMode, SearchUsage
from synth_ai.sdk.index.usage_accounting import CustomerCharge


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
        settlement_state="settled",
        funding_source="index_deep_beta",
        intent_hash="a" * 64,
        allowance_period_key="2026-09",
        terminal_outcome="usable_partial",
    )
    assert receipt.funding_source == usage.funding_source
