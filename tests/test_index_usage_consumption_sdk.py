"""Consumption negotiation preserves old strict clients and unknown old totals."""

from datetime import UTC, datetime

from synth_ai.sdk.index.client import AccountAPI
from synth_ai.sdk.index.usage_accounting import SearchUsageSummaryRow


def test_old_backend_missing_counters_remain_unknown():
    row = SearchUsageSummaryRow(
        mode="fast",
        search_count=1,
        physical_attempt_count=1,
        infrastructure_cost_usd_micros=0,
        customer_charge_microcents=0,
        pending_event_count=1,
    )
    assert row.input_tokens is None
    assert row.colbert_batches is None


def test_summary_and_csv_request_same_consumption_contract():
    calls = []
    api = AccountAPI(calls.append, False)
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 10, 1, tzinfo=UTC)
    api.operation_usage(start, end)
    api.export_operation_usage(start, end)
    assert len(calls) == 2
    assert all(call.params["include_consumption"] == "true" for call in calls)
