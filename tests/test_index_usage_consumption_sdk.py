"""Consumption negotiation preserves old strict clients and unknown old totals."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from synth_ai.sdk.index.client import AccountAPI
from synth_ai.sdk.index.usage_accounting import SearchUsageReceipt, SearchUsageSummary, SearchUsageSummaryRow


def _summary_row():
    return {
        "mode": "deep", "model_identity": "fixture-model",
        "price_version": "synth.index.deep.v1", "funding_source": "wallet",
        "terminal_outcome": "complete", "settlement_state": "refunded",
        "search_count": 1, "physical_attempt_count": 0,
        "reserved_microcents": 100_000_000, "customer_charge_microcents": 75_000_000,
        "released_microcents": 0, "refunded_microcents": 25_000_000,
        "adjustment_microcents": -25_000_000, "pending_event_count": 0,
        "measurement_state": "pending", "unmeasured_search_count": 1,
        "input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
        "colbert_batches": None, "content_read_calls": None,
    }


def test_old_backend_missing_counters_remain_unknown():
    row = SearchUsageSummaryRow(
        mode="fast",
        search_count=1,
        physical_attempt_count=1,
        customer_charge_microcents=0,
        pending_event_count=1,
    )
    assert row.input_tokens is None
    assert row.colbert_batches is None


def test_complete_customer_summary_preserves_net_charge_and_unknown_measurements():
    payload = {
        "period_start": "2026-09-01T00:00:00Z",
        "period_end": "2026-10-01T00:00:00Z",
        "rows": [_summary_row()], "total_rows": 1, "next_offset": None,
    }
    summary = SearchUsageSummary.model_validate(payload)
    assert summary.model_dump(mode="json") == payload


@pytest.mark.parametrize("field", ["infrastructure_cost_usd_micros", "unallocated_cost_usd_micros", "unknown_total"])
def test_customer_summary_rejects_internal_or_unknown_fields(field):
    with pytest.raises(ValidationError) as error:
        SearchUsageSummaryRow.model_validate({**_summary_row(), field: 0})
    assert error.value.errors()[0]["type"] == "extra_forbidden"
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("value", ["75000000", True, -1])
def test_customer_summary_keeps_strict_nonnegative_money(value):
    with pytest.raises(ValidationError):
        SearchUsageSummaryRow.model_validate({**_summary_row(), "customer_charge_microcents": value})


@pytest.mark.parametrize("field", ["infrastructure_cost_usd_micros", "unallocated_cost_usd_micros"])
def test_customer_receipt_rejects_internal_cost_fields(field):
    payload = {
        "search_id": "search-1", "org_id": "org-1", "requested_mode": "deep",
        "effective_mode": "deep", "event_count": 0, "operation_count": 0,
        "physical_attempt_count": 0, "measurement_state": "pending",
        "charge": {}, "generated_at": "2026-09-17T00:00:00Z", field: 0,
    }
    with pytest.raises(ValidationError) as error:
        SearchUsageReceipt.model_validate(payload)
    assert error.value.errors()[0]["type"] == "extra_forbidden"
    assert error.value.errors()[0]["loc"] == (field,)


def test_summary_and_csv_request_same_consumption_contract():
    calls = []
    api = AccountAPI(calls.append, False)
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 10, 1, tzinfo=UTC)
    api.operation_usage(start, end)
    api.export_operation_usage(start, end)
    assert len(calls) == 2
    assert all(call.params["include_consumption"] == "true" for call in calls)
