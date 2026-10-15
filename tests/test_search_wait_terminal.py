"""Search wait fails early and hard on a terminal Search with no result.

v0.23 slot4 2026-10-15: a DEEP Search failed in ~1 s (state ``failed``,
``result_available`` false) and ``wait()`` polled for the full 15-minute
deadline, then raised a bare RequestTimeoutError that hid the cause.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from synth_ai.errors import ContractError, OperationFailedError
from synth_ai.options import ObservationOptions
from synth_ai.sdk.index.search_run import AsyncSearchRun, SearchRun
from synth_ai.sdk.index.types.search import SearchResponse, SearchSnapshot

SEARCH = "40ea7f75-8530-45f8-b05f-f9f0f9cca93a"
METADATA = {
    "contract": "index.v1",
    "request_id": "req-1",
    "operation_id": None,
    "observed_at": datetime(2026, 10, 15, tzinfo=UTC),
}
UNKNOWN = {"state": "unknown", "amount": None}
ZDR = {
    "mode": "zdr",
    "policy_version": "fixture",
    "purpose": "customer_only",
    "content_state": "unavailable",
    "archived_at": None,
    "content_expires_at": None,
    "retention_days": None,
    "history_available": False,
}


def snapshot(state, *, available=False):
    return SearchSnapshot.model_validate(
        {
            "metadata": METADATA,
            "search_id": SEARCH,
            "mode": "deep",
            "state": state,
            "result_available": available,
            "cancellation_requested": state == "cancelled",
            "execution_closed": state in {"completed", "failed", "cancelled"},
            "accounting_final": False,
            "cleanup_complete": False,
            "retention": dict(ZDR, content_state="available") if available else ZDR,
        }
    )


def result(status, failure=None):
    return SearchResponse.model_validate(
        {
            "metadata": METADATA,
            "search_id": SEARCH,
            "mode": "deep",
            "status": status,
            "response": "answer" if status in {"completed", "partial"} else None,
            "citations": (),
            "warnings": (),
            "completeness": "complete" if status == "completed" else "unknown",
            "failure": failure,
            "usage": {
                "metadata": METADATA,
                "operation_id": SEARCH,
                "physical": (),
                "cost": UNKNOWN,
                "customer_charge": UNKNOWN,
                "financial_state": "unknown",
                "execution_source": "unknown",
            },
            "output_ready": status in {"completed", "partial"},
            "accounting_final": False,
            "cleanup_complete": False,
            "retention": dict(ZDR, content_state="available")
            if status in {"completed", "partial"}
            else ZDR,
        }
    )


class Transport:
    def __init__(self, snapshots, terminal):
        self.snapshots = list(snapshots)
        self.terminal = terminal
        self.calls = []

    def request(self, operation, model, *, path, options=None, validate=None, **_):
        self.calls.append(path)
        value = self.terminal if path.endswith("/result") else self.snapshots.pop(0)
        return validate(value) if validate else value


class AsyncTransport(Transport):
    async def request(self, *args, **kwargs):
        return Transport.request(self, *args, **kwargs)


OBSERVE = ObservationOptions(
    timeout_seconds=900,
    idle_timeout_seconds=30,
    reconnect_deadline_seconds=60,
    retry_count_max=1,
    maximum_events=100,
    poll_interval_seconds=0.01,
)


def test_failed_search_raises_its_typed_failure_without_polling_to_the_deadline():
    transport = Transport(
        [snapshot("running"), snapshot("failed")],
        result("failed", {"code": "agents_api_service_refused", "retryable": False}),
    )
    with pytest.raises(OperationFailedError) as raised:
        SearchRun(transport, SEARCH).wait(observation=OBSERVE)
    assert raised.value.code == "agents_api_service_refused"
    assert SEARCH in str(raised.value) and "retryable=False" in str(raised.value)
    assert transport.calls == [
        f"/index/searches/{SEARCH}",
        f"/index/searches/{SEARCH}",
        f"/index/searches/{SEARCH}/result",
    ]


def test_cancelled_search_raises_immediately():
    transport = Transport([snapshot("cancelled")], result("cancelled"))
    with pytest.raises(OperationFailedError) as raised:
        SearchRun(transport, SEARCH).wait(observation=OBSERVE)
    assert raised.value.code == "search_cancelled"


def test_terminal_state_that_disagrees_with_its_result_is_a_contract_error():
    transport = Transport(
        [snapshot("failed")],
        result("cancelled"),
    )
    with pytest.raises(ContractError, match="search_terminal_state_mismatch|reported state"):
        SearchRun(transport, SEARCH).wait(observation=OBSERVE)


def test_failed_result_without_failure_is_refused_by_the_model():
    # The backend must always name the failure; the SDK refuses to decode one without.
    with pytest.raises(ValueError, match="failed result requires a failure"):
        result("failed", None)


def test_completed_search_still_returns_its_result():
    transport = Transport([snapshot("completed", available=True)], result("completed"))
    assert SearchRun(transport, SEARCH).wait(observation=OBSERVE).status == "completed"


def test_async_wait_fails_fast_too():
    transport = AsyncTransport(
        [snapshot("failed")],
        result("failed", {"code": "index_deep_deadline_exceeded", "retryable": True}),
    )
    with pytest.raises(OperationFailedError) as raised:
        asyncio.run(AsyncSearchRun(transport, SEARCH).wait(observation=OBSERVE))
    assert raised.value.code == "index_deep_deadline_exceeded"
