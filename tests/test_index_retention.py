"""ZDR modes and org retention policy in the SDK (SYN-4571), consumer first.

Everything here runs against a fake httpx transport: no network, no backend.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.errors import (
    POLICY_ERROR_TYPES,
    OutcomeUncertainError,
    RetentionModeNotAllowed,
    SearchContentPurged,
    StrictZdrResultUnavailable,
    UnsupportedOperationError,
)
from synth_ai.options import ClientOptions, MutationOptions, ObservationOptions
from synth_ai.sdk.index.client import AsyncIndexClient, IndexClient
from synth_ai.sdk.index.types.retention import SearchRetention
from synth_ai.sdk.index.types.search import KeyedSearchOptions, SearchSnapshot
from synth_ai.types.money import Money

SEARCH = "40ea7f75-8530-45f8-b05f-f9f0f9cca93a"
NOW = datetime(2026, 10, 18, tzinfo=UTC)
METADATA = {
    "contract": "index.v1",
    "request_id": "req-1",
    "operation_id": None,
    "observed_at": NOW.isoformat(),
}
HEADERS = {"X-Synth-Contract": "index.v1"}
UNKNOWN = {"state": "unknown", "amount": None}
USAGE = {
    "metadata": METADATA,
    "operation_id": SEARCH,
    "physical": [],
    "cost": UNKNOWN,
    "customer_charge": UNKNOWN,
    "financial_state": "unknown",
    "execution_source": "unknown",
}
LEGACY_ZDR = {
    "mode": "zdr",
    "policy_version": "fixture",
    "purpose": "customer_only",
    "content_state": "available",
    "archived_at": None,
    "content_expires_at": None,
    "retention_days": None,
    "history_available": False,
}
POLICY = {
    "metadata": METADATA,
    "default_mode": "zdr",
    "allowed_modes": ["zdr", "zdr_strict"],
    "enforced": True,
    "revision": 3,
}


def options(**extra: object) -> KeyedSearchOptions:
    return KeyedSearchOptions(maximum_charge=Money(coefficient=1, scale=0, currency="USD"), **extra)


def snapshot_body(state="running", *, available=False, retention=None, **extra) -> dict:
    return {
        "metadata": METADATA,
        "search_id": SEARCH,
        "mode": "fast",
        "state": state,
        "result_available": available,
        "cancellation_requested": False,
        "execution_closed": state == "completed",
        "accounting_final": False,
        "cleanup_complete": False,
        "retention": retention,
        **extra,
    }


def result_body(retention=None) -> dict:
    return {
        "metadata": METADATA,
        "search_id": SEARCH,
        "mode": "fast",
        "status": "completed",
        "response": "answer",
        "citations": [],
        "warnings": [],
        "completeness": "complete",
        "failure": None,
        "usage": USAGE,
        "output_ready": True,
        "accounting_final": False,
        "cleanup_complete": False,
        "retention": retention,
    }


def error_body(code: str, **extra: object) -> dict:
    return {"error": {"code": code, "request_id": "req-err", **extra}}


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def reply(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, headers=HEADERS, json=payload)


def sync_client(recorder: Recorder) -> IndexClient:
    transport = HttpTransport("https://backend.test", "key", ClientOptions())
    transport.client.close()
    transport.client = httpx.Client(transport=httpx.MockTransport(recorder))
    transport._negotiate = lambda operation, deadline: None  # type: ignore[method-assign]
    return IndexClient(transport, True)


def async_client(recorder: Recorder) -> AsyncIndexClient:
    transport = AsyncHttpTransport("https://backend.test", "key", ClientOptions())
    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))

    async def no_negotiation(operation, deadline):
        return None

    transport._negotiate = no_negotiation  # type: ignore[method-assign]
    return AsyncIndexClient(transport, True)


# --- search options on the wire -------------------------------------------------


def test_retention_option_is_omitted_when_unset_and_sent_when_set():
    assert "retention" not in json.loads(options().model_dump_json())
    assert json.loads(options(retention="zdr_strict").model_dump_json())["retention"] == (
        "zdr_strict"
    )
    assert options(retention="retained").retention == "retained"
    with pytest.raises(ValueError):
        options(retention="off")


def test_create_sends_no_retention_key_when_unset_and_the_mode_when_set():
    strict = dict(LEGACY_ZDR, mode="zdr_strict")
    recorder = Recorder(
        reply(200, snapshot_body(retention=LEGACY_ZDR)), reply(200, snapshot_body(retention=strict))
    )
    client = sync_client(recorder)
    client.searches.create("q", options=options())
    client.searches.create("q", options=options(retention="zdr_strict"))
    first, second = (json.loads(request.content)["options"] for request in recorder.requests)
    assert "retention" not in first
    assert second["retention"] == "zdr_strict"


# --- client.index.retention -----------------------------------------------------


def test_retention_get_and_update_sync():
    recorder = Recorder(reply(200, POLICY))
    client = sync_client(recorder)
    policy = client.retention.get()
    assert (policy.default_mode, policy.allowed_modes, policy.enforced, policy.revision) == (
        "zdr",
        ("zdr", "zdr_strict"),
        True,
        3,
    )
    updated = client.retention.update(
        default_mode="zdr",
        allowed_modes=("zdr", "zdr_strict"),
        mutation=MutationOptions(idempotency_key="key-1"),
    )
    assert updated.revision == 3
    get, put = recorder.requests
    assert (get.method, get.url.path) == ("GET", "/api/v1/index/me/retention")
    assert "Idempotency-Key" not in get.headers
    assert (put.method, put.url.path) == ("PUT", "/api/v1/index/me/retention")
    assert put.headers["Idempotency-Key"] == "key-1"
    # enforced / expected_revision are omitted when not given
    assert json.loads(put.content) == {
        "default_mode": "zdr",
        "allowed_modes": ["zdr", "zdr_strict"],
    }


def test_retention_update_sends_optional_fields_and_validates_locally():
    recorder = Recorder(reply(200, POLICY))
    client = sync_client(recorder)
    client.retention.update(
        default_mode="zdr_strict",
        allowed_modes=("zdr_strict",),
        enforced=True,
        expected_revision=2,
    )
    assert json.loads(recorder.requests[0].content) == {
        "default_mode": "zdr_strict",
        "allowed_modes": ["zdr_strict"],
        "enforced": True,
        "expected_revision": 2,
    }
    with pytest.raises(ValueError):
        client.retention.update(default_mode="retained", allowed_modes=("zdr",))
    with pytest.raises(ValueError):
        client.retention.update(default_mode="zdr", allowed_modes=("zdr", "zdr"))
    assert len(recorder.requests) == 1


def test_retention_get_and_update_async():
    async def run() -> None:
        recorder = Recorder(reply(200, POLICY))
        client = async_client(recorder)
        policy = await client.retention.get()
        assert policy.allowed_modes == ("zdr", "zdr_strict")
        await client.retention.update(
            default_mode="zdr",
            allowed_modes=("zdr", "zdr_strict"),
            enforced=True,
            mutation=MutationOptions(idempotency_key="key-2"),
        )
        get, put = recorder.requests
        assert (get.method, put.method) == ("GET", "PUT")
        assert put.headers["Idempotency-Key"] == "key-2"
        assert json.loads(put.content)["enforced"] is True

    asyncio.run(run())


def test_retention_routes_absent_on_todays_backend_are_a_typed_unsupported_error():
    recorder = Recorder(reply(404, error_body("operation_unavailable")))
    with pytest.raises(UnsupportedOperationError):
        sync_client(recorder).retention.get()


# --- typed definite errors ------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"detail": {"allowed_modes": ["zdr", "zdr_strict"]}},
        {"allowed_modes": ["zdr", "zdr_strict"]},
    ],
    ids=["nested_detail", "flat"],
)
def test_mode_not_allowed_carries_allowed_modes_on_a_mutation(extra):
    recorder = Recorder(reply(403, error_body("retention_mode_not_allowed", **extra)))
    with pytest.raises(RetentionModeNotAllowed) as caught:
        sync_client(recorder).searches.create("q", options=options(retention="retained"))
    error = caught.value
    assert not isinstance(error, OutcomeUncertainError)
    assert (error.status, error.code, error.retryable) == (403, "retention_mode_not_allowed", False)
    assert error.allowed_modes == ("zdr", "zdr_strict")
    assert error.request_id == "req-err"


def test_mode_not_allowed_without_detail_has_no_allowed_modes():
    recorder = Recorder(reply(403, error_body("retention_mode_not_allowed")))
    with pytest.raises(RetentionModeNotAllowed) as caught:
        sync_client(recorder).searches.create("q", options=options())
    assert caught.value.allowed_modes == ()


def test_unrelated_error_with_unknown_detail_is_still_a_contract_error():
    from synth_ai.errors import ContractError

    recorder = Recorder(reply(403, error_body("something_else", detail={"x": 1})))
    with pytest.raises(ContractError):
        sync_client(recorder).searches.get(SEARCH)


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (410, "search_content_purged", SearchContentPurged),
        (409, "result_not_retained_strict_zdr", StrictZdrResultUnavailable),
    ],
)
def test_result_raises_typed_definite_errors(status, code, kind):
    recorder = Recorder(
        reply(200, snapshot_body(retention=LEGACY_ZDR)), reply(status, error_body(code))
    )
    run = sync_client(recorder).searches.create("q", options=options())
    with pytest.raises(kind) as caught:
        run.result()
    assert (caught.value.status, caught.value.code, caught.value.retryable) == (status, code, False)
    assert (status, code) in POLICY_ERROR_TYPES


def test_result_raises_typed_errors_async():
    async def run() -> None:
        recorder = Recorder(
            reply(200, snapshot_body(retention=LEGACY_ZDR)),
            reply(410, error_body("search_content_purged")),
        )
        search = await async_client(recorder).searches.create("q", options=options())
        with pytest.raises(SearchContentPurged):
            await search.result()

    asyncio.run(run())


OBSERVATION = ObservationOptions(
    timeout_seconds=5,
    idle_timeout_seconds=5,
    reconnect_deadline_seconds=5,
    retry_count_max=0,
    maximum_events=10,
    poll_interval_seconds=0.01,
)


def test_wait_raises_content_purged_from_status_without_polling_to_the_deadline():
    purged = dict(LEGACY_ZDR, content_state="purged", purges_at=NOW.isoformat())
    recorder = Recorder(
        reply(200, snapshot_body(retention=LEGACY_ZDR)),
        reply(200, snapshot_body("completed", retention=purged)),
    )
    run = sync_client(recorder).searches.create("q", options=options())
    with pytest.raises(SearchContentPurged) as caught:
        run.wait(observation=OBSERVATION)
    assert caught.value.retryable is False and caught.value.status == 410
    assert len(recorder.requests) == 2


def test_wait_on_zdr_strict_raises_typed_error_instead_of_polling():
    strict = dict(LEGACY_ZDR, mode="zdr_strict", content_state="unavailable")
    recorder = Recorder(
        reply(200, snapshot_body(retention=strict)),
        reply(200, snapshot_body("completed", retention=strict)),
    )
    run = sync_client(recorder).searches.create("q", options=options(retention="zdr_strict"))
    with pytest.raises(StrictZdrResultUnavailable) as caught:
        run.wait(observation=OBSERVATION)
    assert "inline_result" in str(caught.value)
    assert caught.value.retryable is False


def test_wait_on_zdr_strict_raises_typed_error_async():
    async def run() -> None:
        strict = dict(LEGACY_ZDR, mode="zdr_strict", content_state="unavailable")
        recorder = Recorder(
            reply(200, snapshot_body(retention=strict)),
            reply(200, snapshot_body("completed", retention=strict)),
        )
        search = await async_client(recorder).searches.create(
            "q", options=options(retention="zdr_strict")
        )
        with pytest.raises(StrictZdrResultUnavailable):
            await search.wait(observation=OBSERVATION)

    asyncio.run(run())


# --- tolerant parsing -----------------------------------------------------------


def test_legacy_retention_block_still_parses():
    parsed = SearchSnapshot.model_validate_json(json.dumps(snapshot_body(retention=LEGACY_ZDR)))
    assert parsed.retention is not None and parsed.retention.mode == "zdr"
    assert parsed.inline_result is None


def test_retention_block_with_every_field_absent_parses():
    retention = SearchRetention.model_validate({})
    assert retention.mode is None and retention.content_state is None
    assert retention.purges_at is None and retention.expires_at is None
    assert retention.history_available is False


def test_new_fields_and_unknown_content_state_are_preserved_not_rejected():
    block = {"mode": "zdr", "content_state": "quarantined", "purges_at": NOW.isoformat()}
    parsed = SearchSnapshot.model_validate_json(
        json.dumps(snapshot_body("completed", available=True, retention=block))
    )
    assert parsed.retention is not None
    assert parsed.retention.content_state == "quarantined"
    assert parsed.retention.purges_at == NOW
    retained = SearchRetention.model_validate_json(
        json.dumps(
            {"mode": "retained", "content_state": "available", "expires_at": NOW.isoformat()}
        )
    )
    assert retained.expires_at == NOW


def test_known_unreadable_state_still_refuses_to_advertise_a_result():
    with pytest.raises(ValueError):
        SearchSnapshot.model_validate_json(
            json.dumps(
                snapshot_body(
                    "completed",
                    available=True,
                    retention={"mode": "zdr", "content_state": "purged"},
                )
            )
        )


def test_strict_mode_does_not_trip_the_selected_mode_check_when_omitted():
    strict = dict(LEGACY_ZDR, mode="zdr_strict")
    recorder = Recorder(reply(200, snapshot_body(retention=strict)))
    run = sync_client(recorder).searches.create("q", options=options())
    assert run.inline_result is None


def test_inline_result_is_exposed_when_the_create_response_carries_one():
    strict = dict(LEGACY_ZDR, mode="zdr_strict")
    recorder = Recorder(
        reply(200, snapshot_body("completed", retention=strict, inline_result=result_body(strict)))
    )
    run = sync_client(recorder).searches.create("q", options=options(retention="zdr_strict"))
    assert run.inline_result is not None and run.inline_result.response == "answer"
    assert run.inline_result.retention is not None
    assert run.inline_result.retention.mode == "zdr_strict"


# --- MCP ------------------------------------------------------------------------


def test_mcp_search_create_passes_retention_through_and_omits_it_when_unset():
    from synth_ai.mcp.catalog import BY_NAME

    spec = BY_NAME["index_search_create"]
    base = {
        "query": "q",
        "idempotency_key": "k1",
        "options": {"maximum_charge": {"coefficient": 1, "scale": 0, "currency": "USD"}},
    }
    assert "retention" not in spec.validate_arguments(base)["options"]
    with_mode = {**base, "options": {**base["options"], "retention": "zdr_strict"}}
    assert spec.validate_arguments(with_mode)["options"]["retention"] == "zdr_strict"
    bad = {**base, "options": {**base["options"], "retention": "forever"}}
    with pytest.raises(ValueError):
        spec.validate_arguments(bad)
    assert "inline_result" in spec.output_schema()["properties"]


def test_mcp_server_surfaces_purged_and_strict_errors_with_their_own_codes():
    pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")
    from synth_ai.mcp.server import error_code

    purged = SearchContentPurged("x", code="search_content_purged", status=410, retryable=False)
    strict = StrictZdrResultUnavailable(
        "x", code="result_not_retained_strict_zdr", status=409, retryable=False
    )
    refused = RetentionModeNotAllowed(
        "x", code="retention_mode_not_allowed", status=403, retryable=False
    )
    assert error_code(purged) == "search_content_purged"
    assert error_code(strict) == "result_not_retained_strict_zdr"
    assert error_code(refused) == "retention_mode_not_allowed"
