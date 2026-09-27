"""DEEP create/wait survive transient 503s without creating a second Search.

Every request is served by an httpx MockTransport; no network or credential.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from synth_ai.client import AsyncSynthClient, SynthClient
from synth_ai.core.errors import RateLimitedError, TransientServiceError
from synth_ai.sdk.index import IndexErrorCode, IndexRetryPolicy, search_id_from_error
from synth_ai.sdk.index.search import SearchSpec

FAST = IndexRetryPolicy(delay_seconds_initial=0.0, delay_seconds_max=0.0)
SEARCH_ID = "search-7ecc711a"
SPEC = SearchSpec(
    query="exact revision citations",
    mode="deep",
    scope={"visibility": "public"},
    billing={"allow_wallet": True, "max_charge_cents": 25},
)


def _snapshot(state: str = "queued") -> dict[str, Any]:
    return {
        "search_id": SEARCH_ID,
        "spec": SPEC.model_dump(mode="json"),
        "state": state,
        "requested_mode": "deep",
        "created_at": "2026-09-27T00:10:00Z",
        "updated_at": "2026-09-27T00:10:01Z",
    }


def _unavailable(code: str, **extra: Any) -> httpx.Response:
    headers = extra.pop("headers", {})
    return httpx.Response(503, json={"detail": {"code": code, **extra}}, headers=headers)


def _mount(client: Any, responses: list[httpx.Response]) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    transport = client.index._transport
    old = transport.client
    kind = httpx.AsyncClient if isinstance(old, httpx.AsyncClient) else httpx.Client
    transport.client = kind(
        base_url=str(old.base_url), headers=old.headers, transport=httpx.MockTransport(handler)
    )
    return seen


def _client() -> SynthClient:
    return SynthClient(api_key="sk-test", base_url="https://api.example.test")


def test_create_always_sends_a_key_and_replays_it_after_503() -> None:
    client = _client()
    try:
        seen = _mount(
            client,
            [
                _unavailable("monitor_admission_deadline_exceeded"),
                _unavailable("monitor_draining", headers={"Retry-After": "0"}),
                httpx.Response(200, json=_snapshot()),
            ],
        )
        handle = client.index.searches.create(SPEC, retry=FAST)
    finally:
        client.close()
    assert handle.search_id == SEARCH_ID
    keys = [request.headers["Idempotency-Key"] for request in seen]
    assert len(keys) == 3 and len(set(keys)) == 1 and keys[0]
    assert all(request.method == "POST" for request in seen)


def test_caller_key_is_used_verbatim() -> None:
    client = _client()
    try:
        seen = _mount(client, [httpx.Response(200, json=_snapshot())])
        client.index.searches.create(SPEC, idempotency_key="agent-key-1", retry=FAST)
    finally:
        client.close()
    assert seen[0].headers["Idempotency-Key"] == "agent-key-1"


@pytest.mark.parametrize(
    "failure",
    [
        _unavailable("monitor_admission_deadline_exceeded", search_id=SEARCH_ID),
        _unavailable("monitor_draining", headers={"X-Synth-Search-Id": SEARCH_ID}),
    ],
)
def test_failure_naming_the_search_reconnects_instead_of_creating(failure) -> None:
    client = _client()
    try:
        seen = _mount(client, [failure, httpx.Response(200, json=_snapshot("running"))])
        handle = client.index.searches.create(SPEC, retry=FAST)
    finally:
        client.close()
    assert handle.search_id == SEARCH_ID
    assert [(r.method, r.url.path) for r in seen] == [
        ("POST", "/api/v1/index/searches"),
        ("GET", f"/api/v1/index/searches/{SEARCH_ID}"),
    ]


def test_exhausted_retries_raise_with_the_search_id_when_named() -> None:
    client = _client()
    try:
        _mount(
            client,
            [_unavailable("monitor_draining", search_id=SEARCH_ID)]
            + [_unavailable("index_unavailable")] * 3,
        )
        with pytest.raises(TransientServiceError) as raised:
            client.index.searches.create(SPEC, retry=FAST)
    finally:
        client.close()
    assert search_id_from_error(raised.value) == SEARCH_ID


def test_retry_after_beyond_budget_is_surfaced_not_slept() -> None:
    client = _client()
    try:
        seen = _mount(
            client,
            [
                _unavailable(
                    "index_capacity_exhausted",
                    reason="deep_daily_budget",
                    headers={"Retry-After": "3600"},
                )
            ],
        )
        with pytest.raises(TransientServiceError) as raised:
            client.index.searches.create(SPEC, retry=FAST)
    finally:
        client.close()
    assert len(seen) == 1
    assert raised.value.error_code == IndexErrorCode.CAPACITY_EXHAUSTED
    assert raised.value.reason == "deep_daily_budget"
    assert raised.value.retry_after_seconds == 3600


def test_rate_limit_is_not_retried_and_exposes_reason() -> None:
    client = _client()
    try:
        seen = _mount(
            client,
            [
                httpx.Response(
                    429,
                    json={
                        "detail": {
                            "code": "index_concurrency_limited",
                            "reason": "key_concurrency_limit",
                        }
                    },
                    headers={"Retry-After": "5"},
                )
            ],
        )
        with pytest.raises(RateLimitedError) as raised:
            client.index.searches.create(SPEC, retry=FAST)
    finally:
        client.close()
    assert len(seen) == 1
    assert raised.value.reason == "key_concurrency_limit"
    assert raised.value.retry_after_seconds == 5


def test_wait_polls_through_a_transient_503() -> None:
    client = _client()
    try:
        _mount(
            client,
            [
                httpx.Response(200, json=_snapshot()),
                _unavailable("monitor_draining"),
                httpx.Response(
                    200,
                    json={
                        **_snapshot("failed"),
                        "failure": {"code": "index_deep_deadline_exceeded", "retryable": True},
                    },
                ),
            ],
        )
        handle = client.index.searches.create(SPEC, retry=FAST)
        from synth_ai.sdk.index.search import SearchExecutionFailedError

        with pytest.raises(SearchExecutionFailedError):
            handle.wait(timeout_seconds=5, poll_seconds=0.01)
    finally:
        client.close()


def test_async_create_replays_the_same_key() -> None:
    async def run() -> list[httpx.Request]:
        client = AsyncSynthClient(api_key="sk-test", base_url="https://api.example.test")
        try:
            seen = _mount(
                client,
                [_unavailable("index_unavailable"), httpx.Response(200, json=_snapshot())],
            )
            handle = await client.index.searches.create(SPEC, retry=FAST)
            assert handle.search_id == SEARCH_ID
            return seen
        finally:
            await client.close()

    seen = asyncio.run(run())
    assert len({request.headers["Idempotency-Key"] for request in seen}) == 1
    assert len(seen) == 2


def test_new_codes_are_named() -> None:
    assert IndexErrorCode("index_capacity_exhausted") is IndexErrorCode.CAPACITY_EXHAUSTED
    assert IndexErrorCode("index_query_too_long") is IndexErrorCode.QUERY_TOO_LONG


def test_asset_download_unavailable_is_named_and_not_retried() -> None:
    """Asset downloads are not in the public launch: a typed 403, one request."""
    from synth_ai.sdk.index.contracts import ContributionReference

    client = _client()
    try:
        seen = _mount(
            client,
            [
                httpx.Response(
                    403,
                    json={
                        "detail": {
                            "code": "index_asset_download_unavailable",
                            "reason": "not_available_at_launch",
                        }
                    },
                )
            ],
        )
        with pytest.raises(Exception) as raised:
            client.index.contributions.assets.retrieve(
                ContributionReference(contribution_id="c1", revision_id="r1"), "a1"
            )
    finally:
        client.close()
    assert raised.value.error_code == IndexErrorCode.ASSET_DOWNLOAD_UNAVAILABLE
    assert raised.value.reason == "not_available_at_launch"
    assert len(seen) == 1


def test_contribution_withdrawn_is_named() -> None:
    assert IndexErrorCode("contribution_withdrawn") is IndexErrorCode.CONTRIBUTION_WITHDRAWN


def test_wait_bounds_each_status_poll_and_retries_a_hung_poll() -> None:
    """Prod 2026-09-27 P12: a status poll whose response never arrived held the
    client for the 120 s transport timeout. Each poll now carries a <=20 s
    timeout; a timed-out poll is treated as transient and polled again."""
    from synth_ai.sdk.index.client import STATUS_POLL_TIMEOUT_SECONDS
    from synth_ai.sdk.index.search import SearchExecutionFailedError

    client = _client()
    polls: list[dict] = []
    responses = [
        httpx.Response(200, json=_snapshot()),
        "hang",
        httpx.Response(
            200,
            json={
                **_snapshot("failed"),
                "failure": {"code": "index_deep_deadline_exceeded", "retryable": True},
            },
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            polls.append(dict(request.extensions.get("timeout") or {}))
        item = responses.pop(0)
        if item == "hang":
            raise httpx.ReadTimeout("status poll never answered", request=request)
        return item

    try:
        transport = client.index._transport
        old = transport.client
        transport.client = httpx.Client(
            base_url=str(old.base_url),
            headers=old.headers,
            transport=httpx.MockTransport(handler),
        )
        handle = client.index.searches.create(SPEC, retry=FAST)
        with pytest.raises(SearchExecutionFailedError):
            handle.wait(timeout_seconds=60, poll_seconds=0.01)
    finally:
        client.close()
    assert len(polls) == 2
    for timeouts in polls:
        assert 1.0 <= timeouts["read"] <= STATUS_POLL_TIMEOUT_SECONDS
