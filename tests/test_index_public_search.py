"""Index Search v0.2 public contract: SDK client, typed errors, capabilities, MCP tool.

Every request is served by an httpx MockTransport; no network, no credential.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from synth_ai.client import SynthClient
from synth_ai.mcp.research.server import _mcp_structured_core_error_payload
from synth_ai.mcp.research.tools import index as mcp_index
from synth_ai.mcp.research.tools.index import build_index_tools, public_search_tool_description
from synth_ai.sdk.index import (
    AccessFundingAccount,
    Capabilities,
    PublicIndexClient,
    PublicSearchAuthenticatedError,
    PublicSearchBudgetExhaustedError,
    PublicSearchBudgetScope,
    PublicSearchCancelledError,
    PublicSearchCapability,
    PublicSearchDisabledError,
    PublicSearchError,
    PublicSearchFailedError,
    PublicSearchHandle,
    PublicSearchMonitorUnavailableError,
    PublicSearchNotFoundError,
    PublicSearchNotReadyError,
    PublicSearchRateLimitedError,
    PublicSearchRateStoreUnavailableError,
    PublicSearchRequestTooLargeError,
    PublicSearchResult,
    PublicSearchUnavailableError,
    PublicSearchWaitTimeoutError,
    SearchMode,
    WalletConsentRequiredError,
    public_search_copy,
    wallet_search_grant,
)
from synth_ai.sdk.index import public_search as public_search_module

BASE = "https://api.example.test"
SEARCH_ID = "psearch-5f2a"
TOKEN = "tok-do-not-log-8c1d9e"

Handler = Callable[[httpx.Request], httpx.Response]


# Fixtures are built from the backend field lists (backend worktree
# app/api/v1/index/public_search.py and packages/contributions/{search,views}.py):
# PublicSearchDelivery = exactly the Monitor's public key set (backend #1660), with
# search id/token/charge in headers;
# PublicSearchAccepted is the 202 body and the 200 body of a terminal failed/cancelled
# poll; errors are {"detail": {"code", "scope"}}.

RELEASE = "release-77"
# Backend INLINE_CITATION only matches UUID contribution ids.
C1 = "0b6f3c1e-8d2a-4f7b-9c1d-2e3f4a5b6c7d"
C2 = "7a1e2b3c-4d5e-4f60-8172-839405a6b7c8"


def _envelope(mode: str = "fast", **extra: Any) -> dict[str, Any]:
    """``PublicSearchDelivery`` (backend #1660): exactly the Monitor's public key set."""
    return {
        "request_id": "req-41",
        "mode": mode,
        "status": "completed",
        "corpus_generation": "corpus-9",
        "ranker_version": "ranker-3",
        "parser_version": "parser-2",
        "taxonomy_version": "tax-1",
        "response": f"Verifiers reward exact program output [{C1}]; see also [{C2}].",
        "citations": [
            {"contribution_id": C1, "revision_id": "rev-1"},
            {"contribution_id": C2, "revision_id": "rev-7"},
        ],
        "amount_cents": 0,
        **extra,
    }


def _public_headers(**extra: str) -> dict[str, str]:
    """Public-only fields ride headers outside the reviewed body (backend #1660)."""
    return {
        "X-Index-Search-Id": SEARCH_ID,
        "X-Search-Token": TOKEN,
        "X-Search-Token-Expires-At": "2026-09-28T13:00:00+00:00",
        "X-Index-Customer-Charge-Cents": "0",
        "X-Index-Internal-Cost-Recorded": "true",
        **extra,
    }


def _delivered(mode: str = "fast", **extra: Any) -> httpx.Response:
    """200 delivery exactly as the route sends it: public-only fields in headers."""
    return httpx.Response(
        200,
        json=_envelope(mode, **extra),
        headers=_public_headers(
            **{"X-Index-Monitor-Release": RELEASE, "X-Index-Monitor-Delivery": "released"}
        ),
    )


def _accepted(state: str = "queued", **extra: Any) -> dict[str, Any]:
    """``PublicSearchAccepted`` (DEEP start, running poll, terminal failed/cancelled poll)."""
    return {
        "search_id": SEARCH_ID,
        "mode": "deep",
        "state": state,
        "status": state,
        "poll_url": f"/api/v1/index/public/searches/{SEARCH_ID}",
        "cancellation_requested": False,
        "result_available": False,
        "failure": None,
        "search_token": None,
        "search_token_expires_at": None,
        **extra,
    }


def _public_search_block(**overrides: Any) -> dict[str, Any]:
    """``PublicSearchCapability`` exactly as ``public_search_capability()`` emits it."""
    block: dict[str, Any] = {
        "enabled": True,
        "modes": ["fast", "deep"],
        "limits": {
            "fast": {
                "peer_per_minute": 10,
                "peer_per_day": 200,
                "global_per_minute": 600,
                "global_per_day": 20000,
            },
            "deep": {
                "peer_per_minute": 2,
                "peer_per_day": 20,
                "global_per_minute": 60,
                "global_per_day": 1000,
            },
        },
        "daily_budget_cents": 5000,
        "deep_concurrency_max": 4,
        "price_cents": {"fast": 0, "deep": 0},
        "retention": {"public_query_days": 30, "private_processing_minutes": 60},
        "privacy_copy": "Public queries may be reviewed to improve the Index.",
        "token_ttl_seconds": 3600,
        "max_body_bytes": 16384,
    }
    block.update(overrides)
    return block


def _capabilities(public_search: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contribution_schema_versions": ["synth.index.contribution.v1"],
        "taxonomy_version": "tax-1",
        "modes": ["fast", "deep"],
        "search_modes": ["fast", "deep"],
        "visibilities": ["public"],
        "search_filters": True,
        "private_search": {"activated": False},
        "upload": {"enabled": False},
        "review": {"enabled": False},
        "publication": {"enabled": False},
        "limits": {
            "max_results": 10,
            "max_excerpts_per_result": 2,
            "query_max_bytes": 8192,
            "contents_max_bytes": 65536,
        },
    }
    if public_search is not None:
        payload["public_search"] = public_search
    return payload


def _mount(api: Any, handler: Handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = api._transport
    old = transport.client
    transport.client = httpx.Client(
        base_url=str(old.base_url), headers=old.headers, transport=httpx.MockTransport(record)
    )
    return seen


def _access_funding(
    *,
    wallet_enabled: bool = True,
    consent: str | None = "synth-index-wallet-terms-2026-09-27",
    monthly_cap_cents: int = 2000,
) -> dict[str, Any]:
    """``AccessFundingAccount`` as ``GET /api/v1/index/me/access-funding`` returns it."""

    def mode(name: str) -> dict[str, Any]:
        return {
            "mode": name,
            "access": True,
            "wallet_enabled": wallet_enabled,
            "monthly_cap_cents": monthly_cap_cents,
            "concurrency_limit": 2,
            "consent_terms_version": consent if wallet_enabled else None,
            "policy_revision": 3,
        }

    return {
        "org_id": "org-1",
        "can_manage_policy": True,
        "modes": [mode("fast"), mode("deep")],
        "deep_beta": None,
        "live_wallet_holds_microcents": 0,
        "wallet_available_microcents": 1_000_000,
        "generated_at": "2026-09-28T16:00:00Z",
    }


def _paid_fast_result() -> dict[str, Any]:
    """``SearchResult`` from the paid keyed ``POST /api/v1/index/search``."""
    versions = {
        "corpus_generation": "corpus-9",
        "ranker_version": "ranker-3",
        "parser_version": "parser-2",
        "taxonomy_version": "tax-1",
    }
    return {
        "search_id": "search-paid-1",
        "request_id": "req-paid-1",
        "requested_mode": "fast",
        "effective_mode": "fast",
        "status": "completed",
        "partial_reason": None,
        **versions,
        "execution_versions": versions,
        "response": f"Verifiers reward exact program output [{C1}].",
        "citations": [{"contribution_id": C1, "revision_id": "rev-1"}],
        "usage": {
            "mode": "fast",
            "billing_scope": "public",
            "price_version": "synth.index.fast.v2",
            "amount_cents": 5,
            "receipt_id": "receipt-1",
            "funding_source": "wallet",
            "wallet_debit_cents": 5,
        },
    }


def _error(status: int, code: str, **extra: Any) -> httpx.Response:
    headers = extra.pop("headers", {})
    # Backend public_search_http_error: HTTPException(detail={"code", "scope"?}).
    return httpx.Response(status, json={"detail": {"code": code, **extra}}, headers=headers)


@pytest.fixture
def anonymous() -> PublicIndexClient:
    client = PublicIndexClient(base_url=BASE)
    yield client
    client.close()


@pytest.fixture
def keyed() -> SynthClient:
    client = SynthClient(api_key="sk-test", base_url=BASE)
    yield client
    client.close()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(public_search_module.time, "sleep", slept.append)
    return slept


# SDK: Fast -----------------------------------------------------------------------


def test_anonymous_fast_search_sends_no_authorization(anonymous: PublicIndexClient) -> None:
    seen = _mount(anonymous, lambda request: _delivered())
    result = anonymous.public_search("RLVR verifier design", max_results=3)

    assert isinstance(result, PublicSearchResult)
    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/api/v1/index/public/search"
    assert "authorization" not in request.headers
    assert (
        httpx.Request(
            "POST", BASE, json={"mode": "fast", "query": "RLVR verifier design", "max_results": 3}
        ).content
        == request.content
    )
    assert result.search_id == SEARCH_ID
    assert result.customer_charge_cents == 0
    assert result.monitor_release_id == "release-77"
    assert result.mode is SearchMode.FAST
    assert [(c.contribution_id, c.revision_id, c.citation) for c in result.citations] == [
        (C1, "rev-1", f"{C1}@rev-1"),
        (C2, "rev-7", f"{C2}@rev-7"),
    ]
    assert result.results == result.citations
    assert result.response.startswith(f"Verifiers reward exact program output [{C1}]")
    assert result.status == "completed" and result.partial_reason is None
    assert "search_token" not in result.raw
    assert TOKEN not in repr(result)


def test_fast_release_id_comes_from_monitor_header_first(anonymous: PublicIndexClient) -> None:
    # The Monitor's decision travels as a header (reviewed bytes == delivered bytes);
    # the body's monitor.release_id is null and release_header names the header.
    body = _envelope(monitor={"release_id": None, "release_header": "X-Index-Monitor-Release"})
    _mount(
        anonymous,
        lambda request: httpx.Response(
            200,
            json=body,
            headers=_public_headers(
                **{
                    "X-Index-Monitor-Release": "release-hdr-1",
                    "X-Index-Monitor-Delivery": "released",
                }
            ),
        ),
    )
    result = anonymous.public_search("q")
    assert isinstance(result, PublicSearchResult)
    assert result.monitor_release_id == "release-hdr-1"

    # Header wins over a populated body field; the body is the fallback only.
    body_release = _envelope(monitor={"release_id": "release-body"})
    _mount(
        anonymous,
        lambda request: httpx.Response(
            200,
            json=body_release,
            headers=_public_headers(**{"X-Index-Monitor-Release": "release-hdr-2"}),
        ),
    )
    prefer = anonymous.public_search("q")
    assert isinstance(prefer, PublicSearchResult)
    assert prefer.monitor_release_id == "release-hdr-2"

    _mount(anonymous, lambda request: httpx.Response(200, json=body, headers=_public_headers()))
    absent = anonymous.public_search("q")
    assert isinstance(absent, PublicSearchResult)
    assert absent.monitor_release_id is None


def test_deep_completion_reads_monitor_header(
    anonymous: PublicIndexClient, no_sleep: list[float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                202, json=_accepted(search_token=TOKEN), headers={"Retry-After": "1"}
            )
        assert request.headers.get("X-Search-Token") == TOKEN
        return httpx.Response(
            200,
            json=_envelope("deep"),
            headers=_public_headers(**{"X-Index-Monitor-Release": "release-hdr-deep"}),
        )

    _mount(anonymous, handler)
    handle = anonymous.public_search("q", mode="deep", wait=False)
    assert isinstance(handle, PublicSearchHandle)
    result = handle.poll()
    assert isinstance(result, PublicSearchResult)
    assert result.monitor_release_id == "release-hdr-deep"
    assert handle.result is result


def test_keyed_public_search_is_refused_with_typed_409(keyed: SynthClient) -> None:
    # Backend #1708: the public route is anonymous-only; credentials get 409.
    seen = _mount(keyed.index, lambda request: _error(409, "index_public_search_authenticated"))
    with pytest.raises(PublicSearchAuthenticatedError) as info:
        keyed.index.public_search("RLVR verifier design", idempotency_key="key-1")

    (request,) = seen
    assert request.url.path == "/api/v1/index/public/search"
    assert info.value.status == 409
    assert info.value.code == "index_public_search_authenticated"
    assert "IndexAPI.search" in str(info.value)
    assert isinstance(info.value, PublicSearchError)


def test_fast_result_requires_customer_charge(anonymous: PublicIndexClient) -> None:
    envelope = _envelope()
    del envelope["amount_cents"]
    headers = _public_headers()
    del headers["X-Index-Customer-Charge-Cents"]
    _mount(anonymous, lambda request: httpx.Response(200, json=envelope, headers=headers))
    with pytest.raises(PublicSearchError, match="customer charge"):
        anonymous.public_search("q")


def test_fast_search_id_from_header_and_legacy_body_fallback(
    anonymous: PublicIndexClient,
) -> None:
    _mount(anonymous, lambda request: _delivered())
    result = anonymous.public_search("q")
    assert isinstance(result, PublicSearchResult)
    assert result.search_id == SEARCH_ID
    assert set(result.raw) == set(_envelope())
    # Pre-#1660 backends put search_id and usage in the body.
    legacy = {
        **_envelope(),
        "search_id": SEARCH_ID,
        "usage": {"customer_charge_cents": 0},
    }
    _mount(anonymous, lambda request: httpx.Response(200, json=legacy))
    old = anonymous.public_search("q")
    assert isinstance(old, PublicSearchResult)
    assert old.search_id == SEARCH_ID and old.customer_charge_cents == 0
    # No header and no body id is a contract error, not a silent empty id.
    _mount(anonymous, lambda request: httpx.Response(200, json=_envelope()))
    with pytest.raises(PublicSearchError):
        anonymous.public_search("q")


# SDK: Deep -----------------------------------------------------------------------


def _deep_backend(
    *, poll_states: tuple[str, ...] = ("queued", "running")
) -> tuple[Handler, list[str]]:
    polls: list[str] = []
    pending = list(poll_states)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/index/public/search":
            return httpx.Response(
                202,
                json=_accepted(
                    search_token=TOKEN, search_token_expires_at="2026-09-28T13:00:00+00:00"
                ),
                headers={"Retry-After": "2"},
            )
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/index/public/searches/{SEARCH_ID}"
        if request.headers.get("X-Search-Token") != TOKEN:
            return _error(404, "index_search_not_found")
        polls.append(request.headers["X-Search-Token"])
        if pending:
            state = pending.pop(0)
            if state in {"failed", "cancelled"}:
                # Terminal non-delivery: HTTP 200 with the lifecycle body, not a result.
                failure = {"code": "index_deadline_exceeded", "retryable": True}
                return httpx.Response(
                    200,
                    json=_accepted(state, failure=failure if state == "failed" else None),
                )
            return httpx.Response(202, json=_accepted(state), headers={"Retry-After": "0.5"})
        return _delivered("deep")

    return handler, polls


def test_deep_polls_with_token_at_backend_cadence(
    anonymous: PublicIndexClient, no_sleep: list[float]
) -> None:
    handler, polls = _deep_backend()
    _mount(anonymous, handler)
    result = anonymous.public_search("compare retrieval designs", mode="deep", timeout_s=30)

    assert isinstance(result, PublicSearchResult)
    assert result.mode is SearchMode.DEEP
    assert result.search_id == SEARCH_ID
    assert polls == [TOKEN, TOKEN, TOKEN]
    # First wait honors the 202's Retry-After, later waits the poll's Retry-After.
    assert no_sleep == [2.0, 0.5, 0.5]


def test_deep_handle_never_exposes_token(
    anonymous: PublicIndexClient, no_sleep: list[float]
) -> None:
    handler, _polls = _deep_backend(poll_states=("queued", "running"))
    _mount(anonymous, handler)
    handle = anonymous.public_search("q", mode="deep", wait=False)

    assert isinstance(handle, PublicSearchHandle)
    assert handle.search_id == SEARCH_ID
    assert TOKEN not in repr(handle)
    assert TOKEN not in str(handle)
    assert handle.result is None
    with pytest.raises(PublicSearchNotReadyError):
        handle.replay()
    status = handle.poll()
    assert not isinstance(status, PublicSearchResult)
    assert status.state == "running" and status.poll_after_s == 0.5
    assert isinstance(handle.poll(), PublicSearchResult)
    assert handle.result is not None
    assert TOKEN not in repr(handle.result)
    assert handle.replay() is handle.result


def test_deep_wait_timeout_keeps_handle(
    anonymous: PublicIndexClient, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(public_search_module.time, "monotonic", lambda: next(clock))
    handler, _polls = _deep_backend(poll_states=("queued", "running", "running", "running"))
    _mount(anonymous, handler)
    with pytest.raises(PublicSearchWaitTimeoutError) as info:
        anonymous.public_search("q", mode="deep", timeout_s=5)
    assert info.value.search_id == SEARCH_ID
    assert isinstance(info.value.handle, PublicSearchHandle)
    assert TOKEN not in str(info.value)


def test_deep_terminal_failed_poll_raises_typed_failure(
    anonymous: PublicIndexClient, no_sleep: list[float]
) -> None:
    handler, polls = _deep_backend(poll_states=("running", "failed"))
    _mount(anonymous, handler)
    with pytest.raises(PublicSearchFailedError) as info:
        anonymous.public_search("q", mode="deep", timeout_s=30)
    assert type(info.value) is PublicSearchFailedError
    assert info.value.state == "failed"
    assert info.value.failure_code == "index_deadline_exceeded"
    assert info.value.failure_retryable is True
    assert "missing" not in str(info.value)
    assert len(polls) == 2


def test_deep_terminal_cancelled_poll_raises_cancelled(anonymous: PublicIndexClient) -> None:
    handler, _polls = _deep_backend(poll_states=("cancelled",))
    _mount(anonymous, handler)
    handle = anonymous.public_search_handle(SEARCH_ID, TOKEN)
    with pytest.raises(PublicSearchCancelledError) as info:
        handle.poll()
    assert isinstance(info.value, PublicSearchFailedError)
    assert info.value.state == "cancelled"
    assert handle.result is None


def test_deep_start_already_failed_raises(anonymous: PublicIndexClient) -> None:
    # A DEEP create that is already terminal answers 200 with the lifecycle body.
    body = _accepted(
        "failed", failure={"code": "index_search_failed", "retryable": False}, search_token=TOKEN
    )
    _mount(anonymous, lambda request: httpx.Response(200, json=body))
    with pytest.raises(PublicSearchFailedError) as info:
        anonymous.public_search("q", mode="deep")
    assert info.value.failure_code == "index_search_failed"


def test_partial_delivery_keeps_reason(anonymous: PublicIndexClient) -> None:
    _mount(
        anonymous, lambda request: _delivered(status="partial", partial_reason="deadline_exceeded")
    )
    result = anonymous.public_search("q")
    assert isinstance(result, PublicSearchResult)
    assert result.status == "partial" and result.partial_reason == "deadline_exceeded"


def test_wrong_token_is_not_found(anonymous: PublicIndexClient) -> None:
    handler, _polls = _deep_backend()
    _mount(anonymous, handler)
    stale = anonymous.public_search_handle(SEARCH_ID, "wrong-token")
    with pytest.raises(PublicSearchNotFoundError) as info:
        stale.poll()
    assert info.value.code == "index_search_not_found"
    assert info.value.status == 404


def test_handle_cancel_sends_token(anonymous: PublicIndexClient) -> None:
    seen = _mount(
        anonymous,
        lambda request: httpx.Response(
            200, json=_accepted("cancelled", cancellation_requested=True)
        ),
    )
    status = anonymous.public_search_handle(SEARCH_ID, TOKEN).cancel()
    (request,) = seen
    assert request.url.path == f"/api/v1/index/public/searches/{SEARCH_ID}/cancel"
    assert request.headers["X-Search-Token"] == TOKEN
    assert status.state == "cancelled"


# SDK: typed errors -------------------------------------------------------------------


def test_rate_limited_carries_retry_after_and_scope(anonymous: PublicIndexClient) -> None:
    _mount(
        anonymous,
        lambda request: _error(
            429, "index_public_rate_limited", scope="peer_minute", headers={"Retry-After": "37"}
        ),
    )
    with pytest.raises(PublicSearchRateLimitedError) as info:
        anonymous.public_search("q")
    error = info.value
    assert error.retry_after_s == 37.0
    assert error.scope == "peer_minute"
    assert error.code == "index_public_rate_limited"
    assert error.status == 429
    assert error.retry_after_seconds == 37.0  # SynthError surface used by the MCP server
    assert "37" in str(error) and "peer_minute" in str(error)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("index_public_budget_exhausted", PublicSearchBudgetExhaustedError),
        ("index_rate_store_unavailable", PublicSearchRateStoreUnavailableError),
        ("monitor_unavailable", PublicSearchMonitorUnavailableError),
    ],
)
def test_each_503_code_fails_closed_with_its_own_type(
    anonymous: PublicIndexClient, code: str, expected: type[PublicSearchUnavailableError]
) -> None:
    _mount(anonymous, lambda request: _error(503, code))
    with pytest.raises(expected) as info:
        anonymous.public_search("q")
    assert isinstance(info.value, PublicSearchUnavailableError)
    assert info.value.code == code
    assert info.value.status == 503
    assert "nothing was charged" in str(info.value)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("deep_concurrency", PublicSearchBudgetScope.DEEP_CONCURRENCY),
        ("daily_cents", PublicSearchBudgetScope.DAILY_CENTS),
        ("some_future_budget", None),
        (None, None),
    ],
)
def test_budget_exhausted_exposes_typed_scope(
    anonymous: PublicIndexClient, scope: str | None, expected: PublicSearchBudgetScope | None
) -> None:
    extra = {} if scope is None else {"scope": scope}
    _mount(
        anonymous,
        lambda request: _error(
            503, "index_public_budget_exhausted", headers={"Retry-After": "15"}, **extra
        ),
    )
    with pytest.raises(PublicSearchBudgetExhaustedError) as info:
        anonymous.public_search("q")
    assert info.value.scope is expected
    assert info.value.retry_after_s == 15.0
    assert info.value.retry_after_seconds == 15.0
    assert info.value.status == 503
    assert info.value.code == "index_public_budget_exhausted"


def test_fast_search_has_no_reconnectable_handle(anonymous: PublicIndexClient) -> None:
    seen = _mount(anonymous, lambda request: _delivered())
    with pytest.raises(ValueError, match="Fast public search results are final"):
        anonymous.public_search_handle(SEARCH_ID, TOKEN, mode=SearchMode.FAST)
    with pytest.raises(ValueError, match="Fast public search results are final"):
        anonymous.public_search_handle(SEARCH_ID, TOKEN, mode="fast")
    assert seen == []
    result = anonymous.public_search("q")
    assert isinstance(result, PublicSearchResult)
    for name in ("handle", "poll", "wait", "replay", "refetch", "cancel"):
        assert not hasattr(result, name)


def test_request_too_large(anonymous: PublicIndexClient) -> None:
    _mount(anonymous, lambda request: _error(413, "index_request_too_large"))
    with pytest.raises(PublicSearchRequestTooLargeError):
        anonymous.public_search("q")


def test_flag_off_is_disabled(anonymous: PublicIndexClient) -> None:
    _mount(anonymous, lambda request: _error(404, "index_public_search_disabled"))
    with pytest.raises(PublicSearchDisabledError) as info:
        anonymous.public_search("q")
    assert info.value.code == "index_public_search_disabled"


def test_unknown_code_stays_a_public_search_error(anonymous: PublicIndexClient) -> None:
    _mount(anonymous, lambda request: _error(400, "index_something_new"))
    with pytest.raises(PublicSearchError) as info:
        anonymous.public_search("q")
    assert info.value.code == "index_something_new"
    assert type(info.value) is PublicSearchError


def test_private_search_path_is_unchanged(keyed: SynthClient) -> None:
    from synth_ai.core.errors import HTTPError
    from synth_ai.sdk.index import SearchSpec

    seen = _mount(keyed.index, lambda request: httpx.Response(500, json={"code": "x"}))
    with pytest.raises(HTTPError):
        keyed.index.search(SearchSpec(query="q"), idempotency_key="k")
    assert seen[0].url.path == "/api/v1/index/search"
    assert seen[0].headers["Idempotency-Key"] == "k"


# Capabilities and copy ------------------------------------------------------------------


def test_capabilities_public_search_block_is_typed(anonymous: PublicIndexClient) -> None:
    _mount(
        anonymous, lambda request: httpx.Response(200, json=_capabilities(_public_search_block()))
    )
    capability = anonymous.public_search_capability()
    assert isinstance(capability, PublicSearchCapability)
    assert capability.enabled is True
    assert capability.modes == (SearchMode.FAST, SearchMode.DEEP)
    assert capability.limits[SearchMode.FAST].peer_per_minute == 10
    assert capability.limits[SearchMode.DEEP].global_per_day == 1000
    assert capability.daily_budget_cents == 5000
    assert capability.deep_concurrency_max == 4
    assert capability.token_ttl_seconds == 3600
    assert capability.max_body_bytes == 16384
    assert capability.price_cents == {SearchMode.FAST: 0, SearchMode.DEEP: 0}
    assert capability.retention is not None
    assert capability.retention.public_query_days == 30
    assert capability.retention.private_processing_minutes == 60


def test_capability_block_ignores_unknown_future_fields() -> None:
    block = _public_search_block(new_term="x")
    block["limits"]["fast"]["peer_per_hour"] = 99
    block["retention"]["audit_days"] = 7
    capabilities = Capabilities.model_validate(_capabilities(block))
    assert capabilities.public_search is not None
    assert capabilities.public_search.limits[SearchMode.FAST].peer_per_day == 200


def test_capabilities_without_block_parse_on_older_backend() -> None:
    capabilities = Capabilities.model_validate(_capabilities(None))
    assert capabilities.public_search is None
    assert public_search_copy(None).available is False


def test_copy_strings_come_from_capabilities(anonymous: PublicIndexClient) -> None:
    _mount(
        anonymous, lambda request: httpx.Response(200, json=_capabilities(_public_search_block()))
    )
    copy = anonymous.public_search_terms()
    assert copy.available is True
    assert copy.price == "Fast search is free. Deep search is free."
    assert copy.limits == (
        "Fast: 10 per minute and 200 per day per caller; 600 per minute and 20000 per day "
        "platform-wide. Deep: 2 per minute and 20 per day per caller; 60 per minute and "
        "1000 per day platform-wide."
    )
    assert copy.privacy == (
        "Public queries may be reviewed to improve the Index. Public queries are retained "
        "for 30 days. Processing state expires after 60 minutes."
    )
    assert copy.as_dict()["price"] == copy.price


def test_copy_reflects_a_changed_price_without_code_changes() -> None:
    capability = PublicSearchCapability.model_validate(
        _public_search_block(price_cents={"fast": 0, "deep": 3}, modes=["fast", "deep"])
    )
    assert public_search_copy(capability).price == "Fast search is free. Deep search is 3 cents."
    disabled = PublicSearchCapability.model_validate(_public_search_block(enabled=False))
    assert public_search_copy(disabled).available is False
    assert "disabled" in public_search_copy(disabled).price


def test_zero_day_copy_distinguishes_durable_retention_from_processing() -> None:
    block = _public_search_block()
    block["retention"] = {"public_query_days": 0, "private_processing_minutes": 60}
    block["privacy_copy"] = "Customer content uses zero durable retention."
    capability = PublicSearchCapability.model_validate(block)
    copy = public_search_copy(capability)
    assert "Public query content is not retained in durable storage." in copy.privacy
    assert "Processing state expires after 60 minutes." in copy.privacy
    assert "retained for 0 days" not in copy.privacy


# MCP tool ------------------------------------------------------------------------------


@contextmanager
def _factory_client(handler: Handler):
    client = PublicIndexClient(base_url=BASE)
    _mount(client, handler)
    try:
        yield client
    finally:
        client.close()


def _routes(*, capabilities: dict[str, Any] | None, search: Handler) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/index/public/capabilities":
            return httpx.Response(200, json=_capabilities(capabilities))
        return search(request)

    return handler


def _tool(handler: Handler, name: str = "index_search"):
    tools = build_index_tools(lambda: _factory_client(handler), include_lifecycle=False)
    return next(tool for tool in tools if tool.name == name)


_FIVE_CENTS = re.compile(r"\b5\s*(cents?|c)\b|\$0\.05|five cents", re.IGNORECASE)


def test_mcp_description_is_built_from_capabilities_and_never_prices_by_hand() -> None:
    capability = PublicSearchCapability.model_validate(_public_search_block())
    description = public_search_tool_description(capability)
    assert description.startswith("Fast search is free. Deep search is free. Fast: 10 per minute")
    assert "Public queries are retained for 30 days" in description
    assert "peer" not in description  # rendered copy, not raw field names
    assert not _FIVE_CENTS.search(description)
    assert not _FIVE_CENTS.search(public_search_tool_description(None))
    for tool in build_index_tools(lambda: _factory_client(lambda r: httpx.Response(500))):
        assert not _FIVE_CENTS.search(tool.description), tool.name
        assert not _FIVE_CENTS.search(str(tool.input_schema)), tool.name
    source = Path(mcp_index.__file__).read_text()
    assert not _FIVE_CENTS.search(source)
    assert not re.search(r"\b\d+ cents\b", source)


def test_mcp_index_search_is_anonymous_and_returns_terms() -> None:
    seen: list[httpx.Request] = []

    def search(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _delivered()

    tool = _tool(_routes(capabilities=_public_search_block(), search=search))
    out = tool.handler({"query": "RLVR verifier design", "mode": "fast"})

    assert "authorization" not in seen[0].headers
    assert out["search_id"] == SEARCH_ID
    assert out["customer_charge_cents"] == 0
    assert out["monitor_release_id"] == "release-77"
    assert out["citations"] == [
        {"contribution_id": C1, "revision_id": "rev-1"},
        {"contribution_id": C2, "revision_id": "rev-7"},
    ]
    assert out["response"].startswith("Verifiers reward")
    assert out["terms"]["price"] == "Fast search is free. Deep search is free."
    assert "search_token" not in str(out)
    assert TOKEN not in str(out)
    schema = tool.input_schema
    assert set(schema["properties"]) == {"query", "mode", "max_results", "idempotency_key"}
    assert schema["required"] == ["query"]


def test_mcp_index_search_deep_polls_to_done(no_sleep: list[float]) -> None:
    handler, polls = _deep_backend()
    tool = _tool(_routes(capabilities=_public_search_block(), search=handler))
    out = tool.handler({"query": "q", "mode": "deep"})
    assert out["mode"] == "deep"
    assert polls == [TOKEN, TOKEN, TOKEN]


def test_mcp_rate_limit_error_carries_retry_seconds() -> None:
    tool = _tool(
        _routes(
            capabilities=_public_search_block(),
            search=lambda r: _error(
                429, "index_public_rate_limited", scope="global_day", headers={"Retry-After": "90"}
            ),
        )
    )
    with pytest.raises(PublicSearchRateLimitedError) as info:
        tool.handler({"query": "q"})
    payload = _mcp_structured_core_error_payload(info.value)
    assert payload["error"] == "index_public_rate_limited"
    assert payload["retry_after_seconds"] == 90.0
    assert payload["http_status"] == 429
    assert "global_day" in payload["message"]


def test_mcp_budget_exhausted_error_carries_retry_seconds_and_scope() -> None:
    tool = _tool(
        _routes(
            capabilities=_public_search_block(),
            search=lambda r: _error(
                503,
                "index_public_budget_exhausted",
                scope="daily_cents",
                headers={"Retry-After": "3600"},
            ),
        )
    )
    with pytest.raises(PublicSearchBudgetExhaustedError) as info:
        tool.handler({"query": "q"})
    assert info.value.retry_after_s == 3600.0
    assert info.value.scope is PublicSearchBudgetScope.DAILY_CENTS
    payload = _mcp_structured_core_error_payload(info.value)
    assert payload["error"] == "index_public_budget_exhausted"
    assert payload["retry_after_seconds"] == 3600.0
    assert payload["http_status"] == 503
    assert "nothing was charged" in payload["message"]


def test_mcp_503_fails_closed_with_message() -> None:
    tool = _tool(
        _routes(
            capabilities=_public_search_block(),
            search=lambda r: _error(503, "monitor_unavailable"),
        )
    )
    with pytest.raises(PublicSearchMonitorUnavailableError) as info:
        tool.handler({"query": "q"})
    payload = _mcp_structured_core_error_payload(info.value)
    assert payload["error"] == "monitor_unavailable"
    assert payload["http_status"] == 503
    assert "no result was produced" in payload["message"]


def test_mcp_refuses_when_backend_flag_is_off() -> None:
    never = pytest.fail  # a search request must not be sent
    tool = _tool(
        _routes(
            capabilities=_public_search_block(enabled=False),
            search=lambda r: never("search sent while disabled"),
        )
    )
    with pytest.raises(PublicSearchDisabledError):
        tool.handler({"query": "q"})
    older = _tool(_routes(capabilities=None, search=lambda r: never("search sent")))
    with pytest.raises(PublicSearchDisabledError):
        older.handler({"query": "q"})
    fast_only = _tool(
        _routes(
            capabilities=_public_search_block(modes=["fast"]),
            search=lambda r: never("deep sent"),
        )
    )
    with pytest.raises(PublicSearchDisabledError, match="deep"):
        fast_only.handler({"query": "q", "mode": "deep"})


def test_mcp_private_search_is_authenticated_only() -> None:
    names = {
        tool.name
        for tool in build_index_tools(
            lambda: _factory_client(lambda r: httpx.Response(500)), include_lifecycle=False
        )
    }
    assert "index_search" in names
    assert "index_private_search" not in names
    names = {
        tool.name
        for tool in build_index_tools(
            lambda: _factory_client(lambda r: httpx.Response(500)), include_lifecycle=True
        )
    }
    assert "index_private_search" in names


def test_mcp_index_search_keyed_without_wallet_consent_is_refused_without_a_paid_call() -> None:
    keyed_client = SynthClient(api_key="sk-test", base_url=BASE)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/index/me/access-funding":
            return httpx.Response(200, json=_access_funding(wallet_enabled=False))
        return pytest.fail(f"unexpected {request.method} {request.url.path}")

    seen = _mount(keyed_client.index, handler)

    @contextmanager
    def factory():
        yield keyed_client.index

    try:
        tool = next(
            tool
            for tool in build_index_tools(factory, include_lifecycle=False)
            if tool.name == "index_search"
        )
        with pytest.raises(WalletConsentRequiredError) as info:
            tool.handler({"query": "q"})
    finally:
        keyed_client.close()
    assert [request.url.path for request in seen] == ["/api/v1/index/me/access-funding"]
    payload = _mcp_structured_core_error_payload(info.value)
    assert payload["error"] == "index_wallet_consent_required"
    assert payload["detail"]["charged_cents"] == 0
    assert payload["detail"]["mode"] == "fast"
    assert payload["detail"]["reason"] == "wallet_off"
    assert "utm_source=usesynth" in payload["detail"]["docs_url"]
    assert any("without an API key" in step for step in payload["detail"]["steps"])


def test_mcp_index_search_keyed_with_wallet_consent_runs_paid_search_with_a_cap() -> None:
    keyed_client = SynthClient(api_key="sk-test", base_url=BASE)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/index/me/access-funding":
            return httpx.Response(200, json=_access_funding())
        if request.url.path == "/api/v1/index/search":
            return httpx.Response(200, json=_paid_fast_result())
        return pytest.fail(f"unexpected {request.method} {request.url.path}")

    seen = _mount(keyed_client.index, handler)

    @contextmanager
    def factory():
        yield keyed_client.index

    try:
        tool = next(
            tool
            for tool in build_index_tools(factory, include_lifecycle=False)
            if tool.name == "index_search"
        )
        first = tool.handler({"query": "RLVR verifier design", "mode": "fast"})
        tool.handler({"query": "second call reuses the consent read", "mode": "fast"})
    finally:
        keyed_client.close()

    paths = [request.url.path for request in seen]
    # One cached consent read, then the paid route; never the anonymous public route.
    assert paths == [
        "/api/v1/index/me/access-funding",
        "/api/v1/index/search",
        "/api/v1/index/search",
    ]
    body = json.loads(seen[1].content)
    assert body["billing"] == {"allow_wallet": True, "max_charge_cents": 5}
    assert body["mode"] == "fast"
    assert seen[1].headers["authorization"] == "Bearer sk-test"
    assert seen[1].headers["idempotency-key"]
    assert first["paid"] is True
    assert first["customer_charge_cents"] == 5
    assert first["charge"] == {
        "amount_cents": 5,
        "wallet_debit_cents": 5,
        "funding_source": "wallet",
        "receipt_id": "receipt-1",
        "max_charge_cents": 5,
    }
    assert first["citations"] == [{"contribution_id": C1, "revision_id": "rev-1"}]
    assert first["search_id"] == "search-paid-1"


def test_mcp_index_search_anonymous_stays_free_public() -> None:
    seen: list[httpx.Request] = []

    def search(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/api/v1/index/public/search"
        return _delivered()

    tool = _tool(_routes(capabilities=_public_search_block(), search=search))
    out = tool.handler({"query": "q"})
    assert [request.url.path for request in seen] == ["/api/v1/index/public/search"]
    assert "authorization" not in seen[0].headers
    assert out["customer_charge_cents"] == 0
    assert "paid" not in out
    assert "charge" not in out


def test_mcp_index_search_description_says_keyed_calls_are_paid_when_consented() -> None:
    description = public_search_tool_description(None)
    assert "With an API key a call is a PAID search" in description
    assert "index_wallet_consent_required" in description


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"wallet_enabled": False}, "wallet_off"),
        ({"consent": None}, "no_consent"),
        ({"monthly_cap_cents": 0}, "cap_too_low"),
    ],
)
def test_wallet_grant_refuses_without_consent(overrides: dict[str, Any], reason: str) -> None:
    account = AccessFundingAccount.model_validate(_access_funding(**overrides))
    for mode in (SearchMode.FAST, SearchMode.DEEP):
        with pytest.raises(WalletConsentRequiredError) as info:
            wallet_search_grant(account, mode)
        assert info.value.consent_reason.value == reason


def test_wallet_grant_caps_deep_at_docs_ceiling_or_org_cap() -> None:
    wide = AccessFundingAccount.model_validate(_access_funding(monthly_cap_cents=2000))
    assert wallet_search_grant(wide, SearchMode.FAST).max_charge_cents == 5
    assert wallet_search_grant(wide, SearchMode.DEEP).max_charge_cents == 25
    narrow = AccessFundingAccount.model_validate(_access_funding(monthly_cap_cents=15))
    assert wallet_search_grant(narrow, SearchMode.DEEP).max_charge_cents == 15
