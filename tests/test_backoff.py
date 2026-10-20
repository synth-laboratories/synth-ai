"""Capped jitter and server retry hints have identical public-client behavior."""

import random
import warnings
from dataclasses import replace
from email.utils import formatdate

import httpx
import pytest
from synth_ai.core.http.backoff import (
    MAX_RETRY_AFTER_SECONDS,
    delay_seconds,
    retry_after_seconds,
)
from synth_ai.errors import RateLimitError, RequestTimeoutError, TransportError
from synth_ai.options import ClientOptions, _warn_retry_delay_alias
from tests.http_fixtures import Exchanges, invoke, page_response

pytestmark = pytest.mark.unit


def test_seeded_full_jitter_grows_then_caps_and_honors_server_floor():
    rng = random.Random(81)
    expected = random.Random(81)
    for attempt, ceiling in enumerate([0.2, 0.4, 0.8, 1.0, 1.0], 1):
        delay = delay_seconds(attempt, 0.2, 1.0, None, rng=rng)
        assert delay == expected.uniform(0, ceiling)
        assert 0 <= delay <= ceiling
    assert delay_seconds(1, 0.2, 1.0, 2, rng=rng) == 2
    assert delay_seconds(1, 0.2, 1.0, 90000, rng=rng) == MAX_RETRY_AFTER_SECONDS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("NaN", None),
        ("inf", None),
        ("-1", None),
        ("nonsense", None),
        ("x" * 129, None),
        ("1.5", 1.5),
        ("0", 0),
    ],
)
def test_retry_hint_validation(value, expected):
    assert retry_after_seconds(value, now=1_800_000_000) == expected


def test_http_date_retry_hint_and_past_date():
    now = 1_800_000_000
    assert retry_after_seconds(formatdate(now + 2, usegmt=True), now=now) == 2
    assert retry_after_seconds(formatdate(now - 2, usegmt=True), now=now) == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("timeout_seconds", None),
        ("timeout_seconds", 0),
        ("connect_timeout_seconds", True),
        ("index_result_timeout_seconds", -1),
        ("retry_base_delay_seconds", float("nan")),
        ("retry_base_delay_seconds", -1),
        ("retry_max_delay_seconds", 301),
        ("retry_max_delay_seconds", "10"),
        ("retry_delay_seconds", 61),
        ("maximum_response_bytes", 0),
        ("maximum_connections", 0),
        ("read_retry_count", 11),
    ],
)
def test_invalid_options_name_the_field(name, value):
    with pytest.raises(ValueError, match=name):
        ClientOptions(**{name: value})


def test_deprecated_delay_alias_warns_once_and_survives_dataclass_replace():
    _warn_retry_delay_alias.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = ClientOptions(retry_delay_seconds=0.3)
        assert replace(options, timeout_seconds=1).retry_base_delay_seconds == 0.3
        assert ClientOptions(retry_delay_seconds=0.4).retry_base_delay_seconds == 0.4
    assert len(caught) == 1 and caught[0].category is DeprecationWarning
    assert "retry_base_delay_seconds" in str(caught[0].message)
    with pytest.raises(ValueError, match="retry_base_delay_seconds"):
        ClientOptions(retry_delay_seconds=0.3, retry_base_delay_seconds=0.4)


def capture_sleeps(async_mode):
    delays = []

    async def asleep(delay):
        delays.append(delay)

    def configure(transport):
        transport._random = random.Random(81)
        transport._sleep = asleep if async_mode else delays.append

    return delays, configure


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_public_clients_use_the_same_capped_jitter(async_mode):
    delays, configure = capture_sleeps(async_mode)
    handler = Exchanges(httpx.Response(503), httpx.Response(503), page_response())
    result = await invoke(
        async_mode,
        handler,
        configure=configure,
        client_options={
            "retry_base_delay_seconds": 0.2,
            "retry_max_delay_seconds": 0.3,
        },
    )
    rng = random.Random(81)
    assert delays == [rng.uniform(0, 0.2), rng.uniform(0, 0.3)]
    assert result.metadata.client_timing.attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_public_clients_honor_http_date_floor(monkeypatch, async_mode):
    now = 1_800_000_000
    monkeypatch.setattr("synth_ai.core.http.backoff.time.time", lambda: now)
    delays, configure = capture_sleeps(async_mode)
    handler = Exchanges(
        httpx.Response(
            429,
            headers={
                "Retry-After": formatdate(now + 2, usegmt=True),
            },
        ),
        page_response(),
    )
    await invoke(async_mode, handler, configure=configure)
    assert delays == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("hint", [3600, 90000])
@pytest.mark.parametrize("status", [429, 503])
async def test_hint_beyond_deadline_survives_without_sleep(async_mode, hint, status):
    delays, configure = capture_sleeps(async_mode)
    handler = Exchanges(httpx.Response(status, headers={"Retry-After": str(hint)}))
    expected = RateLimitError if status == 429 else RequestTimeoutError
    with pytest.raises(expected) as caught:
        await invoke(async_mode, handler, configure=configure)
    assert caught.value.status == status and caught.value.retry_after_seconds == hint
    assert len(handler.requests) == 1 and delays == []


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_network_fault_retries_use_shared_jitter(async_mode):
    delays, configure = capture_sleeps(async_mode)
    handler = Exchanges()

    def broken(request):
        if request.url.path.endswith("/capabilities"):
            return handler(request)
        handler.requests.append(request)
        raise httpx.ReadError("fixture connection reset", request=request)

    with pytest.raises(TransportError):
        await invoke(
            async_mode,
            broken,
            configure=configure,
            client_options={
                "retry_base_delay_seconds": 0.2,
                "retry_max_delay_seconds": 0.3,
            },
        )
    rng = random.Random(81)
    assert len(handler.requests) == 3
    assert delays == [rng.uniform(0, 0.2), rng.uniform(0, 0.3)]
