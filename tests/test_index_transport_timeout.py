"""Index's default transport must outlast valid monitored delivery."""

from __future__ import annotations

import asyncio

import pytest
from synth_ai.client import AsyncSynthClient, SynthClient
from synth_ai.sdk.index.timeouts import INDEX_TRANSPORT_TIMEOUT_SECONDS


@pytest.mark.parametrize("explicit_seconds", [None, 7.5])
def test_sync_index_timeout_preserves_explicit_override(explicit_seconds: float | None) -> None:
    client = SynthClient(
        api_key="sk-test",
        base_url="https://api.example.test",
        timeout_seconds=explicit_seconds,
    )
    try:
        assert client.timeout_seconds == (30.0 if explicit_seconds is None else explicit_seconds)
        assert client.index._transport.timeout_seconds == (
            INDEX_TRANSPORT_TIMEOUT_SECONDS if explicit_seconds is None else explicit_seconds
        )
    finally:
        client.close()


@pytest.mark.parametrize("explicit_seconds", [None, 7.5])
def test_async_index_timeout_preserves_explicit_override(explicit_seconds: float | None) -> None:
    client = AsyncSynthClient(
        api_key="sk-test",
        base_url="https://api.example.test",
        timeout_seconds=explicit_seconds,
    )
    try:
        assert client.timeout_seconds == (30.0 if explicit_seconds is None else explicit_seconds)
        assert client.index._transport.timeout_seconds == (
            INDEX_TRANSPORT_TIMEOUT_SECONDS if explicit_seconds is None else explicit_seconds
        )
    finally:
        asyncio.run(client.close())
