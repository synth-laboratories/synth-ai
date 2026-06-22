"""Tests for HTTP transport parsing of SMR limit errors."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from synth_ai.managed_research.errors import (
    SmrApiError,
    SmrFundingLaneInvariantError,
    SmrInsufficientCreditsError,
    SmrLimitExceededError,
    SmrManagedInferenceUnavailableError,
    SmrStructuredDenialError,
)
from synth_ai.managed_research.transport.http import SmrHttpTransport


def _transport_with_mock(
    handler: Callable[[httpx.Request], httpx.Response],
) -> SmrHttpTransport:
    transport = SmrHttpTransport(
        base_url="http://smr.test",
        headers={"Authorization": "Bearer sk_test"},
        timeout=5.0,
    )
    transport.client.close()
    transport.client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://smr.test",
        headers=transport.headers,
        timeout=5.0,
    )
    return transport


def test_request_json_raises_smr_limit_exceeded_with_detail() -> None:
    detail = {
        "error_code": "smr_limit_exceeded",
        "message": "Limit exceeded for agent_daytona (daily cap 5.0 USD, current usage 5.0 USD).",
        "resource_id": "agent_daytona",
        "window": "daily",
        "cap": 5.0,
        "current_usage": 5.0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"detail": detail},
        )

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrLimitExceededError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={"work_mode": "standard"})
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == detail
        assert "agent_daytona" in str(exc_info.value)
    finally:
        t.close()


def test_request_json_raises_funding_lane_invariant_error() -> None:
    detail = {
        "error_code": "smr_free_tier_routing_violation",
        "message": "Internal routing error.",
        "invariant": "ga_free_must_not_use_synth_codex_pool",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": detail})

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrFundingLaneInvariantError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={})
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == detail
    finally:
        t.close()


def test_request_json_raises_insufficient_credits_error() -> None:
    detail = {
        "error_code": "smr_insufficient_credits",
        "message": "Insufficient credits to start a new run.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"detail": detail})

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrInsufficientCreditsError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={})
        assert exc_info.value.status_code == 402
        assert exc_info.value.detail == detail
    finally:
        t.close()


def test_request_json_raises_managed_inference_unavailable() -> None:
    detail = {
        "error_code": "smr_managed_inference_unavailable",
        "message": "Managed inference unavailable.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": detail})

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrManagedInferenceUnavailableError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={})
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == detail
    finally:
        t.close()


def test_request_json_unknown_error_code_uses_structured_denial_error() -> None:
    detail = {"error_code": "smr_future_gate_code", "message": "Reserved."}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, json={"detail": detail})

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrStructuredDenialError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={})
        assert exc_info.value.status_code == 418
        assert exc_info.value.detail == detail
    finally:
        t.close()


def test_request_json_structured_detail_without_limit_code_uses_smr_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "detail": {
                    "error": "conflict",
                    "message": "project is paused",
                }
            },
        )

    t = _transport_with_mock(handler)
    try:
        with pytest.raises(SmrApiError) as exc_info:
            t.request_json("POST", "/smr/projects/p/trigger", json_body={})
        assert exc_info.value.status_code == 409
        assert "paused" in str(exc_info.value).lower()
    finally:
        t.close()
