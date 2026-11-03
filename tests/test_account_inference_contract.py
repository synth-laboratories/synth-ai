"""Inference observations preserve account identity and content-free metadata."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from synth_ai.sdk.account.client import _window
from synth_ai.types.account import AccountPolicyLimit
from synth_ai.types.account_cloud import AccountUsageEvent


def test_inference_usage_window_and_token_limit():
    now = datetime.now(UTC)
    assert _window(now, now + timedelta(days=1), "inference", None)["product"] == "inference"
    limit = AccountPolicyLimit.model_validate(
        {
            "limit_id": "inference-tokens",
            "product": "inference",
            "value": 1000,
            "default_value": 1000,
            "source": "policy",
            "revision": None,
            "expires_at": None,
            "dimension": "tokens",
            "window": "minute",
            "observed_at": now,
            "state": "ready",
            "code": None,
            "used": 0,
            "remaining": 1000,
            "resets_at": None,
            "burst_capacity": None,
            "available_requests": None,
            "next_request_at": None,
        }
    )
    assert limit.value == 1000


def test_inference_event_keeps_product_and_content_boundary():
    now = datetime.now(UTC)
    event = {
        "sequence": "1",
        "product": "inference",
        "source": "inference.execution",
        "source_id": "execution-1",
        "source_revision": "a" * 32,
        "operation_id": "operation-1",
        "parent_id": None,
        "actor_id": None,
        "occurred_at": now,
        "recorded_at": now,
        "facts": {"request_id": "request-1", "charge_micros": "25"},
    }
    assert AccountUsageEvent.model_validate(event).product == "inference"
    with pytest.raises(ValidationError):
        AccountUsageEvent.model_validate({**event, "product": "index"})
    with pytest.raises(ValidationError):
        AccountUsageEvent.model_validate({**event, "facts": {"prompt": "private"}})
