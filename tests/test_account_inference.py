"""Inference account observations remain content-free and filterable."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from synth_ai.sdk.account.client import _window
from synth_ai.types.account_cloud import AccountUsageEvent


def test_inference_usage_filter_and_content_free_event():
    now = datetime.now(UTC)
    assert _window(now - timedelta(days=1), now, "inference", None)["product"] == "inference"
    event = {
        "sequence": "1",
        "product": "inference",
        "source": "inference.execution",
        "source_id": "execution-1",
        "source_revision": "a" * 32,
        "operation_id": "request-1",
        "parent_id": None,
        "actor_id": None,
        "occurred_at": now,
        "recorded_at": now,
        "facts": {"request_id": "request-1", "charge_micros": None},
    }
    assert AccountUsageEvent.model_validate(event).facts["charge_micros"] is None
    with pytest.raises(ValidationError):
        AccountUsageEvent.model_validate({**event, "facts": {"prompt": "private"}})
    with pytest.raises(ValidationError):
        AccountUsageEvent.model_validate({**event, "source": "index.usage"})
