"""Message refusals retain their domain type and transport outcome semantics."""

from __future__ import annotations

import httpx
import pytest

from synth_ai import errors


@pytest.mark.parametrize(
    ("code", "status", "error_name"),
    [
        ("message_too_large", 422, "MessageTooLargeError"),
        ("run_message_rate_limited", 429, "RunMessageRateLimitedError"),
        ("run_not_accepting_messages", 409, "RunNotAcceptingMessagesError"),
        ("run_message_queue_full", 409, "RunMessageQueueFullError"),
        ("messaging_unavailable", 503, "MessagingUnavailableError"),
    ],
)
def test_message_refusal_uses_named_domain_error(code, status, error_name):
    response = httpx.Response(
        status,
        headers={"Retry-After": "12"},
        request=httpx.Request("POST", "https://example.test/v1/runs/run_a/messages"),
        json={
            "type": "about:blank",
            "title": code,
            "status": status,
            "detail": code,
            "instance": "urn:request:message",
            "code": code,
            "service": "synth-backend",
            "stage": "run_messages",
            "outcome": "not_accepted",
            "retryable": False,
        },
    )
    with pytest.raises(getattr(errors, error_name)) as caught:
        errors.raise_for_response(response)
    assert caught.value.code == code
    assert caught.value.problem.retryable is False
    assert caught.value.retry_after_seconds == 12
