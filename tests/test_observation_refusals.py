"""HTTP observation refusals retain retry hints without replaying mutations."""

import json

import httpx
import pytest
from synth_ai.core.http.response import decode_response
from synth_ai.errors import OutcomeUncertainError, RateLimitError, RequestTimeoutError
from synth_ai.types.base import ClientTiming, ResponseModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "status,mutation,outcome,expected",
    [
        (408, False, "unknown", RequestTimeoutError),
        (408, True, "not_accepted", RequestTimeoutError),
        (408, True, "unknown", OutcomeUncertainError),
        (429, False, "not_accepted", RateLimitError),
    ],
)
def test_observation_deadline_and_backpressure_classes(status, mutation, outcome, expected):
    body = json.dumps(
        {
            "error": {
                "code": "request_deadline_exceeded" if status == 408 else "db_backpressure",
                "request_id": "request-1",
                "outcome": outcome,
            }
        }
    ).encode()
    response = httpx.Response(
        status,
        headers={
            "X-Synth-Contract": "swarms.v1",
            "Content-Type": "application/json",
            "Retry-After": "3",
        },
        content=body,
    )
    with pytest.raises(expected) as raised:
        decode_response(
            response,
            body,
            ResponseModel,
            ClientTiming(total_ms=1.0, first_byte_ms=1.0, attempts=1),
            None,
            mutation,
            "swarms.v1",
        )
    assert raised.value.status == status
    assert raised.value.retry_after_seconds == 3
    assert raised.value.request_id == "request-1"
