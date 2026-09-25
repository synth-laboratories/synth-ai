from __future__ import annotations

import httpx
import pytest
from synth_ai.core.errors import TransientServiceError
from synth_ai.core.http.transport import raise_http_error


def test_nested_provider_cause_and_retry_directive_survive_http_503() -> None:
    response = httpx.Response(
        503,
        request=httpx.Request(
            "GET", "https://api.example.test/api/v1/index/searches/id?api_key=hidden"
        ),
        json={
            "message": "Index Deep failed",
            "detail": {
                "error_code": "index_deep_provider_request_rejected",
                "message": "provider rejected the request: credit budget exhausted",
                "retryable": False,
                "retry_after_seconds": 9,
            },
        },
    )

    with pytest.raises(TransientServiceError) as caught:
        raise_http_error(response, operation_id="index.deep")

    error = caught.value
    assert error.error_code == "index_deep_provider_request_rejected"
    assert error.retryable is False
    assert error.retry_after_seconds == 9
    assert "provider rejected the request: credit budget exhausted" in str(error)
    assert "api_key=hidden" not in str(error)


def test_plain_text_cause_after_former_excerpt_is_visible_but_secret_is_redacted() -> None:
    cause = "index_deep_provider_request_rejected: upstream budget exhausted"
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://api.example.test/api/v1/index/deep"),
        text="x" * 250 + " " + cause + " api_key=" + secret,
    )

    with pytest.raises(TransientServiceError) as caught:
        raise_http_error(response)

    error = caught.value
    assert cause in str(error)
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret in error.body_snippet  # structured raw detail remains available
