from __future__ import annotations

import base64
import json

from synth_ai.core.errors import HTTPError, PaymentRequiredError


def test_payment_required_error_extracts_challenge_from_detail() -> None:
    error = HTTPError(
        status=402,
        url="/api/v1/graphs/completions",
        message="payment_required",
        detail={
            "x402": {
                "challenge": {"claims": {"challenge_id": "abc"}},
                "payment_required_header": "header-value",
            }
        },
    )
    converted = PaymentRequiredError.from_http_error(error)
    assert converted.challenge == {"claims": {"challenge_id": "abc"}}
    assert converted.payment_required_header == "header-value"


def test_payment_required_error_builds_payment_response_header() -> None:
    error = PaymentRequiredError(
        status=402,
        url="/api/v1/graphs/completions",
        message="payment_required",
        challenge={"claims": {"challenge_id": "abc"}},
    )
    encoded = error.build_payment_response_header(
        payment_reference="mock://payment/proof-1"
    )
    padded = encoded + "=" * (-len(encoded) % 4)
    decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    payload = json.loads(decoded)
    assert payload["payment_reference"] == "mock://payment/proof-1"
    assert payload["challenge"]["claims"]["challenge_id"] == "abc"

