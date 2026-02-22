from __future__ import annotations

import httpx

from synth_ai.core.tunnels.synth_tunnel import _encode_response_headers


def test_encode_response_headers_preserves_duplicate_set_cookie() -> None:
    headers = httpx.Headers(
        [
            ("set-cookie", "a=1"),
            ("set-cookie", "b=2"),
            ("x-trace-id", "trace-1"),
        ]
    )

    encoded = _encode_response_headers(headers)

    assert encoded == [
        ["set-cookie", "a=1"],
        ["set-cookie", "b=2"],
        ["x-trace-id", "trace-1"],
    ]

