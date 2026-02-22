from __future__ import annotations

from synth_ai.core.tunnels.synth_tunnel import _normalize_header_pairs, _strip_hop_by_hop


def test_strip_hop_by_hop_removes_authorization_and_host_headers() -> None:
    headers = [
        ("Authorization", "Bearer relay-worker-token"),
        ("Host", "relay.example"),
        ("Connection", "keep-alive"),
        ("X-Request-Id", "abc"),
    ]

    sanitized = _strip_hop_by_hop(headers)

    assert sanitized == [("X-Request-Id", "abc")]


def test_strip_hop_by_hop_preserves_duplicate_headers() -> None:
    headers = [
        ("X-Demo", "one"),
        ("X-Demo", "two"),
        ("Connection", "keep-alive"),
    ]

    sanitized = _strip_hop_by_hop(headers)

    assert sanitized == [("X-Demo", "one"), ("X-Demo", "two")]


def test_normalize_header_pairs_preserves_duplicate_frame_headers() -> None:
    raw_headers = [
        ["x-demo", "one"],
        ["x-demo", "two"],
        ["x-empty", ""],
        ["x-null", None],
        ["not-a-pair"],
    ]

    normalized = _normalize_header_pairs(raw_headers)

    assert normalized == [
        ("x-demo", "one"),
        ("x-demo", "two"),
        ("x-empty", ""),
    ]
