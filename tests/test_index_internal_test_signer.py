"""The internal-test signer (test tooling) signs only public Search starts."""

from __future__ import annotations

import hashlib
import hmac

import httpx

from tests.support.index_internal_test_signer import (
    HEADER,
    attach_from_env,
    internal_test_header,
    signing_hook,
)

SECRET = "s" * 40
PATH = "/api/v1/index/public/search"


def test_header_matches_backend_construction() -> None:
    value = internal_test_header(SECRET, "post", PATH, now=1_000)
    ts, sig = value.split(".")
    expected = hmac.new(
        SECRET.encode(), f"v1:internal-test:1000:POST:{PATH}".encode(), hashlib.sha256
    ).hexdigest()
    assert ts == "1000" and sig == expected


def test_hook_signs_only_public_search_starts() -> None:
    hook = signing_hook(SECRET)
    start = httpx.Request("POST", "https://api.example" + PATH)
    poll = httpx.Request("GET", "https://api.example/api/v1/index/public/searches/x")
    hook(start)
    hook(poll)
    assert HEADER.lower() in {k.lower() for k in start.headers}
    assert HEADER.lower() not in {k.lower() for k in poll.headers}


def test_unset_secret_attaches_nothing() -> None:
    assert attach_from_env(object(), environ={}) is False
