"""Test tooling only: sign public Search starts with the internal-test credential.

``SYNTH_INDEX_INTERNAL_TEST_SECRET`` set -> every ``POST .../index/public/search``
carries ``X-Synth-Internal-Test: <ts>.<sig>`` (hex HMAC-SHA256 over
``v1:internal-test:{ts}:{METHOD}:{path}``), so live CI/agent runs use the backend's
internal-test quota bucket instead of the runner's IP bucket. Deliberately not
part of the public SDK API; it attaches a standard httpx request event hook.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Callable
from typing import Any

HEADER = "X-Synth-Internal-Test"
ENV = "SYNTH_INDEX_INTERNAL_TEST_SECRET"
_SIGNED_PATH_SUFFIX = "/index/public/search"


def internal_test_header(secret: str, method: str, path: str, now: int | None = None) -> str:
    ts = int(time.time()) if now is None else now
    message = f"v1:internal-test:{ts}:{method.upper()}:{path}".encode()
    return f"{ts}." + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def signing_hook(secret: str) -> Callable[[Any], None]:
    def sign(request: Any) -> None:
        if request.method == "POST" and request.url.path.endswith(_SIGNED_PATH_SUFFIX):
            request.headers[HEADER] = internal_test_header(secret, request.method, request.url.path)

    return sign


def attach_from_env(sdk_client: Any, environ: dict[str, str] | None = None) -> bool:
    """Attach to a PublicIndexClient's httpx client; returns whether signing is on."""
    secret = (os.environ if environ is None else environ).get(ENV, "").strip()
    if not secret:
        return False
    sdk_client._transport.client.event_hooks["request"].append(signing_hook(secret))
    return True
