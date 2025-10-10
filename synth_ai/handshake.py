from __future__ import annotations

import contextlib
import os
import time
import webbrowser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


class HandshakeError(Exception):
    pass


_TRUTHY = {"1", "true", "yes", "on"}


def _origin() -> str:
    """Resolve the dashboard origin for the browser handshake.

    Priority order:
      1. Explicit ``SYNTH_CANONICAL_ORIGIN`` override.
      2. Development flag ``SYNTH_CANONICAL_DEV`` (case-insensitive truthy) → localhost.
      3. Production dashboard at ``https://www.usesynth.ai/dashboard``.
    """

    override = (os.getenv("SYNTH_CANONICAL_ORIGIN") or "").strip()
    if override:
        return override.rstrip("/")

    dev_flag = (os.getenv("SYNTH_CANONICAL_DEV") or "").strip().lower()
    if dev_flag in _TRUTHY:
        print("USING DEV ORIGIN")
        return "http://localhost:3000"

    return "https://www.usesynth.ai/dashboard"


def _split_origin(origin: str) -> tuple[str, str]:
    parsed = urlsplit(origin)
    bare = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    path = parsed.path.rstrip("/")
    return bare, path


def _ensure_verification_uri(data: dict[str, Any], base_with_path: str) -> None:
    uri = data.get("verification_uri")
    if not isinstance(uri, str) or not uri:
        return
    if uri.startswith("http://") or uri.startswith("https://"):
        return
    data["verification_uri"] = urljoin(base_with_path.rstrip("/") + "/", uri.lstrip("/"))


def start_handshake_session(origin: str | None = None) -> tuple[str, str, int, int]:
    base = (origin or _origin()).rstrip("/")
    api_origin, _ = _split_origin(base)
    url = urljoin(api_origin.rstrip("/") + "/", "api/sdk/handshake/init")
    r = requests.post(url, timeout=10)
    if r.status_code != 200:
        raise HandshakeError(f"init failed: {r.status_code} {r.text}")
    try:
        data = r.json()
    except ValueError as exc:  # pragma: no cover - network dependent
        raise HandshakeError(f"init returned malformed JSON: {exc}") from exc
    _ensure_verification_uri(data, base)
    return (
        str(data.get("device_code")),
        str(data.get("verification_uri")),
        int(data.get("expires_in", 600)),
        int(data.get("interval", 3)),
    )


def poll_handshake_token(
    device_code: str, origin: str | None = None, *, timeout_s: int | None = None
) -> dict[str, Any]:
    base = (origin or _origin()).rstrip("/")
    api_origin, _ = _split_origin(base)
    url = urljoin(api_origin.rstrip("/") + "/", "api/sdk/handshake/token")
    deadline = time.time() + (timeout_s or 600)
    while True:
        if time.time() > deadline:
            raise HandshakeError("handshake timed out")
        try:
            r = requests.post(url, json={"device_code": device_code}, timeout=10)
        except Exception:
            time.sleep(2)
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError as exc:  # pragma: no cover - network dependent
                raise HandshakeError(f"token returned malformed JSON: {exc}") from exc
            _ensure_verification_uri(data, base)
            return data
        elif r.status_code in (404, 410):
            raise HandshakeError(f"handshake failed: {r.status_code}")
        # 428 authorization_pending or others → wait and retry
        time.sleep(2)


def run_handshake(origin: str | None = None) -> dict[str, Any]:
    device_code, verification_uri, expires_in, interval = start_handshake_session(origin)
    with contextlib.suppress(Exception):
        webbrowser.open(verification_uri)
    return poll_handshake_token(device_code, origin, timeout_s=expires_in)
