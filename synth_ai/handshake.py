from __future__ import annotations

import json
import os
import time
import webbrowser
from typing import Any, Dict, Tuple

import requests


class HandshakeError(Exception):
    pass


def _origin() -> str:
    # Prefer explicit env; fallback to localhost dashboard
    return (os.getenv("SYNTH_CANONICAL_ORIGIN", "") or "http://localhost:3000").rstrip("/")


def start_handshake_session(origin: str | None = None) -> Tuple[str, str, int, int]:
    base = (origin or _origin()).rstrip("/")
    url = f"{base}/api/sdk/handshake/init"
    r = requests.post(url, timeout=10)
    if r.status_code != 200:
        raise HandshakeError(f"init failed: {r.status_code} {r.text}")
    data = r.json()
    return (
        str(data.get("device_code")),
        str(data.get("verification_uri")),
        int(data.get("expires_in", 600)),
        int(data.get("interval", 3)),
    )


def poll_handshake_token(device_code: str, origin: str | None = None, *, timeout_s: int | None = None) -> Dict[str, Any]:
    base = (origin or _origin()).rstrip("/")
    url = f"{base}/api/sdk/handshake/token"
    deadline = time.time() + (timeout_s or 600)
    while True:
        if time.time() > deadline:
            raise HandshakeError("handshake timed out")
        try:
            r = requests.post(url, json={"device_code": device_code}, timeout=10)
        except Exception as e:
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r.json()
        elif r.status_code in (404, 410):
            raise HandshakeError(f"handshake failed: {r.status_code}")
        # 428 authorization_pending or others → wait and retry
        time.sleep(2)


def run_handshake(origin: str | None = None) -> Dict[str, Any]:
    device_code, verification_uri, expires_in, interval = start_handshake_session(origin)
    try:
        webbrowser.open(verification_uri)
    except Exception:
        pass
    return poll_handshake_token(device_code, origin, timeout_s=expires_in)

