"""Device authentication flow for web browser OAuth-style auth."""

from __future__ import annotations

import contextlib
import time
import webbrowser
from dataclasses import dataclass

from synth_ai.core.utils.urls import FRONTEND_URL_BASE

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for device auth.") from exc

POLL_INTERVAL = 3


@dataclass
class AuthSession:
    device_code: str
    verification_uri: str
    expires_at: float


def init_auth_session() -> AuthSession:
    data = synth_ai_py.init_device_auth(FRONTEND_URL_BASE)
    return AuthSession(
        device_code=str(data.get("device_code") or ""),
        verification_uri=str(data.get("verification_uri") or ""),
        expires_at=float(data.get("expires_at") or 0),
    )


def fetch_credentials_from_web_browser() -> dict:
    print(f"Fetching your credentials from {FRONTEND_URL_BASE}")

    auth_session = init_auth_session()

    with contextlib.suppress(Exception):
        webbrowser.open(auth_session.verification_uri)

    timeout_seconds = max(0, int(auth_session.expires_at - time.time())) or None
    creds = synth_ai_py.poll_device_token(
        FRONTEND_URL_BASE,
        auth_session.device_code,
        POLL_INTERVAL,
        timeout_seconds,
    )
    print(f"Connected to {FRONTEND_URL_BASE}")
    return {
        "SYNTH_API_KEY": str(creds.get("SYNTH_API_KEY") or ""),
        "ENVIRONMENT_API_KEY": str(creds.get("ENVIRONMENT_API_KEY") or ""),
    }
