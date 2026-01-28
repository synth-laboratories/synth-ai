"""Device authentication flow for web browser OAuth-style auth."""

from __future__ import annotations

import contextlib
import time
import webbrowser
from dataclasses import dataclass

try:
    import synth_ai_py as _synth_ai_py
except Exception:  # pragma: no cover - optional rust bindings
    _synth_ai_py = None

import requests
from requests import RequestException

from synth_ai.core.utils.urls import FRONTEND_URL_BASE

INIT_URL = FRONTEND_URL_BASE + "/api/auth/device/init"
TOKEN_URL = FRONTEND_URL_BASE + "/api/auth/device/token"
POLL_INTERVAL = 3


@dataclass
class AuthSession:
    device_code: str
    verification_uri: str
    expires_at: float


def init_auth_session() -> AuthSession:
    if _synth_ai_py is not None:
        data = _synth_ai_py.init_device_auth(FRONTEND_URL_BASE)
        device_code = str(data.get("device_code") or "").strip()
        verification_uri = str(data.get("verification_uri") or "").strip()
        expires_at = float(data.get("expires_at") or 0.0)
        if not device_code or not verification_uri or not expires_at:
            raise RuntimeError(
                "Handshake init response missing device_code or verification_uri or expires_at."
            )
        return AuthSession(
            device_code=device_code,
            verification_uri=verification_uri,
            expires_at=expires_at,
        )

    try:
        res = requests.post(INIT_URL, timeout=10)
    except RequestException as exc:
        raise RuntimeError(f"Failed to reach handshake init endpoint: {exc}") from exc
    if res.status_code != 200:
        body = res.text.strip()
        raise RuntimeError(
            f"Handshake init failed ({res.status_code}): {body or 'no response body'}"
        )

    try:
        data = res.json()
    except ValueError as err:
        raise RuntimeError("Handshake init returned malformed JSON.") from err

    device_code = str(data.get("device_code") or "").strip()
    verification_uri = str(data.get("verification_uri") or "").strip()
    expires_in = int(data.get("expires_in") or 600)
    if not device_code or not verification_uri or not expires_in:
        raise RuntimeError(
            "Handshake init response missing device_code or verification_uri or expires_in."
        )

    return AuthSession(
        device_code=device_code,
        verification_uri=verification_uri,
        expires_at=time.time() + expires_in,
    )


def fetch_data(device_code: str) -> requests.Response | None:
    if _synth_ai_py is not None:
        return None
    try:
        return requests.post(
            TOKEN_URL,
            json={"device_code": device_code},
            timeout=10,
        )
    except RequestException:
        return None


def fetch_credentials_from_web_browser() -> dict:
    print(f"Fetching your credentials from {FRONTEND_URL_BASE}")

    auth_session = init_auth_session()

    with contextlib.suppress(Exception):
        webbrowser.open(auth_session.verification_uri)

    if _synth_ai_py is not None:
        data = _synth_ai_py.poll_device_token(
            FRONTEND_URL_BASE,
            auth_session.device_code,
            POLL_INTERVAL,
            max(1, int(auth_session.expires_at - time.time())),
        )
    else:
        data = None
    while time.time() <= auth_session.expires_at:
        if _synth_ai_py is not None:
            break
        res = fetch_data(auth_session.device_code)
        if not res:
            time.sleep(POLL_INTERVAL)
            continue
        if res.status_code == 200:
            try:
                data = res.json()
            except ValueError as err:
                raise RuntimeError("Handshake token returned malformed JSON.") from err
            break
        if res.status_code in (404, 410):
            raise RuntimeError("Handshake failed: device code expired or was revoked.")
        time.sleep(POLL_INTERVAL)
    if data is None:
        raise TimeoutError("Handshake timed out before credentials were returned.")

    print(f"Connected to {FRONTEND_URL_BASE}")
    synth_key = str(data.get("synth_api_key") or "").strip()
    legacy_keys = data.get("keys") if isinstance(data, dict) else {}
    if not isinstance(legacy_keys, dict):
        legacy_keys = {}

    env_key = str(
        data.get("environment_api_key")
        or legacy_keys.get("rl_env")
        or legacy_keys.get("environment_api_key")
        or ""
    ).strip()
    if not synth_key:
        synth_key = str(legacy_keys.get("synth") or "").strip()

    return {
        "SYNTH_API_KEY": synth_key,
        "ENVIRONMENT_API_KEY": env_key,
    }
