import contextlib
import time
import webbrowser
from dataclasses import dataclass

import requests
from requests import RequestException

from synth_ai.core.urls import FRONTEND_URL_BASE

INIT_URL = FRONTEND_URL_BASE + "/api/sdk/handshake/init"
TOKEN_URL = FRONTEND_URL_BASE + "/api/sdk/handshake/token"
POLL_INTERVAL = 3


@dataclass
class AuthSession:
    device_code: str
    verification_uri: str
    expires_at: float


def init_auth_session() -> AuthSession:    
    try:
        res = requests.post(INIT_URL, timeout=10)
    except RequestException as exc:
        raise RuntimeError(f"Failed to reach handshake init endpoint: {exc}") from exc
    if res.status_code != 200:
        body = res.text.strip()
        raise RuntimeError(f"Handshake init failed ({res.status_code}): {body or 'no response body'}")

    try:
        data = res.json()
    except ValueError as err:
        raise RuntimeError("Handshake init returned malformed JSON.") from err

    device_code = str(data.get("device_code") or "").strip()
    verification_uri = str(data.get("verification_uri") or "").strip()
    expires_in = int(data.get("expires_in") or 600)
    if not device_code or not verification_uri or not expires_in:
        raise RuntimeError("Handshake init response missing device_code or verification_uri or expires_in.")

    return AuthSession(
        device_code=device_code,
        verification_uri=verification_uri,
        expires_at=time.time() + expires_in
    )


def fetch_data(device_code: str) -> requests.Response | None:
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

    data = None
    while time.time() <= auth_session.expires_at:
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
    credentials = data.get("keys")
    if not isinstance(credentials, dict):
        credentials = {}

    return {
        "SYNTH_API_KEY": str(credentials.get("synth") or "").strip(),
        "ENVIRONMENT_API_KEY": str(credentials.get("rl_env") or "").strip()
    }
