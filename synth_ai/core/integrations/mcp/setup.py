import json
import os

from synth_ai.core.auth import (
    AuthSession,
    extract_environment_api_key,
    extract_synth_api_key,
    fetch_data,
    init_auth_session,
)
from synth_ai.core.env import mask_str, write_env_var_to_json
from synth_ai.core.paths import SYNTH_USER_CONFIG_PATH


def setup_start() -> str:
    try:
        session: AuthSession = init_auth_session()
    except RuntimeError as err:
        return json.dumps(
            {"status": "error", "message": str(err), "code": "init_auth_session"},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "pending",
            "verification_uri": session.verification_uri,
            "device_code": session.device_code,
            "expires_at": session.expires_at,
        },
        ensure_ascii=False,
    )


def setup_fetch(device_code: str) -> str:
    stage = "poll"
    res = fetch_data(device_code)
    if res is None:
        return json.dumps(
            {
                "status": "error",
                "stage": stage,
                "message": "Network error while contacting handshake token endpoint",
            },
            ensure_ascii=False,
        )
    if res.status_code == 200:
        try:
            payload = res.json()
        except Exception:
            return json.dumps(
                {
                    "status": "error",
                    "stage": stage,
                    "message": "Handshake token returned malformed JSON",
                },
                ensure_ascii=False,
            )
        synth_key = extract_synth_api_key(payload)
        if not synth_key:
            return json.dumps(
                {
                    "status": "error",
                    "stage": stage,
                    "message": "Handshake token response missing Synth API key",
                },
                ensure_ascii=False,
            )
        credentials = {"SYNTH_API_KEY": synth_key}
        env_key = extract_environment_api_key(payload)
        if env_key:
            credentials["ENVIRONMENT_API_KEY"] = env_key
        for k, v in credentials.items():
            write_env_var_to_json(k, v, str(SYNTH_USER_CONFIG_PATH))
            os.environ[k] = v
        masked_credentials = {key: mask_str(value or "") for key, value in credentials.items()}
        return json.dumps(
            {"status": "success", "credentials": masked_credentials}, ensure_ascii=False
        )
    if res.status_code in (404, 410):
        return json.dumps(
            {
                "status": "error",
                "stage": stage,
                "message": "Device code expired or was revoked. Restart setup",
                "code": res.status_code,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "status": "pending",
            "stage": stage,
            "message": f"Authorization pending (HTTP {res.status_code})",
        },
        ensure_ascii=False,
    )
