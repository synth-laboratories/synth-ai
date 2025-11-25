import json
import os

from synth_ai.cli.lib.env import mask_str, write_env_var_to_dotenv, write_env_var_to_json
from synth_ai.core.auth import AuthSession, fetch_data, init_auth_session


def setup_start() -> str:
    try:
        session: AuthSession = init_auth_session()
    except RuntimeError as err:
        return json.dumps(
            {
                "status": "error",
                "message": str(err),
                "code": "init_auth_session"
            },
            ensure_ascii=False
        )
    return json.dumps(
        {
            "status": "pending",
            "verification_uri": session.verification_uri,
            "device_code": session.device_code,
            "expires_at": session.expires_at
        },
        ensure_ascii=False
    )


def setup_fetch(device_code: str) -> str:
    stage = "poll"
    res = fetch_data(device_code)
    if res is None:
        return json.dumps(
            {
                "status": "error",
                "stage": stage,
                "message": "Network error while contacting handshake token endpoint"
            },
            ensure_ascii=False
        )
    if res.status_code == 200:
        try:
            payload = res.json()
        except Exception:
            return json.dumps(
                {
                    "status": "error",
                    "stage": stage,
                    "message": "Handshake token returned malformed JSON"
                },
                ensure_ascii=False
            )
        raw_keys = payload.get("keys") or {}
        if not isinstance(raw_keys, dict):
            return json.dumps(
                {
                    "status": "error",
                    "stage": stage,
                    "message": "Handshake token response missing keys dictionary",
                },
                ensure_ascii=False,
            )
        credentials = {
            "SYNTH_API_KEY": str(raw_keys.get("synth") or '').strip(),
            "ENVIRONMENT_API_KEY": str(raw_keys.get("rl_env") or '').strip()
        }
        for k, v in credentials.items():
            write_env_var_to_dotenv(k, v)
            write_env_var_to_json(k, v, "~/.synth-ai/config.json")
            os.environ[k] = v
        masked_credentials = {
            key: mask_str(value or "") for key, value in credentials.items()
        }
        return json.dumps(
            {
                "status": "success",
                "credentials": masked_credentials
            },
            ensure_ascii=False
        )
    if res.status_code in (404, 410):
        return json.dumps(
            {
                "status": "error",
                "stage": stage,
                "message": "Device code expired or was revoked. Restart setup",
                "code": res.status_code
            },
            ensure_ascii=False
        )
    return json.dumps(
        {
            "status": "pending",
            "stage": stage,
            "message": f"Authorization pending (HTTP {res.status_code})"
        },
        ensure_ascii=False
    )
