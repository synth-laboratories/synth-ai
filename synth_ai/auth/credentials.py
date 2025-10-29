import contextlib
import os
import time
import webbrowser

import requests
from requests import RequestException
from synth_ai.utils.env import resolve_env_var, write_env_var_to_dotenv, write_env_var_to_json


def fetch_credentials_from_web_browser_session(
    browser: bool = True,
    prod: bool = True
) -> None:
    synth_api_key = ''
    env_api_key = ''
    org_name = ''

    if browser:
        origin = "https://www.usesynth.ai" if prod else "http://localhost:3000"
        init_url = f"{origin}/api/sdk/handshake/init"
        token_url =f"{origin}/api/sdk/handshake/token"

        print(f"\n🌐 Connecting to {origin} to fetch your Synth credentials")

        # 1. Initialize browser handshake
        try:
            init_res = requests.post(init_url, timeout=10)
        except RequestException as exc:
            raise RuntimeError(f"Failed to reach handshake init endpoint: {exc}") from exc

        if init_res.status_code != 200:
            body = init_res.text.strip()
            raise RuntimeError(f"Handshake init failed ({init_res.status_code}): {body or 'no response body'}")

        try:
            init_data = init_res.json()
        except ValueError as exc:
            raise RuntimeError("Handshake init returned malformed JSON.") from exc

        device_code = str(init_data.get("device_code") or "").strip()
        verification_uri = str(init_data.get("verification_uri") or "").strip()
        if not device_code or not verification_uri:
            raise RuntimeError("Handshake init response missing device_code or verification_uri.")

        try:
            expires_in = int(init_data.get("expires_in") or 600)
        except (TypeError, ValueError):
            expires_in = 120
        try:
            interval = max(int(init_data.get("interval") or 3), 1)
        except (TypeError, ValueError):
            interval = 3

        # 2. Open browser to verification URL
        with contextlib.suppress(Exception):
            webbrowser.open(verification_uri)

        deadline = time.time() + expires_in
        handshake_data = None

        # 3. Poll handshake token endpoint
        while time.time() <= deadline:
            try:
                handshake_res = requests.post(
                    token_url,
                    json={"device_code": device_code},
                    timeout=10,
                )
            except RequestException:
                time.sleep(interval)
                continue

            if handshake_res.status_code == 200:
                try:
                    handshake_data = handshake_res.json()
                except ValueError as exc:
                    raise RuntimeError("Handshake token returned malformed JSON.") from exc
                break

            if handshake_res.status_code in (404, 410):
                raise RuntimeError("Handshake failed: device code expired or was revoked.")

            time.sleep(interval)

        if handshake_data is None:
            raise TimeoutError("Handshake timed out before credentials were returned.")

        # 4. Extract credentials from handshake payload
        org = handshake_data.get("org")
        if not isinstance(org, dict):
            org = {}
        org_name = str(org.get("name") or "your organization").strip()

        credentials = handshake_data.get("keys")
        if not isinstance(credentials, dict):
            credentials = {}

        synth_api_key = str(credentials.get("synth") or "").strip()
        env_api_key = str(credentials.get("rl_env") or "").strip()

        print(f"\n✅ Connected to {org_name}")

    # Load credentials to process environment and save credentials to .env and ~/synth-ai/config.json
    if synth_api_key:
        print("\nLoading SYNTH_API_KEY into process environment")
        os.environ["SYNTH_API_KEY"] = synth_api_key
    synth_api_key = resolve_env_var("SYNTH_API_KEY")
    if env_api_key:
        print("\nLoading ENVIRONMENT_API_KEY into process environment")
        os.environ["ENVIRONMENT_API_KEY"] = env_api_key
    env_api_key = resolve_env_var("ENVIRONMENT_API_KEY")

    if browser:
        print('')
        write_env_var_to_json("SYNTH_API_KEY", synth_api_key, "~/.synth-ai/config.json")
        write_env_var_to_dotenv("SYNTH_API_KEY", synth_api_key)
        write_env_var_to_json("ENVIRONMENT_API_KEY", env_api_key, "~/.synth-ai/config.json")
        write_env_var_to_dotenv("ENVIRONMENT_API_KEY", env_api_key)
