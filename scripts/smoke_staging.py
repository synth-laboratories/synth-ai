"""Live smoke: GET /healthz and GET /api/v1/account on SYNTH_API_BASE_URL.

Reads SYNTH_API_KEY from the environment when set. Prints status codes only.
"""

from __future__ import annotations

import os
import sys

import httpx

PATHS = ("/healthz", "/api/v1/account", "/v1/account")


def main() -> int:
    base = os.environ.get("SYNTH_API_BASE_URL", "https://api.usesynth.ai").rstrip("/")
    key = os.environ.get("SYNTH_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    print(f"api_key_present={bool(key)}")
    with httpx.Client(timeout=20.0) as http:
        for path in PATHS:
            try:
                status = http.get(f"{base}{path}", headers=headers).status_code
            except httpx.HTTPError as error:
                print(f"GET {path} error={type(error).__name__}")
                continue
            print(f"GET {path} {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
