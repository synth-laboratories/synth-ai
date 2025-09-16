from __future__ import annotations

"""
One-off diagnostics: probe backend health and Hatchet wiring.

Checks:
- GET {BACKEND}/api/health
- GET {BACKEND}/api/orchestration/hatchet/health

Env resolution mirrors examples/rl/openai_in_task_app.py.
"""

import argparse
import json
import os
import sys
from typing import Any

import httpx


def _load_env_files() -> None:
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for p in (os.path.join(root, ".env"), os.path.join(os.path.dirname(__file__), ".env")):
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass


def _resolve_backend_url() -> str:
    try:
        from examples.common.backend import resolve_backend_url as _rb  # type: ignore
        return _rb()
    except Exception:
        raw = (os.getenv("PROD_BACKEND_URL") or os.getenv("BACKEND_URL") or "https://agent-learning.onrender.com/api").strip()
        raw = raw.rstrip("/")
        return raw if raw.endswith("/api") else f"{raw}/api"


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


def _dump(obj: Any, prefix: str) -> None:
    try:
        print(prefix, json.dumps(obj))
    except Exception:
        print(prefix, obj)


def main() -> int:
    _load_env_files()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-url", type=str, default=_resolve_backend_url())
    ap.add_argument("--api-key", type=str, default=(os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "").strip()))
    args = ap.parse_args()

    backend_url = args.backend_url
    api_key = args.api_key
    if not backend_url:
        print("Missing backend URL (set BACKEND_URL or PROD_BACKEND_URL)")
        return 2
    if not api_key:
        print("Missing SYNTH_API_KEY (or TESTING_LOCAL_SYNTH_API_KEY)")
        return 2

    headers = {"Authorization": f"Bearer {api_key}"}
    api = _api_base(backend_url)

    ok = True
    # /health
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{api}/health", headers=headers)
        print(f"backend /health: HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                _dump(r.json(), "health:")
            except Exception:
                pass
        else:
            ok = False
    except Exception as e:
        ok = False
        print(f"backend /health error: {type(e).__name__}: {e}")

    # /orchestration/hatchet/health
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(f"{api}/orchestration/hatchet/health", headers=headers)
        print(f"hatchet health: HTTP {r.status_code}")
        if r.status_code == 200:
            try:
                js = r.json()
            except Exception:
                js = {"raw": r.text[:400]}
            _dump(js, "hatchet:")
            key_present = bool(js.get("api_key_present")) if isinstance(js, dict) else False
            if not key_present:
                print("WARN: hatchet api_key_present=false — backend missing HATCHET_API_KEY/HATCHET_CLIENT_TOKEN")
        else:
            ok = False
    except Exception as e:
        ok = False
        print(f"hatchet health error: {type(e).__name__}: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


