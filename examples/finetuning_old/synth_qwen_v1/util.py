from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict

from synth_ai.config.base_url import get_backend_from_env

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover

    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


STATE_PATH = Path(__file__).parent / "state.json"


def _default_backend_url() -> str:
    base, _ = get_backend_from_env()
    base = base.rstrip("/")
    return base if base.endswith("/api") else f"{base}/api"


def load_env(mode: str | None = None) -> tuple[str, str]:
    """Resolve backend base_url and api_key.

    Precedence:
    - SYNTH_BACKEND_URL_OVERRIDE=local|dev|prod (preferred)
    - explicit mode arg (local|dev|prod)
    - default prod
    """
    load_dotenv()
    # Prefer global override if present
    override = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    if override in {"local", "dev", "prod"}:
        base, key = get_backend_from_env()
        base = base.rstrip("/")
        print(f"SYNTH backend: {base} (override={override})")
        # Print masked API key and source for clarity
        src = ""
        if override == "prod":
            if key and key == os.getenv("PROD_SYNTH_API_KEY", "").strip():
                src = "PROD_SYNTH_API_KEY"
            elif key and key == os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip():
                src = "TESTING_PROD_SYNTH_API_KEY"
            elif key and key == os.getenv("SYNTH_API_KEY", "").strip():
                src = "SYNTH_API_KEY"
        elif override == "dev":
            if key and key == os.getenv("DEV_SYNTH_API_KEY", "").strip():
                src = "DEV_SYNTH_API_KEY"
        else:  # local
            if key and key == os.getenv("DEV_SYNTH_API_KEY", "").strip():
                src = "DEV_SYNTH_API_KEY"
            elif key and key == os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "").strip():
                src = "TESTING_LOCAL_SYNTH_API_KEY"
        masked = ("*" * max(0, len(key) - 6)) + key[-6:] if key else "<empty>"
        print(f"SYNTH api key: {masked} (len={len(key)}, src={src or '<unknown>'})")
        return base, key

    # Fallback to explicit mode
    if mode is None:
        mode = os.getenv("SYNTH_MODE", "prod").strip().lower()
    if mode == "local":
        base_url = os.getenv("LOCAL_BACKEND_URL", "").strip()
        # Prefer DEV_SYNTH_API_KEY for local development; fall back to legacy var
        api_key = (
            os.getenv("DEV_SYNTH_API_KEY", "").strip()
            or os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "").strip()
        )
        if not base_url or not api_key:
            raise RuntimeError(
                "Missing LOCAL_BACKEND_URL or DEV_SYNTH_API_KEY/TESTING_LOCAL_SYNTH_API_KEY in environment/.env"
            )
    elif mode == "dev":
        base_url = os.getenv("DEV_BACKEND_URL", "").strip()
        api_key = os.getenv("DEV_SYNTH_API_KEY", "").strip()
        if not base_url or not api_key:
            raise RuntimeError("Missing DEV_BACKEND_URL or DEV_SYNTH_API_KEY in environment/.env")
    else:  # prod
        base_url = os.getenv("PROD_BACKEND_URL", "").strip() or _default_backend_url()
        api_key = (
            os.getenv("PROD_SYNTH_API_KEY", "").strip()
            or os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip()
            or os.getenv("SYNTH_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError(
                "Missing PROD_SYNTH_API_KEY/TESTING_PROD_SYNTH_API_KEY/SYNTH_API_KEY in environment/.env"
            )
    base_url = base_url.rstrip("/")
    print(f"SYNTH backend: {base_url} (mode={mode})")
    # Also print masked API key and source
    src = ""
    if mode == "prod":
        if api_key and api_key == os.getenv("PROD_SYNTH_API_KEY", "").strip():
            src = "PROD_SYNTH_API_KEY"
        elif api_key and api_key == os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip():
            src = "TESTING_PROD_SYNTH_API_KEY"
        elif api_key and api_key == os.getenv("SYNTH_API_KEY", "").strip():
            src = "SYNTH_API_KEY"
    elif mode == "dev":
        if api_key and api_key == os.getenv("DEV_SYNTH_API_KEY", "").strip():
            src = "DEV_SYNTH_API_KEY"
    else:
        if api_key and api_key == os.getenv("DEV_SYNTH_API_KEY", "").strip():
            src = "DEV_SYNTH_API_KEY"
        elif api_key and api_key == os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "").strip():
            src = "TESTING_LOCAL_SYNTH_API_KEY"
    masked = ("*" * max(0, len(api_key) - 6)) + api_key[-6:] if api_key else "<empty>"
    print(f"SYNTH api key: {masked} (len={len(api_key)}, src={src or '<unknown>'})")
    return base_url, api_key


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "dev", "local"], default=None, help="Backend mode")
    return p.parse_args()


def save_state(obj: Dict[str, Any]) -> None:
    prev: Dict[str, Any] = {}
    if STATE_PATH.exists():
        try:
            prev = json.loads(STATE_PATH.read_text())
        except Exception:
            prev = {}
    prev.update(obj)
    STATE_PATH.write_text(json.dumps(prev, indent=2))


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def validate_jsonl(path: str | Path) -> None:
    """Backwards-compatible wrapper that delegates to shared SDK validator.

    Prefer synth_ai.learning.validators.validate_training_jsonl to keep a single source
    of JSONL validation rules used across examples and tests.
    """
    from synth_ai.learning import validate_training_jsonl

    validate_training_jsonl(path)
