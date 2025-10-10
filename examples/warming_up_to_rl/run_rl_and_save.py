#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import requests
from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"config not found: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create clustered RL training job via backend RL endpoint"
    )
    p.add_argument(
        "--backend", default=os.getenv("BACKEND_BASE_URL", f"{PROD_BASE_URL_DEFAULT}/api")
    )
    p.add_argument("--config", required=True, help="Path to RL TOML config")
    p.add_argument(
        "--task-url",
        default=os.getenv("TASK_APP_URL", ""),
        help="Override task service URL (or set TASK_APP_URL)",
    )
    p.add_argument(
        "--idempotency",
        default=os.getenv("RL_IDEMPOTENCY_KEY", ""),
        help="Optional Idempotency-Key header value",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = _load_toml(cfg_path)

    services = cfg.get("services", {}) if isinstance(cfg.get("services"), dict) else {}

    # Resolve task app base URL for the job
    cli_task_url = (args.task_url or "").strip()
    env_task_url = (os.getenv("TASK_APP_URL") or "").strip()
    task_url = (
        cli_task_url
        or env_task_url
        or ((services.get("task_url") or "").strip() if isinstance(services, dict) else "")
    )
    if not task_url:
        print(
            "Missing task service URL. Provide --task-url or set TASK_APP_URL or services.task_url in TOML",
            file=sys.stderr,
        )
        sys.exit(2)

    # TOML-only model selection validation
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    has_source = bool((model_cfg.get("source") or "").strip())
    has_base = bool((model_cfg.get("base") or "").strip())
    if has_source == has_base:
        print(
            "Model selection must specify exactly one of [model].source or [model].base in TOML",
            file=sys.stderr,
        )
        sys.exit(2)

    # Build create-job payload. Send full TOML under data.config, plus endpoint_base_url.
    payload: dict[str, Any] = {
        "job_type": "rl",
        # Optional: compute pass-through
        "compute": cfg.get("compute", {}) if isinstance(cfg.get("compute"), dict) else {},
        "data": {
            "endpoint_base_url": task_url,
            "config": cfg,
        },
        "tags": {"source": "warming_up_to_rl"},
    }

    backend = str(args.backend).rstrip("/")
    url = f"{backend}/rl/jobs"
    api_key = (os.getenv("SYNTH_API_KEY") or os.getenv("SYNTH_KEY") or "").strip()
    if not api_key:
        print("Missing SYNTH_API_KEY in env", file=sys.stderr)
        sys.exit(2)

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    idem = (args.idempotency or "").strip()
    if idem:
        headers["Idempotency-Key"] = idem

    print(f"[INFO] POST {url}")
    try:
        preview = dict(payload)
        preview_data = dict(preview.get("data", {}))
        cfg_keys = list(cfg.keys())
        preview_data["config"] = {"keys": cfg_keys}
        preview["data"] = preview_data
        print(f"[INFO] Payload: {json.dumps(preview)[:500]}")
    except Exception:
        print("[INFO] Payload: <unavailable>")

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    ok = r.status_code in (200, 201)
    try:
        snippet = r.json()
    except Exception:
        snippet = r.text[:300]
    print(f"[INFO] Response: {r.status_code} {snippet}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
