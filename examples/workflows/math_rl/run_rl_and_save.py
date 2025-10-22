#!/usr/bin/env python3
"""Submit math RL training jobs via Synth backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import requests


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"config not found: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create math RL job via backend RL endpoint")
    parser.add_argument(
        "--backend", default=os.getenv("BACKEND_BASE_URL", "http://localhost:8000/api")
    )
    parser.add_argument("--config", required=True, help="Path to RL TOML config")
    parser.add_argument(
        "--task-url", default=os.getenv("TASK_APP_URL", ""), help="Override task service URL"
    )
    parser.add_argument(
        "--idempotency",
        default=os.getenv("RL_IDEMPOTENCY_KEY", ""),
        help="Optional Idempotency-Key header",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    cfg = _load_toml(cfg_path)

    services = cfg.get("services") if isinstance(cfg.get("services"), dict) else {}

    task_url = (
        (args.task_url or "").strip()
        or (os.getenv("TASK_APP_URL") or "").strip()
        or (services.get("task_url") or "").strip()
    )
    if not task_url:
        print(
            "Missing task service URL. Provide --task-url or set TASK_APP_URL or services.task_url in TOML",
            file=sys.stderr,
        )
        sys.exit(2)

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    has_source = bool((model_cfg.get("source") or "").strip())
    has_base = bool((model_cfg.get("base") or "").strip())
    if has_source == has_base:
        print(
            "Model section must specify exactly one of [model].source or [model].base",
            file=sys.stderr,
        )
        sys.exit(2)

    payload: dict[str, Any] = {
        "job_type": "rl",
        "compute": cfg.get("compute", {}),
        "data": {
            "endpoint_base_url": task_url,
            "config": cfg,
        },
        "tags": cfg.get("tags", {}),
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
        preview = {"job_type": payload["job_type"], "data": {"config_keys": list(cfg.keys())}}
        print(f"[INFO] Payload preview: {json.dumps(preview)}")
    except Exception:
        pass

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    ok = resp.status_code in (200, 201)
    try:
        snippet = resp.json()
    except Exception:
        snippet = resp.text[:300]
    print(f"[INFO] Response: {resp.status_code} {snippet}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
