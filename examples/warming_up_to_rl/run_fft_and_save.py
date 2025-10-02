#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

import tomllib
import re
import requests


def mask(val: str) -> str:
    if not isinstance(val, str) or not val:
        return "<unset>"
    return f"{val[:6]}…{val[-4:]}" if len(val) >= 10 else "****"


def post_multipart(base: str, api_key: str, path: str, file_field: str, filepath: Path) -> Dict[str, Any]:
    """Upload a file, trying backend-specific endpoints with fallbacks.

    Priority:
    - {BASE}/learning/files (Modal Learning v2 style)
    - {BASE}/files (OpenAI-style)
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {file_field: (filepath.name, filepath.read_bytes(), "application/jsonl")}
    data = {"purpose": "fine-tune"}

    endpoints = [
        f"{base.rstrip('/')}/{path.lstrip('/')}",  # e.g., /learning/files
        f"{base.rstrip('/')}/files",               # OpenAI-style
    ]
    last_err: Dict[str, Any] | None = None
    for ep in endpoints:
        try:
            r = requests.post(ep, headers=headers, files=files, data=data, timeout=300)
            # Success fast-path
            try:
                js = r.json()
            except Exception:
                js = {"status": r.status_code, "text": r.text[:800]}

            if r.status_code < 400 and (js.get("id") or js.get("object") in ("file",)):
                return js

            # 404/405 -> try next endpoint
            if r.status_code in (404, 405):
                last_err = {"status": r.status_code, "body": (r.text or "")[:800], "endpoint": ep}
                continue

            # Other errors: return rich error
            return {
                "error": True,
                "status": r.status_code,
                "endpoint": ep,
                "body": (r.text or "")[:1200],
            }
        except requests.RequestException as e:
            last_err = {"error": True, "exception": str(e), "endpoint": ep}
            continue

    return last_err or {"error": True, "detail": "upload_failed_all_endpoints"}


def post_json(base: str, api_key: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
    try:
        return r.json()
    except Exception:
        return {"status": r.status_code, "text": r.text[:400]}


def get_json(base: str, api_key: str, path: str) -> Dict[str, Any]:
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=30)
    try:
        return r.json()
    except Exception:
        return {"status": r.status_code, "text": r.text[:400]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit FFT job and save resulting model id")
    parser.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL", "http://localhost:8000/api"))
    parser.add_argument("--toml", required=True, help="Path to FFT TOML config")
    parser.add_argument("--data", default="", help="Override dataset JSONL path")
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--env-file", default="", help="Optional path to .env file with SYNTH_API_KEY")
    args = parser.parse_args()

    config_path = Path(args.toml).expanduser().resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(2)
    with config_path.open("rb") as fh:
        cfg = tomllib.load(fh)

    job_cfg = cfg.get("job", {}) if isinstance(cfg.get("job"), dict) else {}
    compute_cfg = cfg.get("compute", {}) if isinstance(cfg.get("compute"), dict) else {}
    data_cfg_full = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    topo_cfg = (data_cfg_full or {}).get("topology", {}) if isinstance(data_cfg_full, dict) else {}
    validation_local_path = (data_cfg_full or {}).get("validation_path") if isinstance(data_cfg_full, dict) else None
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    hp_cfg = cfg.get("hyperparameters", {}) if isinstance(cfg.get("hyperparameters"), dict) else {}

    model = str(job_cfg.get("model") or os.getenv("SFT_MODEL") or "Qwen/Qwen3-4B")

    # Resolve dataset path
    data_path = args.data or job_cfg.get("data") or job_cfg.get("data_path")
    data_file: Path | None = None
    if isinstance(data_path, str) and data_path.strip():
        p = Path(data_path).expanduser()
        if not p.is_absolute():
            p = (config_path.parent / p).resolve()
        data_file = p
    if data_file is None:
        print("Missing dataset path in --data or [job].data", file=sys.stderr)
        sys.exit(2)
    if not data_file.exists():
        print(f"Dataset not found: {data_file}", file=sys.stderr)
        sys.exit(2)

    synth_key = (os.getenv("SYNTH_API_KEY") or "").strip()
    # Fallback: try to load from .env if not present in environment
    if not synth_key:
        candidate_env: Path | None = None
        if isinstance(args.env_file, str) and args.env_file.strip():
            candidate_env = Path(args.env_file).expanduser().resolve()
        else:
            # Prefer .env next to the TOML config
            candidate_env = (config_path.parent / ".env").resolve()
        if candidate_env and candidate_env.exists():
            try:
                env_text = candidate_env.read_text(encoding="utf-8", errors="ignore")
                # Match lines like: SYNTH_API_KEY=..., or export SYNTH_API_KEY=...
                key_val: str | None = None
                for line in env_text.splitlines():
                    m = re.match(r"^\s*(?:export\s+)?SYNTH_API_KEY\s*=\s*(.*)$", line)
                    if m:
                        raw = m.group(1).strip()
                        # Trim surrounding quotes if present
                        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                            raw = raw[1:-1]
                        key_val = raw.strip()
                        break
                if key_val:
                    synth_key = key_val
                    os.environ["SYNTH_API_KEY"] = synth_key
                    print(f"[INFO] Loaded SYNTH_API_KEY from {candidate_env}")
            except Exception as _e:
                # Ignore and fall through to error below
                pass
    if not synth_key:
        print("Missing SYNTH_API_KEY (set in env or provide --env-file pointing to .env)", file=sys.stderr)
        sys.exit(2)

    backend = args.backend.rstrip("/")
    print(f"[INFO] Using backend={backend} key_fp={mask(synth_key)} data={data_file}")
    if isinstance(validation_local_path, str) and validation_local_path.strip():
        print(f"[INFO] Using validation path={validation_local_path}")

    # 1) Upload training file
    print("[INFO] Uploading training file…")
    upf = post_multipart(backend, synth_key, "/learning/files", "file", data_file)
    try:
        print(f"[INFO] Upload response: {json.dumps(upf, indent=2)[:400]}")
    except Exception:
        print(f"[INFO] Upload response (raw): {str(upf)[:400]}")
    file_id = str((upf or {}).get("id") or "").strip()
    if not file_id:
        # Rich diagnostics
        err_status = (upf or {}).get("status")
        err_body = (upf or {}).get("body") or (upf or {}).get("text")
        err_ep = (upf or {}).get("endpoint")
        print(f"Upload failed (status={err_status} endpoint={err_ep}) body={str(err_body)[:200]}", file=sys.stderr)
        sys.exit(4)

    # Optionally upload validation file
    val_file_id: str | None = None
    if isinstance(validation_local_path, str) and validation_local_path.strip():
        vpath = Path(validation_local_path).expanduser()
        if not vpath.is_absolute():
            vpath = (config_path.parent / vpath).resolve()
        if not vpath.exists():
            print(f"[WARN] Validation file not found: {vpath} (skipping validation)")
        else:
            print("[INFO] Uploading validation file…")
            upv = post_multipart(backend, synth_key, "/learning/files", "file", vpath)
            try:
                print(f"[INFO] Validation upload response: {json.dumps(upv, indent=2)[:300]}")
            except Exception:
                print(f"[INFO] Validation upload response (raw): {str(upv)[:300]}")
            val_file_id = str((upv or {}).get("id") or "").strip() or None
            if not val_file_id:
                err_status = (upv or {}).get("status")
                err_body = (upv or {}).get("body") or (upv or {}).get("text")
                err_ep = (upv or {}).get("endpoint")
                print(f"[WARN] Validation upload failed (status={err_status} endpoint={err_ep}) body={str(err_body)[:180]} — continuing without validation")

    # 2) Build job payload
    hp_block: Dict[str, Any] = {
        "n_epochs": int(hp_cfg.get("n_epochs") or 1),
    }
    # Optional extras if present
    for k in (
        "batch_size",
        "global_batch",
        "per_device_batch",
        "gradient_accumulation_steps",
        "sequence_length",
        "learning_rate",
        "warmup_ratio",
        "train_kind",
    ):
        if k in hp_cfg:
            hp_block[k] = hp_cfg[k]

    parallel = hp_cfg.get("parallelism") if isinstance(hp_cfg.get("parallelism"), dict) else None
    if parallel:
        hp_block["parallelism"] = parallel

    compute_block: Dict[str, Any] = {}
    for k in ("gpu_type", "gpu_count", "nodes"):
        if k in compute_cfg:
            compute_block[k] = compute_cfg[k]

    effective = {
        "compute": compute_block,
        "data": {"topology": topo_cfg or {}},
        "training": {k: v for k, v in train_cfg.items() if k in ("mode", "use_qlora")},
    }
    # If TOML includes a [training.validation] block, forward relevant knobs into hyperparameters
    validation_cfg = train_cfg.get("validation") if isinstance(train_cfg.get("validation"), dict) else None
    if isinstance(validation_cfg, dict):
        # Enable evaluation and map keys as-is; backend trainer maps metric_for_best_model 'val.loss'→'eval_loss'
        hp_block.update({
            "evaluation_strategy": validation_cfg.get("evaluation_strategy", "steps"),
            "eval_steps": int(validation_cfg.get("eval_steps", 0) or 0),
            "save_best_model_at_end": bool(validation_cfg.get("save_best_model_at_end", True)),
            "metric_for_best_model": validation_cfg.get("metric_for_best_model", "val.loss"),
            "greater_is_better": bool(validation_cfg.get("greater_is_better", False)),
        })
        # Also surface validation enable flag into effective_config for visibility (optional)
        effective.setdefault("training", {})["validation"] = {"enabled": bool(validation_cfg.get("enabled", True))}

    body = {
        "model": model,
        "training_file_id": file_id,
        "training_type": "sft_offline",
        "hyperparameters": hp_block,
        "metadata": {"effective_config": effective},
    }
    if val_file_id:
        # Shared API expects top-level validation_file? Tests mention legacy; prefer placing into metadata.effective_config.data
        # Put into effective_config.data so downstream loader can read it; keep top-level off unless required.
        effective.setdefault("data", {})["validation_files"] = [val_file_id]

    # 3) Create and start job
    print("[INFO] Creating FFT job…")
    cj = post_json(backend, synth_key, "/learning/jobs", body)
    print(f"[INFO] Create response: {json.dumps(cj, indent=2)[:200]}")
    job_id = str(cj.get("job_id") or cj.get("id") or "").strip()
    if not job_id:
        print("Create job failed", file=sys.stderr)
        sys.exit(5)

    print(f"[INFO] Starting job {job_id}…")
    _ = post_json(backend, synth_key, f"/learning/jobs/{job_id}/start", {})

    # 4) Poll until terminal
    deadline = time.time() + max(30, int(job_cfg.get("poll_seconds") or args.poll_seconds))
    status = "queued"
    ft_model = None
    queued_since = time.time()
    while time.time() < deadline:
        info = get_json(backend, synth_key, f"/learning/jobs/{job_id}")
        status = (info.get("status") or "").lower()
        ft_model = info.get("fine_tuned_model")
        print(f"[INFO] poll status={status} ft_model={ft_model}")
        if status in ("succeeded", "failed", "canceled", "cancelled"):
            break
        # Warn if stuck queued for >10 minutes
        if status == "queued" and (time.time() - queued_since) > 600:
            print("[WARN] Job has remained queued for >10 minutes. Backend may be capacity constrained.")
            queued_since = time.time()
        time.sleep(5)

    # 5) Save model id
    out_file = Path(__file__).parent / "ft_model_id.txt"
    if ft_model:
        with out_file.open("a") as fh:
            fh.write(str(ft_model) + "\n")
        print(f"[INFO] Saved model id to {out_file}: {ft_model}")
        sys.exit(0 if status == "succeeded" else 1)
    else:
        print(f"[WARN] No fine_tuned_model found; final status={status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
