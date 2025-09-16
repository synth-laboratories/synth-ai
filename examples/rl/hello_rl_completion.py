from __future__ import annotations

"""
examples/rl/hello_rl_completion.py

Hello-world chat completion using the backend's OpenAI-compatible endpoint
with an RL'ed model ID. Resolves backend URL and API key like run_rl_job.py.

Usage (env-resolved backend):

  uv run python examples/rl/hello_rl_completion.py \
    --model "rl:Qwen-Qwen3-0.6B:job_XXXXXXXX:checkpoint-epoch-1"

Or override explicitly:

  uv run python examples/rl/hello_rl_completion.py \
    --backend-url "$DEV_BACKEND_URL" \
    --api-key "$SYNTH_API_KEY" \
    --model "rl:Qwen-Qwen3-0.6B:job_...:checkpoint-epoch-1"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root on sys.path for namespace imports under examples/
ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.finetuning.synth_qwen_v1.util import load_env  # type: ignore
from synth_ai.inference import InferenceClient  # type: ignore
import httpx


def _load_rl_env() -> None:
    """Load env from project .env, examples/rl/.env, and monorepo backend/.env.dev (best-effort)."""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for p in (os.path.join(root, ".env"), os.path.join(os.path.dirname(__file__), ".env")):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        monorepo_env = os.path.join(root, "..", "monorepo", "backend", ".env.dev")
        if os.path.exists(monorepo_env):
            with open(monorepo_env, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    _load_rl_env()
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "dev", "local"], default="local", help="Backend mode override (default: local)")
    p.add_argument("--backend-url", type=str, default="", help="Override backend base URL")
    p.add_argument("--api-key", type=str, default="", help="Override API key")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID to query (e.g., rl:Qwen-Qwen3-0.6B:job_xxx:checkpoint-epoch-1)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("HTTP_TIMEOUT", "60") or 60.0),
        help="HTTP timeout seconds",
    )
    p.add_argument(
        "--weights-path",
        type=str,
        default=os.getenv("RL_WEIGHTS_PATH", "").strip(),
        help="Explicit Wasabi key for RL checkpoint (models/.../checkpoint-*.tar.gz)",
    )
    return p.parse_args()


async def _amain() -> int:
    args = _parse_args()
    # Prefer explicit overrides
    if args.backend_url and args.api_key:
        base_url, api_key = args.backend_url.rstrip("/"), args.api_key
    else:
        # Respect SYNTH_BACKEND_URL_OVERRIDE like openai_in_task_app: use SYNTH_API_KEY primarily
        override = (os.getenv("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
        if override in {"local", "dev", "prod"}:
            if override == "local":
                base_url = (os.getenv("LOCAL_BACKEND_URL", "http://localhost:8000") or "").rstrip("/")
                api_key = os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("TESTING_LOCAL_SYNTH_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("SYNTH_API_KEY or TESTING_LOCAL_SYNTH_API_KEY is required for local override")
            elif override == "dev":
                base_url = (os.getenv("DEV_BACKEND_URL", "") or "").rstrip("/")
                api_key = os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("DEV_SYNTH_API_KEY", "").strip()
                if not base_url or not api_key:
                    raise RuntimeError("DEV_BACKEND_URL and SYNTH_API_KEY/DEV_SYNTH_API_KEY required for dev override")
            else:  # prod
                base_url = (os.getenv("PROD_BACKEND_URL", "https://agent-learning.onrender.com") or "").rstrip("/")
                api_key = os.getenv("SYNTH_API_KEY", "").strip() or os.getenv("TESTING_PROD_SYNTH_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("SYNTH_API_KEY or TESTING_PROD_SYNTH_API_KEY required for prod override")
        else:
            base_url, api_key = load_env(args.mode)
    # Ensure '/api' suffix for compatibility with local/dev backends exposing /api/v1
    if not base_url.endswith("/api"):
        base_url = f"{base_url.rstrip('/')}/api"
    # Debug: show where we are sending the request
    try:
        key_prefix = api_key[:8]
    except Exception:
        key_prefix = "<na>"
    print(f"Backend: {base_url}")
    print(f"API key prefix: {key_prefix}")
    client = InferenceClient(base_url=base_url, api_key=api_key, timeout=float(args.timeout))

    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please say: Hello, world!"},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }

    # If model is an RL alias, pre-resolve artifacts and ensure model on Modal via backend routes
    async def _resolve_qwen_base(model_id: str) -> str | None:
        try:
            if not model_id.lower().startswith("rl:"):
                return None
            parts = model_id.split(":")
            if len(parts) < 2:
                return None
            base_slug = parts[1]  # e.g., Qwen-Qwen3-0.6B
            if not (base_slug.startswith("Qwen-Qwen3-") and base_slug.endswith("B")):
                return None
            first_dash = base_slug.find("-")
            if first_dash <= 0:
                return None
            return f"{base_slug[:first_dash]}/{base_slug[first_dash+1:]}"
        except Exception:
            return None

    async def _extract_job_token(model_id: str) -> tuple[str | None, str | None]:
        try:
            parts = model_id.split(":", 2)
            if len(parts) < 3:
                return None, None
            tail = parts[2]
            segs = tail.split(":", 1)
            job_id = segs[0]
            ckpt = segs[1] if len(segs) > 1 else None
            return job_id, ckpt
        except Exception:
            return None, None

    async def _ensure_rl_model():
        if not str(args.model).lower().startswith("rl:"):
            return
        base_model_hint = await _resolve_qwen_base(args.model)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # 1) Try to find artifact for this job
        job_id, ckpt = await _extract_job_token(args.model)
        weights_path: str | None = None
        tokenizer_path: str | None = None
        # 0) Honor explicit override if provided (avoids any DB/event lookups)
        if args.weights_path:
            weights_path = args.weights_path
            tokenizer_path = args.weights_path
            # proceed to promote/ensure
        # 1a) Query RL job artifacts directly (authoritative, API-backed)
        if job_id:
            try:
                async with httpx.AsyncClient(timeout=30.0) as hc:
                    ra = await hc.get(f"{base_url}/orchestration/jobs/{job_id}/artifacts", headers=headers)
                if ra.status_code == 200:
                    js = ra.json()
                    arts = js.get("artifacts") if isinstance(js, dict) else js
                    if isinstance(arts, list):
                        def _pick(a: dict) -> bool:
                            try:
                                label = (a.get("label") or a.get("role") or "").lower()
                                ctype = (a.get("content_type") or "").lower()
                                key_like = any(str(a.get(k) or "") for k in ("s3_uri","uri","url","storage_path","key"))
                                return ("model" in label or label in ("final_model","artifact","model")) or ("tar" in ctype or "gzip" in ctype) or key_like
                            except Exception:
                                return False
                        cand = next((a for a in arts if isinstance(a, dict) and _pick(a)), None)
                        if isinstance(cand, dict):
                            s3_uri = cand.get("s3_uri") or cand.get("uri") or cand.get("url")
                            bucket = cand.get("bucket")
                            key = cand.get("key") or cand.get("storage_path")
                            # Normalize s3 uri → key
                            if isinstance(s3_uri, str) and s3_uri.startswith("s3://"):
                                try:
                                    _tmp = s3_uri[len("s3://"):]
                                    parts = _tmp.split("/", 1)
                                    bucket = parts[0]
                                    key = parts[1] if len(parts) > 1 else None
                                except Exception:
                                    pass
                            if isinstance(key, str) and key:
                                weights_path = key
                                tokenizer_path = key
            except Exception:
                pass

        # 1b) Parse RL job events for s3 URL (no DB creds needed; API-backed)
        if not weights_path and job_id:
            for ev_path in (
                f"{base_url}/rl/jobs/{job_id}/events",
                f"{base_url}/orchestration/jobs/{job_id}/events",
            ):
                try:
                    async with httpx.AsyncClient(timeout=20.0) as hc:
                        evr = await hc.get(ev_path, headers=headers, params={"limit": 200})
                    if evr.status_code != 200:
                        continue
                    data = evr.json() or {}
                    events = data.get("events") or data.get("data") or []
                    for e in events:
                        # Search both message and data for s3:// URLs
                        txt = " ".join(str(e.get(k) or "") for k in ("message","data","payload"))
                        if "s3://" in txt:
                            try:
                                import re
                                m = re.search(r"s3://[^\s'\"]+", txt)
                                if m:
                                    s3_uri = m.group(0)
                                    _tmp = s3_uri[len("s3://"):]
                                    parts = _tmp.split("/", 1)
                                    key = parts[1] if len(parts) > 1 else None
                                    if key:
                                        weights_path = key
                                        tokenizer_path = key
                                        break
                            except Exception:
                                pass
                    if weights_path:
                        break
                except Exception:
                    pass
        try:
            async with httpx.AsyncClient(timeout=30.0) as hc:
                r = await hc.get(f"{base_url}/learning/models/rl", headers=headers, params={"status": "succeeded", "limit": 200})
            if r.status_code == 200:
                data = r.json() or {}
                for item in data.get("models", []):
                    if item.get("job_id") == job_id and item.get("storage_path"):
                        weights_path = str(item.get("storage_path"))
                        tokenizer_path = weights_path
                        break
        except Exception:
            pass

        # Try registry for direct mapping first (preferred)
        if not weights_path:
            try:
                async with httpx.AsyncClient(timeout=15.0) as hc:
                    rr = await hc.get(f"{base_url}/learning/models/rl/registry", headers=headers)
                if rr.status_code == 200:
                    rows = (rr.json() or {}).get("models", [])
                    for row in rows:
                        if row.get("id") == args.model:
                            weights_path = row.get("weights_path") or row.get("weights")
                            tokenizer_path = row.get("tokenizer_path") or weights_path
                            break
            except Exception:
                pass

        # As a last resort, synthesize the expected S3 path from alias convention
        if not weights_path and base_model_hint and job_id:
            base_slug = base_model_hint.replace("/", "-")
            ck = ckpt or "checkpoint-epoch-1"
            weights_path = f"s3://synth-artifacts/models/{base_slug}/rl-{job_id}/{ck}.tar.gz"
            tokenizer_path = weights_path

        # 2) Promote into registry to record mapping (best-effort)
        if weights_path:
            promote_body = {
                "base_model": base_model_hint or "Qwen/Qwen3-0.6B",
                "job_id": job_id or "",
                "weights_path": weights_path,
                "tokenizer_path": tokenizer_path or weights_path,
                "dtype": "bfloat16",
                "model_id": args.model,
                "create_latest_alias": True,
            }
            try:
                async with httpx.AsyncClient(timeout=20.0) as hc:
                    _pr = await hc.post(f"{base_url}/learning/models/rl/promote", headers=headers, json=promote_body)
                    # Ignore non-200 if already promoted
            except Exception:
                pass

        # 3) Ensure on Modal via backend route; must include weights for rl:*
        ensure_body: Dict[str, Any] = {"model_id": args.model, "prefer_merged": True, "verbose": True}
        if base_model_hint:
            ensure_body["base_model"] = base_model_hint
            payload.setdefault("tokenizer", base_model_hint)
        if weights_path:
            ensure_body["weights_path"] = weights_path
            ensure_body["tokenizer_path"] = tokenizer_path or weights_path
        ensure_result: Dict[str, Any] | None = None
        try:
            async with httpx.AsyncClient(timeout=180.0) as hc:
                er = await hc.post(f"{base_url}/learning/models/ensure-on-modal", headers=headers, json=ensure_body)
            if er.status_code != 200:
                # Print a concise diagnostic but continue – upstream chat may still work if already present
                try:
                    print(f"ensure-on-modal failed: HTTP {er.status_code} {er.text[:200]}")
                except Exception:
                    pass
            else:
                try:
                    ensure_result = er.json()
                    kind = ensure_result.get("kind")
                    local_path = ensure_result.get("local_path")
                    src = ensure_result.get("source")
                    print(f"Ensured model on Modal: kind={kind} local_path={local_path} source={src}")
                except Exception:
                    pass
        except Exception as e:
            print(f"ensure-on-modal error: {type(e).__name__}: {e}")

        # 4) Best-effort registry proof: show the exact mapping for this RL alias
        try:
            async with httpx.AsyncClient(timeout=15.0) as hc:
                rr = await hc.get(f"{base_url}/learning/models/rl/registry", headers=headers)
            if rr.status_code == 200:
                rows = (rr.json() or {}).get("models", [])
                hit = next((r for r in rows if str(r.get("id")) == str(args.model)), None)
                if hit:
                    print("Registry mapping:")
                    try:
                        import json as _json
                        print(_json.dumps(hit, indent=2))
                    except Exception:
                        print(hit)
        except Exception:
            pass

    await _ensure_rl_model()

    try:
        resp = await client.create_chat_completion(**payload)
        print("Response:")
        print(json.dumps(resp, indent=2))
        try:
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
            if content:
                print("\nAssistant:")
                print(content)
        except Exception:
            pass
        return 0
    except Exception as e:
        import traceback
        print("\n===== Inference Error =====")
        print(f"Type: {type(e).__name__}")
        print(f"Repr: {repr(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        try:
            from synth_ai.http import HTTPError  # type: ignore
            if isinstance(e, HTTPError):
                print("HTTPError details:")
                print(f"  status={e.status}")
                print(f"  url={e.url}")
                print(f"  message={e.message}")
                if getattr(e, 'detail', None) is not None:
                    print(f"  detail={e.detail}")
                if getattr(e, 'body_snippet', None):
                    print(f"  body_snippet={e.body_snippet}")
        except Exception:
            pass
        print("===== End Inference Error =====\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))


