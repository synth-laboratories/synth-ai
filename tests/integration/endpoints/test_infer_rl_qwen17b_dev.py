import os
import time
from pathlib import Path

import asyncio
import pytest
import httpx


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.asyncio
async def test_infer_rl_qwen17b_dev() -> None:
    """
    Dev smoke test for RL-trained model chat completions.

    Uses RL model id by default:
      rl:Qwen/Qwen3-1.7B:job_199c249ff6488eb3a88:checkpoint-epoch-10
    """
    # Load .env for dev first, then prod, then legacy fallback
    env_candidates = [
        os.path.join(os.getcwd(), ".env.test.dev"),
        os.path.join(os.getcwd(), ".env.test.prod"),
        os.path.join(os.getcwd(), ".env.test"),
    ]
    for env_path in env_candidates:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        # For dev, do not override variables already set in the shell
                        if env_path.endswith(".env.test.dev"):
                            os.environ.setdefault(k, v)
                        else:
                            os.environ.setdefault(k, v)
        except Exception:
            pass

    # Resolve a single dev endpoint: POST /api/inference/v1/chat/completions (proxy)
    dev_backend = os.getenv("DEV_BACKEND_URL")
    api_key = os.getenv("DEV_ACTIONS_SYNTH_API_KEY") or os.getenv("SYNTH_API_KEY")

    if not dev_backend or not api_key:
        pytest.skip("DEV_ACTIONS_SYNTH_API_KEY/SYNTH_API_KEY and DEV_BACKEND_URL required for RL dev test")

    base = dev_backend.rstrip("/")
    # Normalize to root (strip trailing /api if present), then use /api/inference/v1
    if base.endswith("/api"):
        base = base[:-4]
    candidates = [f"{base}/api/inference/v1/chat/completions"]

    print("[DEV RL TEST] Request candidates (in order):")
    for u in candidates:
        print(f"  - {u}")

    # Model id (override with RL_MODEL_ID if needed)
    # model_id = os.getenv(
    #     "RL_MODEL_ID",
    #     ,
    # )
    model_id = "rl:Qwen/Qwen3-1.7B:job_199c249ff6488eb3a88:checkpoint-epoch-10"

    # Single request only: call chat completions and finish
    client_timeout_s = float(os.getenv("SYNTH_TIMEOUT", "300"))
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "temperature": 0.2,
        "max_tokens": 256,
        "thinking_budget": 256,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(client_timeout_s)) as http:
        last_error = None
        for request_url in candidates:
            try:
                print(f"[DEV RL TEST] Trying: {request_url}")
                resp = await http.post(
                    request_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except Exception as request_err:
                last_error = f"{request_url} -> request error: {request_err}"
                print(f"[DEV RL TEST]  -> Request error: {request_err}")
                continue

            if resp.status_code == 404:
                last_error = f"404 at {request_url}"
                print(f"[DEV RL TEST]  -> 404 Not Found")
                continue

            if resp.status_code != 200:
                reason = resp.reason_phrase or ""
                snippet_limit = 1000
                body_snippet = resp.text[:snippet_limit]
                clean_snippet = body_snippet.replace("\n", " ")
                interesting_headers = {}
                for key in ("date", "content-type", "content-length", "x-request-id", "cf-ray", "server"):
                    if key in resp.headers:
                        interesting_headers[key] = resp.headers[key]
                print(
                    "[DEV RL TEST]  -> Non-200 response: status=%s reason=%s" % (resp.status_code, reason)
                )
                if interesting_headers:
                    print(f"[DEV RL TEST]  -> Headers: {interesting_headers}")
                print(
                    f"[DEV RL TEST]  -> Body snippet (first {snippet_limit} chars): {clean_snippet}"
                )
                try:
                    log_dir = Path("temp") / "integration_failures"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / (
                        f"rl_chat_completion_{int(time.time())}_{resp.status_code}.txt"
                    )
                    log_path.write_text(resp.text, encoding="utf-8", errors="ignore")
                    print(f"[DEV RL TEST]  -> Full response saved to: {log_path}")
                except Exception as write_err:
                    print(f"[DEV RL TEST]  -> Failed to persist response body: {write_err}")
                last_error = f"{request_url} -> {resp.status_code} {reason}".strip()
                continue

            try:
                data = resp.json()
            except Exception as decode_err:
                snippet = resp.text[:500]
                print(f"[DEV RL TEST]  -> JSON decode failed: {decode_err}")
                print(f"[DEV RL TEST]  -> Response snippet: {snippet}")
                last_error = f"{request_url} -> JSON decode failed: {decode_err}"
                continue

            if not isinstance(data, dict) or "choices" not in data:
                print(f"[DEV RL TEST]  -> Unexpected payload structure: {type(data)}")
                print(f"[DEV RL TEST]  -> Keys: {list(data) if isinstance(data, dict) else 'n/a'}")
                last_error = f"{request_url} -> unexpected payload"
                continue

            print(f"[DEV RL TEST] Success at: {request_url}")
            return
        pytest.fail(f"All endpoints failed. Last error: {last_error}")
