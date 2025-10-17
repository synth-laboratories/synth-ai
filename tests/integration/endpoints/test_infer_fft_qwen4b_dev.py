import os
import asyncio
import pytest
import httpx


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.asyncio
async def test_fft_qwen4b_chat_completion_smoke() -> None:
    """
    Smoke-test the Synth OpenAI-compatible chat completions endpoint using a fine-tuned model.

    Requirements (skips if missing):
    - DEV_BACKEND_URL: dev backend root (with or without /api)
    - DEV_ACTIONS_SYNTH_API_KEY or SYNTH_API_KEY: bearer token

    Optional overrides:
    - FFT_MODEL_ID: model id to query (defaults to the provided job id)
    - SYNTH_TIMEOUT: per-request timeout seconds (defaults to 600 for this test)
    """
    # Load .env.test manually if present (avoid extra test-time deps)
    env_candidates = [
        os.path.join(os.getcwd(), ".env.test.dev"),
        os.path.join(os.getcwd(), ".env.test.prod"),
        os.path.join(os.getcwd(), ".env.test"),  # legacy fallback
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
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

    # Resolve a single dev endpoint: POST /api/inference/v1/chat/completions (proxy)
    dev_backend = os.getenv("DEV_BACKEND_URL")
    api_key = os.getenv("DEV_ACTIONS_SYNTH_API_KEY") or os.getenv("SYNTH_API_KEY")

    if not dev_backend or not api_key:
        pytest.skip("DEV_ACTIONS_SYNTH_API_KEY/SYNTH_API_KEY and DEV_BACKEND_URL required for dev test")

    base = dev_backend.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    candidates = [f"{base}/api/inference/v1/chat/completions"]

    print("[DEV TEST] Request candidates (in order):")
    for u in candidates:
        print(f"  - {u}")

    model_id = "fft:Qwen/Qwen3-4B:job_291bc0cfa2f641ee"  # or override with FFT_MODEL_ID

    # Single request only: call chat completions directly and finish
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
                print(f"[DEV TEST] Trying: {request_url}")
                resp = await http.post(
                    request_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code == 404:
                    last_error = f"404 at {request_url}"
                    print(f"[DEV TEST]  -> 404 Not Found")
                    continue
                assert resp.status_code == 200, f"{request_url} -> {resp.status_code}: {resp.text[:200]}"
                data = resp.json()
                assert isinstance(data, dict) and "choices" in data
                print(f"[DEV TEST] Success at: {request_url}")
                return
            except Exception as e:
                last_error = str(e)
                print(f"[DEV TEST]  -> Error: {last_error}")
                continue
        pytest.fail(f"All endpoints failed. Last error: {last_error}")

