import os
import asyncio
import pytest
import httpx


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_fft_qwen06b_chat_completion_smoke_prod() -> None:
    """
    Prod smoke test for the Synth OpenAI-compatible chat completions endpoint.

    Prefers .env.test.prod and PROD_BACKEND_URL, with 10-minute timeout + warmup.
    Default model is the provided fine-tuned job id for Qwen3-0.6B.
    """
    # Load only prod env to avoid accidental dev fallback
    prod_env = os.path.join(os.getcwd(), ".env.test.prod")
    if os.path.exists(prod_env):
        try:
            with open(prod_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

    # Force localhost testing; allow override via LOCAL_BACKEND_URL
    synth_base = os.getenv("LOCAL_BACKEND_URL") or "http://127.0.0.1:8000"
    prod_backend = None
    dev_backend = None
    api_key = os.getenv("DEV_ACTIONS_SYNTH_API_KEY") or os.getenv("SYNTH_API_KEY")

    bases: list[str] = []
    if synth_base:
        bases.append(synth_base.rstrip("/"))

    if not api_key or not bases:
        pytest.skip("SYNTH_API_KEY and base URL required for prod test")

    # Build request candidates per backend routing rules (avoid duplicate /api)
    candidates: list[str] = []
    seen: set[str] = set()
    for b in bases:
        base = b.rstrip("/")

        def _add(u: str) -> None:
            if u not in seen:
                candidates.append(u)
                seen.add(u)

        if base.endswith("/api"):
            _add(f"{base}/v1/chat/completions")
        else:
            _add(f"{base}/api/v1/chat/completions")

    print("[LOCAL TEST] Request candidates (in order):")
    for u in candidates:
        print(f"  - {u}")

    # Use provided fine-tuned job id by default
    model_id = "rl:Qwen/Qwen3-0.6B:job_009fa6565fa4457d"#os.getenv("FFT_MODEL_ID", )

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
        # Preflight: skip if localhost backend is not reachable
        reachable = False
        for b in bases:
            try:
                h = await http.get(b.rstrip('/') + "/api/health", timeout=2.0)
                if h.status_code in (200, 401, 403, 404):
                    reachable = True
                    break
            except Exception:
                continue
        if not reachable:
            pytest.skip("Local backend not reachable at /api/health; ensure it is running on localhost:8000")
        last_error = None
        for request_url in candidates:
            try:
                print(f"[PROD TEST] Trying: {request_url}")
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
                    print(f"[PROD TEST]  -> 404 Not Found")
                    continue
                    # No proxy fallback locally; require 200
                assert resp.status_code == 200, f"{request_url} -> {resp.status_code}: {resp.text[:200]}"
                data = resp.json()
                assert isinstance(data, dict) and "choices" in data
                print(f"[PROD TEST] Success at: {request_url}")
                return
            except Exception as e:
                last_error = str(e)
                print(f"[PROD TEST]  -> Error: {last_error}")
                continue
        pytest.fail(f"All endpoints failed. Last error: {last_error}")


