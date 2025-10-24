import os
import time
from typing import Dict, Any, cast

import httpx
import pytest

from synth_ai._utils.base_url import get_backend_from_env


def _resolve_endpoint(base: str, path: str) -> str:
    base = base.rstrip("/")
    # If base already ends with /api, just append the path
    if base.endswith("/api"):
        return f"{base}{path}"
    # Otherwise, add /api between base and path
    return f"{base}/api{path}"


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_QWEN_CHAT") != "1", reason="Set RUN_QWEN_CHAT=1 to run")
@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])  # test Qwen/Qwen3-0.6B only
def test_chat_completions_dev(backend_base_url: str, auth_headers: Dict[str, str], model: str):
    # Treat the loaded backend as the target (dev by default)
    url = _resolve_endpoint(backend_base_url, "/chat/completions")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok'."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    timeout = float(os.getenv("CHAT_COMPLETIONS_TIMEOUT", "60"))

    print(f"🚀 DEBUG: About to make request to: {url}")
    print(f"🚀 DEBUG: Headers: {auth_headers}")
    print(f"🚀 DEBUG: Payload: {payload}")
    print(f"🚀 DEBUG: Timeout: {timeout}")

    # One warmup try to mitigate cold starts
    with httpx.Client(timeout=timeout) as client:
        try:
            warmup_resp = client.post(url, headers=auth_headers, json={**payload, "max_tokens": 1})
            print(f"🚀 DEBUG: Warmup response: {warmup_resp.status_code}")
        except Exception as e:
            print(f"🚀 DEBUG: Warmup failed: {e}")

    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=auth_headers, json=payload)
    latency = time.time() - t0

    print(f"🚀 DEBUG: Main response status: {resp.status_code}")
    print(f"🚀 DEBUG: Main response text: {resp.text[:500]}")
    print(f"🚀 DEBUG: Latency: {latency:.2f}s")

    # If backend returns 5xx, retry once after short sleep
    if resp.status_code >= 500:
        print(f"🚀 DEBUG: Got 5xx, retrying...")
        time.sleep(2.0)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=auth_headers, json=payload)
        print(f"🚀 DEBUG: Retry response: {resp.status_code}")

    assert resp.status_code == 200, f"chat/completions failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    assert "choices" in data and isinstance(data["choices"], list)
    assert latency < timeout


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_QWEN_CHAT") != "1", reason="Set RUN_QWEN_CHAT=1 to run")
@pytest.mark.prod
@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])  # test Qwen/Qwen3-0.6B only
def test_chat_completions_prod(auth_headers: Dict[str, str], model: str):
    base_env = (
        os.getenv("PROD_BACKEND_URL")
        or os.getenv("BACKEND_URL")
        or os.getenv("LEARNING_V2_BASE_URL")
        or os.getenv("PROD_API_URL")
        or os.getenv("API_URL")
        or ""
    ).strip()
    base = base_env.rstrip("/")
    api_key = os.getenv("PROD_SYNTH_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not base:
        base, _ = get_backend_from_env()
    if not api_key:
        _, key = get_backend_from_env()
        api_key = key
    if not base or not api_key:
        pytest.skip("Missing PROD backend URL or API key; set PROD_BACKEND_URL and PROD_SYNTH_API_KEY")

    base = base.rstrip("/")
    url = _resolve_endpoint(base, "/chat/completions")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok'."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    timeout = float(os.getenv("CHAT_COMPLETIONS_TIMEOUT", "60"))

    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
    latency = time.time() - t0

    assert resp.status_code == 200, f"chat/completions failed ({resp.status_code}): {resp.text}"
    data = resp.json()
    assert "choices" in data and isinstance(data["choices"], list)
    assert latency < timeout
