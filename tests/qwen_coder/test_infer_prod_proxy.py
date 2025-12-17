import os
import time
import pytest
import httpx


@pytest.mark.skipif(not os.getenv("SYNTH_API_KEY"), reason="requires SYNTH_API_KEY in env")
@pytest.mark.slow
def test_infer_prod_proxy_base_model():
    base_url = os.getenv("BACKEND_BASE_URL", "https://api.usesynth.ai/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = base_url + "/api"
    url = base_url + "/inference/v1/chat/completions"

    payload = {
        "model": os.getenv("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "messages": [{"role": "user", "content": "Say 'ok'."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "thinking_budget": 64,
    }
    headers = {"Authorization": f"Bearer {os.environ['SYNTH_API_KEY']}", "Content-Type": "application/json"}
    resp = None
    backoffs = [1.0, 2.0, 3.0, 5.0, 8.0]
    for attempt, delay in enumerate(backoffs, start=1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            if resp.status_code < 500:
                break
        except Exception:
            # transient network error, retry
            pass
        time.sleep(delay)
    assert resp is not None
    if resp.status_code >= 500:
        pytest.xfail(f"inference upstream returned {resp.status_code}: {resp.text[:200]}")
    assert resp.status_code == 200
    js = resp.json()
    assert isinstance(js.get("choices"), list)


@pytest.mark.skipif(not (os.getenv("SYNTH_API_KEY") and os.getenv("FT_MODEL_ID")), reason="requires SYNTH_API_KEY and FT_MODEL_ID")
def test_infer_prod_proxy_ft_model():
    base_url = os.getenv("BACKEND_BASE_URL", "https://api.usesynth.ai/api").rstrip("/")
    if not base_url.endswith("/api"):
        base_url = base_url + "/api"
    url = base_url + "/inference/v1/chat/completions"
    model_id = os.environ["FT_MODEL_ID"].strip()

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say 'ok'."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "thinking_budget": 64,
    }
    headers = {"Authorization": f"Bearer {os.environ['SYNTH_API_KEY']}", "Content-Type": "application/json"}
    resp = None
    backoffs = [1.0, 2.0, 3.0, 5.0, 8.0]
    for attempt, delay in enumerate(backoffs, start=1):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            if resp.status_code < 500:
                break
        except Exception:
            pass
        time.sleep(delay)
    assert resp is not None
    if resp.status_code >= 500:
        pytest.xfail(f"inference upstream returned {resp.status_code}: {resp.text[:200]}")
    assert resp.status_code == 200
    js = resp.json()
    assert isinstance(js.get("choices"), list)


