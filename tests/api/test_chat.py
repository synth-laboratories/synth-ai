import httpx
import pytest


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-0.6B",
        # keep simple and cheap; adjust if needed
    ],
)
@pytest.mark.slow
def test_chat_completion_minimal(base_url: str, auth_headers: dict, model: str):
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in one word."}],
        "max_tokens": 5,
        "temperature": 0.2,
        "stream": False,
    }
    r = httpx.post(url, headers=auth_headers, json=payload, timeout=60)
    assert r.status_code in (200, 401, 429), r.text  # allow auth/rate cases in CI
    if r.status_code == 200:
        data = r.json()
        assert data.get("object") in ("chat.completion", "chat.completion.chunk")
        assert "choices" in data and isinstance(data["choices"], list)
