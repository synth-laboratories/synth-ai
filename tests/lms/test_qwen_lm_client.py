import os
import asyncio
import pytest

from synth_ai.lm.core.main_v3 import LM


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_QWEN_CHAT") != "1", reason="Set RUN_QWEN_CHAT=1 to run")
@pytest.mark.asyncio
async def test_lm_client_qwen_dev(backend_base_url: str, synth_api_key: str):
    # Map to OpenAI-compatible env consumed by Synth client
    base = backend_base_url.rstrip("/")
    os.environ["OPENAI_API_BASE"] = base
    os.environ["OPENAI_BASE_URL"] = base  # some clients prefer this var name
    os.environ["OPENAI_API_KEY"] = synth_api_key

    lm = LM(model_name="Qwen/Qwen3-0.6B", provider="synth", enable_v3_tracing=False)
    resp = await lm.respond_async(messages=[{"role": "user", "content": "Say 'ok'."}], temperature=0.0, max_tokens=8)
    assert resp is not None
    # The LM returns a vendor-agnostic response wrapper; ensure content exists
    # Try different attribute names based on the actual response structure
    text = ""
    if hasattr(resp, 'content') and resp.content:
        text = resp.content
    elif hasattr(resp, 'raw_response') and resp.raw_response:
        text = str(resp.raw_response)
    elif hasattr(resp, 'text'):
        text = resp.text
    else:
        # Fallback: convert the whole response to string
        text = str(resp)

    assert text, f"Response should contain text content, got: {resp}"
    assert isinstance(text, str) and len(text) > 0


