import os

import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_lm_injection_traced():
    load_dotenv()
    from synth_ai.lm.core.main_v3 import LM
    from synth_ai.lm.overrides import LMOverridesContext
    from synth_ai.tracing_v3.abstractions import LMCAISEvent
    from synth_ai.tracing_v3.session_tracer import SessionTracer

    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "groq")

    tracer = SessionTracer()
    await tracer.start_session(metadata={"test": "lm_injection"})
    await tracer.start_timestep(step_id="lm_test")

    lm = LM(model=model, vendor=vendor, temperature=0.0, session_tracer=tracer)

    messages = [
        {"role": "system", "content": "You will echo the user message."},
        {"role": "user", "content": "I used the atm to withdraw cash."},
    ]

    overrides = [
        {
            "match": {"contains": "atm", "role": "user"},
            "injection_rules": [{"find": "atm", "replace": "ATM"}],
        }
    ]
    with LMOverridesContext(overrides):
        _ = await lm.respond_async(messages=messages)

    # Inspect v3 tracing for the substitution
    events = [
        e
        for e in (tracer.current_session.event_history if tracer.current_session else [])
        if isinstance(e, LMCAISEvent)
    ]
    assert events, "No LMCAISEvent recorded by SessionTracer"
    cr = events[-1].call_records[0]
    traced_user = ""
    for m in cr.input_messages:
        if m.role == "user":
            for part in m.parts:
                if getattr(part, "type", None) == "text":
                    traced_user += part.text or ""
    assert "ATM" in traced_user, f"Expected substitution in traced prompt; got: {traced_user!r}"

    await tracer.end_timestep()
    await tracer.end_session()


@pytest.mark.asyncio
async def test_openai_wrapper_injection_call():
    load_dotenv()
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("No Groq/OpenAI API key set; skipping")

    from openai import AsyncOpenAI

    import synth_ai.lm.provider_support.openai as _synth_openai_patch  # noqa: F401
    from synth_ai.lm.overrides import LMOverridesContext

    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    messages = [
        {"role": "system", "content": "Answer with a Banking77 label."},
        {"role": "user", "content": "I used the atm to withdraw cash."},
    ]
    overrides = [
        {
            "match": {"contains": "atm", "role": "user"},
            "injection_rules": [{"find": "atm", "replace": "ATM"}],
        }
    ]
    try:
        with LMOverridesContext(overrides):
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
        assert hasattr(resp, "choices")
    except Exception as e:
        # Accept unauthorized as expected xfail to not fail CI when key is invalid
        if "invalid api key" in str(e).lower():
            pytest.xfail("Invalid API key for Groq/OpenAI; wrapper path xfailed")
        raise


@pytest.mark.asyncio
async def test_anthropic_wrapper_injection_call():
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("No Anthropic API key set; skipping")

    import anthropic

    import synth_ai.lm.provider_support.anthropic as _synth_anthropic_patch  # noqa: F401
    from synth_ai.lm.overrides import LMOverridesContext

    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
    client = anthropic.AsyncClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "I used the atm to withdraw cash."}]},
    ]
    overrides = [
        {"match": {"contains": "atm"}, "injection_rules": [{"find": "atm", "replace": "ATM"}]}
    ]

    try:
        with LMOverridesContext(overrides):
            resp = await client.messages.create(
                model=model,
                system="Answer with a Banking77 label.",
                max_tokens=64,
                temperature=0,
                messages=messages,
            )
        assert hasattr(resp, "content")
    except Exception as e:
        if "invalid" in str(e).lower() and "api key" in str(e).lower():
            pytest.xfail("Invalid Anthropic API key; wrapper path xfailed")
        raise
