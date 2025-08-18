import asyncio

import pytest
from pydantic import BaseModel

# Run these as part of the ultra-fast unit suite
pytestmark = [pytest.mark.unit]

from synth_ai.lm.core.main_v3 import LM as LMv3
from synth_ai.lm.vendors.base import BaseLMResponse


class ExampleModel(BaseModel):
    message: str


class FakeStructuredClient:
    """
    Minimal fake client that supports the structured output call paths
    used by StructuredOutputHandler in forced_json mode.
    """

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages,
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        tools=None,
        reasoning_effort: str = "high",
    ):
        instance = response_model(message="ok")
        return BaseLMResponse(raw_response="", structured_output=instance, tool_calls=None)

    # Provide a no-op standard async path in case it's selected
    async def _hit_api_async(
        self,
        messages,
        model,
        lm_config,
        use_ephemeral_cache_only=False,
        tools=None,
        reasoning_effort="high",
    ):
        return BaseLMResponse(
            raw_response='{"message":"ok"}', structured_output=None, tool_calls=None
        )


@pytest.mark.fast
def test_main_v3_structured_output_forced_json(monkeypatch):
    """
    Fail->Pass guard: constructing LMv3 with response_format used to crash due to
    incorrect StructuredOutputHandler init. This asserts it now succeeds and returns
    a structured output when routed through forced_json mode.
    """

    # Monkeypatch the get_client symbol used within main_v3
    import synth_ai.lm.core.main_v3 as main_v3_module

    def fake_get_client(
        model_name: str, with_formatting: bool = False, synth_logging: bool = True, provider=None
    ):
        return FakeStructuredClient()

    monkeypatch.setattr(main_v3_module, "get_client", fake_get_client)

    lm = LMv3(vendor="openai", model="gpt-5-nano", response_format=ExampleModel)

    async def run():
        return await lm.respond_async(system_message="system", user_message="hello")

    result = asyncio.run(run())
    assert isinstance(result, BaseLMResponse)
    assert result.structured_output is not None
    assert isinstance(result.structured_output, ExampleModel)
    assert result.structured_output.message == "ok"

    # Also verify providing response_model at call-time works
    lm2 = LMv3(vendor="openai", model="gpt-5-nano")

    async def run2():
        return await lm2.respond_async(
            system_message="system", user_message="hello", response_model=ExampleModel
        )

    result2 = asyncio.run(run2())
    assert isinstance(result2, BaseLMResponse)
    assert isinstance(result2.structured_output, ExampleModel)
    assert result2.structured_output.message == "ok"
