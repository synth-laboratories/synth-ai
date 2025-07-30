import pytest
import openai

from synth_ai.zyk import LM
from synth_ai.lm.vendors.core.anthropic_api import AnthropicAPI
from synth_ai.lm.vendors.openai_standard import OpenAIStandard

TEST_PROMPT = "What is 2+2? Answer with just the number."


@pytest.mark.slow
def test_openai_text():
    client = OpenAIStandard(
        sync_client=openai.OpenAI(),
        async_client=openai.AsyncOpenAI(),
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    response = client._hit_api_sync(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": TEST_PROMPT}],
        lm_config={"temperature": 0},
    )

    assert response.raw_response.strip() == "4"


@pytest.mark.slow
def test_openai_text_lm():
    lm = LM(
        model_name="gpt-4o-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="",
        user_message=TEST_PROMPT,
    )

    assert response.raw_response.strip() == "4"


@pytest.mark.slow
def test_anthropic_text():
    client = AnthropicAPI(
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    response = client._hit_api_sync(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides direct answers.",
            },
            {"role": "user", "content": TEST_PROMPT},
        ],
        lm_config={"temperature": 0},
    )

    assert response.raw_response.strip() == "4"


@pytest.mark.slow
def test_anthropic_text_lm():
    lm = LM(
        model_name="claude-3-haiku-20240307",
        formatting_model_name="claude-3-haiku-20240307",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that provides direct answers.",
        user_message=TEST_PROMPT,
    )

    assert response.raw_response.strip() == "4"


if __name__ == "__main__":
    test_openai_text_lm()
    test_anthropic_text_lm()
    test_openai_text()
    test_anthropic_text()
