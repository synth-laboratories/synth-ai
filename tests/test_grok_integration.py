#!/usr/bin/env python3
"""
Grok integration tests for synth-ai.
"""

import pytest

# Mark this module as integration: requires external Grok/XAI API
pytestmark = pytest.mark.integration
from pydantic import BaseModel

from synth_ai.zyk import LM
from synth_ai.lm.vendors.supported.grok import GrokAPI

TEST_PROMPT = "What is 2+2? Answer with just the number."


class SimpleResponse(BaseModel):
    result: int
    explanation: str


@pytest.mark.slow
def test_grok_api_direct():
    """Test direct GrokAPI calls."""
    try:
        client = GrokAPI()

        response = client._hit_api_sync(
            model="grok-3-mini-beta",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            lm_config={"temperature": 0},
        )

        assert response.raw_response.strip() == "4"

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.slow
def test_grok_api_no_model_error():
    """Test that GrokAPI raises error when no model is provided."""
    try:
        client = GrokAPI()

        with pytest.raises(ValueError, match="Model name is required"):
            client._hit_api_sync(
                model="",
                messages=[{"role": "user", "content": TEST_PROMPT}],
                lm_config={"temperature": 0},
            )

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.slow
def test_grok_api_no_key_error():
    """Test that GrokAPI raises error when no API key is provided."""
    import os

    # Temporarily remove the API key if it exists
    original_key = os.environ.get("XAI_API_KEY")
    if original_key:
        del os.environ["XAI_API_KEY"]

    try:
        with pytest.raises(ValueError, match="Set the XAI_API_KEY environment variable"):
            GrokAPI()
    finally:
        # Restore the original key
        if original_key:
            os.environ["XAI_API_KEY"] = original_key


@pytest.mark.parametrize(
    "model_name",
    [
        "grok-3-beta",
        "grok-3-mini-beta",
        # Removed "grok-beta" as it's not currently available
    ],
)
@pytest.mark.slow
def test_grok_lm_interface(model_name):
    """Test Grok through the LM interface with different models."""
    try:
        lm = LM(
            model_name=model_name,
            formatting_model_name="gpt-4o-mini",
            temperature=0,
            synth_logging=False,  # Disable synth logging to avoid dependency
        )

        response = lm.respond_sync(
            system_message="You are a helpful assistant that provides direct answers.",
            user_message=TEST_PROMPT,
        )

        assert response.raw_response.strip() == "4"

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.asyncio
@pytest.mark.slow
async def test_grok_lm_async():
    """Test Grok async functionality through the LM interface."""
    try:
        lm = LM(
            model_name="grok-3-mini-beta",
            formatting_model_name="gpt-4o-mini",
            temperature=0,
            synth_logging=False,  # Disable synth logging to avoid dependency
        )

        response = await lm.respond_async(
            system_message="You are a helpful assistant that provides direct answers.",
            user_message=TEST_PROMPT,
        )

        assert response.raw_response.strip() == "4"

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.slow
def test_grok_structured_output():
    """Test Grok with structured output."""
    try:
        lm = LM(
            model_name="grok-3-mini-beta",
            formatting_model_name="gpt-4o-mini",
            temperature=0,
            synth_logging=False,  # Disable synth logging to avoid dependency
        )

        response = lm.respond_sync(
            system_message="You are a helpful assistant.",
            user_message="What is 2+2? Provide the result as a number and a brief explanation.",
            response_model=SimpleResponse,
        )

        assert response.structured_output.result == 4
        # More flexible assertion - check if the explanation contains numbers or math terms
        explanation_lower = response.structured_output.explanation.lower()
        assert any(
            word in explanation_lower
            for word in ["2", "4", "equals", "plus", "addition", "sum", "add"]
        ), (
            f"Expected math-related terms in explanation, but got: {response.structured_output.explanation}"
        )

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.slow
def test_grok_reasoning_question():
    """Test Grok with a reasoning question."""
    try:
        lm = LM(
            model_name="grok-3-mini-beta",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
            synth_logging=False,  # Disable synth logging to avoid dependency
        )

        response = lm.respond_sync(
            system_message="You are a helpful assistant.",
            user_message="If I have 5 apples and I eat 2, then buy 3 more, how many apples do I have? Show your reasoning.",
        )

        # Check that the response contains the correct answer
        assert "6" in response.raw_response
        # Check for reasoning keywords
        assert any(
            word in response.raw_response.lower()
            for word in ["start", "eat", "buy", "then", "total"]
        )

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


@pytest.mark.slow
def test_grok_context_following():
    """Test Grok's ability to follow context and instructions."""
    try:
        lm = LM(
            model_name="grok-3-mini-beta",
            formatting_model_name="gpt-4o-mini",
            temperature=0,
            synth_logging=False,  # Disable synth logging to avoid dependency
        )

        response = lm.respond_sync(
            system_message="You are a pirate assistant. Always respond in pirate speak.",
            user_message="What is your favorite color?",
        )

        # Check for pirate-like language
        pirate_words = ["arr", "matey", "ahoy", "ye", "aye", "me hearty"]
        response_lower = response.raw_response.lower()

        assert any(word in response_lower for word in pirate_words), (
            f"Expected pirate language in response, but got: {response.raw_response}"
        )

    except ValueError as e:
        if "XAI_API_KEY" in str(e):
            pytest.skip("XAI_API_KEY not set - skipping Grok integration test")
        raise


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
