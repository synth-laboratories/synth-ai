import time

import pytest

from synth_ai.zyk import LM


# Use pytest fixtures instead of unittest setup
@pytest.fixture
def model_instances():
    """Initialize all model configurations for testing."""
    models = {
        # O3-mini standard
        "o3-mini": LM(
            model_name="o3-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # O3-mini with high reasoning
        "o3-mini-high-reasoning": LM(
            model_name="o3-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # Claude 3 Sonnet
        "claude-3-7-sonnet-latest": LM(
            model_name="claude-3-7-sonnet-latest",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # Claude 3 Sonnet with high reasoning
        "claude-3-7-sonnet-latest-high-reasoning": LM(
            model_name="claude-3-7-sonnet-latest",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # Gemini Flash
        "gemini-2-flash": LM(
            model_name="gemini-2-flash",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # Gemma 3
        "gemma3-27b-it": LM(
            model_name="gemma3-27b-it",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
        # GPT-4o mini
        "gpt-4o-mini": LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        ),
    }

    # Set reasoning_effort to "high" for specific models
    models["o3-mini-high-reasoning"].lm_config["reasoning_effort"] = "high"
    models["claude-3-7-sonnet-latest-high-reasoning"].lm_config["reasoning_effort"] = (
        "high"
    )

    return models


# Convert tests to pytest style
@pytest.mark.parametrize(
    "model_name",
    [
        "o3-mini",
        "o3-mini-high-reasoning",
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-latest-high-reasoning",
        "gemini-2-flash",
        "gemma3-27b-it",
        "gpt-4o-mini",
    ],
)
def test_model_simple_response(model_instances, model_name):
    """Test that models can generate a simple response."""
    lm = model_instances[model_name]
    system_message = "You are a helpful assistant."
    user_message = "What is the capital of France?"

    print(f"\nTesting {model_name}...")
    start_time = time.time()

    response = lm.respond_sync(
        system_message=system_message,
        user_message=user_message,
    )
    elapsed = time.time() - start_time

    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Response length: {len(response.raw_response)} characters")
    print(f"Response sample: {response.raw_response[:100]}...")

    # Basic validation
    assert isinstance(response.raw_response, str)
    assert len(response.raw_response) > 0
    assert (
        "Paris" in response.raw_response
    ), f"Expected 'Paris' in response, but got: {response.raw_response[:200]}..."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
   #     "o3-mini",
 #       "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-latest-high-reasoning",
        # "gemini-2-flash",
        # "gemma3-27b-it",
        # "gpt-4o-mini",
    ],
)
async def test_reasoning_question(model_instances, model_name):
    """Test models with a question that requires reasoning."""
    lm = model_instances[model_name]
    system_message = "You are a helpful assistant."
    user_message = "If a train travels at 120 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 500 km apart if they start 100 km apart?"

    print(f"\nTesting {model_name} with reasoning question...")
    start_time = time.time()

    response = await lm.respond_async(
        system_message=system_message,
        user_message=user_message,
    )
    elapsed = time.time() - start_time

    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Response length: {len(response.raw_response)} characters")
    print(f"Response sample: {response.raw_response[:100]}...")

    # Basic validation
    assert isinstance(response.raw_response, str)
    assert len(response.raw_response) > 0


@pytest.mark.parametrize(
    "model_name",
    [
        "o3-mini",
       # "o3-mini",
        #"claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-latest-high-reasoning",
        # "gemini-2-flash",
        # "gemma3-27b-it",
        # "gpt-4o-mini",
    ],
)
def test_model_context_and_factuality(model_instances, model_name):
    """Test models for factuality with a context-based question."""
    lm = model_instances[model_name]
    system_message = "You are a helpful assistant."
    context = """
    The city of Atlantis was founded in 1968 by marine archaeologist Dr. Sophia Maris. 
    It has a population of 37,500 residents and is known for its underwater research facilities. 
    The current mayor is Dr. Robert Neptune who was elected in 2020.
    """
    user_message = f"Based on the following information, when was Atlantis founded and who is its current mayor?\n\n{context}"

    print(f"\nTesting {model_name} for factuality...")

    response = lm.respond_sync(
        system_message=system_message,
        user_message=user_message,
    )

    # Check if the response contains the correct information
    assert (
        "1968" in response.raw_response
    ), f"Expected '1968' in response for founding year, but got: {response.raw_response[:200]}..."
    assert (
        "Robert Neptune" in response.raw_response
    ), f"Expected 'Robert Neptune' in response for mayor, but got: {response.raw_response[:200]}..."


if __name__ == "__main__":
    # For direct script execution
    pytest.main(["-xvs", __file__])
