import asyncio
import os
import time

from pydantic import BaseModel

from synth_ai.zyk import LM


def get_openai_api_key():
    # Get API key from environment variable
    return os.environ.get("OPENAI_API_KEY")


class TestOutput(BaseModel):
    answer: str
    reasoning: str


async def test_reasoning_effort():
    # Define a question that requires reasoning
    question = "If a train travels at 120 km/h and another train travels at 80 km/h in the opposite direction, how long will it take for them to be 500 km apart if they start 100 km apart?"

    print("Testing o3-mini with different reasoning_effort settings:")
    print("-" * 60)

    # Create an instance for HIGH reasoning
    print("Testing with reasoning_effort='high'")
    lm_high = LM(
        model_name="o3-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=1,
    )
    # Set reasoning_effort in lm_config
    lm_high.lm_config["reasoning_effort"] = "high"

    # Time the API call
    start_time = time.time()
    high_result = await lm_high.respond_async(
        system_message="You are a helpful assistant.",
        user_message=question,
    )
    high_time = time.time() - start_time

    print(f"Time taken: {high_time:.2f} seconds")
    print(f"Response length: {len(high_result.raw_response)} characters")
    print("-" * 60)

    # Create a separate instance for LOW reasoning
    print("Testing with reasoning_effort='low'")
    lm_low = LM(
        model_name="o3-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=1,
    )
    # Set reasoning_effort in lm_config
    lm_low.lm_config["reasoning_effort"] = "low"

    # Time the API call
    start_time = time.time()
    low_result = await lm_low.respond_async(
        system_message="You are a helpful assistant.",
        user_message=question,
    )
    low_time = time.time() - start_time

    print(f"Time taken: {low_time:.2f} seconds")
    print(f"Response length: {len(low_result.raw_response)} characters")
    print("-" * 60)

    # Print comparison
    print("Results comparison:")
    print(f"High reasoning time: {high_time:.2f} seconds")
    print(f"Low reasoning time: {low_time:.2f} seconds")
    print(
        f"Difference: {high_time - low_time:.2f} seconds ({(high_time/low_time - 1)*100:.1f}% difference)"
    )
    print(f"High response length: {len(high_result.raw_response)} characters")
    print(f"Low response length: {len(low_result.raw_response)} characters")
    print(
        f"Response length ratio: {len(high_result.raw_response)/len(low_result.raw_response):.2f}x"
    )

    # Print response samples
    print("\nHIGH Response Sample (first 200 chars):")
    print(high_result.raw_response[:200] + "...")
    print("\nLOW Response Sample (first 200 chars):")
    print(low_result.raw_response[:200] + "...")


if __name__ == "__main__":
    asyncio.run(test_reasoning_effort())
