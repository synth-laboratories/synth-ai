"""
Template for creating a LocalAPI with proper trace correlation handling.

This template demonstrates the recommended patterns for building a LocalAPI
that correctly handles trace correlation IDs, which are required for proper
trace hydration in the Synth AI backend.

Key patterns:
1. Install guards to detect direct LLM provider calls
2. Use normalize_inference_url() before calling LLM APIs
3. Use build_rollout_response() helper to automatically extract trace correlation ID
4. Implement proper error handling

Usage:
    1. Copy this template to your project
    2. Customize the dataset loading and task-specific logic
    3. Update the scoring function for your task
    4. Run with: ENVIRONMENT_API_KEY=<your-key> python your_localapi.py
"""

from fastapi import Request
import httpx

from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import (
    normalize_inference_url,
    build_rollout_response,
    install_all_guards,
)
from synth_ai.sdk.task.contracts import (
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)


# =============================================================================
# INSTALL GUARDS (Do this early, before any LLM calls)
# =============================================================================

# Install guards to detect direct LLM provider calls that bypass trace capture
# This will warn if code tries to call api.openai.com, api.anthropic.com, etc. directly
install_all_guards()


# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_ID = "my_task"
APP_NAME = "My Task Name"


# =============================================================================
# DATASET
# =============================================================================

def get_dataset_size() -> int:
    """Return the total number of samples in your dataset."""
    return 100  # Replace with actual dataset size


def get_sample(seed: int) -> dict:
    """Get a specific sample from your dataset based on seed.

    Args:
        seed: Seed value used to deterministically select a sample

    Returns:
        dict with sample data including:
        - input: The input to process (query, text, etc.)
        - expected_output: The expected output (for scoring)
        - Any other metadata needed for evaluation
    """
    idx = seed % get_dataset_size()
    # Replace with actual dataset loading logic
    return {
        "index": idx,
        "input": f"Sample input {idx}",
        "expected_output": f"Sample output {idx}",
    }


# =============================================================================
# SCORING
# =============================================================================

def score_response(predicted: str, sample: dict) -> float:
    """Score the LLM's response against the expected output.

    Args:
        predicted: The LLM's generated response
        sample: The original sample data with expected_output

    Returns:
        float between 0.0 and 1.0 representing match quality
    """
    expected = sample["expected_output"]
    # Replace with task-specific scoring logic
    # Examples: exact match, F1 score, semantic similarity, etc.
    return 1.0 if predicted.strip() == expected.strip() else 0.0


# =============================================================================
# TASK APP PROVIDERS
# =============================================================================

def provide_taskset_description() -> dict:
    """Provide metadata about available dataset splits."""
    return {
        "splits": ["test"],
        "sizes": {"test": get_dataset_size()},
    }


def provide_task_instances(seeds: list[int]):
    """Generate TaskInfo objects for each seed."""
    for seed in seeds:
        sample = get_sample(seed)
        yield TaskInfo(
            task={"id": APP_ID, "name": APP_NAME},
            dataset={"id": APP_ID, "split": "test", "index": sample["index"]},
            inference={},
            limits={"max_turns": 1},  # Adjust based on your task
            task_metadata={
                "input": sample["input"],
                "expected_output": sample["expected_output"],
            },
        )


# =============================================================================
# LLM CALL
# =============================================================================

async def call_llm(
    user_prompt: str,
    inference_url: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini"
) -> str:
    """Call the LLM to process the input.

    IMPORTANT: This function demonstrates the correct pattern:
    1. Use normalize_inference_url() to add proper path structure
    2. Make the API call to the normalized URL
    3. Extract and return the response text
    """
    system_prompt = "You are a helpful assistant."  # Customize for your task

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # IMPORTANT: Normalize the inference URL before calling
        url = normalize_inference_url(inference_url)
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# =============================================================================
# ROLLOUT HANDLER
# =============================================================================

async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """Handle a single evaluation rollout.

    IMPORTANT: This function demonstrates the correct pattern:
    1. Extract policy config and inference URL
    2. Call LLM with normalized URL
    3. Score the response
    4. Use build_rollout_response() to automatically handle trace correlation
    """
    seed = request.env.seed
    sample = get_sample(seed)

    policy_config = request.policy.config or {}
    inference_url = policy_config.get("inference_url")

    if not inference_url:
        raise ValueError("No inference_url provided in policy config")

    # Call LLM to process the input
    predicted = await call_llm(
        user_prompt=sample["input"],
        inference_url=inference_url,
        api_key=policy_config.get("api_key"),
        model=policy_config.get("model", "gpt-4o-mini"),
    )

    # Score the prediction
    score = score_response(predicted, sample)

    # IMPORTANT: Use build_rollout_response() helper
    # This automatically extracts trace_correlation_id and handles the
    # complex policy_config filtering required for proper trace hydration
    return build_rollout_response(
        request=request,
        outcome_reward=score,
        policy_config=policy_config,
        inference_url=inference_url,
        # Optional: Include trace payload if you're building one
        # trace=my_trace_dict,
    )


# =============================================================================
# CREATE THE APP
# =============================================================================

app = create_local_api(LocalAPIConfig(
    app_id=APP_ID,
    name=APP_NAME,
    description="Description of what this task evaluates",
    provide_taskset_description=provide_taskset_description,
    provide_task_instances=provide_task_instances,
    rollout=run_rollout,
    cors_origins=["*"],
))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
