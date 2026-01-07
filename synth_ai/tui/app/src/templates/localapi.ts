/**
 * Template for new LocalAPI task apps.
 */

export const LOCALAPI_TEMPLATE = `"""
LocalAPI Task App - Define your evaluation task for Synth AI.

This file creates a task app that Synth AI uses to evaluate prompts.
The backend calls your /rollout endpoint with different seeds (test cases)
and aggregates the scores.
"""

from fastapi import Request
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutRequest,
    RolloutResponse,
    RolloutMetrics,
    TaskInfo,
)


# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_ID = "my-task"
APP_NAME = "My Evaluation Task"


# =============================================================================
# TODO: IMPLEMENT YOUR DATASET
# =============================================================================
#
# Load your evaluation dataset here. Examples:
#
#   # Option 1: HuggingFace datasets
#   from datasets import load_dataset
#   DATASET = load_dataset("your-dataset", split="test")
#
#   # Option 2: JSON file
#   import json
#   with open("data.json") as f:
#       DATASET = json.load(f)
#
#   # Option 3: Inline list
#   DATASET = [
#       {"input": "What is 2+2?", "expected": "4"},
#       {"input": "Capital of France?", "expected": "Paris"},
#   ]
#
# =============================================================================


def get_dataset_size() -> int:
    """Return the total number of samples in your dataset."""
    raise NotImplementedError(
        "Implement get_dataset_size() to return len(DATASET)"
    )


def get_sample(seed: int) -> dict:
    """
    Get a test case by seed index.

    Args:
        seed: The seed/index for this evaluation (from request.env.seed)

    Returns:
        Dict with your test case fields (e.g. {"input": ..., "expected": ...})
    """
    raise NotImplementedError(
        "Implement get_sample() to return a test case for the given seed.\\n"
        "Example: return DATASET[seed % len(DATASET)]"
    )


# =============================================================================
# TODO: IMPLEMENT YOUR SCORING LOGIC
# =============================================================================


def score_response(response: str, sample: dict) -> float:
    """
    Score the model response. Returns 0.0 to 1.0.

    Args:
        response: The model's response text
        sample: The test case dict from get_sample()

    Returns:
        Score between 0.0 (wrong) and 1.0 (correct)
    """
    raise NotImplementedError(
        "Implement score_response() to score the model output.\\n"
        "Example: return 1.0 if sample['expected'] in response else 0.0"
    )


# =============================================================================
# TASK APP PROVIDERS (required by Synth backend)
# =============================================================================


def provide_taskset_description() -> dict:
    """Return metadata about your task set (splits, sizes, etc.)."""
    return {
        "splits": ["default"],
        "sizes": {"default": get_dataset_size()},
    }


def provide_task_instances(seeds: list[int]):
    """Yield TaskInfo for each seed. Called by Synth to get task metadata."""
    for seed in seeds:
        sample = get_sample(seed)
        yield TaskInfo(
            task={"id": APP_ID, "name": APP_NAME},
            dataset={"id": APP_ID, "split": "default", "index": seed % get_dataset_size()},
            inference={},
            limits={"max_turns": 1},
            task_metadata=sample,
        )


# =============================================================================
# LLM CALL HELPER
# =============================================================================

async def call_llm(prompt: str, inference_url: str, api_key: str | None = None) -> str:
    """Call the LLM via the inference URL provided by Synth."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "model": "gpt-4.1-nano",  # Or get from policy_config
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(inference_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# =============================================================================
# ROLLOUT HANDLER
# =============================================================================

async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """
    Handle a single evaluation rollout.

    Args:
        request: Contains seed, policy config, env config
        fastapi_request: The FastAPI request object

    Returns:
        RolloutResponse with the evaluation score
    """
    # Get test case for this seed
    seed = request.env.seed
    sample = get_sample(seed)

    # Get inference URL from policy config (Synth provides this)
    policy_config = request.policy.config or {}
    inference_url = policy_config.get("inference_url")

    if not inference_url:
        raise ValueError("No inference_url provided in policy config")

    # Call the LLM
    response = await call_llm(
        prompt=sample["input"],
        inference_url=inference_url,
        api_key=policy_config.get("api_key"),
    )

    # Score the response
    score = score_response(response, sample)

    return RolloutResponse(
        run_id=request.run_id,
        metrics=RolloutMetrics(outcome_reward=score),
        trace=None,
        trace_correlation_id=None,
        inference_url=inference_url,
    )


# =============================================================================
# CREATE THE APP
# =============================================================================

app = create_local_api(LocalAPIConfig(
    app_id=APP_ID,
    name=APP_NAME,
    description="",
    provide_taskset_description=provide_taskset_description,
    provide_task_instances=provide_task_instances,
    rollout=run_rollout,
    cors_origins=["*"],
))


# =============================================================================
# RUNNING LOCALLY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
`
