"""Integration tests for RLM v1 (zero_shot_verifier_rubric_rlm).

These tests verify that the RLM v1 verifier graph works correctly against
the dev backend. They use small context sizes to keep tests fast.

Environment variables:
    DEV_BACKEND_URL: Backend URL (from CI secrets)
    DEV_ACTIONS_SYNTH_API_KEY: API key (from CI secrets)
"""

import os
import time

import pytest

# Skip if required env vars not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("DEV_BACKEND_URL") or not os.environ.get("DEV_ACTIONS_SYNTH_API_KEY"),
    reason="DEV_BACKEND_URL and DEV_ACTIONS_SYNTH_API_KEY required",
)


@pytest.fixture
def backend_url() -> str:
    return os.environ["DEV_BACKEND_URL"]


@pytest.fixture
def api_key() -> str:
    return os.environ["DEV_ACTIONS_SYNTH_API_KEY"]


@pytest.fixture
def graph_client(backend_url: str, api_key: str):
    """Create GraphCompletionsAsyncClient."""
    from synth_ai.sdk.graphs.completions import GraphCompletionsAsyncClient

    return GraphCompletionsAsyncClient(backend_url, api_key, timeout=300.0)


def create_test_context(size_chars: int = 5_000) -> str:
    """Create a small test context with hidden answer for fast testing."""
    filler = "Lorem ipsum dolor sit amet. " * 10

    answer_section = """
=== Q3 2024 REPORT ===
Revenue: $4.2 billion
Net income: $1.5 billion
=== END REPORT ===
"""

    parts = []
    current = 0

    while current < size_chars * 0.4:
        parts.append(filler)
        current += len(filler)

    parts.append(answer_section)
    current += len(answer_section)

    while current < size_chars:
        parts.append(filler)
        current += len(filler)

    return "".join(parts)


def get_test_rubric() -> str:
    """Simple rubric for testing."""
    return """
task_description = "Answer a question from a document"

[[outcome]]
id = "answer_accuracy"
description = "Answer is accurate"
weight = 1.0
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rlm_v1_basic_completion(graph_client):
    """Test that RLM v1 can process a basic query and return a reward."""
    context = create_test_context(size_chars=5_000)
    rubric = get_test_rubric()
    question = "What was the Q3 2024 revenue?"

    trace_data = {
        "session_id": f"test-rlm-v1-{int(time.time())}",
        "metadata": {"test": True},
        "event_history": [
            {
                "type": "user_message",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        "markov_blanket_message_history": [],
    }

    input_data = {
        "trace": trace_data,
        "rubric": rubric,
        "query": f"Evaluate: {question}",
        "options": {
            "rlm_limits": {
                "max_iterations": 100,
                "max_root_calls": 50,
                "max_subcalls": 200,
                "max_time_ms": 60_000,
                "max_cost_usd": 0.10,
            },
            "timeout_s": 120,
        },
    }

    result = await graph_client.run(
        input_data=input_data,
        job_id="zero_shot_verifier_rubric_rlm",
        model="gpt-4.1-mini",
    )

    # Verify we got a response
    assert result is not None
    assert isinstance(result, dict)

    # Check for output
    output = result.get("output", {})
    assert output is not None

    # Verify no error
    error = output.get("error") if isinstance(output, dict) else None
    assert error is None, f"RLM returned error: {error}"

    # Try to extract reward (may be nested in different locations)
    reward = None
    if isinstance(output, dict):
        outcome_review = output.get("outcome_review", {})
        if isinstance(outcome_review, dict):
            reward = outcome_review.get("total")

        if reward is None and "answer" in output:
            answer = output.get("answer", {})
            if isinstance(answer, dict):
                outcome_review = answer.get("outcome_review", {})
                if isinstance(outcome_review, dict):
                    reward = outcome_review.get("total")

    # Reward should be a number (can be 0)
    assert reward is not None, f"No reward found in output: {output}"
    assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rlm_v1_handles_empty_context(graph_client):
    """Test that RLM v1 handles empty/minimal context gracefully."""
    rubric = get_test_rubric()

    trace_data = {
        "session_id": f"test-rlm-v1-empty-{int(time.time())}",
        "metadata": {"test": True},
        "event_history": [
            {
                "type": "user_message",
                "content": "No context provided. Question: What is 2+2?",
            },
        ],
        "markov_blanket_message_history": [],
    }

    input_data = {
        "trace": trace_data,
        "rubric": rubric,
        "query": "Evaluate the answer",
        "options": {
            "rlm_limits": {
                "max_iterations": 50,
                "max_root_calls": 25,
                "max_subcalls": 100,
                "max_time_ms": 30_000,
                "max_cost_usd": 0.05,
            },
            "timeout_s": 60,
        },
    }

    result = await graph_client.run(
        input_data=input_data,
        job_id="zero_shot_verifier_rubric_rlm",
        model="gpt-4.1-mini",
    )

    # Should complete without crashing
    assert result is not None
    assert isinstance(result, dict)
