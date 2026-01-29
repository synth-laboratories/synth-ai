#!/usr/bin/env python3
"""Run RLM v2 verifier on OOLONG-style large context benchmark.

This demo tests the RLM v2 verifier's ability to handle large context
retrieval tasks. RLM v2 includes enhanced features:
- AgentFS integration for file operations
- Multi-agent coordination for parallel reasoning
- Enhanced message summarization

Usage:
    uv run python demos/rlm/run_rlm_v2_oolong.py           # Production mode
    uv run python demos/rlm/run_rlm_v2_oolong.py --local   # Local mode
    uv run python demos/rlm/run_rlm_v2_oolong.py --local --context-size 50000
"""

import argparse
import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Parse args first to set up environment
parser = argparse.ArgumentParser(description="Run RLM v2 OOLONG benchmark")
parser.add_argument(
    "--local",
    action="store_true",
    help="Run against localhost:8000 instead of production",
)
parser.add_argument(
    "--context-size",
    type=int,
    default=100_000,
    help="Size of OOLONG context in characters (default: 100000)",
)
parser.add_argument(
    "--num-questions",
    type=int,
    default=3,
    help="Number of questions to test (default: 3)",
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4.1-mini",
    help="Model to use for RLM (default: gpt-4.1-mini)",
)
parser.add_argument(
    "--parallel",
    action="store_true",
    help="Run questions in parallel (default: sequential)",
)
args = parser.parse_args()

# SDK imports after env setup
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.graphs.completions import GraphCompletionsAsyncClient

# Configuration
LOCAL_MODE = args.local
BACKEND_URL = "http://localhost:8000" if LOCAL_MODE else PROD_BASE_URL
VERIFIER_GRAPH_ID = "zero_shot_verifier_rubric_rlm_v2"  # RLM v2
VERIFIER_MODEL = args.model
CONTEXT_SIZE = args.context_size
NUM_QUESTIONS = args.num_questions
PARALLEL_MODE = args.parallel

# RLM v2 limits (with v2-specific options)
RLM_LIMITS = {
    "max_iterations": 1000,
    "max_root_calls": 1000,
    "max_subcalls": 5000,
    "max_time_ms": 600_000,  # 10 minutes
    "max_cost_usd": 1.0,
    "max_messages_before_summary": 30,  # v2-specific
}


def create_oolong_context(size_chars: int = 100_000) -> str:
    """Create OOLONG-style large context with hidden answers.

    The context contains financial data buried in filler text,
    testing the RLM's ability to search and retrieve specific information.
    """
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    ) * 10

    answer_section = """

=== Q3 2024 FINANCIAL REPORT ===
Revenue Summary:
- Total quarterly revenue: $4.2 billion
- Operating expenses: $2.1 billion
- Net income: $1.5 billion (15% YoY increase)
- Gross margin: 42.3%

Regional Breakdown:
- North America: $2.1B (50%)
- Europe: $1.26B (30%)
- Asia Pacific: $840M (20%)

Key Metrics:
- Customer acquisition cost: $125
- Monthly active users: 45 million
- Employee count: 12,500
=== END FINANCIAL REPORT ===

"""

    # Build context with answer buried at ~40% point
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


# Test questions and ground truth
OOLONG_QUESTIONS = [
    {
        "question": "What was the Q3 2024 quarterly revenue?",
        "gold_answer": "$4.2 billion",
    },
    {
        "question": "What were the operating expenses in Q3 2024?",
        "gold_answer": "$2.1 billion",
    },
    {
        "question": "What was the gross margin percentage?",
        "gold_answer": "42.3%",
    },
    {
        "question": "What was the net income and its YoY change?",
        "gold_answer": "$1.5 billion (15% YoY increase)",
    },
    {
        "question": "How many monthly active users does the company have?",
        "gold_answer": "45 million",
    },
]


def get_rubric_for_oolong() -> str:
    """Generate rubric for OOLONG question answering."""
    return """
task_description = "Answer questions by finding information in a large document"

[[event]]
id = "search_strategy"
description = "Agent used effective search tools (grep, view_lines) to locate relevant sections"
weight = 0.4

[[event]]
id = "evidence_gathering"
description = "Agent gathered appropriate evidence before answering"
weight = 0.3

[[outcome]]
id = "answer_accuracy"
description = "Answer is accurate and matches the information in the document"
weight = 1.0
"""


async def run_rlm_v2_on_question(
    client: GraphCompletionsAsyncClient,
    context: str,
    question: str,
    rubric: str,
    question_idx: int = 0,
) -> Dict[str, Any]:
    """Run RLM v2 verifier on a single question."""
    start_time = time.time()

    # Create trace in SessionTraceInput format (with session_id and event_history)
    trace_data = {
        "session_id": f"oolong-v2-{int(time.time())}-{question_idx}",
        "metadata": {
            "task_type": "question_answering",
            "context_size": len(context),
            "question_idx": question_idx,
        },
        "event_history": [
            {
                "type": "user_message",
                "content": f"Context document:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        "markov_blanket_message_history": [],
    }

    try:
        input_data = {
            "trace": trace_data,
            "rubric": rubric,
            "query": f"Evaluate the agent's ability to answer: {question}",
            "options": {
                "rlm_limits": RLM_LIMITS,
                "timeout_s": RLM_LIMITS["max_time_ms"] / 1000 + 60,
            },
        }

        result = await client.run(
            input_data=input_data,
            job_id=VERIFIER_GRAPH_ID,
            model=VERIFIER_MODEL,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Extract results
        output = result.get("output", {}) if isinstance(result, dict) else {}
        reward = None
        error = None
        rlm_stats = {}

        if isinstance(output, dict):
            # Extract RLM stats
            rlm_stats = output.get("rlm_stats", {})

            # Check direct outcome_review
            outcome_review = output.get("outcome_review", {})
            if isinstance(outcome_review, dict):
                reward = outcome_review.get("total")

            # Check inside answer wrapper
            if reward is None and "answer" in output:
                answer = output.get("answer", {})
                if isinstance(answer, dict):
                    outcome_review = answer.get("outcome_review", {})
                    if isinstance(outcome_review, dict):
                        reward = outcome_review.get("total")

            # Check for error
            error = output.get("error") or result.get("error")

        return {
            "question": question,
            "reward": reward,
            "elapsed_ms": elapsed_ms,
            "error": str(error)[:200] if error else None,
            "output": output,
            "rlm_stats": rlm_stats,
        }

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "question": question,
            "reward": None,
            "elapsed_ms": elapsed_ms,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "output": None,
            "rlm_stats": {},
        }


async def main():
    print("=" * 70)
    print("RLM V2 OOLONG BENCHMARK")
    print("=" * 70)
    print(f"Backend: {BACKEND_URL}")
    print(f"Model: {VERIFIER_MODEL}")
    print(f"Context size: {CONTEXT_SIZE:,} chars")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Execution: {'parallel' if PARALLEL_MODE else 'sequential'}")
    print()

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key and LOCAL_MODE:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=BACKEND_URL)

    if not api_key:
        print("ERROR: No SYNTH_API_KEY set and not in local mode")
        return

    # Create context
    print("Creating OOLONG context...")
    context = create_oolong_context(CONTEXT_SIZE)
    print(f"  Context created: {len(context):,} chars (~{len(context)//4:,} tokens)")

    # Get rubric
    rubric = get_rubric_for_oolong()

    # Select questions
    questions = OOLONG_QUESTIONS[:NUM_QUESTIONS]

    # Run verifier on each question
    client = GraphCompletionsAsyncClient(BACKEND_URL, api_key, timeout=900.0)

    print(f"\nRunning RLM v2 verifier on {len(questions)} questions...")
    print("-" * 70)

    total_start = time.time()

    if PARALLEL_MODE:
        # Run all questions in parallel
        tasks = [
            run_rlm_v2_on_question(client, context, q["question"], rubric, i)
            for i, q in enumerate(questions)
        ]
        results = await asyncio.gather(*tasks)
        for r, q in zip(results, questions):
            r["gold_answer"] = q["gold_answer"]
            print(f"\n  {q['question'][:50]}...")
            print(f"    Reward: {r['reward']}, Time: {r['elapsed_ms']:.0f}ms")
    else:
        # Run sequentially
        results = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {q['question']}")
            result = await run_rlm_v2_on_question(
                client, context, q["question"], rubric, i
            )
            result["gold_answer"] = q["gold_answer"]
            results.append(result)

            status = (
                "OK"
                if result["reward"] is not None
                else f"ERROR: {result['error'][:50]}"
            )
            print(f"  Reward: {result['reward']}")
            print(f"  Time: {result['elapsed_ms']:.0f}ms")
            if result["rlm_stats"]:
                cost = result["rlm_stats"].get("cost_usd", 0)
                print(f"  Cost: ${cost:.4f}")
            print(f"  Status: {status}")

    total_elapsed = (time.time() - total_start) * 1000

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    valid_rewards = [r["reward"] for r in results if r["reward"] is not None]
    errors = [r for r in results if r["error"]]
    total_cost = sum(
        r.get("rlm_stats", {}).get("cost_usd", 0)
        for r in results
        if r.get("rlm_stats")
    )

    print(f"Questions tested: {len(results)}")
    print(f"Successful: {len(valid_rewards)}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_elapsed:.0f}ms")
    print(f"Avg time per question: {total_elapsed/len(results):.0f}ms")
    print(f"Total cost: ${total_cost:.4f}")

    if valid_rewards:
        avg_reward = sum(valid_rewards) / len(valid_rewards)
        print(f"\nAverage reward: {avg_reward:.4f}")

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  - {r['question'][:40]}...: {r['error'][:60]}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
