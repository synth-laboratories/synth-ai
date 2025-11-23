#!/usr/bin/env python3
"""Run MIPRO locally on Banking77 with hosted judge scoring (no task-app reward).

Usage (example):
    uv run python run_mipro_judges_local.py \\
      --task-app-url http://127.0.0.1:8104 \\
      --backend-url http://localhost:8000 \\
      --rollout-budget 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional
import tempfile

from synth_ai.api.train.prompt_learning import PromptLearningJob


def build_toml(
    *,
    task_app_url: str,
    rollout_budget: int,
    backend_base: str,
    policy_model: str,
    meta_model: str,
    judge_model: str,
        bootstrap_seeds: list[int],
        online_pool: list[int],
        test_pool: list[int],
        val_seeds: list[int],
        reference_pool: list[int],
    judge_rubric_id: str,
    reward_source: str,
    judge_only: bool,
) -> str:
    """Create TOML content with judge-only reward."""
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not task_app_api_key:
        raise RuntimeError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")

    # Auto-scale iterations based on budget
    num_iterations = max(2, min(5, rollout_budget // 10))
    num_evaluations_per_iteration = max(2, min(5, rollout_budget // (num_iterations * 2)))
    batch_size = max(5, min(20, rollout_budget // num_iterations))

    toml_lines: list[str] = [
        "[prompt_learning]",
        'algorithm = "mipro"',
        f'task_app_url = "{task_app_url}"',
        f'task_app_api_key = "{task_app_api_key}"',
        "",
        "[prompt_learning.initial_prompt]",
        'id = "banking77_pattern"',
        'name = "Banking77 Judge Demo"',
        "",
        "[[prompt_learning.initial_prompt.messages]]",
        'role = "system"',
        (
            'pattern = "You are an expert banking assistant that classifies customer queries into '
            'banking intents. Given a customer message, respond with exactly one intent label from '
            'the provided list using the `banking77_classify` tool."'
        ),
        "order = 0",
        "",
        "[[prompt_learning.initial_prompt.messages]]",
        'role = "user"',
        (
            'pattern = "Customer Query: {query}\\n\\nAvailable Intents:\\n'
            '{available_intents}\\n\\nClassify this query into one of the above banking '
            'intents using the tool call."'
        ),
        "order = 1",
        "",
        "[prompt_learning.initial_prompt.wildcards]",
        'query = "REQUIRED"',
        'available_intents = "REQUIRED"',
        "",
        "[prompt_learning.policy]",
        'inference_mode = "synth_hosted"',
        f'provider = "groq"',
        f'model = "{policy_model}"',
        "temperature = 0.0",
        "max_completion_tokens = 256",
        "",
        "[prompt_learning.mipro]",
        'env_name = "banking77"',
        f"num_iterations = {num_iterations}",
        f"num_evaluations_per_iteration = {num_evaluations_per_iteration}",
        f"batch_size = {batch_size}",
        "max_concurrent = 5",
        f'meta_model = "{meta_model}"',
        'meta_model_provider = "groq"',
        'meta_model_inference_url = "https://api.groq.com"',
        "meta_model_temperature = 0.7",
        "meta_model_max_tokens = 512",
        "few_shot_score_threshold = 0.8",
        "max_instructions = 10",
        "max_demo_set_size = 20",
        "instructions_per_batch = 30",
        f"bootstrap_train_seeds = {bootstrap_seeds}",
        f"online_pool = {online_pool}",
        f"test_pool = {test_pool}",
        f"val_seeds = {val_seeds}",  # Validation seeds for top-K evaluation
        f"reference_pool = {reference_pool}",  # Reference pool for meta-prompt context
        "",
        "[prompt_learning.judge]",
        "enabled = true",
        f'reward_source = "{reward_source}"',  # task_app | judge | fused
        # BackendJudgeClient expects base_url without /api, it adds /judge/v1/score
        # The router is at /api/judge/v1, so base_url should be http://host:port/api
        f'backend_base = "{backend_base.rstrip("/")}/api"',
        'backend_api_key_env = "SYNTH_API_KEY"',
        # Determine provider from judge model
        f'backend_provider = "{("groq" if "gpt-oss" in judge_model or "qwen" in judge_model else "google" if "gemini" in judge_model else "openai")}"',
        f'backend_model = "{judge_model}"',
        f'backend_rubric_id = "{judge_rubric_id}"',
        "backend_event_enabled = true",
        "backend_outcome_enabled = true",
        "concurrency = 4",
        "timeout = 60.0",
        "",
        "[prompt_learning.termination_config]",
        f"max_cost_usd = {max(0.10, rollout_budget * 0.01):.2f}",
        f"max_trials = {rollout_budget * 2}",
        f"max_rollouts = {rollout_budget * 10}",
    ]
    return "\n".join(toml_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MIPRO + judge run for Banking77")
    parser.add_argument("--task-app-url", required=True, help="Running task app URL (e.g., http://127.0.0.1:8104)")
    parser.add_argument("--backend-url", required=True, help="Backend base URL (e.g., http://localhost:8000)")
    parser.add_argument("--rollout-budget", type=int, default=20, help="Rollout budget (default: 20)")
    parser.add_argument("--judge-rubric-id", default="banking77-rubric-v1", help="Judge rubric id")
    parser.add_argument(
        "--reward-source",
        choices=["task_app", "judge", "fused"],
        default="judge",
        help="Reward source: task_app, judge, or fused (default: judge)",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Force judge-only scoring (sets reward_source=judge, weight_env=0)",
    )
    parser.add_argument(
        "--policy-model",
        default=os.getenv("POLICY_MODEL", "llama-3.1-8b-instant"),
        help="Policy model (default: env POLICY_MODEL or llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--meta-model",
        default=os.getenv("META_MODEL", "llama-3.3-70b-versatile"),
        help="Meta model for instruction generation (default: env META_MODEL or llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"),
        help="Judge model - must be one of: gemini-2.5-flash, gpt-5, gpt-5-mini, gpt-5-nano, qwen/qwen3-32b, openai/gpt-oss-20b, openai/gpt-oss-120b (default: env JUDGE_MODEL or openai/gpt-oss-120b)",
    )
    args = parser.parse_args()

    # Simple seed pools for a quick hello-world run
    bootstrap_seeds = list(range(0, 5))
    online_pool = list(range(5, 15))
    val_seeds = list(range(15, 20))  # Validation seeds for top-K evaluation
    test_pool = list(range(20, 25))  # Test pool for final evaluation
    reference_pool = list(range(25, 30))  # Reference pool for meta-prompt context (must not overlap with other pools)

    # Validate judge model is in allowed list
    ALLOWED_JUDGE_MODELS = {
        "gemini-2.5-flash",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    }
    judge_model_lower = args.judge_model.lower().strip()
    if judge_model_lower not in {m.lower() for m in ALLOWED_JUDGE_MODELS}:
        raise ValueError(
            f"Judge model '{args.judge_model}' is not supported. "
            f"Allowed models: {', '.join(sorted(ALLOWED_JUDGE_MODELS))}"
        )
    
    toml_content = build_toml(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        backend_base=args.backend_url,
        policy_model=args.policy_model,
        meta_model=args.meta_model,
        judge_model=args.judge_model,
        bootstrap_seeds=bootstrap_seeds,
        online_pool=online_pool,
        test_pool=test_pool,
        val_seeds=val_seeds,
        reference_pool=reference_pool,
        judge_rubric_id=args.judge_rubric_id,
        reward_source="judge" if args.judge_only else args.reward_source,
        judge_only=args.judge_only,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
        tmp.write(toml_content)
        tmp_path = Path(tmp.name)

    job = PromptLearningJob.from_config(
        config_path=tmp_path,
        backend_url=args.backend_url,
        api_key=os.getenv("SYNTH_API_KEY"),
    )
    job_id = job.submit()
    print(f"Submitted job: {job_id}")
    result = job.poll_until_complete(timeout=1800.0, interval=5.0)
    # High-level summary (avoid verbose dumps)
    best_prompt = None
    best_score = result.get("best_score")
    best_validation = result.get("best_validation_score") or result.get("judge_reward")
    try:
        best_prompt = job.get_best_prompt_text(rank=1)
    except Exception:
        best_prompt = None

    # Fetch prompt results for judge visibility
    judge_score = None
    try:
        results = job.get_results()
        # Extract judge_score from top-level results (stored separately from best_score)
        judge_score = results.get("judge_score")
        # Fallback: try validation_results if judge_score not available
        if judge_score is None:
            validation_results = results.get("validation_results") or []
            if validation_results:
                judge_score = validation_results[0].get("score") if isinstance(validation_results[0], dict) else None
        # override bests if available from results
        best_score = results.get("best_score") or best_score
        if not best_prompt:
            best_prompt = results.get("best_prompt")
    except Exception:
        pass

    print(
        "Outcome:",
        {
            "status": result.get("status"),
            "best_score": best_score,
            "judge_score": judge_score or best_validation,
            "best_prompt": best_prompt,
        },
    )


if __name__ == "__main__":
    main()

