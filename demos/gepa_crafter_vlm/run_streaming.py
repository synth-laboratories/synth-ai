#!/usr/bin/env python3
"""Run GEPA demo with streaming progress events."""

import asyncio
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from synth_ai.core.urls import synth_prompt_learning_events_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_user_key
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.helpers import extract_api_key
from synth_ai.sdk.task import TaskInfo, run_server_background
from synth_ai.sdk.task.contracts import (
    DatasetInfo,
    InferenceInfo,
    LimitsInfo,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskDescriptor,
)
from synth_ai.sdk.tunnels import wait_for_health_check
from synth_ai.sdk.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

# Add script directory to path and load env
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))
os.chdir(_script_dir)
load_dotenv(_script_dir.parent.parent / ".env")

# Import local module dynamically after path setup
_crafter_logic = importlib.import_module("crafter_logic")
ACTION_STRING_TO_INT = _crafter_logic.ACTION_STRING_TO_INT
CRAFTER_ALLOWED_ACTIONS = _crafter_logic.CRAFTER_ALLOWED_ACTIONS
CrafterEnvironmentWrapper = _crafter_logic.CrafterEnvironmentWrapper
CrafterScorer = _crafter_logic.CrafterScorer
CrafterVLMReActPolicy = _crafter_logic.CrafterVLMReActPolicy
normalize_action_name = _crafter_logic.normalize_action_name

# Config
SYNTH_USER_KEY = get_or_mint_synth_user_key()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
POLICY_MODEL = "gpt-4.1-nano"
EVAL_MODEL = "gpt-4o-mini"
ROLLOUT_BUDGET = 6
NUM_GENERATIONS = 1
MAX_TURNS = 20
COMPARISON_SEEDS = list(range(30))  # 30 seeds for fair comparison
COMPARISON_MAX_TURNS = 15  # Fewer turns for faster comparison


def log(msg: str) -> None:
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def create_localapi_app(system_prompt: str):
    """Create Crafter VLM task app."""
    app_id = "crafter_vlm"
    app_name = "Crafter VLM"
    tool_name = "crafter_interact"

    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        seed = request.env.seed or 0
        env_config = request.env.config or {}
        max_steps = int(env_config.get("max_steps_per_episode", 200))
        max_turns = int(env_config.get("max_turns", MAX_TURNS))

        log(f"  Rollout seed={seed} starting (max_turns={max_turns})")

        env = CrafterEnvironmentWrapper(seed=seed, max_steps=max_steps)
        observation = await env.reset()

        policy = CrafterVLMReActPolicy(
            system_prompt=system_prompt,
            use_vision=True,
            image_only_mode=True,
        )

        inference_url = policy_config.get("inference_url", "")
        if inference_url:
            os.environ["OPENAI_BASE_URL"] = inference_url

        # Use SDK helper to extract API key from headers or env (like banking77 demo)
        api_key = extract_api_key(fastapi_request, policy_config) or OPENAI_API_KEY
        if not api_key:
            raise ValueError("No API key available")
        client = AsyncOpenAI(api_key=api_key)

        history = []
        episode_rewards = []

        turns_completed = 0
        for _ in range(max_turns):
            turns_completed += 1
            messages = policy.build_messages(observation, history)

            response = await client.chat.completions.create(
                model=policy_config.get("model", POLICY_MODEL),
                messages=messages,
                tools=policy.tools,
                tool_choice="required",
                max_completion_tokens=512,
            )

            message = response.choices[0].message
            response_text = message.content or ""
            tool_calls: list[dict[str, Any]] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (message.tool_calls or [])
            ]

            next_observation = observation
            tool_responses = []

            if tool_calls:
                for tc in tool_calls:
                    tool_call_id = tc["id"]
                    actions_list = []

                    if tc["function"]["name"] == tool_name:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            raw_actions = args.get("actions_list", [])
                            actions_list = [str(a) for a in raw_actions if str(a).strip()][:5]
                        except Exception:
                            pass

                    if not actions_list:
                        actions_list = ["noop"]

                    action_results = []
                    for action_str in actions_list:
                        normalized = normalize_action_name(action_str) or "noop"
                        action = ACTION_STRING_TO_INT.get(normalized, 0)
                        next_observation = await env.step(action)
                        reward = next_observation.get("reward", 0.0)
                        episode_rewards.append(float(reward))
                        action_results.append({"action": normalized, "reward": reward})
                        if next_observation.get("terminated") or next_observation.get("truncated"):
                            break

                    tool_responses.append({"tool_call_id": tool_call_id, "results": action_results})
                    if next_observation.get("terminated") or next_observation.get("truncated"):
                        break

            history.append(
                {"role": "assistant", "content": response_text, "tool_calls": tool_calls}
            )
            for resp in tool_responses:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": resp["tool_call_id"],
                        "content": json.dumps(resp["results"]),
                    }
                )

            observation = next_observation
            if observation.get("terminated") or observation.get("truncated"):
                break

        score, details = CrafterScorer.score_episode(observation, len(episode_rewards), max_steps)
        log(f"  Rollout seed={seed} done: score={score:.3f}, turns={turns_completed}")

        return RolloutResponse(
            reward_info=RolloutMetrics(outcome_reward=score, details=details),
            trace=None,
            trace_correlation_id=policy_config.get("trace_correlation_id", ""),
        )

    def provide_taskset_description():
        return {"splits": ["train", "test"]}

    def provide_task_instances(seeds):
        for seed in seeds:
            yield TaskInfo(
                task=TaskDescriptor(id=app_id, name=app_name),
                dataset=DatasetInfo(id=app_id, split="train", index=seed),
                inference=InferenceInfo(tool=tool_name),
                limits=LimitsInfo(max_turns=MAX_TURNS),
                task_metadata={"seed": seed},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=app_id,
            name=app_name,
            description=f"{app_name} task app",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


async def run_local_rollout(system_prompt: str, seed: int, max_turns: int = 15) -> dict:
    """Run a single local rollout and return score + achievement details."""
    env = CrafterEnvironmentWrapper(seed=seed, max_steps=200)
    observation = await env.reset()

    policy = CrafterVLMReActPolicy(
        system_prompt=system_prompt,
        use_vision=True,
        image_only_mode=True,
    )

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    history = []
    episode_rewards = []

    turns_completed = 0
    for _ in range(max_turns):
        turns_completed += 1
        messages = policy.build_messages(observation, history)

        try:
            response = await client.chat.completions.create(
                model=EVAL_MODEL,
                messages=messages,
                tools=policy.tools,
                tool_choice="required",
                max_completion_tokens=512,
            )
        except Exception as e:
            log(f"    API error on seed {seed}: {e}")
            break

        message = response.choices[0].message
        response_text = message.content or ""
        tool_calls: list[dict[str, Any]] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (message.tool_calls or [])
        ]

        next_observation = observation
        tool_responses: list[dict[str, Any]] = []

        if tool_calls:
            for tc in tool_calls:
                tool_call_id = tc["id"]
                actions_list = []

                if tc["function"]["name"] == "crafter_interact":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        raw_actions = args.get("actions_list", [])
                        actions_list = [str(a) for a in raw_actions if str(a).strip()][:5]
                    except Exception:
                        pass

                if not actions_list:
                    actions_list = ["noop"]

                action_results = []
                for action_str in actions_list:
                    normalized = normalize_action_name(action_str) or "noop"
                    action = ACTION_STRING_TO_INT.get(normalized, 0)
                    next_observation = await env.step(action)
                    reward = next_observation.get("reward", 0.0)
                    episode_rewards.append(float(reward))
                    action_results.append({"action": normalized, "reward": reward})
                    if next_observation.get("terminated") or next_observation.get("truncated"):
                        break

                tool_responses.append({"tool_call_id": tool_call_id, "results": action_results})
                if next_observation.get("terminated") or next_observation.get("truncated"):
                    break

        history.append({"role": "assistant", "content": response_text, "tool_calls": tool_calls})
        for resp in tool_responses:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": resp["tool_call_id"],
                    "content": json.dumps(resp["results"]),
                }
            )

        observation = next_observation
        if observation.get("terminated") or observation.get("truncated"):
            break

    score, details = CrafterScorer.score_episode(observation, len(episode_rewards), 200)
    return {
        "seed": seed,
        "score": score,
        "details": details,
        "achievements": details.get("achievements", {}),
        "turns": turns_completed,
    }


async def run_comparison_eval(
    baseline_prompt: str, optimized_prompt: str, seeds: list[int], max_turns: int = 15
) -> dict:
    """Run baseline vs optimized comparison on same seeds."""
    log("")
    log("=" * 60)
    log(f"COMPARISON EVAL: {len(seeds)} seeds, {max_turns} turns/rollout")
    log("=" * 60)

    baseline_results = []
    optimized_results = []

    # Run baseline
    log("")
    log("Running BASELINE rollouts...")
    for i, seed in enumerate(seeds):
        result = await run_local_rollout(baseline_prompt, seed, max_turns)
        baseline_results.append(result)
        log(f"  [{i + 1}/{len(seeds)}] seed={seed}: score={result['score']:.3f}")

    # Run optimized
    log("")
    log("Running OPTIMIZED rollouts...")
    for i, seed in enumerate(seeds):
        result = await run_local_rollout(optimized_prompt, seed, max_turns)
        optimized_results.append(result)
        log(f"  [{i + 1}/{len(seeds)}] seed={seed}: score={result['score']:.3f}")

    # Compute statistics
    baseline_scores = [r["score"] for r in baseline_results]
    optimized_scores = [r["score"] for r in optimized_results]

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    optimized_mean = sum(optimized_scores) / len(optimized_scores)
    baseline_max = max(baseline_scores)
    optimized_max = max(optimized_scores)

    # Per-seed comparison
    wins = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=True) if o > b)
    ties = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=True) if o == b)
    losses = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=True) if o < b)

    # Achievement frequencies
    all_achievements = set()
    for r in baseline_results + optimized_results:
        all_achievements.update(r.get("achievements", {}).keys())

    baseline_achievement_counts = dict.fromkeys(all_achievements, 0)
    optimized_achievement_counts = dict.fromkeys(all_achievements, 0)

    for r in baseline_results:
        for a, v in r.get("achievements", {}).items():
            if v:
                baseline_achievement_counts[a] += 1

    for r in optimized_results:
        for a, v in r.get("achievements", {}).items():
            if v:
                optimized_achievement_counts[a] += 1

    # Print results
    log("")
    log("=" * 60)
    log("COMPARISON RESULTS")
    log("=" * 60)
    log("")
    log(f"{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Delta':>12}")
    log("-" * 56)
    log(
        f"{'Mean Score':<20} {baseline_mean:>12.3f} {optimized_mean:>12.3f} {optimized_mean - baseline_mean:>+12.3f}"
    )
    log(
        f"{'Max Score':<20} {baseline_max:>12.3f} {optimized_max:>12.3f} {optimized_max - baseline_max:>+12.3f}"
    )
    log("")
    log(f"Per-seed: Optimized wins {wins}/{len(seeds)}, ties {ties}, losses {losses}")
    log("")
    log(f"Achievement Frequencies (count out of {len(seeds)}):")
    log(f"{'Achievement':<25} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    log("-" * 56)

    for achievement in sorted(all_achievements):
        b_count = baseline_achievement_counts.get(achievement, 0)
        o_count = optimized_achievement_counts.get(achievement, 0)
        delta = o_count - b_count
        log(f"{achievement:<25} {b_count:>10} {o_count:>10} {delta:>+10}")

    return {
        "baseline": {
            "mean": baseline_mean,
            "max": baseline_max,
            "scores": baseline_scores,
            "achievement_counts": baseline_achievement_counts,
        },
        "optimized": {
            "mean": optimized_mean,
            "max": optimized_max,
            "scores": optimized_scores,
            "achievement_counts": optimized_achievement_counts,
        },
        "comparison": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_improvement": optimized_mean - baseline_mean,
            "max_improvement": optimized_max - baseline_max,
        },
    }


async def stream_job_events(job_id: str) -> None:
    """Stream events from GEPA job."""
    url = synth_prompt_learning_events_url(job_id)
    headers = {"Authorization": f"Bearer {SYNTH_USER_KEY}"}

    client = httpx.AsyncClient(timeout=None)
    async with client, client.stream("GET", url, headers=headers) as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                try:
                    event = json.loads(line[5:].strip())
                    event_type = event.get("type", "unknown")

                    if event_type == "gepa.generation_started":
                        gen = event.get("payload", {}).get("generation", 0)
                        log(f"Generation {gen} started")
                    elif event_type == "gepa.candidate_evaluated":
                        payload = event.get("payload", {})
                        score = payload.get("score", 0)
                        log(f"  Candidate evaluated: score={score:.3f}")
                    elif event_type == "gepa.generation_completed":
                        payload = event.get("payload", {})
                        best = payload.get("best_score", 0)
                        log(f"Generation completed: best_score={best:.3f}")
                    elif event_type == "job.completed":
                        log("Job completed!")
                        return
                    elif event_type == "job.failed":
                        log(f"Job failed: {event.get('payload', {}).get('error', 'unknown')}")
                        return
                except Exception:
                    pass


async def main() -> None:
    log("=" * 60)
    log("GEPA Crafter VLM Demo (Streaming)")
    log("=" * 60)
    log(f"Rollout budget: {ROLLOUT_BUDGET}")
    log(f"Generations: {NUM_GENERATIONS}")
    log(f"Max turns/rollout: {MAX_TURNS}")
    log("")

    # Create preliminary job to get localapi_key (SDK auto-provisions it)
    prelim_config = {
        "prompt_learning": {
            "algorithm": "gepa",
            "localapi_url": "http://localhost:8001",
            "env_name": "crafter",
            "initial_prompt": {
                "messages": [{"role": "system", "order": 0, "pattern": "placeholder"}],
                "wildcards": {},
            },
            "policy": {"model": POLICY_MODEL, "provider": "openai"},
            "gepa": {
                "env_name": "crafter",
                "evaluation": {"seeds": [0]},
                "rollout": {"budget": 1},
                "population": {"initial_size": 1, "num_generations": 1},
            },
        },
    }
    prelim_job = PromptLearningJob.from_dict(
        config_dict=prelim_config, synth_user_key=SYNTH_USER_KEY
    )
    env_key = prelim_job.config.localapi_key
    log("Environment key configured")

    # Baseline prompt
    allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
    baseline_prompt = (
        "You are an agent playing Crafter, a survival crafting game. "
        "Your goal is to survive and unlock achievements. "
        "Analyze images to understand surroundings, inventory, health, resources. "
        "Use crafter_interact tool. "
        "Key: 'do' only works adjacent to resources (tree, stone, cow, plant). "
        "Craft progression: wood -> table -> wood_pickaxe -> stone -> stone_pickaxe. "
        f"Actions: {allowed_actions}. Return 2-5 actions per decision."
    )

    # Start task app
    log("Starting task app...")
    app = create_localapi_app(baseline_prompt)
    run_server_background(app, port=8001)
    await wait_for_health_check("127.0.0.1", 8001, env_key, timeout=30.0)
    log("Task app ready on port 8001")

    # Create tunnel
    log("Creating tunnel...")
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareManagedTunnel,
        synth_user_key=SYNTH_USER_KEY,
        progress=False,
    )
    log(f"Tunnel ready: {tunnel.url}")

    # Submit GEPA job
    log("")
    log("Submitting GEPA job...")
    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "localapi_url": tunnel.url,
            "env_name": "crafter",
            "initial_prompt": {
                "messages": [{"role": "system", "order": 0, "pattern": baseline_prompt}],
                "wildcards": {},
            },
            "policy": {
                "inference_mode": "synth_hosted",
                "model": POLICY_MODEL,
                "provider": "openai",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
            "gepa": {
                "env_name": "crafter",
                "evaluation": {"seeds": list(range(15)), "validation_seeds": list(range(50, 56))},
                "rollout": {"budget": ROLLOUT_BUDGET, "max_concurrent": 3, "minibatch_size": 3},
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": NUM_GENERATIONS,
                    "children_per_generation": 2,
                },
                "archive": {"size": 5, "pareto_set_size": 10},
                "token": {"max_limit": 4000, "counting_model": "gpt-4", "max_spend_usd": 50.0},
            },
            "env": {
                "max_turns": MAX_TURNS,
                "max_steps_per_episode": 200,
            },
            "verifier": {
                "enabled": False,
                "reward_source": "localapi",
            },
        },
    }

    job = PromptLearningJob.from_dict(
        config_dict=config_body,
    )
    job_id = job.submit()
    log(f"Job submitted: {job_id}")
    log("")

    # Poll with progress
    log("Polling for completion...")
    result = job.poll_until_complete(timeout=600.0, interval=3.0, progress=True)

    log("")
    log(f"Job status: {result.status.value}")

    if result.succeeded:
        log("Extracting optimized prompt...")
        pl_client = PromptLearningClient(synth_user_key=SYNTH_USER_KEY)
        prompt_results = await pl_client.get_prompts(job_id)

        optimized = None
        if prompt_results.best_prompt:
            for msg in prompt_results.best_prompt.get("messages", []):
                if msg.get("role") == "system":
                    optimized = msg.get("pattern") or msg.get("content")
                    break

        if optimized:
            log("")
            log("=" * 60)
            log("OPTIMIZED PROMPT:")
            log("=" * 60)
            print(optimized[:500] + "..." if len(optimized) > 500 else optimized)

            Path("results").mkdir(exist_ok=True)
            with open("results/optimized_prompt.txt", "w") as f:
                f.write(optimized)
            log("Saved to results/optimized_prompt.txt")

        # Run fair comparison on same seeds (only if we have optimized prompt)
        comparison_results = None
        if optimized:
            log("")
            log("Running fair comparison eval (same seeds for both prompts)...")
            comparison_results = await run_comparison_eval(
                baseline_prompt=baseline_prompt,
                optimized_prompt=optimized,
                seeds=COMPARISON_SEEDS,
                max_turns=COMPARISON_MAX_TURNS,
            )
        else:
            log("No optimized prompt available - skipping comparison")

        # Save all results
        results_data = {
            "job_id": job_id,
            "status": result.status.value,
            "baseline_prompt": baseline_prompt,
            "optimized_prompt": optimized,
            "comparison": comparison_results,
        }
        with open("results/gepa_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        log("Saved results to results/gepa_results.json")
    else:
        log(f"Job failed: {result.error}")

    log("")
    log("Done!")


if __name__ == "__main__":
    asyncio.run(main())
