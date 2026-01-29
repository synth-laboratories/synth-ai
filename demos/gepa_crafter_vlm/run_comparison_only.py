#!/usr/bin/env python3
"""Run fair comparison between baseline and optimized prompt (no GEPA)."""

import asyncio
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client

# Setup - add directory to path and change to it for local imports
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
os.chdir(_THIS_DIR)

# Import local module dynamically after path setup
_crafter_logic = importlib.import_module("crafter_logic")
ACTION_STRING_TO_INT = _crafter_logic.ACTION_STRING_TO_INT
CRAFTER_ALLOWED_ACTIONS = _crafter_logic.CRAFTER_ALLOWED_ACTIONS
CrafterEnvironmentWrapper = _crafter_logic.CrafterEnvironmentWrapper
CrafterScorer = _crafter_logic.CrafterScorer
CrafterVLMReActPolicy = _crafter_logic.CrafterVLMReActPolicy
normalize_action_name = _crafter_logic.normalize_action_name

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EVAL_MODEL = "gpt-4o-mini"
COMPARISON_SEEDS = list(range(30))
COMPARISON_MAX_TURNS = 15


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Prompts
allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
BASELINE_PROMPT = (
    "You are an agent playing Crafter, a survival crafting game. "
    "Your goal is to survive and unlock achievements. "
    "Analyze images to understand surroundings, inventory, health, resources. "
    "Use crafter_interact tool. "
    "Key: 'do' only works adjacent to resources (tree, stone, cow, plant). "
    "Craft progression: wood -> table -> wood_pickaxe -> stone -> stone_pickaxe. "
    f"Actions: {allowed_actions}. Return 2-5 actions per decision."
)

# Load optimized prompt
optimized_path = Path("results/optimized_prompt.txt")
if optimized_path.exists():
    OPTIMIZED_PROMPT = optimized_path.read_text().strip()
    log(f"Loaded optimized prompt ({len(OPTIMIZED_PROMPT)} chars)")
else:
    raise FileNotFoundError("No optimized prompt found - run run_streaming.py first")


async def run_local_rollout(system_prompt: str, seed: int, max_turns: int = 15) -> dict:
    """Run a single local rollout and return score + achievement details."""
    env = CrafterEnvironmentWrapper(seed=seed, max_steps=200)
    observation = await env.reset()

    policy = CrafterVLMReActPolicy(
        system_prompt=system_prompt,
        use_vision=True,
        image_only_mode=True,
    )

    client = AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=get_shared_http_client())
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
        tool_calls = [
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


async def main():
    log("=" * 60)
    log(f"COMPARISON EVAL: {len(COMPARISON_SEEDS)} seeds, {COMPARISON_MAX_TURNS} turns/rollout")
    log("=" * 60)

    baseline_results = []
    optimized_results = []

    # Run baseline
    log("")
    log("Running BASELINE rollouts...")
    for i, seed in enumerate(COMPARISON_SEEDS):
        result = await run_local_rollout(BASELINE_PROMPT, seed, COMPARISON_MAX_TURNS)
        baseline_results.append(result)
        log(f"  [{i + 1}/{len(COMPARISON_SEEDS)}] seed={seed}: score={result['score']:.3f}")

    # Run optimized
    log("")
    log("Running OPTIMIZED rollouts...")
    for i, seed in enumerate(COMPARISON_SEEDS):
        result = await run_local_rollout(OPTIMIZED_PROMPT, seed, COMPARISON_MAX_TURNS)
        optimized_results.append(result)
        log(f"  [{i + 1}/{len(COMPARISON_SEEDS)}] seed={seed}: score={result['score']:.3f}")

    # Compute statistics
    baseline_scores = [r["score"] for r in baseline_results]
    optimized_scores = [r["score"] for r in optimized_results]

    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    optimized_mean = sum(optimized_scores) / len(optimized_scores)
    baseline_max = max(baseline_scores)
    optimized_max = max(optimized_scores)

    # Per-seed comparison
    wins = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=False) if o > b)
    ties = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=False) if o == b)
    losses = sum(1 for b, o in zip(baseline_scores, optimized_scores, strict=False) if o < b)

    # Achievement frequencies (achievements is a list of achieved names)
    all_achievements = set()
    for r in baseline_results + optimized_results:
        achievements = r.get("achievements", [])
        if isinstance(achievements, list):
            all_achievements.update(achievements)
        elif isinstance(achievements, dict):
            all_achievements.update(k for k, v in achievements.items() if v)

    baseline_achievement_counts = dict.fromkeys(all_achievements, 0)
    optimized_achievement_counts = dict.fromkeys(all_achievements, 0)

    for r in baseline_results:
        achievements = r.get("achievements", [])
        if isinstance(achievements, list):
            for a in achievements:
                if a in baseline_achievement_counts:
                    baseline_achievement_counts[a] += 1
        elif isinstance(achievements, dict):
            for a, v in achievements.items():
                if v and a in baseline_achievement_counts:
                    baseline_achievement_counts[a] += 1

    for r in optimized_results:
        achievements = r.get("achievements", [])
        if isinstance(achievements, list):
            for a in achievements:
                if a in optimized_achievement_counts:
                    optimized_achievement_counts[a] += 1
        elif isinstance(achievements, dict):
            for a, v in achievements.items():
                if v and a in optimized_achievement_counts:
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
    log(f"Per-seed: Optimized wins {wins}/{len(COMPARISON_SEEDS)}, ties {ties}, losses {losses}")
    log("")
    log(f"Achievement Frequencies (count out of {len(COMPARISON_SEEDS)}):")
    log(f"{'Achievement':<25} {'Baseline':>10} {'Optimized':>10} {'Delta':>10}")
    log("-" * 56)

    for achievement in sorted(all_achievements):
        b_count = baseline_achievement_counts.get(achievement, 0)
        o_count = optimized_achievement_counts.get(achievement, 0)
        delta = o_count - b_count
        log(f"{achievement:<25} {b_count:>10} {o_count:>10} {delta:>+10}")

    # Save results
    comparison_results = {
        "baseline": {
            "prompt": BASELINE_PROMPT,
            "mean": baseline_mean,
            "max": baseline_max,
            "scores": baseline_scores,
            "achievement_counts": baseline_achievement_counts,
        },
        "optimized": {
            "prompt": OPTIMIZED_PROMPT,
            "mean": optimized_mean,
            "max": optimized_max,
            "scores": optimized_scores,
            "achievement_counts": optimized_achievement_counts,
        },
        "comparison": {
            "seeds": COMPARISON_SEEDS,
            "max_turns": COMPARISON_MAX_TURNS,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "mean_improvement": optimized_mean - baseline_mean,
            "max_improvement": optimized_max - baseline_max,
        },
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/comparison_results.json", "w") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    log("Saved to results/comparison_results.json")

    log("")
    log("Done!")


if __name__ == "__main__":
    asyncio.run(main())
