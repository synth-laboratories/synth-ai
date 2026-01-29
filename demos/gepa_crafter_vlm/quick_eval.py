#!/usr/bin/env python3
"""Quick eval comparison: baseline vs optimized prompt (skips GEPA)."""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.auth import get_or_mint_synth_api_key
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi._impl.http_pool import get_shared_http_client
from synth_ai.sdk.task import TaskInfo, run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse
from synth_ai.sdk.tunnels import wait_for_health_check
from synth_ai.sdk.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

# Add this directory to path and change to it for local imports
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
SYNTH_API_BASE = BACKEND_URL_BASE
SYNTH_API_KEY = get_or_mint_synth_api_key(backend_url=SYNTH_API_BASE)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EVAL_MODEL = "gpt-4o-mini"
EVAL_SEEDS = [100, 101, 102]  # Just 3 seeds for speed
MAX_TURNS = 10  # Even fewer turns per rollout

# Prompts
allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
BASELINE_PROMPT = (
    "You are an agent playing Crafter, a survival crafting game. "
    "Your goal is to survive and unlock achievements by exploring, crafting, and building. "
    "You can see the game state through images. Analyze each image carefully. "
    "Use the crafter_interact tool to execute actions. "
    f"Available actions: {allowed_actions}. "
    "Return 2-5 actions per decision."
)

# Load optimized prompt from previous run
optimized_path = Path("results/optimized_prompt.txt")
if optimized_path.exists():
    OPTIMIZED_PROMPT = optimized_path.read_text().strip()
    print(f"Loaded optimized prompt ({len(OPTIMIZED_PROMPT)} chars)")
else:
    print("No optimized prompt found - will compare baseline only")
    OPTIMIZED_PROMPT = None


def create_task_app(system_prompt: str):
    """Create a Crafter VLM task app."""
    app_id = "crafter_vlm"
    app_name = "Crafter VLM"
    tool_name = "crafter_interact"

    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        seed = request.env.seed or 0
        env_config = request.env.config or {}
        max_steps = int(env_config.get("max_steps_per_episode", 200))
        max_turns = int(env_config.get("max_turns", MAX_TURNS))

        env = CrafterEnvironmentWrapper(seed=seed, max_steps=max_steps)
        observation = await env.reset()

        policy = CrafterVLMReActPolicy(
            system_prompt=system_prompt,
            use_vision=True,
            image_only_mode=True,
        )

        api_key = policy_config.get("api_key") or OPENAI_API_KEY
        client = AsyncOpenAI(api_key=api_key, http_client=get_shared_http_client())

        history = []
        episode_rewards = []

        for _turn in range(max_turns):
            messages = policy.build_messages(observation, history)

            response = await client.chat.completions.create(
                model=policy_config.get("model", EVAL_MODEL),
                messages=messages,
                tools=policy.tools,
                tool_choice="required",
                max_completion_tokens=256,
            )

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
        return RolloutResponse(
            run_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=score, details=details),
            trace=None,
            trace_correlation_id=policy_config.get("trace_correlation_id"),
        )

    def provide_taskset_description():
        return {"splits": ["train", "test"]}

    def provide_task_instances(seeds):
        for seed in seeds:
            yield TaskInfo(
                task={"id": app_id, "name": app_name},
                dataset={"id": app_id, "split": "train", "index": seed},
                inference={"tool": tool_name},
                limits={"max_turns": MAX_TURNS},
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


def run_eval(local_api_url: str, seeds: list[int], mode: str):
    """Run eval job."""
    config = EvalJobConfig(
        task_app_url=local_api_url,
        backend_url=SYNTH_API_BASE,
        api_key=SYNTH_API_KEY,
        env_name="crafter",
        seeds=seeds,
        policy_config={
            "model": EVAL_MODEL,
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
        },
        env_config={
            "max_steps_per_episode": 200,
            "max_turns": MAX_TURNS,
        },
        concurrency=3,
    )
    job = EvalJob(config)
    job_id = job.submit()
    print(f"  {mode} eval: {job_id}")
    return job.poll_until_complete(timeout=300.0, interval=2.0, progress=True)


async def main():
    print("=" * 50)
    print("Quick Eval: Baseline vs Optimized Prompt")
    print("=" * 50)
    print(f"Seeds: {EVAL_SEEDS}")
    print(f"Max turns per rollout: {MAX_TURNS}")
    print()

    # Setup env key
    env_key = ensure_localapi_auth(
        backend_base=SYNTH_API_BASE,
        synth_api_key=SYNTH_API_KEY,
    )

    # Start baseline API
    print("Starting baseline API...")
    baseline_app = create_task_app(BASELINE_PROMPT)
    run_server_background(baseline_app, port=8001)
    await wait_for_health_check("127.0.0.1", 8001, env_key, timeout=30.0)

    baseline_tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareManagedTunnel,
        api_key=SYNTH_API_KEY,
        backend_url=SYNTH_API_BASE,
        progress=True,
    )
    print(f"Baseline URL: {baseline_tunnel.url}")

    results = {}

    # Run baseline eval
    print("\nRunning BASELINE eval...")
    baseline_result = run_eval(baseline_tunnel.url, EVAL_SEEDS, "baseline")
    results["baseline"] = baseline_result.raw
    print(f"Baseline: {baseline_result.raw}")

    # Run optimized eval if prompt exists
    if OPTIMIZED_PROMPT:
        print("\nStarting optimized API...")
        optimized_app = create_task_app(OPTIMIZED_PROMPT)
        run_server_background(optimized_app, port=8002)
        await wait_for_health_check("127.0.0.1", 8002, env_key, timeout=30.0)

        optimized_tunnel = await TunneledLocalAPI.create(
            local_port=8002,
            backend=TunnelBackend.CloudflareManagedTunnel,
            api_key=SYNTH_API_KEY,
            backend_url=SYNTH_API_BASE,
            progress=True,
        )
        print(f"Optimized URL: {optimized_tunnel.url}")

        print("\nRunning OPTIMIZED eval...")
        optimized_result = run_eval(optimized_tunnel.url, EVAL_SEEDS, "optimized")
        results["optimized"] = optimized_result.raw
        print(f"Optimized: {optimized_result.raw}")

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/quick_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
