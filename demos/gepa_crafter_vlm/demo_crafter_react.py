import asyncio
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import httpx
from crafter_logic import (
    ACTION_STRING_TO_INT,
    CRAFTER_ALLOWED_ACTIONS,
    CrafterEnvironmentWrapper,
    CrafterScorer,
    CrafterVLMReActPolicy,
    normalize_action_name,
)
from fastapi import Request
from synth_ai.core.urls import BACKEND_URL_BASE, backend_health_url
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig, EvalResult
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob, PromptLearningResult
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi import LocalAPIConfig, RubricBundle, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi.helpers import (
    call_chat_completion_api,
    create_http_client_hooks,
    extract_api_key,
)
from synth_ai.sdk.task import TaskInfo, run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse
from synth_ai.sdk.task.rubrics import Criterion, Rubric
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.tunnels import (
    cleanup_all,
    is_port_available,
    open_quick_tunnel_with_dns_verification,
    track_process,
    wait_for_health_check,
)

SYNTH_API_BASE = BACKEND_URL_BASE
TASK_APP_PORT = int(os.environ.get("CRAFTER_TASK_APP_PORT", "8001"))
OPTIMIZED_TASK_APP_PORT = int(os.environ.get("CRAFTER_OPT_TASK_APP_PORT", "8002"))
POLICY_MODEL = os.environ.get("CRAFTER_POLICY_MODEL", "gpt-4.1-nano")
PROPOSER_TYPE = os.environ.get("CRAFTER_PROPOSER_TYPE", "dspy")
PROPOSER_EFFORT = os.environ.get("CRAFTER_PROPOSER_EFFORT", "MEDIUM")
PROPOSER_OUTPUT_TOKENS = os.environ.get("CRAFTER_PROPOSER_OUTPUT_TOKENS", "FAST")
TOKEN_COUNT_MODEL = os.environ.get("CRAFTER_TOKEN_MODEL", "gpt-4")
VERIFIER_MODEL = os.environ.get("CRAFTER_VERIFIER_MODEL", "gpt-4.1-nano")
VERIFIER_PROVIDER = os.environ.get("CRAFTER_VERIFIER_PROVIDER", "openai")
MIN_PARETO_SET_SIZE = 10
MIN_FEEDBACK_SEEDS = 3
PARETO_SET_SIZE = max(int(os.environ.get("CRAFTER_PARETO_SET_SIZE", "10")), MIN_PARETO_SET_SIZE)
ARCHIVE_SIZE = int(os.environ.get("CRAFTER_ARCHIVE_SIZE", "5"))
ROLLOUT_BUDGET = int(os.environ.get("CRAFTER_ROLLOUT_BUDGET", "30"))
MAX_CONCURRENT = int(os.environ.get("CRAFTER_MAX_CONCURRENT", "3"))
MINIBATCH_SIZE = int(os.environ.get("CRAFTER_MINIBATCH_SIZE", "3"))
POPULATION_SIZE = int(os.environ.get("CRAFTER_POPULATION_SIZE", "3"))
NUM_GENERATIONS = int(os.environ.get("CRAFTER_NUM_GENERATIONS", "2"))
CHILDREN_PER_GEN = int(os.environ.get("CRAFTER_CHILDREN_PER_GEN", "2"))
TRAIN_SEED_COUNT = max(
    int(os.environ.get("CRAFTER_TRAIN_SEED_COUNT", "10")),
    PARETO_SET_SIZE + MIN_FEEDBACK_SEEDS,
)
VALIDATION_SEED_COUNT = int(os.environ.get("CRAFTER_VAL_SEED_COUNT", "5"))
TRAIN_SEEDS = list(range(TRAIN_SEED_COUNT))
VALIDATION_SEEDS = list(range(TRAIN_SEED_COUNT, TRAIN_SEED_COUNT + VALIDATION_SEED_COUNT))
EVAL_SEED_START = int(os.environ.get("CRAFTER_EVAL_SEED_START", "100"))
EVAL_SEED_COUNT = int(os.environ.get("CRAFTER_EVAL_SEED_COUNT", "10"))
EVAL_SEEDS = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SEED_COUNT))
EVAL_MAX_CONCURRENT = int(os.environ.get("CRAFTER_EVAL_MAX_CONCURRENT", "5"))

IS_DEV_MODE = SYNTH_API_BASE.startswith("http://localhost") or SYNTH_API_BASE.startswith(
    "http://127.0.0.1"
)
USE_TUNNEL = os.environ.get("CRAFTER_USE_TUNNEL", "").lower() in ("1", "true", "yes")

CRAFTER_RUBRICS = RubricBundle(
    outcome=Rubric(
        version="1.0",
        goal_text="Evaluate gameplay quality at the end of the rollout",
        criteria=[
            Criterion(
                id="task_progress",
                description="Agent makes progress toward in-game objectives and avoids stalling.",
                weight=1.0,
                required=False,
            )
        ],
        aggregation="weighted_sum",
    )
)


def _build_trace_event(
    messages: List[Dict[str, Any]], response_json: Dict[str, Any], *, turn: int
) -> Dict[str, Any]:
    return {
        "type": "lm_call",
        "event_type": "lm_call",
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_request": {"messages": messages},
        "llm_response": response_json,
        "metadata": {"turn": turn},
    }


def create_crafter_vlm_local_api(system_prompt: str):
    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=60.0,
        log_prefix="crafter_vlm_local_api",
    )

    async def rollout_executor(
        request: RolloutRequest, fastapi_request: Request
    ) -> RolloutResponse:
        policy_config = request.policy.config or {}
        seed = request.env.seed or 0
        env_config = request.env.config or {}
        max_steps = int(env_config.get("max_steps_per_episode", 200))
        max_turns = int(env_config.get("max_turns", 50))
        max_steps_override = os.environ.get("CRAFTER_MAX_STEPS", "").strip()
        max_turns_override = os.environ.get("CRAFTER_MAX_TURNS", "").strip()
        if max_steps_override:
            max_steps = int(max_steps_override)
        if max_turns_override:
            max_turns = int(max_turns_override)

        env = CrafterEnvironmentWrapper(seed=seed, max_steps=max_steps)
        observation = await env.reset()

        policy = CrafterVLMReActPolicy(
            system_prompt=system_prompt,
            use_vision=True,
            image_only_mode=True,
        )

        api_key = extract_api_key(fastapi_request, policy_config) or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        http_client = getattr(fastapi_request.app.state, "http_client", None)

        trace_events: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        episode_rewards: List[float] = []

        for turn in range(max_turns):
            messages = policy.build_messages(observation, history)
            response_text, response_json, tool_calls = await call_chat_completion_api(
                policy_config=policy_config,
                messages=messages,
                tools=policy.tools,
                tool_choice="required",
                api_key=api_key,
                http_client=http_client,
                enable_dns_preresolution=True,
                expected_tool_name="crafter_interact",
                log_prefix="[CRAFTER_VLM]",
            )

            trace_events.append(_build_trace_event(messages, response_json, turn=turn))

            next_observation = observation
            tool_responses: List[Dict[str, Any]] = []
            if tool_calls:
                for tc in tool_calls:
                    tool_call_id = tc.get("id") or tc.get("tool_call_id")
                    tool_name = tc.get("function", {}).get("name")
                    actions_list: List[str] = []
                    if tool_name == "crafter_interact":
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        try:
                            args = json.loads(args_str)
                            raw_actions = args.get("actions_list", [])
                            if isinstance(raw_actions, list):
                                actions_list = [str(a) for a in raw_actions if str(a).strip()]
                        except Exception:
                            actions_list = []
                    if not actions_list:
                        actions_list = ["noop"]

                    actions_list = actions_list[:5]
                    normalized_actions: List[str] = []
                    action_results: List[Dict[str, Any]] = []

                    for action_str in actions_list:
                        normalized = normalize_action_name(action_str) or "noop"
                        normalized_actions.append(normalized)
                        action = ACTION_STRING_TO_INT.get(normalized, 0)
                        next_observation = await env.step(action)
                        reward = next_observation.get("reward", 0.0)
                        episode_rewards.append(float(reward))
                        action_results.append(
                            {
                                "action": normalized,
                                "reward": reward,
                                "step_count": next_observation.get("step_count"),
                                "terminated": next_observation.get("terminated"),
                                "truncated": next_observation.get("truncated"),
                            }
                        )
                        if next_observation.get("terminated") or next_observation.get("truncated"):
                            break

                    if tool_call_id:
                        tool_responses.append(
                            {
                                "tool_call_id": tool_call_id,
                                "actions": normalized_actions,
                                "results": action_results,
                            }
                        )
                    if next_observation.get("terminated") or next_observation.get("truncated"):
                        break
            else:
                next_observation = await env.step(0)
                reward = next_observation.get("reward", 0.0)
                episode_rewards.append(float(reward))

            history.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": tool_calls or [],
                }
            )
            if tool_responses:
                for response in tool_responses:
                    payload = {
                        "actions": response.get("actions", []),
                        "results": response.get("results", []),
                        "terminated": next_observation.get("terminated"),
                        "truncated": next_observation.get("truncated"),
                        "step_count": next_observation.get("step_count"),
                    }
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": response["tool_call_id"],
                            "content": json.dumps(payload),
                        }
                    )

            observation = next_observation
            if observation.get("terminated") or observation.get("truncated"):
                break

        outcome_reward, details = CrafterScorer.score_episode(
            observation,
            len(episode_rewards),
            max_steps,
        )

        trace_correlation_id = extract_trace_correlation_id(
            policy_config,
            policy_config.get("inference_url"),
        )

        trace_metadata: Dict[str, Any] = {
            "session_id": f"crafter-{seed}-{int(time.time())}",
            "env": "crafter",
            "seed": seed,
            "episode_length": len(episode_rewards),
            **details,
        }
        if trace_correlation_id:
            trace_metadata["trace_correlation_id"] = trace_correlation_id
            trace_metadata["correlation_ids"] = {"trace_correlation_id": trace_correlation_id}

        trace = {
            "schema_version": "4.0",
            "event_history": trace_events,
            "markov_blanket_message_history": [],
            "metadata": trace_metadata,
        }

        metrics = RolloutMetrics(
            outcome_reward=outcome_reward,
            event_rewards=episode_rewards,
            details=details,
        )

        return RolloutResponse(
            metrics=metrics,
            trace_correlation_id=trace_correlation_id,
            trace=trace,
            inference_url=policy_config.get("inference_url", ""),
        )

    def describe_taskset() -> Dict[str, Any]:
        return {
            "id": "crafter_vlm",
            "name": "Crafter VLM ReAct Agent",
            "splits": ["train", "test"],
            "description": "Vision-language model playing Crafter using image-only observations",
        }

    def provide_task_instances(seeds: List[int]):
        for seed in seeds:
            yield TaskInfo(
                task={"id": "crafter_vlm", "name": "Crafter VLM", "version": "1.0.0"},
                environment="crafter",
                dataset={"id": "crafter_vlm", "split": "train", "index": seed},
                inference={"supports_proxy": True, "tool": "crafter_interact"},
                limits={"max_turns": 50, "max_steps_per_episode": 200},
                task_metadata={"seed": seed, "format": "vlm_image_only"},
            )

    config = LocalAPIConfig(
        app_id="crafter_vlm",
        name="Crafter VLM ReAct Agent",
        description="Crafter local API for VLM ReAct agent with image-only observations.",
        base_task_info=TaskInfo(
            task={"id": "crafter_vlm", "name": "Crafter VLM", "version": "1.0.0"},
            environment="crafter",
            dataset={"id": "crafter_vlm", "splits": ["train", "test"]},
            inference={"supports_proxy": True, "tool": "crafter_interact"},
            limits={"max_turns": 50, "max_steps_per_episode": 200},
            task_metadata={"format": "vlm_image_only"},
        ),
        provide_taskset_description=describe_taskset,
        provide_task_instances=provide_task_instances,
        rollout=rollout_executor,
        rubrics=CRAFTER_RUBRICS,
        app_state={},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )

    return create_local_api(config)


async def run_gepa_job(
    *,
    api_key: str,
    task_app_url: str,
    baseline_system_prompt: str,
) -> PromptLearningResult:
    pareto_set_size = max(PARETO_SET_SIZE, 10)
    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_app_url,
            "env_name": "crafter",
            "initial_prompt": {
                "messages": [{"role": "system", "order": 0, "pattern": baseline_system_prompt}],
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
                "proposer_type": PROPOSER_TYPE,
                "proposer_effort": PROPOSER_EFFORT,
                "proposer_output_tokens": PROPOSER_OUTPUT_TOKENS,
                "evaluation": {"seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS},
                "rollout": {
                    "budget": ROLLOUT_BUDGET,
                    "max_concurrent": MAX_CONCURRENT,
                    "minibatch_size": MINIBATCH_SIZE,
                },
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": POPULATION_SIZE,
                    "num_generations": NUM_GENERATIONS,
                    "children_per_generation": CHILDREN_PER_GEN,
                },
                "archive": {"size": ARCHIVE_SIZE, "pareto_set_size": pareto_set_size},
                "token": {"counting_model": TOKEN_COUNT_MODEL},
            },
            "verifier": {
                "enabled": True,
                "reward_source": "verifier",
                "backend_base": SYNTH_API_BASE,
                "backend_provider": VERIFIER_PROVIDER,
                "backend_model": VERIFIER_MODEL,
                "verifier_graph_id": "zero_shot_verifier_crafter_vlm",
                "backend_event_enabled": False,
                "backend_outcome_enabled": True,
                "weight_env": 0.0,
                "weight_event": 0.0,
                "weight_outcome": 1.0,
            },
        },
    }

    def _submit_and_poll() -> PromptLearningResult:
        job = PromptLearningJob.from_dict(
            config_dict=config_body,
        )
        job_id = job.submit()
        print(f"GEPA job created: {job_id}")
        result = job.poll_until_complete(timeout=3600.0, interval=3.0, progress=True)
        print(f"GEPA job finished: {result.status.value}")
        return result

    return await asyncio.to_thread(_submit_and_poll)


async def run_eval_job(
    *,
    api_key: str,
    task_app_url: str,
    seeds: List[int],
    mode: str,
) -> EvalResult:
    def _submit_and_poll() -> EvalResult:
        config = EvalJobConfig(
            task_app_url=task_app_url,
            backend_url=SYNTH_API_BASE,
            api_key=api_key,
            app_id=f"crafter_vlm_{mode}",
            env_name="crafter",
            seeds=seeds,
            policy_config={"model": POLICY_MODEL, "provider": "openai"},
            env_config={"max_steps_per_episode": 200},
            concurrency=EVAL_MAX_CONCURRENT,
        )
        job = EvalJob(config)
        job_id = job.submit()
        print(f"Eval job ({mode}): {job_id}")
        return job.poll_until_complete(timeout=600.0, interval=2.0, progress=True)

    return await asyncio.to_thread(_submit_and_poll)


def _extract_system_prompt(best_prompt: Optional[Dict[str, Any]]) -> Optional[str]:
    if not best_prompt:
        return None
    messages = best_prompt.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("pattern") or msg.get("content")
    sections = best_prompt.get("sections", [])
    for sec in sections:
        if sec.get("role") == "system":
            return sec.get("content")
    return None


def _ensure_available_port(port: int, label: str) -> int:
    def _port_in_use(candidate: int) -> bool:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{candidate}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return bool(result.stdout.strip())
        except FileNotFoundError:
            return not is_port_available(candidate)

    if not _port_in_use(port):
        return port

    for offset in range(1, 100):
        candidate = port + offset
        if not _port_in_use(candidate):
            print(f"{label} port {port} busy; using {candidate}.")
            return candidate

    raise RuntimeError(f"{label} port {port} busy; no available port found.")


async def main() -> None:
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        print("SYNTH_API_KEY not set; skipping GEPA job and evals.")
        return

    # Set API key in environment for SDK to use (in case it wasn't already set)
    os.environ["SYNTH_API_KEY"] = api_key

    print(f"Backend: {SYNTH_API_BASE} (dev_mode={IS_DEV_MODE})")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(backend_health_url(SYNTH_API_BASE))
        resp.raise_for_status()
        print(f"Backend health: {resp.json()}")

    environment_api_key = ensure_localapi_auth(
        backend_base=SYNTH_API_BASE,
        synth_api_key=api_key,
    )

    allowed_actions = ", ".join(CRAFTER_ALLOWED_ACTIONS)
    baseline_prompt = (
        "You are an agent playing Crafter, a survival crafting game. "
        "Your goal is to survive and unlock achievements by exploring, crafting, and building. "
        "You can see the game state through images. Analyze each image carefully to understand "
        "your surroundings, inventory, health, and available resources. "
        "Use the crafter_interact tool to execute actions. "
        "Key mechanics: use 'do' only when adjacent to a resource (tree, stone, cow, plant); "
        "it does nothing on grass or water. "
        "Craft progression: wood -> table -> wood_pickaxe -> stone -> stone_pickaxe -> iron tools. "
        "Sleep when energy is low to restore and unlock wake_up. "
        f"Available actions: {allowed_actions}. "
        "Only use these action names and return 2-5 actions per decision. "
        "Strategy: move toward trees to collect wood; place a table once you have wood; "
        "craft a wood pickaxe, then collect stone and craft a stone pickaxe; "
        "progress toward iron tools and combat when safe."
    )

    baseline_port = _ensure_available_port(TASK_APP_PORT, "Baseline")
    baseline_app = create_crafter_vlm_local_api(baseline_prompt)
    run_server_background(baseline_app, port=baseline_port)
    await wait_for_health_check("127.0.0.1", baseline_port, environment_api_key, timeout=60.0)

    if USE_TUNNEL or not IS_DEV_MODE:
        baseline_url, proc = await open_quick_tunnel_with_dns_verification(
            TASK_APP_PORT,
            api_key=environment_api_key,
        )
        track_process(proc)
    else:
        baseline_url = f"http://127.0.0.1:{baseline_port}"

    print(f"Baseline local API URL: {baseline_url}")

    job_result = await run_gepa_job(
        api_key=api_key,
        task_app_url=baseline_url,
        baseline_system_prompt=baseline_prompt,
    )

    if not job_result.succeeded:
        status = job_result.status.value
        print(f"GEPA job did not succeed ({status}); skipping optimized eval.")
        if job_result.error:
            print(f"GEPA error: {job_result.error}")
        cleanup_all()
        return

    pl_client = PromptLearningClient(SYNTH_API_BASE, api_key)
    prompt_results = await pl_client.get_prompts(job_result.job_id)
    optimized_prompt = _extract_system_prompt(prompt_results.best_prompt)
    if not optimized_prompt:
        raise RuntimeError(
            "Failed to extract optimized system prompt from prompt learning results."
        )

    optimized_port = _ensure_available_port(OPTIMIZED_TASK_APP_PORT, "Optimized")
    optimized_app = create_crafter_vlm_local_api(optimized_prompt)
    run_server_background(optimized_app, port=optimized_port)
    await wait_for_health_check("127.0.0.1", optimized_port, environment_api_key, timeout=60.0)

    if USE_TUNNEL or not IS_DEV_MODE:
        optimized_url, proc = await open_quick_tunnel_with_dns_verification(
            OPTIMIZED_TASK_APP_PORT,
            api_key=environment_api_key,
        )
        track_process(proc)
    else:
        optimized_url = f"http://127.0.0.1:{optimized_port}"

    eval_seeds = EVAL_SEEDS
    if not eval_seeds:
        print("No eval seeds configured; skipping eval.")
        cleanup_all()
        return

    baseline_eval = await run_eval_job(
        api_key=api_key,
        task_app_url=baseline_url,
        seeds=eval_seeds,
        mode="baseline",
    )
    optimized_eval = await run_eval_job(
        api_key=api_key,
        task_app_url=optimized_url,
        seeds=eval_seeds,
        mode="optimized",
    )

    print("Baseline eval:", baseline_eval.raw)
    print("Optimized eval:", optimized_eval.raw)
    cleanup_all()


if __name__ == "__main__":
    asyncio.run(main())
