from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from fastapi import HTTPException, Request
import httpx

from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.examples.red.taskset import INSTANCE as RED_DEFAULT_INSTANCE
from synth_ai.task.apps import TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.server import ProxyConfig, TaskAppConfig


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={"id": "pokemon_red", "name": "Pokémon Red", "version": "0.1.0"},
        environments=["pokemon_red"],
        action_space={
            "type": "tool_call",
            "tools": [
                {
                    "name": "press_button",
                    "schema": {"button": "string", "frames": "int"},
                }
            ],
            "max_calls": 1,
        },
        observation={
            "summary": "GB memory-derived state with reward fields.",
            "keys": [
                "position",
                "badges_earned",
                "badges_bitfield",
                "hp_status",
                "party_level",
                "party_xp",
                "in_battle",
                "step_count",
                "reward_last_step",
                "total_reward",
                "terminated",
            ],
        },
        dataset={"id": "pokemon_red_default", "name": "Pokémon Red Default", "version": "0.1.0"},
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={
            "supports_proxy": True,
            "tool": {"name": "press_button", "parallel_tool_calls": False},
            "endpoints": {
                "openai": "/proxy/v1/chat/completions",
                "groq": "/proxy/groq/v1/chat/completions",
            },
        },
        capabilities={"supports_rollout": True, "supports_env_lifecycle": True},
        limits={"max_steps": 1000},
    )


def _describe_taskset() -> dict[str, Any]:
    return {"id": "pokemon_red_default", "name": "Pokémon Red Default"}


def _provide_task_instances(seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base = _base_task_info()
    for s in seeds:
        yield TaskInfo(
            task=base.task,
            environments=base.environments,
            action_space=base.action_space,
            observation={**base.observation, "seed": s},
            dataset=base.dataset,
            rubric=base.rubric,
            inference=base.inference,
            capabilities=base.capabilities,
            limits=base.limits,
        )


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    async def _call_inference(policy_cfg: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are controlling Pokémon Red. Respond with a single tool call named 'press_button' "
                    "with JSON arguments {button: 'A|B|UP|DOWN|LEFT|RIGHT|START|SELECT', frames: 1-120}."
                ),
            },
            {
                "role": "user",
                "content": (
                    "State summary: " + str({k: observation.get(k) for k in observation.keys() if k != "error"})
                ),
            },
        ]
        payload = {
            "model": policy_cfg.get("model") or "qwen-2.5-7b",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "press_button",
                        "description": "Press a Game Boy button for N frames",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {"type": "string"},
                                "frames": {"type": "integer", "minimum": 1, "maximum": 120},
                            },
                            "required": ["button"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "press_button"}},
            "temperature": float(policy_cfg.get("temperature") or 0.0),
            "top_p": float(policy_cfg.get("top_p") or 1.0),
            "max_tokens": int(policy_cfg.get("max_tokens") or 64),
        }
        inference_url = str(policy_cfg.get("inference_url") or "").rstrip("/")
        if not inference_url:
            # Prefer built-in proxy endpoints from app if no external URL
            provider = (policy_cfg.get("provider") or "").lower()
            if provider == "groq":
                inference_url = "/proxy/groq/v1/chat/completions"
            else:
                inference_url = "/proxy/v1/chat/completions"
        async with httpx.AsyncClient(base_url="http://127.0.0.1:" + str(fastapi_request.url.port or 8913), timeout=httpx.Timeout(60.0)) as client:  # best-effort
            resp = await client.post(inference_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Extract first tool call
        choices = data.get("choices") or []
        if not choices:
            return {}
        message = choices[0].get("message") or {}
        raw_calls = message.get("tool_calls") or []
        if not raw_calls:
            return {}
        f = raw_calls[0].get("function") or {}
        args = f.get("arguments")
        import json as _json
        try:
            parsed_args = _json.loads(args) if isinstance(args, str) else dict(args or {})
        except Exception:
            parsed_args = {}
        return {"button": parsed_args.get("button"), "frames": int(parsed_args.get("frames") or 1)}
    
    env = PokemonRedEnvironment(RED_DEFAULT_INSTANCE)
    obs0 = await env.initialize()

    steps: list[RolloutStep] = [
        RolloutStep(obs=obs0, tool_calls=[], reward=0.0, done=False, info={}),
    ]

    # Process all ops (explicit actions)
    final_obs = obs0
    for op in (request.ops or []):
        macro = None
        if isinstance(op, dict):
            macro = op.get("action") or op

        if isinstance(macro, dict):
            button = macro.get("button") or "A"
            frames = int(macro.get("frames") or 1)
            obs1 = await env.step(EnvToolCall(tool="press_button", args={"button": button, "frames": frames}))
            steps.append(
                RolloutStep(
                    obs=obs1,
                    tool_calls=[{"tool": "press_button", "args": {"button": button, "frames": frames}}],
                    reward=0.0,
                    done=False,
                    info={},
                )
            )
            final_obs = obs1
        else:
            # Attempt policy-driven step if policy.config present
            policy_cfg = request.policy.config or {}
            if policy_cfg:
                try:
                    action = await _call_inference(policy_cfg, final_obs if isinstance(final_obs, Mapping) else {})
                    if action.get("button"):
                        obs1 = await env.step(EnvToolCall(tool="press_button", args=action))
                        steps.append(
                            RolloutStep(
                                obs=obs1,
                                tool_calls=[{"tool": "press_button", "args": action}],
                                reward=0.0,
                                done=False,
                                info={"proxy": True},
                            )
                        )
                        final_obs = obs1
                except Exception:
                    pass

    metrics = RolloutMetrics(
        episode_returns=[0.0],
        mean_return=0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=0.0,
        details={"note": "demo rollout"},
    )

    trajectory = RolloutTrajectory(
        env_id="pokemon_red",
        policy_id=request.policy.policy_id or "policy",
        steps=steps,
        final={"observation": final_obs, "reward": 0.0},
        length=len(steps),
    )

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(request.ops or []),
    )


def build_config() -> TaskAppConfig:
    base_info = _base_task_info()
    return TaskAppConfig(
        app_id="pokemon_red",
        name="Pokémon Red Task App",
        description="Expose Pokémon Red via Synth task framework (demo).",
        base_task_info=base_info,
        describe_taskset=_describe_taskset,
        provide_task_instances=_provide_task_instances,
        rollout=rollout_executor,
        dataset_registry=None,
        proxy=ProxyConfig(
            enable_openai=True,
            enable_groq=True,
            system_hint=(
                "You control Pokémon Red. Respond with a single 'press_button' tool call."
            ),
        ),
        app_state={},
        require_api_key=False,
        expose_debug_env=True,
        cors_origins=["*"],
    )


register_task_app(
    entry=TaskAppEntry(
        app_id="pokemon_red",
        description="Pokémon Red demo task app",
        config_factory=build_config,
        aliases=("pokemon_red_demo",),
        env_files=(),
        modal=None,
    )
)


