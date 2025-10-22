from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from fastapi import HTTPException, Request
import httpx

from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.examples.red.taskset import INSTANCE as RED_DEFAULT_INSTANCE
from synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_progression import (
    PalletTownProgressionCompositeReward,
)
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


def _build_action_context(prev_state: dict[str, Any], current_state: dict[str, Any]) -> dict[str, Any]:
    """Build action context dict with prev_ fields for reward calculation."""
    return {
        "prev_map_id": prev_state.get("map_id", 0),
        "prev_player_x": prev_state.get("player_x", 0),
        "prev_player_y": prev_state.get("player_y", 0),
        "prev_party_count": prev_state.get("party_count", 0),
        "prev_in_battle": prev_state.get("in_battle", False),
        "prev_text_box_active": prev_state.get("text_box_active", False),
        "prev_enemy_hp_current": prev_state.get("enemy_hp_current", 0),
        "prev_enemy_hp_percentage": prev_state.get("enemy_hp_percentage", 0.0),
        "prev_badges": prev_state.get("badges", 0),
        "prev_party_level": prev_state.get("party_level", 0),
        "prev_party_xp": prev_state.get("party_xp", 0),
    }


def _describe_milestone(current_state: dict[str, Any], prev_state: dict[str, Any], reward: float) -> str:
    """Generate human-readable milestone description."""
    descriptions = []
    
    # Map transitions
    prev_map = prev_state.get("map_id", -1)
    curr_map = current_state.get("map_id", -1)
    if prev_map != curr_map:
        map_names = {0: "Pallet Town", 1: "Bedroom", 2: "House", 3: "Oak's Lab"}
        descriptions.append(f"Moved from {map_names.get(prev_map, f'Map{prev_map}')} to {map_names.get(curr_map, f'Map{curr_map}')}")
    
    # Party changes
    prev_party = prev_state.get("party_count", 0)
    curr_party = current_state.get("party_count", 0)
    if curr_party > prev_party:
        descriptions.append(f"Received Pokémon (party: {prev_party}→{curr_party})")
    
    # Battle state
    prev_battle = prev_state.get("in_battle", False)
    curr_battle = current_state.get("in_battle", False)
    if not prev_battle and curr_battle:
        descriptions.append("Entered battle")
    elif prev_battle and not curr_battle:
        battle_outcome = current_state.get("battle_outcome", 0)
        if battle_outcome == 1:
            descriptions.append("Won battle")
        elif battle_outcome == 2:
            descriptions.append("Lost battle")
    
    # HP damage
    prev_enemy_hp = prev_state.get("enemy_hp_current", 0)
    curr_enemy_hp = current_state.get("enemy_hp_current", 0)
    if prev_enemy_hp > curr_enemy_hp > 0:
        damage = prev_enemy_hp - curr_enemy_hp
        descriptions.append(f"Dealt {damage} damage to enemy")
    
    return " | ".join(descriptions) if descriptions else f"Progress (+{reward:.0f})"


def _calculate_outcome_score(final_state: dict[str, Any], total_reward: float) -> float:
    """Calculate outcome score based on final state and total reward."""
    # Normalize reward to 0-1 scale (max expected is ~700)
    reward_score = min(total_reward / 700.0, 1.0)
    
    # Bonus for having Pokemon
    has_pokemon = 1.0 if final_state.get("party_count", 0) > 0 else 0.0
    
    # Bonus for being in Oak's lab or having left it
    map_id = final_state.get("map_id", -1)
    map_bonus = 0.5 if map_id in [0, 3] else 0.0  # Pallet Town or Oak's Lab
    
    # Weighted combination
    return (reward_score * 0.7) + (has_pokemon * 0.2) + (map_bonus * 0.1)


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
    
    # Initialize reward function
    reward_fn = PalletTownProgressionCompositeReward()
    
    env = PokemonRedEnvironment(RED_DEFAULT_INSTANCE)
    obs0 = await env.initialize()

    # Track cumulative stats
    total_reward = 0.0
    all_reward_components: list[dict[str, Any]] = []
    milestone_events: list[dict[str, Any]] = []
    
    steps: list[RolloutStep] = [
        RolloutStep(obs=obs0, tool_calls=[], reward=0.0, done=False, info={"step_type": "initial"}),
    ]

    # Track previous state for reward calculation
    prev_state = dict(obs0) if isinstance(obs0, Mapping) else {}
    
    # Process all ops (explicit actions)
    final_obs = obs0
    for step_idx, op in enumerate(request.ops or []):
        macro = None
        if isinstance(op, dict):
            macro = op.get("action") or op

        if isinstance(macro, dict):
            button = macro.get("button") or "A"
            frames = int(macro.get("frames") or 1)
            obs1 = await env.step(EnvToolCall(tool="press_button", args={"button": button, "frames": frames}))
            
            # Calculate step reward
            current_state = dict(obs1) if isinstance(obs1, Mapping) else {}
            action_context = _build_action_context(prev_state, current_state)
            step_reward = await reward_fn.score(current_state, action_context)
            total_reward += step_reward
            
            # Track reward components if non-zero
            step_info: dict[str, Any] = {"step_type": "action", "step_idx": step_idx}
            if step_reward > 0:
                reward_component = {
                    "step": step_idx + 1,
                    "reward": step_reward,
                    "button": button,
                    "map_id": current_state.get("map_id"),
                    "position": f"({current_state.get('player_x')},{current_state.get('player_y')})",
                }
                all_reward_components.append(reward_component)
                step_info["reward_component"] = reward_component
                
                # Track milestone events
                milestone_events.append({
                    "type": "milestone",
                    "step": step_idx + 1,
                    "reward": step_reward,
                    "description": _describe_milestone(current_state, prev_state, step_reward),
                })
            
            step_info["cumulative_reward"] = total_reward
            
            steps.append(
                RolloutStep(
                    obs=obs1,
                    tool_calls=[{"tool": "press_button", "args": {"button": button, "frames": frames}}],
                    reward=step_reward,
                    done=False,
                    info=step_info,
                )
            )
            final_obs = obs1
            prev_state = current_state
        else:
            # Attempt policy-driven step if policy.config present
            policy_cfg = request.policy.config or {}
            if policy_cfg:
                try:
                    action = await _call_inference(policy_cfg, final_obs if isinstance(final_obs, Mapping) else {})
                    if action.get("button"):
                        obs1 = await env.step(EnvToolCall(tool="press_button", args=action))
                        
                        # Calculate step reward
                        current_state = dict(obs1) if isinstance(obs1, Mapping) else {}
                        action_context = _build_action_context(prev_state, current_state)
                        step_reward = await reward_fn.score(current_state, action_context)
                        total_reward += step_reward
                        
                        step_info_policy: dict[str, Any] = {
                            "step_type": "policy",
                            "step_idx": step_idx,
                            "cumulative_reward": total_reward,
                            "proxy": True,
                        }
                        if step_reward > 0:
                            step_info_policy["reward_earned"] = step_reward
                        
                        steps.append(
                            RolloutStep(
                                obs=obs1,
                                tool_calls=[{"tool": "press_button", "args": action}],
                                reward=step_reward,
                                done=False,
                                info=step_info_policy,
                            )
                        )
                        final_obs = obs1
                        prev_state = current_state
                except Exception:
                    pass

    # Calculate outcome score based on milestones achieved
    final_state = dict(final_obs) if isinstance(final_obs, Mapping) else {}
    outcome_score = _calculate_outcome_score(final_state, total_reward)
    
    metrics = RolloutMetrics(
        episode_returns=[total_reward],
        mean_return=total_reward,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=outcome_score,
        details={
            "total_reward": total_reward,
            "reward_components": all_reward_components,
            "milestone_events": milestone_events,
            "final_map": final_state.get("map_id"),
            "party_count": final_state.get("party_count", 0),
            "badges": final_state.get("badges", 0),
        },
    )

    trajectory = RolloutTrajectory(
        env_id="pokemon_red",
        policy_id=request.policy.policy_id or "policy",
        steps=steps,
        final={"observation": final_obs, "reward": total_reward},
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


