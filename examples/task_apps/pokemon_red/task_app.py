from __future__ import annotations

import logging
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
from synth_ai.task.tracing_utils import (
    build_tracer_factory,
    resolve_sft_output_dir,
    resolve_tracing_db_url,
    tracing_env_enabled,
)
from synth_ai.tracing_v3.session_tracer import SessionTracer

logger = logging.getLogger(__name__)


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={"id": "pokemon_red", "name": "Pokémon Red", "version": "0.1.0"},
        environment="pokemon_red",
        action_space={
            "type": "tool_call",
            "tools": [
                {
                    "name": "press_button",
                    "schema": {"button": "string", "frames": "int"},
                },
                {
                    "name": "execute_sequence",
                    "description": "Execute multiple button presses in sequence. More efficient than separate calls. Recommended: 5-10 actions per call.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]},
                                        "frames": {"type": "integer", "minimum": 1, "maximum": 120}
                                    },
                                    "required": ["button", "frames"]
                                },
                                "minItems": 1,
                                "maxItems": 20
                            }
                        },
                        "required": ["actions"]
                    },
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
        limits={"max_steps": 1000},
    )


def _describe_taskset() -> dict[str, Any]:
    return {"id": "pokemon_red_default", "name": "Pokémon Red Default"}


def _provide_task_instances(seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base = _base_task_info()
    for s in seeds:
        yield TaskInfo(
            task=base.task,
            environment=base.environment,
            action_space=base.action_space,
            observation={**base.observation, "seed": s},
            dataset=base.dataset,
            rubric=base.rubric,
            inference=base.inference,
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
    # Initialize SessionTracer for this rollout
    tracer_factory = getattr(fastapi_request.app.state, "session_tracer_factory", None)
    tracer_instance: SessionTracer | None = None
    if callable(tracer_factory):
        try:
            inst = tracer_factory()
            tracer_instance = inst if isinstance(inst, SessionTracer) else None
        except Exception as exc:
            logger.debug(f"TRACER_FACTORY_FAIL: {exc}")
    
    # Start tracing session
    if tracer_instance is not None:
        try:
            await tracer_instance.initialize()
            await tracer_instance.start_session(
                session_id=request.run_id,
                metadata={
                    "run_id": request.run_id,
                    "env_name": "pokemon_red",
                    "policy_name": request.policy.policy_name or "default",
                    "seed": request.env.seed,
                }
            )
            logger.info(f"[pokemon_red] tracing enabled for run_id={request.run_id}")
        except Exception as exc:
            logger.warning(f"[pokemon_red] tracing init failed: {exc}")
            tracer_instance = None
    
    async def _call_inference(policy_cfg: Mapping[str, Any], observation: Mapping[str, Any]) -> Mapping[str, Any]:
        # Check if vision mode is enabled
        use_vision = bool(policy_cfg.get("use_vision", False))
        image_only_mode = bool(policy_cfg.get("image_only_mode", False))
        
        # Build user message content
        if use_vision and "observation_image_data_url" in observation:
            # Extract image data URL
            image_data_url = observation["observation_image_data_url"]
            
            # Build state summary (text observation)
            state_summary = "State summary: " + str({
                k: observation.get(k) 
                for k in observation.keys() 
                if k not in ["error", "observation_image_base64", "observation_image_data_url", 
                            "observation_image_format", "observation_image_width", "observation_image_height"]
            })
            
            # Image-only mode: only send image, no text
            if image_only_mode:
                user_content = [
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            else:
                # Vision mode with text: send both text and image
                user_content = [
                    {"type": "text", "text": state_summary},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
        else:
            # Text-only mode (default)
            state_summary = "State summary: " + str({
                k: observation.get(k) for k in observation.keys() if k != "error"
            })
            user_content = state_summary
        
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
                "content": user_content,
            },
        ]
        payload = {
            "model": policy_cfg.get("model") or "qwen-2.5-7b",
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_sequence",
                        "description": "Execute multiple button presses in sequence. More efficient than separate calls. Recommended: 5-10 actions per call.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "button": {
                                                "type": "string",
                                                "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"],
                                                "description": "Game Boy button to press"
                                            },
                                            "frames": {
                                                "type": "integer",
                                                "minimum": 1,
                                                "maximum": 120,
                                                "description": "Number of frames to hold the button (30 frames = 0.5 seconds)"
                                            }
                                        },
                                        "required": ["button", "frames"]
                                    },
                                    "minItems": 1,
                                    "maxItems": 20,
                                    "description": "Sequence of button presses to execute"
                                }
                            },
                            "required": ["actions"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "press_button",
                        "description": "Press a single Game Boy button for N frames (use execute_sequence for multiple actions)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]},
                                "frames": {"type": "integer", "minimum": 1, "maximum": 120},
                            },
                            "required": ["button"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "execute_sequence"}},
            "temperature": float(policy_cfg.get("temperature") or 0.0),
            "top_p": float(policy_cfg.get("top_p") or 1.0),
            "max_tokens": int(policy_cfg.get("max_tokens") or 500),
        }
        inference_url = str(policy_cfg.get("inference_url") or "").rstrip("/")
        
        # Determine if this is an external URL or internal proxy
        is_external = inference_url.startswith("http://") or inference_url.startswith("https://")
        
        if not inference_url:
            # Prefer built-in proxy endpoints from app if no external URL
            provider = (policy_cfg.get("provider") or "").lower()
            if provider == "groq":
                inference_url = "/proxy/groq/v1/chat/completions"
            else:
                inference_url = "/proxy/v1/chat/completions"
            is_external = False
        elif is_external:
            # Add /v1/chat/completions if using OpenAI directly
            if "api.openai.com" in inference_url and not inference_url.endswith("/chat/completions"):
                inference_url = inference_url + "/v1/chat/completions"
        
        if is_external:
            # External API: use direct HTTP client with auth header
            headers = {}
            if "api.openai.com" in inference_url:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                resp = await client.post(inference_url, json=payload, headers=headers)
        else:
            # Internal proxy: use local base_url
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:" + str(fastapi_request.url.port or 8913),
                timeout=httpx.Timeout(60.0)
            ) as client:
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
        tool_name = f.get("name", "")
        args = f.get("arguments")
        import json as _json
        try:
            parsed_args = _json.loads(args) if isinstance(args, str) else dict(args or {})
        except Exception:
            parsed_args = {}
        
        # Handle execute_sequence tool
        if tool_name == "execute_sequence":
            return {"actions": parsed_args.get("actions", [])}
        
        # Handle press_button tool (legacy single action)
        return {"button": parsed_args.get("button"), "frames": int(parsed_args.get("frames") or 30)}
    
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
            # Check if this is an execute_sequence call
            if "actions" in macro:
                # Handle execute_sequence: multiple actions in one call
                actions_list = macro.get("actions", [])
                sequence_reward = 0.0
                sequence_tool_calls = []
                
                for action_item in actions_list:
                    button = action_item.get("button", "A")
                    frames = int(action_item.get("frames", 1))
                    
                    obs1 = await env.step(EnvToolCall(tool="press_button", args={"button": button, "frames": frames}))
                    current_state = dict(obs1) if isinstance(obs1, Mapping) else {}
                    action_context = _build_action_context(prev_state, current_state)
                    step_reward = await reward_fn.score(current_state, action_context)
                    
                    sequence_reward += step_reward
                    sequence_tool_calls.append({"tool": "press_button", "args": {"button": button, "frames": frames}})
                    
                    if step_reward > 0:
                        reward_component = {
                            "step": step_idx + 1,
                            "reward": step_reward,
                            "button": button,
                            "map_id": current_state.get("map_id"),
                            "position": f"({current_state.get('player_x')},{current_state.get('player_y')})",
                        }
                        all_reward_components.append(reward_component)
                        milestone_events.append({
                            "type": "milestone",
                            "step": step_idx + 1,
                            "reward": step_reward,
                            "description": _describe_milestone(current_state, prev_state, step_reward),
                        })
                    
                    final_obs = obs1
                    prev_state = current_state
                
                total_reward += sequence_reward
                step_info = {
                    "step_type": "sequence",
                    "step_idx": step_idx,
                    "actions_count": len(actions_list),
                    "cumulative_reward": total_reward,
                }
                if sequence_reward > 0:
                    step_info["sequence_reward"] = sequence_reward
                
                steps.append(
                    RolloutStep(
                        obs=final_obs,
                        tool_calls=sequence_tool_calls,
                        reward=sequence_reward,
                        done=False,
                        info=step_info,
                    )
                )
            else:
                # Handle single press_button call
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
                    
                    # Handle execute_sequence from policy
                    if "actions" in action:
                        actions_list = action.get("actions", [])
                        sequence_reward = 0.0
                        sequence_tool_calls = []
                        
                        for action_item in actions_list:
                            button = action_item.get("button", "A")
                            frames = int(action_item.get("frames", 30))
                            
                            obs1 = await env.step(EnvToolCall(tool="press_button", args={"button": button, "frames": frames}))
                            current_state = dict(obs1) if isinstance(obs1, Mapping) else {}
                            action_context = _build_action_context(prev_state, current_state)
                            step_reward = await reward_fn.score(current_state, action_context)
                            
                            sequence_reward += step_reward
                            sequence_tool_calls.append({"tool": "press_button", "args": {"button": button, "frames": frames}})
                            
                            if step_reward > 0:
                                reward_component = {
                                    "step": step_idx + 1,
                                    "reward": step_reward,
                                    "button": button,
                                    "map_id": current_state.get("map_id"),
                                    "position": f"({current_state.get('player_x')},{current_state.get('player_y')})",
                                }
                                all_reward_components.append(reward_component)
                                milestone_events.append({
                                    "type": "milestone",
                                    "step": step_idx + 1,
                                    "reward": step_reward,
                                    "description": _describe_milestone(current_state, prev_state, step_reward),
                                })
                            
                            final_obs = obs1
                            prev_state = current_state
                        
                        total_reward += sequence_reward
                        step_info = {
                            "step_type": "policy_sequence",
                            "step_idx": step_idx,
                            "actions_count": len(actions_list),
                            "cumulative_reward": total_reward,
                        }
                        if sequence_reward > 0:
                            step_info["sequence_reward"] = sequence_reward
                        
                        steps.append(
                            RolloutStep(
                                obs=final_obs,
                                tool_calls=sequence_tool_calls,
                                reward=sequence_reward,
                                done=False,
                                info=step_info,
                            )
                        )
                    
                    # Handle single button press from policy
                    elif action.get("button"):
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

    # Extract inference_url from policy config
    inference_url = (policy_cfg or {}).get("inference_url")
    
    trajectory = RolloutTrajectory(
        env_id="pokemon_red",
        policy_id=request.policy.policy_id or "policy",
        steps=steps,
        final={"observation": final_obs, "reward": total_reward},
        length=len(steps),
        inference_url=inference_url,  # NEW: Required for trace correlation
    )

    # Record outcome rewards and end session
    trace_payload = None
    if tracer_instance is not None:
        try:
            # Count achievements (milestones)
            achievements_count = len(milestone_events)
            
            # Build metadata with all relevant info
            reward_metadata = {
                "run_id": request.run_id,
                "env_name": "pokemon_red",
                "final_map": final_state.get("map_id", -1),
                "party_count": final_state.get("party_count", 0),
                "badges": final_state.get("badges", 0),
                "steps": len(steps),
                "milestone_events": milestone_events,
                "reward_components": all_reward_components,
            }
            
            # Record outcome reward to Turso
            await tracer_instance.record_outcome_reward(
                total_reward=int(total_reward),
                achievements_count=achievements_count,
                total_steps=len(steps),
                reward_metadata=reward_metadata,
            )
            logger.info(f"[pokemon_red] recorded outcome: reward={total_reward}, achievements={achievements_count}")
            
            # End session and get trace
            session_trace = await tracer_instance.end_session()
            
            # Build trace payload if requested
            record_config = getattr(request, 'record', None)
            if record_config and getattr(record_config, 'return_trace', False) and session_trace:
                trace_payload = {
                    "session_id": session_trace.session_id,
                    "created_at": session_trace.created_at.isoformat() if session_trace.created_at else None,
                    "metadata": dict(session_trace.metadata or {}),
                    "num_timesteps": session_trace.num_timesteps,
                    "num_events": session_trace.num_events,
                    "num_messages": session_trace.num_messages,
                }
        except Exception as exc:
            logger.warning(f"[pokemon_red] tracing finalization failed: {exc}")
    
    # Fallback trace payload if no tracer but CLI needs it
    if trace_payload is None:
        record_config = getattr(request, 'record', None)
        if record_config and getattr(record_config, 'return_trace', False):
            trace_payload = {
                "session_id": request.run_id,
                "created_at": import_datetime().now().isoformat(),
                "metadata": {
                    "run_id": request.run_id,
                    "env_name": "pokemon_red",
                    "total_reward": int(total_reward),
                    "final_map": final_state.get("map_id", -1),
                    "party_count": final_state.get("party_count", 0),
                    "badges": final_state.get("badges", 0),
                    "steps": len(steps),
                },
                "num_timesteps": len(steps),
                "num_events": len(steps),
                "num_messages": len(steps) * 2,
            }
    
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(request.ops or []),
        trace=trace_payload,
    )


def import_datetime():
    """Helper to import datetime for trace timestamps."""
    from datetime import datetime
    return datetime


def build_config() -> TaskAppConfig:
    base_info = _base_task_info()
    
    # Set up tracing
    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(
        SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url
    )
    sft_output_dir = resolve_sft_output_dir()
    
    app_state: dict[str, Any] = {
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir
    
    if tracing_enabled:
        status_msg = f"[task:tracing] enabled (db={tracing_db_url or 'default'})"
        logger.info(status_msg)
        print(status_msg, flush=True)
    
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
                "You control Pokémon Red. Use 'execute_sequence' with 5-10 actions to play efficiently. "
                "Plan ahead: navigate rooms, advance dialogue, battle strategically. "
                "Example: {\"tool\": \"execute_sequence\", \"args\": {\"actions\": [{\"button\": \"DOWN\", \"frames\": 30}, ...]}}"
            ),
        ),
        app_state=app_state,
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


