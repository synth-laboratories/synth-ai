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
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
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
from synth_ai.tracing_v3.abstractions import EnvironmentEvent, TimeRecord
from datetime import datetime, UTC

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
                    "You are controlling Pokémon Red, a classic Game Boy game. You can see the game screen in the images provided. "
                    "Your goal is to make progress in the game. "
                    "IMPORTANT: Always use the 'execute_sequence' tool to submit 5-10 actions per call. "
                    "Do not reason about which tool to use - execute_sequence is the only tool available. "
                    "Choose appropriate button presses based on what you see in the game screen. "
                    "Plan 5-10 actions ahead to play efficiently. "
                    "CRITICAL: If stuck in a text box (text_box_active=True), try pressing B button first, then try A. "
                    "Always respond with exactly one tool call containing 5-10 actions."
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
                        "description": "Execute multiple button presses in sequence. More efficient than separate calls. ALWAYS use this tool. Plan 5-10 actions ahead to play efficiently.",
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
                                    "minItems": 5,
                                    "maxItems": 10,
                                    "description": "Sequence of 5-10 button presses to execute. Plan ahead to navigate efficiently."
                                }
                            },
                            "required": ["actions"],
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
        
        # Debug: print exact payload being sent
        import json as _json_debug
        print(f"\n{'='*80}")
        print(f"[pokemon_red] INFERENCE REQUEST DEBUG")
        print(f"{'='*80}")
        print(f"Inference URL: {inference_url}")
        print(f"Payload keys: {list(payload.keys())}")
        print(f"Payload (formatted):")
        print(_json_debug.dumps(payload, indent=2)[:2000])
        print(f"{'='*80}\n")
        
        
        if is_external:
            # External API: use direct HTTP client with auth header
            headers = {}
            import os
            if "api.openai.com" in inference_url:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
            elif "modal.run" in inference_url or "synth" in inference_url.lower():
                # Synth API: use SYNTH_API_KEY
                api_key = os.getenv("SYNTH_API_KEY")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                print(f"[pokemon_red] Using Synth API auth: {'Bearer ' + api_key[:10] + '...' if api_key else 'NONE'}")
                # For 30B-A3B models, require H200 (A100 doesn't have enough memory)
                model_id = payload.get("model", "")
                if "30B-A3B" in model_id or "A3B" in model_id:
                    headers["X-GPU-Preference"] = "H200"
                    print(f"[pokemon_red] Setting X-GPU-Preference: H200 (required for A3B MoE)")
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=1800.0, write=60.0, pool=60.0)) as client:  # 30 min read timeout for cold starts
                resp = await client.post(inference_url, json=payload, headers=headers)
        else:
            # Internal proxy: use local base_url
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:" + str(fastapi_request.url.port or 8913),
                timeout=httpx.Timeout(connect=30.0, read=1800.0, write=60.0, pool=60.0)  # 30 min read timeout for cold starts
            ) as client:
                resp = await client.post(inference_url, json=payload)
        
        resp.raise_for_status()
        data = resp.json()
        
        # Record user message (system + user)
        if tracer_instance is not None:
            try:
                print(f"[pokemon_red] Recording messages: tracer_instance={tracer_instance is not None}", flush=True)
                # Record system message
                await tracer_instance.record_message(
                    content=messages[0].get("content", ""),
                    message_type="system",
                )
                # Record user message
                user_msg_content = messages[1].get("content", "")
                if isinstance(user_msg_content, list):
                    # For multimodal content, extract text summary
                    text_parts = [item.get("text", "") for item in user_msg_content if item.get("type") == "text"]
                    user_msg_content = " ".join(text_parts) if text_parts else str(user_msg_content)
                await tracer_instance.record_message(
                    content=user_msg_content,
                    message_type="user",
                )
                print(f"[pokemon_red] Recorded user messages", flush=True)
            except Exception as exc:
                logger.debug(f"[pokemon_red] Failed to record user messages: {exc}")
                print(f"[pokemon_red] ERROR recording user messages: {exc}", flush=True)
        
        # Debug logging for tool calls
        print(f"\n{'='*80}")
        print(f"[pokemon_red] INFERENCE RESPONSE DEBUG")
        print(f"{'='*80}")
        print(f"Response status: {resp.status_code}")
        print(f"Response keys: {list(data.keys())}")
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            print(f"Message keys: {list(message.keys())}")
            print(f"Message content preview: {str(message.get('content', ''))[:200]}")
            print(f"Tool calls: {message.get('tool_calls', [])}")
            print(f"Full message (formatted):")
            print(_json_debug.dumps(message, indent=2)[:1500])
        print(f"{'='*80}\n")
        
        # Record assistant message/tool calls
        if tracer_instance is not None:
            try:
                message = choices[0].get("message", {}) if choices else {}
                tool_calls = message.get("tool_calls", [])
                content = message.get("content", "")
                
                if tool_calls:
                    # Record tool calls as assistant message
                    import json as _json_record
                    await tracer_instance.record_message(
                        content=_json_record.dumps(tool_calls) if tool_calls else (content or ""),
                        message_type="assistant",
                        metadata={"is_tool_call": True} if tool_calls else {},
                    )
                elif content:
                    # Record text content as assistant message
                    await tracer_instance.record_message(
                        content=content,
                        message_type="assistant",
                    )
            except Exception as exc:
                logger.debug(f"[pokemon_red] Failed to record assistant message: {exc}")
        
        # Extract first tool call
        if not choices:
            print("[pokemon_red] WARNING: No choices in inference response")
            return {}
        message = choices[0].get("message") or {}
        raw_calls = message.get("tool_calls") or []
        
        # If no structured tool_calls, try parsing XML tool calls from content
        if not raw_calls:
            content = message.get("content", "")
            if content and "<tool_call>" in content:
                import re as _re
                import json as _json_parse
                # Parse XML tool calls: <tool_call>{...}</tool_call>
                xml_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
                matches = _re.findall(xml_pattern, content, _re.DOTALL)
                if matches:
                    print(f"[pokemon_red] Parsed {len(matches)} XML tool call(s) from content")
                    try:
                        tool_data = _json_parse.loads(matches[0])
                        tool_name = tool_data.get("name", "")
                        args = tool_data.get("arguments", {})
                        
                        print(f"[pokemon_red] Parsed tool: {tool_name}, args: {str(args)[:200]}")
                        
                        # Handle execute_sequence tool
                        if tool_name == "execute_sequence":
                            return {"actions": args.get("actions", [])}
                        
                        # Handle press_button tool (legacy single action)
                        if tool_name == "press_button":
                            return {"button": args.get("button"), "frames": int(args.get("frames") or 30)}
                    except Exception as parse_err:
                        print(f"[pokemon_red] Error parsing XML tool call: {parse_err}")
        
        if not raw_calls:
            print(f"[pokemon_red] WARNING: No tool_calls in response. Content: {message.get('content', '')[:200]}")
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
                    
                    # Record environment event
                    if tracer_instance is not None:
                        try:
                            event = EnvironmentEvent(
                                system_instance_id="environment:pokemon_red",
                                time_record=TimeRecord(event_time=datetime.now(UTC).timestamp()),
                                reward=step_reward,
                                terminated=False,
                                truncated=False,
                                system_state_before={"map_id": prev_state.get("map_id"), "position": f"({prev_state.get('player_x')},{prev_state.get('player_y')})"},
                                system_state_after={"map_id": current_state.get("map_id"), "position": f"({current_state.get('player_x')},{current_state.get('player_y')})"},
                                metadata={"step": step_idx + 1, "button": button, "run_id": request.run_id},
                            )
                            await tracer_instance.record_event(event)
                        except Exception as exc:
                            logger.debug(f"[pokemon_red] Failed to record environment event: {exc}")
                    
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
                
                # Record environment event
                if tracer_instance is not None:
                    try:
                        event = EnvironmentEvent(
                            system_instance_id="environment:pokemon_red",
                            time_record=TimeRecord(event_time=datetime.now(UTC).timestamp()),
                            reward=step_reward,
                            terminated=False,
                            truncated=False,
                            system_state_before={"map_id": prev_state.get("map_id"), "position": f"({prev_state.get('player_x')},{prev_state.get('player_y')})"},
                            system_state_after={"map_id": current_state.get("map_id"), "position": f"({current_state.get('player_x')},{current_state.get('player_y')})"},
                            metadata={"step": step_idx + 1, "button": button, "run_id": request.run_id},
                        )
                        await tracer_instance.record_event(event)
                    except Exception as exc:
                        logger.debug(f"[pokemon_red] Failed to record environment event: {exc}")
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
                print(f"[pokemon_red] Calling _call_inference: tracer_instance={tracer_instance is not None}", flush=True)
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
                            
                            # Record environment event
                            if tracer_instance is not None:
                                try:
                                    event = EnvironmentEvent(
                                        system_instance_id="environment:pokemon_red",
                                        time_record=TimeRecord(event_time=datetime.now(UTC).timestamp()),
                                        reward=step_reward,
                                        terminated=False,
                                        truncated=False,
                                        system_state_before={"map_id": prev_state.get("map_id"), "position": f"({prev_state.get('player_x')},{prev_state.get('player_y')})"},
                                        system_state_after={"map_id": current_state.get("map_id"), "position": f"({current_state.get('player_x')},{current_state.get('player_y')})"},
                                        metadata={"step": step_idx + 1, "button": button, "run_id": request.run_id},
                                    )
                                    await tracer_instance.record_event(event)
                                except Exception as exc:
                                    logger.debug(f"[pokemon_red] Failed to record environment event: {exc}")
                            
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
            
            # Build trace payload if requested - ALWAYS use full format when return_trace=True
            # This ensures markov_blanket_message_history is always included
            record_config = getattr(request, 'record', None)
            print(f"[pokemon_red] TRACE DEBUG: record_config={record_config}, return_trace={getattr(record_config, 'return_trace', None) if record_config else None}, session_trace={session_trace is not None}", flush=True)
            if session_trace:
                print(f"[pokemon_red] TRACE DEBUG: IMMEDIATELY AFTER end_session: session_trace has {len(session_trace.markov_blanket_message_history)} messages, {len(session_trace.event_history)} events", flush=True)
                print(f"[pokemon_red] TRACE DEBUG: session_trace.markov_blanket_message_history type: {type(session_trace.markov_blanket_message_history)}", flush=True)
                if session_trace.markov_blanket_message_history:
                    print(f"[pokemon_red] TRACE DEBUG: First message type: {type(session_trace.markov_blanket_message_history[0])}, content: {str(session_trace.markov_blanket_message_history[0].content)[:100]}", flush=True)
                else:
                    print(f"[pokemon_red] TRACE DEBUG: WARNING - markov_blanket_message_history is EMPTY RIGHT AFTER end_session!", flush=True)
            
            if record_config and getattr(record_config, 'return_trace', False) and session_trace:
                # Always return full trace with all messages and events (no compact format)
                import dataclasses
                trace_payload = session_trace.to_dict()
                print(f"[pokemon_red] TRACE DEBUG: to_dict() returned keys: {list(trace_payload.keys())}", flush=True)
                print(f"[pokemon_red] TRACE DEBUG: to_dict() markov_blanket_message_history length: {len(trace_payload.get('markov_blanket_message_history', []))}", flush=True)
                
                # Always manually serialize messages and events to ensure they're included
                # asdict() may not recursively serialize nested dataclasses correctly
                from synth_ai.tracing_v3.abstractions import SessionEventMarkovBlanketMessage, BaseEvent
                if session_trace.markov_blanket_message_history:
                    print(f"[pokemon_red] TRACE DEBUG: Manually serializing {len(session_trace.markov_blanket_message_history)} messages", flush=True)
                    trace_payload["markov_blanket_message_history"] = [
                        dataclasses.asdict(msg) if isinstance(msg, SessionEventMarkovBlanketMessage) else (msg if isinstance(msg, dict) else str(msg))
                        for msg in session_trace.markov_blanket_message_history
                    ]
                else:
                    print(f"[pokemon_red] TRACE DEBUG: WARNING - session_trace.markov_blanket_message_history is EMPTY!", flush=True)
                if session_trace.event_history:
                    print(f"[pokemon_red] TRACE DEBUG: Manually serializing {len(session_trace.event_history)} events", flush=True)
                    trace_payload["event_history"] = [
                        dataclasses.asdict(evt) if isinstance(evt, BaseEvent) else (evt if isinstance(evt, dict) else str(evt))
                        for evt in session_trace.event_history
                    ]
                else:
                    print(f"[pokemon_red] TRACE DEBUG: WARNING - session_trace.event_history is EMPTY!", flush=True)
                print(f"[pokemon_red] TRACE DEBUG: Final trace payload has {len(trace_payload.get('markov_blanket_message_history', []))} messages, {len(trace_payload.get('event_history', []))} events", flush=True)
                print(f"[pokemon_red] TRACE DEBUG: Final trace payload keys: {list(trace_payload.keys())}", flush=True)
            else:
                print(f"[pokemon_red] TRACE DEBUG: SKIPPING trace payload build - record_config={record_config}, return_trace={getattr(record_config, 'return_trace', None) if record_config else None}, session_trace={session_trace is not None}", flush=True)
        except Exception as exc:
            logger.warning(f"[pokemon_red] tracing finalization failed: {exc}")
            print(f"[pokemon_red] TRACE DEBUG EXCEPTION: {exc}", flush=True)
            import traceback
            print(f"[pokemon_red] TRACE DEBUG EXCEPTION TRACEBACK: {traceback.format_exc()}", flush=True)
    
    # Fallback trace payload if no tracer but CLI needs it
    if trace_payload is None:
        record_config = getattr(request, 'record', None)
        print(f"[pokemon_red] TRACE DEBUG: trace_payload is None, using fallback. record_config={record_config}, return_trace={getattr(record_config, 'return_trace', None) if record_config else None}", flush=True)
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
            print(f"[pokemon_red] TRACE DEBUG: Created fallback trace_payload with keys: {list(trace_payload.keys())}", flush=True)
    
    print(f"[pokemon_red] TRACE DEBUG: About to return RolloutResponse with trace_payload={trace_payload is not None}, keys={list(trace_payload.keys()) if trace_payload else []}", flush=True)
    if trace_payload:
        import json as _json_final
        markov_msgs = trace_payload.get('markov_blanket_message_history', [])
        event_history = trace_payload.get('event_history', [])
        print(f"[pokemon_red] TRACE DEBUG: trace_payload markov_blanket_message_history length: {len(markov_msgs)}", flush=True)
        print(f"[pokemon_red] TRACE DEBUG: trace_payload event_history length: {len(event_history)}", flush=True)
        if markov_msgs:
            print(f"[pokemon_red] TRACE DEBUG: First markov message type: {type(markov_msgs[0]) if markov_msgs else None}", flush=True)
            print(f"[pokemon_red] TRACE DEBUG: First markov message (first 500 chars): {_json_final.dumps(markov_msgs[0] if markov_msgs else {}, indent=2, default=str)[:500]}", flush=True)
        else:
            print(f"[pokemon_red] TRACE DEBUG: WARNING - markov_blanket_message_history is EMPTY in final trace_payload!", flush=True)
    
    response = RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(request.ops or []),
        trace=trace_payload,
    )
    
    # Final check: inspect what's actually in the response
    if response.trace:
        import json as _json_response
        resp_markov = response.trace.get('markov_blanket_message_history', []) if isinstance(response.trace, dict) else []
        print(f"[pokemon_red] TRACE DEBUG: Response.trace markov_blanket_message_history length: {len(resp_markov)}", flush=True)
    
    return response


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
        description="Pokémon Red demo task app with vision support",
        config_factory=build_config,
        aliases=("pokemon_red_demo",),
        env_files=(),
        modal=ModalDeploymentConfig(
            app_name="pokemon-red-vision-task-app",
            python_version="3.11",
            pip_packages=(
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0",
                "numpy>=1.24.0",
                "aiohttp>=3.8.0",
                "httpx>=0.24.0",
                "python-dotenv>=1.0.1",
                # Tracing/DB runtime deps
                "sqlalchemy>=2.0.42",
                "aiosqlite>=0.21.0",
                "greenlet>=3.2.3",
                # Pokemon Red environment
                "pyboy>=2.0.0",
                "pillow>=9.0.0",
            ),
            extra_local_dirs=(
                # Mount repo root so local modules resolve when deployed on Modal
                ("/Users/joshpurtell/Documents/GitHub/synth-ai", "/opt/synth_ai_repo"),
                ("/Users/joshpurtell/Documents/GitHub/synth-ai/synth_ai", "/opt/synth_ai_repo/synth_ai"),
                ("/Users/joshpurtell/Documents/GitHub/synth-ai/examples/task_apps/pokemon_red", "/opt/synth_ai_repo/examples/task_apps/pokemon_red"),
            ),
            secret_names=("openai-api-key", "groq-api-key"),
            memory=16384,
            cpu=4.0,
            max_containers=10,
        ),
    )
)


