"""Pokemon Red baseline file for Game Boy emulation evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.inference import InferenceClient
import os
import httpx

try:
    from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
    from synth_ai.environments.examples.red.taskset import (
        PokemonRedTaskInstance,
        PokemonRedTaskInstanceMetadata,
    )
    POKEMON_RED_AVAILABLE = True
except ImportError:
    POKEMON_RED_AVAILABLE = False


class PokemonRedTaskRunner(BaselineTaskRunner):
    """Task runner for Pokemon Red Game Boy emulation."""
    
    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)
        
        if not POKEMON_RED_AVAILABLE:
            raise ImportError(
                "Pokemon Red environment not available. "
                "Install synth-ai with Pokemon Red support."
            )
        
        # Store config for inference
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 512)
        self.inference_url = policy_config.get("inference_url")
        
        # Tool definition
        self.tools = [{
            "type": "function",
            "function": {
                "name": "execute_sequence",
                "description": "Execute multiple button presses in sequence",
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
                                    },
                                    "frames": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 120,
                                        "description": "Frames to hold button (60fps)",
                                    },
                                },
                                "required": ["button", "frames"],
                            },
                            "minItems": 1,
                            "maxItems": 20,
                        },
                    },
                    "required": ["actions"],
                },
            },
        }]
    
    def _format_observation(self, obs: Dict[str, Any], step: int, max_steps: int) -> str:
        """Format observation for LLM."""
        lines = [
            f"Pokemon Red - Step {step}/{max_steps}",
            "",
        ]
        
        # Position
        if "map_id" in obs:
            lines.append(f"Location: Map {obs['map_id']}")
        if "player_x" in obs and "player_y" in obs:
            lines.append(f"Position: ({obs['player_x']}, {obs['player_y']})")
        
        # Party
        if "party_count" in obs:
            lines.append(f"Party Size: {obs['party_count']}")
        if "party_pokemon" in obs and obs["party_pokemon"]:
            pokemon = obs["party_pokemon"][0]
            lines.append(
                f"First Pokemon: Level {pokemon.get('level', '?')}, "
                f"HP {pokemon.get('hp_current', '?')}/{pokemon.get('hp_max', '?')}"
            )
        
        # Battle
        if obs.get("in_battle"):
            lines.append("=== IN BATTLE ===")
            if "enemy_hp_current" in obs:
                lines.append(
                    f"Enemy HP: {obs['enemy_hp_current']}/{obs.get('enemy_hp_max', '?')}"
                )
            if "battle_turn" in obs:
                lines.append(f"Battle Turn: {obs['battle_turn']}")
        
        # Progress
        if "badges" in obs:
            lines.append(f"Badges: {obs['badges']}")
        if "money" in obs:
            lines.append(f"Money: ${obs['money']}")
        
        # Dialogue
        if obs.get("text_box_active"):
            lines.append("Text box is active - press A to advance dialogue")
        
        lines.append("")
        lines.append("What actions should we take?")
        
        return "\n".join(lines)
    
    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Pokemon Red episode."""
        
        # Create task instance
        rom_path = self.env_config.get("rom_path")
        if not rom_path:
            raise ValueError("rom_path required in env_config for Pokemon Red")
        
        init_state_path = self.env_config.get("init_state_path")
        max_steps = self.env_config.get("max_steps", 500)
        
        metadata = PokemonRedTaskInstanceMetadata(
            seed=seed,
            rom_path=rom_path,
            init_state_path=init_state_path,
            reward_type=self.env_config.get("reward_type", "pallet_town_progression"),
        )
        
        task_instance = PokemonRedTaskInstance(
            id=f"pokemon-red-{seed}",
            metadata=metadata,
        )
        
        # Create environment
        env = PokemonRedEnvironment(task_instance=task_instance)
        
        # Initialize environment
        raw_obs = await env.initialize()
        observation = getattr(raw_obs, "observation", raw_obs) if hasattr(raw_obs, "observation") else raw_obs
        obs_dict = observation if isinstance(observation, dict) else {}
        
        # Episode loop
        total_reward = 0.0
        total_steps = 0
        event_rewards: List[Dict[str, Any]] = []
        battle_won = False
        game_over = False
        
        for step in range(max_steps):
            # Format observation
            prompt = self._format_observation(obs_dict, step, max_steps)
            
            # Add image if available
            messages = [{"role": "user", "content": prompt}]
            if obs_dict.get("observation_image_base64"):
                messages[0]["content"] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{obs_dict['observation_image_base64']}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            
            # Get action from LLM
            if self.inference_url and self.inference_url.startswith("http"):
                api_key = os.getenv("SYNTH_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
                base_url = self.inference_url.rstrip("/")
                if not base_url.endswith("/api"):
                    base_url = f"{base_url}/api" if "/api" not in base_url else base_url
                client = InferenceClient(base_url=base_url, api_key=api_key)
                response = await client.create_chat_completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice={"type": "function", "function": {"name": "execute_sequence"}},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
                base_url = "https://api.openai.com/v1" if "openai" in self.model.lower() else "https://api.groq.com/openai/v1"
                async with httpx.AsyncClient() as http_client:
                    resp = await http_client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "tools": self.tools,
                            "tool_choice": {"type": "function", "function": {"name": "execute_sequence"}},
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                    )
                    response = resp.json()
            
            # Extract actions
            actions = []
            tool_calls = []
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0].get("message", {})
                tool_calls = message.get("tool_calls", [])
            elif "tool_calls" in response:
                tool_calls = response["tool_calls"]
            
            if tool_calls:
                tool_call = tool_calls[0]
                actions = tool_call["function"]["arguments"].get("actions", [])
            
            if not actions:
                break
            
            # Execute actions
            for action_spec in actions:
                if total_steps >= max_steps:
                    break
                
                # Convert to tool call format
                from synth_ai.environments.environment.tools import EnvToolCall
                
                tool_call = EnvToolCall(
                    name="execute_sequence",
                    arguments={"actions": [action_spec]},
                )
                
                # Step environment
                step_result = await env.step([tool_call])
                total_steps += 1
                
                # Get observation
                step_obs = (
                    getattr(step_result, "observation", step_result)
                    if hasattr(step_result, "observation")
                    else step_result
                )
                obs_dict = step_obs if isinstance(step_obs, dict) else {}
                
                # Extract reward
                reward = getattr(step_result, "reward", 0.0)
                total_reward += reward
                
                if reward > 0:
                    event_rewards.append({
                        "step": total_steps,
                        "reward": reward,
                    })
                
                # Check termination
                if getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False):
                    game_over = True
                    break
                
                # Check battle outcome
                if obs_dict.get("battle_outcome") == 1:
                    battle_won = True
                elif obs_dict.get("battle_outcome") == 2:
                    game_over = True
            
            if game_over:
                break
        
        # Cleanup
        await env.terminate()
        
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=total_reward,
            event_rewards=event_rewards,
            total_steps=total_steps,
            metadata={
                "battle_won": battle_won,
                "game_over": game_over,
                "final_map": obs_dict.get("map_id"),
                "badges": obs_dict.get("badges", 0),
                "party_size": obs_dict.get("party_count", 0),
            },
        )


# Define baseline config (only if Pokemon Red is available)
if POKEMON_RED_AVAILABLE:
    pokemon_vl_baseline = BaselineConfig(
        baseline_id="pokemon_vl",
        name="Pokemon VL - Pokemon Red",
        description="Pokemon Red Game Boy emulation baseline for vision-language agents",
        task_runner=PokemonRedTaskRunner,
        splits={
            "train": DataSplit(name="train", seeds=list(range(20))),
            "val": DataSplit(name="val", seeds=list(range(20, 25))),
            "test": DataSplit(name="test", seeds=list(range(25, 30))),
        },
        default_policy_config={
            "model": "groq:llama-3.1-70b-versatile",
            "temperature": 0.0,
            "max_tokens": 512,
        },
        default_env_config={
            "rom_path": None,  # Must be provided
            "init_state_path": None,  # Optional
            "reward_type": "pallet_town_progression",
            "max_steps": 500,
        },
        metadata={
            "environment": "pokemon_red",
            "task_type": "emulation",
            "requires_rom": True,
        },
    )

