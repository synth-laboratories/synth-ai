"""Crafter baseline file for self-contained evaluation.

This baseline file defines how to evaluate agents on Crafter without
requiring a deployed task app. It includes train/val/test splits and
computes both event rewards (achievement deltas) and outcome rewards
(total unique achievements).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
)
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.inference import InferenceClient
from synth_ai.tracing_v3.session_tracer import SessionTracer
import os


# Action mapping: string names to action indices
CRAFTER_ACTION_MAP: Dict[str, int] = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}


def format_crafter_observation(obs: Dict[str, Any]) -> str:
    """Format Crafter observation as text for LLM."""
    health = obs.get("health") or obs.get("inventory", {}).get("health", 0)
    inventory = obs.get("inventory", {})
    pos = obs.get("player_position", [0, 0])
    achievements_status = obs.get("achievements_status", {})
    
    # Format inventory (skip health)
    inv_items = [f"{k}:{v}" for k, v in inventory.items() if v > 0 and k != "health"]
    inventory_str = ", ".join(inv_items) if inv_items else "empty"
    
    # Format achievements
    achieved_list = [k for k, v in achievements_status.items() if v]
    achievements_str = ", ".join(achieved_list) if achieved_list else "none"
    
    return f"""Crafter Game State:
- Health: {health}/10
- Hunger: {inventory.get('hunger', 0)}/10
- Position: {pos}
- Inventory: {inventory_str}
- Achievements unlocked: {len(achieved_list)}/22
- Achievements: {achievements_str}

What actions should we take?"""


class CrafterTaskRunner(BaselineTaskRunner):
    """Task runner for Crafter survival game."""
    
    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)
        
        # Initialize inference client
        inference_url = policy_config.get("inference_url")
        if inference_url and inference_url.startswith("http"):
            # External URL - use InferenceClient
            api_key = os.getenv("SYNTH_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = inference_url.rstrip("/")
            if not base_url.endswith("/api"):
                base_url = f"{base_url}/api" if "/api" not in base_url else base_url
            self.client = InferenceClient(base_url=base_url, api_key=api_key)
            self.use_inference_client = True
        else:
            # For OpenAI/Groq direct APIs, we'll use httpx
            import httpx
            self.http_client = httpx.AsyncClient()
            self.use_inference_client = False
        
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 512)
        
        # System prompt
        self.system_prompt = """You are playing Crafter, a survival game. Your goal is to unlock achievements.

Core rules:
- The world contains trees (wood), stone, coal, iron, plants, cows, zombies, and water.
- Movement constraints: you cannot walk onto blocking tiles (tree, stone, water, lava, coal, iron).
- You start with empty hands and low health/hunger.
- Interact ('do') only when adjacent to a resource.
- Movement is essential: move multiple steps in one turn to explore.

Available actions: noop, move_up, move_down, move_left, move_right, do, sleep, 
place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, 
make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword

Always return a tool call: interact_many({actions: [...]})
Use 2-5 actions per call. Prefer long movement sequences."""
        
        # Tool definition
        self.tools = [{
            "type": "function",
            "function": {
                "name": "interact_many",
                "description": "Execute multiple Crafter actions in sequence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(CRAFTER_ACTION_MAP.keys())},
                            "description": "List of actions to execute",
                        }
                    },
                    "required": ["actions"],
                },
            },
        }]
    
    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Crafter episode and return results."""
        
        # Create task instance
        difficulty = self.env_config.get("difficulty", "normal")
        max_steps = self.env_config.get("max_steps", 100)
        
        impetus = Impetus(instructions="Survive and unlock achievements.")
        intent = Intent(
            rubric={"goal": "Unlock achievements"},
            gold_trajectories=None,
            gold_state_diff={},
        )
        metadata = CrafterTaskInstanceMetadata(
            difficulty=difficulty,
            seed=seed,
            num_trees_radius=0,
            num_cows_radius=0,
            num_hostiles_radius=0,
        )
        task_instance = CrafterTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        
        # Attach config
        task_instance.config = {"seed": seed, "length": 256, "area": [64, 64]}
        
        # Create environment
        env = CrafterClassicEnvironment(task_instance=task_instance)
        
        # Setup tracing
        tracer: Optional[SessionTracer] = None
        session_id: Optional[str] = None
        if self.env_config.get("enable_tracing", True):
            tracer = SessionTracer(db_url=None, auto_save=False)
            await tracer.initialize()
            session_id = tracer.create_session(metadata={
                "seed": seed,
                "difficulty": difficulty,
                "model": self.policy_config["model"],
            })
        
        # Initialize environment
        raw_obs = await env.initialize()
        observation = getattr(raw_obs, "observation", raw_obs) if hasattr(raw_obs, "observation") else raw_obs
        obs_dict = observation if isinstance(observation, dict) else {}
        
        # Track achievements
        prev_achievements: Set[str] = set()
        if isinstance(obs_dict.get("achievements_status"), dict):
            prev_achievements = {
                k for k, v in obs_dict.get("achievements_status", {}).items() if v
            }
        
        event_rewards: List[Dict[str, Any]] = []
        total_steps = 0
        tool_calls_history: List[Dict[str, Any]] = []
        
        # Episode loop
        for step in range(max_steps):
            # Format observation
            obs_text = format_crafter_observation(obs_dict)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{obs_text}\n\nPrevious tool calls: {tool_calls_history[-3:]}"},
            ]
            
            # Record LLM event
            llm_event_id = None
            if tracer and session_id:
                llm_event_id = tracer.record_event(
                    session_id=session_id,
                    event_type="cais",
                    data={"messages": messages, "step": step},
                )
            
            # Get action from LLM
            if self.use_inference_client:
                response = await self.client.create_chat_completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice={"type": "function", "function": {"name": "interact_many"}},
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                # Fallback: use OpenAI-compatible API
                import httpx
                import json as json_lib
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""
                base_url = "https://api.openai.com/v1" if "openai" in self.model.lower() else "https://api.groq.com/openai/v1"
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "tools": self.tools,
                            "tool_choice": {"type": "function", "function": {"name": "interact_many"}},
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                    )
                    response = resp.json()
            
            # Parse tool call
            tool_calls = []
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0].get("message", {})
                tool_calls = message.get("tool_calls", [])
            elif "tool_calls" in response:
                tool_calls = response["tool_calls"]
            
            if not tool_calls:
                break
            
            tool_call = tool_calls[0]
            actions = tool_call["function"]["arguments"].get("actions", [])
            tool_calls_history.append({"step": step, "actions": actions})
            
            # Execute actions
            for action_name in actions:
                if total_steps >= max_steps:
                    break
                
                # Map action string to index
                action_idx = CRAFTER_ACTION_MAP.get(action_name, 0)
                
                # Step environment
                step_result = await env.step(action_idx)
                total_steps += 1
                
                # Get observation from step result
                step_obs = getattr(step_result, "observation", step_result) if hasattr(step_result, "observation") else step_result
                obs_dict = step_obs if isinstance(step_obs, dict) else {}
                
                # Record environment event
                env_event_id = None
                if tracer and session_id:
                    env_event_id = tracer.record_event(
                        session_id=session_id,
                        event_type="environment",
                        data={
                            "action": action_name,
                            "reward": getattr(step_result, "reward", 0.0),
                            "terminated": getattr(step_result, "terminated", False),
                            "step": total_steps,
                        },
                    )
                
                # Check for new achievements
                current_achievements: Set[str] = set()
                if isinstance(obs_dict.get("achievements_status"), dict):
                    current_achievements = {
                        k for k, v in obs_dict.get("achievements_status", {}).items() if v
                    }
                
                new_achievements = current_achievements - prev_achievements
                
                if new_achievements:
                    event_reward_value = len(new_achievements)
                    if tracer and session_id and env_event_id:
                        tracer.record_event_reward(
                            session_id=session_id,
                            event_id=env_event_id,
                            reward_value=float(event_reward_value),
                            reward_type="achievement_delta",
                            key="achievements",
                            annotation={"new_achievements": list(new_achievements)},
                            source="environment",
                        )
                    event_rewards.append({
                        "step": total_steps,
                        "reward": event_reward_value,
                        "achievements": list(new_achievements),
                    })
                
                prev_achievements = current_achievements
                
                # Check termination
                if getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False):
                    break
            
            if getattr(step_result, "terminated", False) or getattr(step_result, "truncated", False):
                break
        
        # Compute outcome reward
        unique_achievements = len(prev_achievements)
        if tracer and session_id:
            tracer.record_outcome_reward(
                session_id=session_id,
                total_reward=unique_achievements,
                achievements_count=unique_achievements,
                total_steps=total_steps,
                reward_metadata={
                    "achievements": list(prev_achievements),
                },
            )
            
            # Export trace
            trace_dict = await tracer.export_session(session_id)
        else:
            trace_dict = None
        
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=float(unique_achievements),
            event_rewards=event_rewards,
            total_steps=total_steps,
            metadata={
                "achievements": list(prev_achievements),
                "achievement_count": unique_achievements,
                "difficulty": difficulty,
            },
            trace=trace_dict,
        )


# Define baseline config
crafter_baseline = BaselineConfig(
    baseline_id="crafter",
    name="Crafter Survival",
    description="Crafter survival game with achievement tracking",
    task_runner=CrafterTaskRunner,
    splits={
        "train": DataSplit(
            name="train",
            seeds=list(range(100)),
            metadata={"difficulty": "normal"},
        ),
        "val": DataSplit(
            name="val",
            seeds=list(range(100, 150)),
            metadata={"difficulty": "normal"},
        ),
        "test": DataSplit(
            name="test",
            seeds=list(range(150, 200)),
            metadata={"difficulty": "hard"},
        ),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    default_env_config={
        "difficulty": "normal",
        "max_steps": 100,
        "enable_tracing": True,
    },
    metadata={
        "environment": "crafter",
        "reward_type": "achievements",
        "max_achievements": 22,
    },
    tags=["rl", "gym", "survival", "achievements"],
)

