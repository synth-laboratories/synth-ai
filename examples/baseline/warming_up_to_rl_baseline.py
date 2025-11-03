"""Warming Up to RL baseline file for Gymnasium environments."""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.inference import InferenceClient
import os
import httpx


class WarmingUpToRLTaskRunner(BaselineTaskRunner):
    """Task runner for Gymnasium environments (CartPole, FrozenLake, etc.)."""
    
    def __init__(self, policy_config: Dict[str, Any], env_config: Dict[str, Any]):
        super().__init__(policy_config, env_config)
        
        # Store config for inference
        self.model = policy_config["model"]
        self.temperature = policy_config.get("temperature", 0.0)
        self.max_tokens = policy_config.get("max_tokens", 128)
        self.inference_url = policy_config.get("inference_url")
        
        # Environment name
        self.env_name = env_config.get("env_name", "CartPole-v1")
    
    def _get_action_tool(self, env: gym.Env) -> Dict[str, Any]:
        """Generate tool schema based on environment action space."""
        if isinstance(env.action_space, gym.spaces.Discrete):
            return {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": f"Take action in {env.spec.id if env.spec else self.env_name}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": env.action_space.n - 1,
                                "description": "Action index",
                            }
                        },
                        "required": ["action"],
                    },
                },
            }
        else:
            # Default for unknown action spaces
            return {
                "type": "function",
                "function": {
                    "name": "take_action",
                    "description": "Take action in the environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "integer",
                                "description": "Action index",
                            }
                        },
                        "required": ["action"],
                    },
                },
            }
    
    def _format_observation(self, obs: Any, env: gym.Env, step: int, max_steps: int) -> str:
        """Format observation for LLM."""
        obs_str = str(obs)
        if hasattr(env, "spec") and env.spec:
            env_id = env.spec.id
        else:
            env_id = self.env_name
        
        return f"""Environment: {env_id}
Step: {step}/{max_steps}
Observation: {obs_str}

What action should we take?"""
    
    async def run_task(self, seed: int) -> TaskResult:
        """Run a single Gymnasium episode."""
        
        # Create environment
        env = gym.make(self.env_name)
        
        # Reset with seed
        obs, info = env.reset(seed=seed)
        
        # Get action tool
        action_tool = self._get_action_tool(env)
        
        # Episode loop
        total_reward = 0.0
        total_steps = 0
        max_steps = self.env_config.get("max_steps", 500)
        
        terminated = False
        truncated = False
        
        for step in range(max_steps):
            # Format observation
            prompt = self._format_observation(obs, env, step, max_steps)
            
            # Get action from LLM
            messages = [{"role": "user", "content": prompt}]
            
            if self.inference_url and self.inference_url.startswith("http"):
                api_key = os.getenv("SYNTH_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
                base_url = self.inference_url.rstrip("/")
                if not base_url.endswith("/api"):
                    base_url = f"{base_url}/api" if "/api" not in base_url else base_url
                client = InferenceClient(base_url=base_url, api_key=api_key)
                response = await client.create_chat_completion(
                    model=self.model,
                    messages=messages,
                    tools=[action_tool],
                    tool_choice={"type": "function", "function": {"name": "take_action"}},
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
                            "tools": [action_tool],
                            "tool_choice": {"type": "function", "function": {"name": "take_action"}},
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        },
                        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                    )
                    response = resp.json()
            
            # Extract action
            action = 0
            tool_calls = []
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0].get("message", {})
                tool_calls = message.get("tool_calls", [])
            elif "tool_calls" in response:
                tool_calls = response["tool_calls"]
            
            if tool_calls:
                action = tool_calls[0]["function"]["arguments"].get("action", 0)
            else:
                # Fallback: sample random action
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=total_reward,
            total_steps=total_steps,
            metadata={
                "env_name": self.env_name,
                "episode_length": total_steps,
                "terminated": terminated,
                "truncated": truncated,
                "final_reward": total_reward,
            },
        )


# Define baseline configs for different environments
cartpole_baseline = BaselineConfig(
    baseline_id="cartpole",
    name="CartPole-v1",
    description="Balance a pole on a cart using Gymnasium",
    task_runner=WarmingUpToRLTaskRunner,
    splits={
        "train": DataSplit(name="train", seeds=list(range(100))),
        "val": DataSplit(name="val", seeds=list(range(100, 120))),
        "test": DataSplit(name="test", seeds=list(range(120, 140))),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 128,
    },
    default_env_config={
        "env_name": "CartPole-v1",
        "max_steps": 500,
    },
    metadata={
        "environment": "CartPole-v1",
        "task_type": "control",
        "max_reward": 500,
    },
    tags=["rl", "gymnasium", "control"],
)

frozenlake_baseline = BaselineConfig(
    baseline_id="frozenlake",
    name="FrozenLake-v1",
    description="Navigate a frozen lake to reach goal using Gymnasium",
    task_runner=WarmingUpToRLTaskRunner,
    splits={
        "train": DataSplit(name="train", seeds=list(range(100))),
        "val": DataSplit(name="val", seeds=list(range(100, 120))),
        "test": DataSplit(name="test", seeds=list(range(120, 140))),
    },
    default_policy_config={
        "model": "groq:llama-3.1-70b-versatile",
        "temperature": 0.0,
        "max_tokens": 128,
    },
    default_env_config={
        "env_name": "FrozenLake-v1",
        "max_steps": 100,
    },
    metadata={
        "environment": "FrozenLake-v1",
        "task_type": "navigation",
        "max_reward": 1,
    },
    tags=["rl", "gymnasium", "navigation"],
)

