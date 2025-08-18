#!/usr/bin/env python3
"""
Comprehensive script to run Crafter rollouts for multiple models and compare their performance.
Updated to use tracing_v3 with async architecture.

Runs experiments for:
- gpt-4o-mini
- gpt-4.1-mini  
- gpt-4.1-nano
- gemini-1.5-flash
- gemini-2.5-flash-lite
- qwen3/32b

Analyzes and compares:
- Invalid action rates
- Achievement frequencies by step
- Achievement counts across models
- Performance metrics
- Cost analysis
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio as atqdm

# Disable httpx logging for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

# Disable v1 logging to see v3 tracing clearly
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["SYNTH_LOGGING"] = "false"

# Import enhanced LM with v3 tracing
from synth_ai.lm.core.main_v3 import LM
from synth_ai.tracing_v3.abstractions import (
    EnvironmentEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    TimeRecord,
)
from synth_ai.tracing_v3.decorators import set_turn_number

# Import session tracer for v3 tracing
from synth_ai.tracing_v3.session_tracer import SessionTracer

# from synth_ai.tracing_v3.utils import create_experiment_context  # Not needed
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager

# Import Crafter hooks
try:
    from synth_ai.environments.examples.crafter_classic.trace_hooks_v3 import CRAFTER_HOOKS
    print(f"✅ Loaded {len(CRAFTER_HOOKS.hooks)} Crafter achievement hooks (Easy, Medium, Hard)")
except ImportError:
    print("Warning: Could not import CRAFTER_HOOKS for v3")
    from synth_ai.tracing_v3.hooks import HookManager
    CRAFTER_HOOKS = HookManager()

import random

import httpx

# Global buckets for sessions
_SESSIONS: dict[str, tuple[str, object]] = {}  # session_id -> (experiment_id, trace)

# Configuration
MODELS_TO_TEST = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
]

# Service URLs (modify these based on your setup)
CRAFTER_SERVICE_URL = "http://localhost:8901"

# Database configuration - uses the centralized config which matches serve.sh
from synth_ai.tracing_v3.db_config import get_default_db_config

db_config = get_default_db_config()
DATABASE_URL = db_config.database_url

# Retry configuration for HTTP requests
MAX_RETRIES = 3
BASE_DELAY = 0.1
MAX_DELAY = 2.0
HTTP_TIMEOUT = 30.0

class ExperimentConfig:
    """Configuration for the multi-model experiment."""
    
    def __init__(self):
        self.num_episodes = 10  # Number of episodes per model
        self.max_turns = 100    # Max turns per episode
        self.difficulty = "easy"
        self.save_traces = True
        self.verbose = True
        self.quiet = False      # Default to verbose mode
        self.enable_v3_tracing = True
        self.v3_trace_dir = "./traces"
        self.crafter_service_url = CRAFTER_SERVICE_URL
        self.database_url = DATABASE_URL
        self.base_seed = 1000   # Base seed for episode generation
        self.turn_timeout = 30.0  # Timeout per turn in seconds
        self.episode_timeout = 300.0  # Total timeout per episode in seconds


async def retry_http_request(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Any:
    """Retry HTTP requests with exponential backoff and jitter."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter
                await asyncio.sleep(total_delay)
            
            response = await client.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            
            if response.status_code < 500:
                return response
            
            last_exception = Exception(f"HTTP {response.status_code}: {response.text}")
            
        except httpx.ConnectError as e:
            last_exception = Exception(f"Connection failed to {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1.0 * (2 ** attempt))
        except httpx.ReadError as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                read_error_delay = min(1.0 * (2 ** attempt), 5.0)
                await asyncio.sleep(read_error_delay)
        except Exception as e:
            last_exception = e
    
    print(f"    ❌ HTTP request failed after {MAX_RETRIES} attempts: {method} {url}")
    print(f"    ❌ Error: {type(last_exception).__name__}: {str(last_exception)[:200]}")
    raise last_exception


# Crafter action mapping
CRAFTER_ACTIONS = {
    "noop": 0, "move_left": 1, "move_right": 2, "move_up": 3, "move_down": 4,
    "do": 5, "sleep": 6, "place_stone": 7, "place_table": 8, "place_furnace": 9,
    "place_plant": 10, "make_wood_pickaxe": 11, "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13, "make_wood_sword": 14, "make_stone_sword": 15,
    "make_iron_sword": 16, "eat_cow": 17, "eat_plant": 18
}

# Create reverse mapping for validation
INT_TO_ACTION_STRING = {v: k for k, v in CRAFTER_ACTIONS.items()}


def compress_observation_for_trace(obs: dict[str, Any]) -> str:
    """Compress observation data for storage in traces."""
    try:
        return json.dumps({
            "inv": {k: v for k, v in obs.get("inventory", {}).items() if v > 0},
            "nearby": obs.get("nearby", []),
            "hp": obs.get("status", {}).get("health", 0),
            "food": obs.get("status", {}).get("food", 0),
            "ach": sum(1 for v in obs.get("achievements_status", {}).values() if v)
        }, separators=(',', ':'))
    except Exception as e:
        return f"{{\"error\": \"{str(e)}\"}}"


def create_message(content: str, message_type: str, system_id: str, turn: int) -> SessionEventMarkovBlanketMessage:
    """Create a SessionEventMarkovBlanketMessage with metadata."""
    return SessionEventMarkovBlanketMessage(
        content=content,
        message_type=message_type,
        metadata={"system_id": system_id, "turn": turn},
        time_record=TimeRecord(
            event_time=time.time(),
            message_time=turn
        )
    )


async def run_episode(config: ExperimentConfig, 
                     model_name: str, 
                     episode_num: int,
                     experiment_id: str) -> dict[str, Any]:
    """Run a single episode with a specific model using v3 tracing."""
    # Create a new session tracer for this episode
    session_tracer = SessionTracer(hooks=CRAFTER_HOOKS, db_url=config.database_url)
    
    # Start session with metadata
    session_id = await session_tracer.start_session(
        metadata={
            "model": model_name,
            "episode": episode_num,
            "experiment_id": experiment_id,
            "difficulty": config.difficulty
        }
    )
    
    # Started tracing session (output disabled for clean UI)
    
    # Store session in global bucket
    _SESSIONS[session_id] = (experiment_id, session_tracer)
    
    # Initialize LM with session tracer
    lm = LM(
        vendor="openai",
        model=model_name,
        temperature=0.1,  # Low temperature for more consistent gameplay
        session_tracer=session_tracer,
        system_id=f"crafter_agent_{model_name}",
        enable_v3_tracing=True
    )
    
    # Create HTTP client
    async with httpx.AsyncClient() as client:
        try:
            # Initialize environment with consecutive seed
            seed = config.base_seed + episode_num  # Base seed + episode number for consecutive seeds
            request_data = {"config": {"difficulty": config.difficulty, "seed": seed}}
            init_response = await retry_http_request(
                client, "POST", f"{config.crafter_service_url}/env/CrafterClassic/initialize",
                json=request_data
            )
            init_data = init_response.json()
            
            # Debug the response format (removed for clean output)
            
            # Handle different possible response formats
            if "instance_id" in init_data:
                instance_id = init_data["instance_id"]
            elif "env_id" in init_data:
                instance_id = init_data["env_id"]
            elif "id" in init_data:
                instance_id = init_data["id"]
            else:
                # If none of the expected keys exist, print the response and raise a clear error
                print(f"❌ Unexpected response format from Crafter service: {init_data}")
                raise KeyError(f"Could not find environment ID in response. Available keys: {list(init_data.keys())}")
            
            # Get initial observation (from initialize response)
            obs = init_data["observation"]
            
            prev_obs = obs
            done = False
            invalid_actions = 0
            total_actions = 0
            episode_start_time = time.time()
            
            for turn in range(config.max_turns):
                if done:
                    break
                
                # Check episode timeout
                if time.time() - episode_start_time > config.episode_timeout:
                    print(f"    ⏰ Episode {episode_num} timed out after {config.episode_timeout}s")
                    done = True
                    break
                
                # Update progress bar
                if hasattr(config, '_pbar'):
                    current_achievements = sum(1 for v in obs.get("achievements_status", {}).values() if v)
                    config._pbar.set_postfix({
                        f"ep{episode_num}": f"step {turn+1}/{config.max_turns}, ach: {current_achievements}"
                    })
                
                set_turn_number(turn)
                
                # Start timestep for this turn
                await session_tracer.start_timestep(f"turn_{turn}")
                
                # Prepare context for the agent
                inventory_str = ", ".join([f"{k}: {v}" for k, v in obs.get("inventory", {}).items() if v > 0])
                if not inventory_str:
                    inventory_str = "empty"
                
                nearby_str = ", ".join(obs.get("nearby", []))
                if not nearby_str:
                    nearby_str = "nothing"
                
                status = obs.get("status", {})
                health = status.get("health", 0)
                hunger = status.get("food", 0)
                
                # Get more detailed game state
                position = obs.get("position", [0, 0])
                achievements = obs.get("achievements_status", {})
                unlocked = [name for name, status in achievements.items() if status]
                achievements_str = ", ".join(unlocked) if unlocked else "none"
                
                # Get semantic map if available
                semantic_map = obs.get("semantic_map", None)
                map_str = ""
                if semantic_map is not None:
                    # Simple 5x5 view around player
                    try:
                        px, py = position
                        view_size = 5
                        half = view_size // 2
                        map_lines = []
                        for dy in range(-half, half + 1):
                            row = []
                            for dx in range(-half, half + 1):
                                x, y = px + dx, py + dy
                                if dx == 0 and dy == 0:
                                    row.append("@")  # Player
                                elif 0 <= x < len(semantic_map) and 0 <= y < len(semantic_map[0]):
                                    cell = semantic_map[x][y]
                                    # Map common items
                                    if cell == 0:
                                        row.append(".")  # Empty/grass
                                    elif cell == 1:
                                        row.append("T")  # Tree
                                    elif cell == 2:
                                        row.append("S")  # Stone
                                    elif cell == 3:
                                        row.append("C")  # Cow
                                    elif cell == 4:
                                        row.append("W")  # Water
                                    else:
                                        row.append("?")
                                else:
                                    row.append("#")  # Out of bounds
                            map_lines.append(" ".join(row))
                        map_str = "\nMap (5x5 view, @ = you):\n" + "\n".join(map_lines)
                    except Exception:
                        map_str = "\nMap view unavailable"
                
                # Create agent prompt
                prompt = f"""Game State (Turn {turn}):
- Position: {position}
- Health: {health}/9
- Hunger: {hunger}/9
- Inventory: {inventory_str}
- Nearby objects: {nearby_str}
- Achievements unlocked: {achievements_str}
{map_str}

Choose your next actions based on what you see. Use the 'interact' tool with a list of action IDs.

Tips:
- Look at the map! T=tree (wood), S=stone, C=cow (food), W=water
- To collect resources: move to them (actions 1-4) then use action 5 (do)
- To craft: place table (8) first, then craft tools (11-16)
- If hungry and see cow (C), move to it and eat (17)

What actions do you want to take?"""

                # Send observation as message
                obs_msg = create_message(
                    f"Observation: {compress_observation_for_trace(obs)}",
                    "system",
                    f"crafter_env_{instance_id}",
                    turn
                )
                await session_tracer.record_message(
                    content=obs_msg.content,
                    message_type=obs_msg.message_type,
                    event_time=obs_msg.time_record.event_time,
                    message_time=obs_msg.time_record.message_time,
                    metadata=obs_msg.metadata
                )
                
                # Get action from LM with tools (with timeout)
                turn_start_time = time.time()
                try:
                    # Define the interact tool for Crafter
                    from pydantic import BaseModel, Field
                    from synth_ai.lm.tools.base import BaseTool
                    
                    class InteractArgs(BaseModel):
                        actions: list[int] = Field(..., description="List of action IDs to execute")
                    
                    interact_tool = BaseTool(
                        name="interact",
                        arguments=InteractArgs,
                        description="Execute actions in the Crafter game"
                    )
                    
                    # Create system message that explains available actions
                    action_list = "\n".join([f"{action_id}: {action}" for action, action_id in CRAFTER_ACTIONS.items()])
                    system_message = f"""You are an agent playing Crafter, a 2D survival game. Your goal is to survive and unlock achievements.

You MUST use the 'interact' tool to execute actions. The tool takes a list of action IDs.

Action ID mapping:
{action_list}

Strategy tips:
- Start by collecting wood (move to trees and use action 5)
- Place a crafting table (action 8) to unlock crafting recipes
- Craft tools to collect resources more efficiently
- Eat when hungry, sleep when tired
- Explore to find different resources

IMPORTANT: Always use the 'interact' tool with a list of action IDs. For example: interact(actions=[2, 2, 5]) to move right twice and collect."""
                    
                    # Get actions from LM using tools with timeout
                    try:
                        action_response = await asyncio.wait_for(
                            lm.respond_async(
                                system_message=system_message,
                                user_message=prompt,
                                tools=[interact_tool],
                                turn_number=turn
                            ),
                            timeout=config.turn_timeout
                        )
                    except asyncio.TimeoutError:
                        print(f"    ⏰ Turn {turn} timed out for episode {episode_num} after {config.turn_timeout}s")
                        action_response = None
                        done = True
                        break
                    
                    # Debug: print response (removed for clean output)
                    
                    # Extract tool calls from response
                    if hasattr(action_response, 'tool_calls') and action_response.tool_calls:
                        tool_calls = action_response.tool_calls
                        
                        # Process each tool call
                        for tool_call in tool_calls:
                            if tool_call.get('function', {}).get('name') == 'interact':
                                # Extract actions from the tool call
                                import json
                                args = json.loads(tool_call.get('function', {}).get('arguments', '{}'))
                                actions = args.get('actions', [])
                                
                                if not actions:
                                    # If no actions provided, use noop
                                    actions = [0]
                                
                                # Execute each action separately
                                for action_id in actions:
                                    total_actions += 1
                                    
                                    # Validate action ID
                                    if action_id not in INT_TO_ACTION_STRING:
                                        # Invalid action logging removed for clean output
                                        action_id = 0
                                        invalid_actions += 1
                                    
                                    # Send action to Crafter service with timeout
                                    try:
                                        step_response = await asyncio.wait_for(
                                            retry_http_request(
                                                client, "POST", f"{config.crafter_service_url}/env/CrafterClassic/step",
                                                json={
                                                    "env_id": instance_id, 
                                                    "action": {
                                                        "tool_calls": [
                                                            {"tool": "interact", "args": {"action": action_id}}
                                                        ]
                                                    }
                                                }
                                            ),
                                            timeout=5.0  # 5 second timeout for individual action
                                        )
                                    except asyncio.TimeoutError:
                                        print(f"    ⏰ Action execution timed out in episode {episode_num}")
                                        done = True
                                        break
                    
                                    if step_response.status_code != 200:
                                        print(f"    ❌ Step failed: {step_response.status_code} - {step_response.text}")
                                        done = True
                                        break
                                        
                                    step_data = step_response.json()
                                    
                                    # Extract data from response
                                    new_obs = step_data["observation"]
                                    reward = step_data["reward"]
                                    done = step_data["done"]
                                    
                                    # Record runtime event for action
                                    action_name = INT_TO_ACTION_STRING.get(action_id, "unknown")
                                    runtime_event = RuntimeEvent(
                                        system_instance_id=f"crafter_env_{instance_id}",
                                        time_record=TimeRecord(
                                            event_time=time.time(),
                                            message_time=turn
                                        ),
                                        actions=[action_id],
                                        metadata={
                                            "action_name": action_name,
                                            "valid": action_name != "noop" or invalid_actions == 0
                                        }
                                    )
                                    await session_tracer.record_event(runtime_event)
                                    
                                    # Record environment event
                                    env_event = EnvironmentEvent(
                                        system_instance_id=f"crafter_env_{instance_id}",
                                        time_record=TimeRecord(
                                            event_time=time.time(),
                                            message_time=turn
                                        ),
                                        reward=reward,
                                        terminated=done,
                                        system_state_before={"observation": prev_obs},
                                        system_state_after={"observation": new_obs, "public_state": {"achievements_status": new_obs.get("achievements_status", {})}}
                                    )
                                    await session_tracer.record_event(env_event)
                                    
                                    # Update for next turn
                                    prev_obs = obs
                                    obs = new_obs
                                    
                                    if done:
                                        break
                                
                                # Update progress bar after each action
                                if hasattr(config, '_pbar'):
                                    config._pbar.update(1)
                    else:
                        # No tool calls provided, use noop
                        action_id = 0
                        total_actions += 1
                        invalid_actions += 1
                        
                        # Send noop action with timeout
                        try:
                            step_response = await asyncio.wait_for(
                                retry_http_request(
                                    client, "POST", f"{config.crafter_service_url}/env/CrafterClassic/step",
                                    json={
                                        "env_id": instance_id, 
                                        "action": {
                                            "tool_calls": [
                                                {"tool": "interact", "args": {"action": action_id}}
                                            ]
                                        }
                                    }
                                ),
                                timeout=5.0  # 5 second timeout
                            )
                        except asyncio.TimeoutError:
                            print(f"    ⏰ Noop action timed out in episode {episode_num}")
                            done = True
                            break
                        
                        if step_response.status_code != 200:
                            print(f"    ❌ Step failed: {step_response.status_code} - {step_response.text}")
                            done = True
                        else:
                            step_data = step_response.json()
                            new_obs = step_data["observation"]
                            reward = step_data["reward"]
                            done = step_data["done"]
                            
                            # Update observation
                            prev_obs = obs
                            obs = new_obs
                    
                    # End timestep
                    await session_tracer.end_timestep(f"turn_{turn}")
                    
                except Exception as e:
                    print(f"    ❌ Environment step error: {e}")
                    done = True
                    
            # Update progress bar for remaining steps if episode ended early
            if hasattr(config, '_pbar') and turn < config.max_turns - 1:
                remaining_steps = config.max_turns - turn - 1
                config._pbar.update(remaining_steps)
            
            # Calculate invalid action rate
            invalid_rate = invalid_actions / total_actions if total_actions > 0 else 0
            
            # Calculate achievements
            final_achievements = obs.get("achievements_status", {})
            total_achievements = sum(1 for v in final_achievements.values() if v)
            
            # Terminate environment
            try:
                await retry_http_request(
                    client, "POST", f"{config.crafter_service_url}/env/CrafterClassic/terminate",
                    json={"env_id": instance_id}
                )
            except Exception as e:
                print(f"    ⚠️  Failed to terminate environment: {e}")
            
            # End session
            await session_tracer.end_session(save=config.save_traces)
            # Close the tracer for this episode
            await session_tracer.close()
            
            return {
                "model": model_name,
                "episode": episode_num,
                "total_achievements": total_achievements,
                "achievements": final_achievements,
                "invalid_action_rate": invalid_rate,
                "total_actions": total_actions,
                "invalid_actions": invalid_actions,
                "session_id": session_id
            }
            
        except Exception as e:
            print(f"    ❌ Episode failed: {e}")
            import traceback
            traceback.print_exc()
            
            # End session even if failed
            await session_tracer.end_session(save=config.save_traces)
            # Close the tracer for this episode
            await session_tracer.close()
            
            return {
                "model": model_name,
                "episode": episode_num,
                "total_achievements": 0,
                "achievements": {},
                "invalid_action_rate": 1.0,
                "total_actions": 0,
                "invalid_actions": 0,
                "session_id": session_id,
                "error": str(e)
            }


async def run_model_experiment(config: ExperimentConfig, model_name: str, experiment_id: str) -> list[dict[str, Any]]:
    """Run multiple episodes for a single model in parallel."""
    print(f"\n🚀 Running {config.num_episodes} episodes for {model_name} in parallel...\n")
    
    # Create a progress bar for all steps across all episodes
    total_steps = config.num_episodes * config.max_turns
    pbar = atqdm(total=total_steps, desc=f"{model_name}", unit="steps", leave=True)
    config._pbar = pbar  # Store in config so episodes can update it
    
    try:
        # Create tasks for all episodes (each will create its own tracer)
        tasks = []
        for i in range(config.num_episodes):
            task = run_episode(config, model_name, i, experiment_id)
            tasks.append(task)
        
        # Run all episodes in parallel
        results = await asyncio.gather(*tasks)
        
        # Calculate summary stats
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            avg_achievements = sum(r["total_achievements"] for r in successful_results) / len(successful_results)
            avg_invalid_rate = sum(r["invalid_action_rate"] for r in successful_results) / len(successful_results)
            pbar.set_postfix({
                "avg_achievements": f"{avg_achievements:.1f}",
                "avg_invalid_rate": f"{avg_invalid_rate:.1%}",
                "success_rate": f"{len(successful_results)}/{len(results)}"
            })
    finally:
        pbar.close()
    
    return results


async def analyze_results(config: ExperimentConfig, all_results: dict[str, list[dict[str, Any]]]):
    """Analyze results across all models using v3 database."""
    print("\n📊 Analysis Results:")
    print("=" * 80)
    
    # Initialize database manager
    db_manager = AsyncSQLTraceManager(config.database_url)
    await db_manager.initialize()
    
    try:
        # Basic statistics by model
        model_stats = {}
        for model, results in all_results.items():
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                achievements = [r["total_achievements"] for r in valid_results]
                invalid_rates = [r["invalid_action_rate"] for r in valid_results]
                
                model_stats[model] = {
                    "avg_achievements": np.mean(achievements),
                    "std_achievements": np.std(achievements),
                    "max_achievements": max(achievements),
                    "avg_invalid_rate": np.mean(invalid_rates),
                    "success_rate": len(valid_results) / len(results)
                }
        
        # Print model comparison
        print("\n📈 Model Performance Summary:")
        print(f"{'Model':<20} {'Avg Achievements':<18} {'Max Achievements':<18} {'Invalid Rate':<15} {'Success Rate':<15}")
        print("-" * 86)
        
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["avg_achievements"], reverse=True):
            print(f"{model:<20} {stats['avg_achievements']:>6.2f} ± {stats['std_achievements']:>4.2f}     "
                  f"{stats['max_achievements']:>16}     {stats['avg_invalid_rate']:>12.2%}     {stats['success_rate']:>12.2%}")
        
        # Achievement frequency analysis
        print("\n🏆 Achievement Frequencies:")
        achievement_counts = defaultdict(lambda: defaultdict(int))
        
        for model, results in all_results.items():
            for result in results:
                if "error" not in result:
                    for achievement, unlocked in result["achievements"].items():
                        if unlocked:
                            achievement_counts[model][achievement] += 1
        
        # Get all unique achievements
        all_achievements = set()
        for model_achievements in achievement_counts.values():
            all_achievements.update(model_achievements.keys())
        
        # Print achievement table
        if all_achievements:
            print(f"\n{'Achievement':<25} " + " ".join(f"{model[:8]:>10}" for model in sorted(all_results.keys())))
            print("-" * (25 + 11 * len(all_results)))
            
            for achievement in sorted(all_achievements):
                row = f"{achievement:<25}"
                for model in sorted(all_results.keys()):
                    count = achievement_counts[model].get(achievement, 0)
                    total = len([r for r in all_results[model] if "error" not in r])
                    pct = (count / total * 100) if total > 0 else 0
                    row += f" {count:>3}/{total:<3} ({pct:>3.0f}%)"
                print(row)
        
        # Query model usage from database - filter to only show models used in this experiment
        print("\n💰 Model Usage Statistics from Current Experiment:")
        model_usage_df = await db_manager.get_model_usage()
        
        if model_usage_df is not None and not model_usage_df.empty:
            # Filter to only show models from this experiment
            experiment_models = set(all_results.keys())
            filtered_df = model_usage_df[model_usage_df['model_name'].isin(experiment_models)]
            
            if not filtered_df.empty:
                # Format model usage statistics as table
                print(f"{'Model':<20} {'Provider':<10} {'Usage Count':<12} {'Avg Latency (ms)':<18} {'Total Cost':<12}")
                print("-" * 72)
                for _, row in filtered_df.iterrows():
                    avg_latency = row['avg_latency_ms']
                    if pd.notna(avg_latency):
                        print(f"{row['model_name']:<20} {row['provider'] or 'N/A':<10} {row['usage_count']:<12} "
                              f"{avg_latency:<18.2f} ${row['total_cost_usd']:<11.4f}")
                    else:
                        print(f"{row['model_name']:<20} {row['provider'] or 'N/A':<10} {row['usage_count']:<12} "
                              f"{'N/A':<18} ${row['total_cost_usd']:<11.4f}")
        
        # Export detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"crafter_experiment_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "config": {
                    "num_episodes": config.num_episodes,
                    "max_turns": config.max_turns,
                    "difficulty": config.difficulty,
                    "models": list(all_results.keys())
                },
                "results": all_results,
                "statistics": model_stats,
                "timestamp": timestamp
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    finally:
        await db_manager.close()


async def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description="Run Crafter experiments with multiple models")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per model")
    parser.add_argument("--max-turns", type=int, default=100, help="Maximum turns per episode")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", help="Game difficulty")
    parser.add_argument("--models", nargs="+", default=MODELS_TO_TEST, help="Models to test")
    parser.add_argument("--no-save", action="store_true", help="Don't save traces to database")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--db-url", default=DATABASE_URL, help="Database URL for tracing")
    parser.add_argument("--base-seed", type=int, default=1000, help="Base seed for episodes (episodes use base_seed+episode_num)")
    parser.add_argument("--turn-timeout", type=float, default=30.0, help="Timeout per turn in seconds")
    parser.add_argument("--episode-timeout", type=float, default=300.0, help="Total timeout per episode in seconds")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig()
    config.num_episodes = args.episodes
    config.max_turns = args.max_turns
    config.difficulty = args.difficulty
    config.save_traces = not args.no_save
    config.verbose = not args.quiet
    config.quiet = args.quiet
    config.database_url = args.db_url
    config.base_seed = args.base_seed
    config.turn_timeout = args.turn_timeout
    config.episode_timeout = args.episode_timeout
    
    # Generate experiment ID
    experiment_id = f"crafter_multi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🎮 Crafter Multi-Model Experiment")
    print("=" * 50)
    print(f"Experiment ID: {experiment_id}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Episodes per model: {config.num_episodes}")
    print(f"Max turns per episode: {config.max_turns}")
    print(f"Difficulty: {config.difficulty}")
    print(f"Seeds: {config.base_seed} to {config.base_seed + config.num_episodes - 1}")
    print(f"Turn timeout: {config.turn_timeout}s")
    print(f"Episode timeout: {config.episode_timeout}s")
    print(f"Save traces: {config.save_traces}")
    print(f"Database URL: {config.database_url}")
    print("=" * 50)
    
    # Check Crafter service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.crafter_service_url}/health", timeout=5.0)
            if response.status_code != 200:
                print(f"❌ Crafter service not healthy at {config.crafter_service_url}")
                return
    except Exception as e:
        print(f"❌ Cannot connect to Crafter service at {config.crafter_service_url}: {e}")
        print("Please ensure the Crafter service is running.")
        return
    
    print("✅ Crafter service is running")
    
    # Run experiments for each model
    all_results = {}
    
    for model in args.models:
        results = await run_model_experiment(config, model, experiment_id)
        all_results[model] = results
    
    # Analyze and compare results
    await analyze_results(config, all_results)
    
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    asyncio.run(main())