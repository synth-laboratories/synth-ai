#!/usr/bin/env python3
"""
Test script to run ReAct agents against Crafter environment on synth service (port 8901)
Tests on multiple easy Crafter instances with enhanced debugging
"""

import asyncio
import json
import uuid
import math
import argparse
import toml
import logging
import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel, Field
from httpx import AsyncClient
import httpx
import sys
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import random
from collections import defaultdict

# Disable Langfuse completely to prevent hangs and warnings
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = "dummy"  # Prevent the warning about missing key
os.environ["LANGFUSE_SECRET_KEY"] = "dummy"  # Prevent the secret key warning
# Disable all Langfuse logging
import logging
logging.getLogger("langfuse").setLevel(logging.ERROR)

# Monkey patch Langfuse to disable all warnings
import warnings
warnings.filterwarnings("ignore", message=".*Langfuse.*")

from langfuse.openai import openai
from langfuse import Langfuse

# Override Langfuse client to silence it completely
class SilentLangfuse(Langfuse):
    def __init__(self, *args, **kwargs):
        # Set dummy values to prevent warnings
        kwargs['public_key'] = kwargs.get('public_key', 'dummy')
        kwargs['secret_key'] = kwargs.get('secret_key', 'dummy')
        kwargs['enabled'] = False
        super().__init__(*args, **kwargs)
        
# Replace Langfuse with silent version
import langfuse
langfuse.Langfuse = SilentLangfuse

# --- Prevent Langfuse background threads from blocking shutdown ---
try:
    import langfuse._task_manager.task_manager as _lftm
    # Override methods that try to join background threads during interpreter shutdown
    _lftm.TaskManager.shutdown = lambda self: None  # type: ignore[attr-defined]
    _lftm.TaskManager.join = lambda self, *a, **k: None  # type: ignore[attr-defined]
    import langfuse.prompt_cache as _lfpc
    _lfpc.PromptCacheTaskManager.shutdown = lambda self: None  # type: ignore[attr-defined]
    _lfpc.PromptCacheTaskManager.join = lambda self, *a, **k: None  # type: ignore[attr-defined]
except Exception:
    # If internals change or Langfuse not present, proceed without hard failure.
    pass

import numpy as np

# Import session tracer for CAIS event capture
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    RuntimeEvent, EnvironmentEvent, CAISEvent
)
from synth_ai.tracing_v2.utils import create_experiment_context
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from datetime import datetime

# Retry configuration for HTTP requests
MAX_RETRIES = 3      # Increase to 3 retries for better reliability
BASE_DELAY = 0.1     # 100ms base delay
MAX_DELAY = 2.0      # Max 2 seconds delay
HTTP_TIMEOUT = 10.0  # 10 seconds timeout for slower connections

async def retry_http_request(client: AsyncClient, method: str, url: str, **kwargs) -> Any:
    """
    Retry HTTP requests with exponential backoff and jitter.
    
    Args:
        client: httpx AsyncClient
        method: HTTP method ('GET', 'POST', etc.)
        url: Request URL
        **kwargs: Additional arguments for the request
        
    Returns:
        Response object
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Calculate delay with exponential backoff and jitter
            if attempt > 0:
                delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
                jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
                total_delay = delay + jitter
                # Don't print retry messages - only print if all retries fail
                await asyncio.sleep(total_delay)
            
            # Make the request with timeout
            start_request = time.time()
            response = await client.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            end_request = time.time()
            
            # Check if response is successful
            if response.status_code < 500:  # Don't retry client errors (4xx)
                return response
            
            # For server errors (5xx), continue retrying
            last_exception = Exception(f"HTTP {response.status_code}: {response.text}")
            
        except httpx.ReadError as e:
            # Specific handling for ReadErrors (connection issues)
            last_exception = e
            # For ReadErrors, wait longer with exponential backoff
            if attempt < MAX_RETRIES - 1:
                read_error_delay = min(1.0 * (2 ** attempt), 5.0)  # 1s, 2s, 4s (max 5s)
                await asyncio.sleep(read_error_delay)
        except Exception as e:
            last_exception = e
            # Don't log intermediate failures - only final failure
    
    # All retries failed
    print(f"    ❌ HTTP request failed after {MAX_RETRIES} attempts: {type(last_exception).__name__}: {str(last_exception)[:200]}")
    raise last_exception

# Import Crafter hooks
try:
    from synth_ai.environments.examples.crafter_classic.trace_hooks import CRAFTER_HOOKS
    print(f"✅ Loaded {len(CRAFTER_HOOKS)} Crafter achievement hooks (Easy, Medium, Hard)")
except ImportError:
    print("Warning: Could not import CRAFTER_HOOKS")
    CRAFTER_HOOKS = []


# Create a proper message structure with origin_system_id
def create_message(content: Any, message_type: str, origin_system_id: Any, turn: int) -> SessionEventMessage:
    """Create a message with origin system ID embedded in content."""
    return SessionEventMessage(
        content={
            "origin_system_id": str(origin_system_id),
            "payload": content
        },
        message_type=message_type,
        time_record=TimeRecord(
            event_time=datetime.now().isoformat(),
            message_time=turn
        )
    )


def compress_observation_for_trace(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Compress observation data for efficient trace storage."""
    import base64
    obs_compressed = obs.copy()
    
    # Convert semantic map to text
    if "semantic_map" in obs_compressed:
        map_view = format_semantic_map_view(obs_compressed, view_size=7)
        obs_compressed["semantic_map_text"] = map_view
        del obs_compressed["semantic_map"]
    
    # Skip heavy fields instead of base64 encoding - just store shape/hash
    heavy_fields = ["observation_image", "world_material_map", "rgb", "image"]
    for field in heavy_fields:
        if field in obs_compressed and isinstance(obs_compressed[field], (list, np.ndarray)):
            arr = np.array(obs_compressed[field], dtype=np.uint8)
            # Just store metadata instead of full data
            obs_compressed[f"{field}_shape"] = arr.shape
            obs_compressed[f"{field}_size_kb"] = arr.nbytes / 1024
            obs_compressed[f"{field}_hash"] = hash(arr.tobytes()) % 1000000  # Simple hash for tracking
            del obs_compressed[field]
    
    return obs_compressed


def print_hook_legend():
    """Print the legend for hook codes."""
    print("\n📖 Hook Legend:")
    print("  E = Easy achievement    (e.g., collect_wood, place_table)")
    print("  M = Medium achievement  (e.g., make_wood_pickaxe, collect_coal)")
    print("  H = Hard achievement    (e.g., make_iron_sword, defeat_zombie)")
    print("  X = Invalid action      (action had no effect)")
    print("  # = Regular step")
    print("")  # Add blank line to separate from progress bars


# NOTE: These custom progress display functions are no longer used - replaced with tqdm
# def create_progress_bar(episode_num: int, steps: List[str], max_steps: int) -> str:
#     """Create a progress bar string with hook codes."""
#     # Pad with spaces if fewer steps than max
#     padded_steps = steps + [' '] * (max_steps - len(steps))
#     bar = ''.join(padded_steps[:max_steps])
#     return f"Episode {episode_num:2d}: [{bar}] {len(steps)}/{max_steps}"


# def update_progress_display(episode_bars: Dict[int, List[str]], max_steps: int):
#     """Update the progress display in place."""
#     # Clear previous lines
#     num_episodes = len(episode_bars)
#     if num_episodes > 0:
#         # Move cursor up to overwrite previous display
#         print(f"\033[{num_episodes}A", end='')
#     
#     # Print all episode progress bars
#     for episode_num in sorted(episode_bars.keys()):
#         steps = episode_bars[episode_num]
#         print(create_progress_bar(episode_num, steps, max_steps))
#     
#     # Ensure we don't leave cursor in wrong position
#     sys.stdout.flush()

def clear_progress_display():
    """Clear the progress display area to prevent overlap with error messages."""
    print("\n" * 3)  # Add extra spacing


def print_achievements_table(all_achievements: Dict[str, int], num_episodes: int):
    """Print a beautiful table of achievements across all episodes."""
    if not all_achievements:
        return
        
    print("\n" + "=" * 80)
    print("🏆 ACHIEVEMENTS SUMMARY")
    print("=" * 80)
    print(f"{'Achievement':<30} {'Count':<10} {'Percentage':<15}")
    print("-" * 55)
    
    # Sort achievements by count (descending) then by name
    sorted_achievements = sorted(all_achievements.items(), key=lambda x: (-x[1], x[0]))
    
    for achievement, count in sorted_achievements:
        percentage = (count / num_episodes) * 100
        print(f"{achievement:<30} {count:<10} {percentage:>6.1f}%")
    
    print("-" * 55)
    print(f"{'TOTAL UNIQUE':<30} {len(all_achievements):<10}")
    print("=" * 80)


def print_invalid_actions_table(invalid_actions: Dict[str, int], total_actions: Dict[str, int] = None):
    """Print a table of invalid actions by type with failure rates."""
    if not invalid_actions:
        return
    
    print("\n" + "=" * 90)
    print("❌ INVALID ACTIONS SUMMARY")
    print("=" * 90)
    print(f"{'Action Type':<20} {'Invalid/Total':<15} {'Failure %':<12} {'Description':<35}")
    print("-" * 90)
    
    # Sort by count (descending) then by name
    sorted_actions = sorted(invalid_actions.items(), key=lambda x: (-x[1], x[0]))
    
    action_descriptions = {
        'move_left': 'Movement blocked (wall/edge)',
        'move_right': 'Movement blocked (wall/edge)',
        'move_up': 'Movement blocked (wall/edge)',
        'move_down': 'Movement blocked (wall/edge)',
        'do': 'Nothing to collect/attack',
        'sleep': 'Energy already full or conditions not met',
        'place_stone': 'No stone or invalid location',
        'place_table': 'No wood or invalid location',
        'place_furnace': 'No stone or invalid location',
        'place_plant': 'No sapling or invalid location',
        'make_wood_pickaxe': 'Missing materials or no table',
        'make_stone_pickaxe': 'Missing materials or no table',
        'make_iron_pickaxe': 'Missing materials or no furnace',
        'make_wood_sword': 'Missing materials or no table',
        'make_stone_sword': 'Missing materials or no table',
        'make_iron_sword': 'Missing materials or no furnace'
    }
    
    total_invalid = 0
    for action, invalid_count in sorted_actions:
        total_count = total_actions.get(action, invalid_count) if total_actions else invalid_count
        failure_rate = (invalid_count / total_count * 100) if total_count > 0 else 0
        fraction = f"{invalid_count}/{total_count}"
        description = action_descriptions.get(action, 'Unknown reason')
        print(f"{action:<20} {fraction:<15} {failure_rate:>6.1f}%     {description:<35}")
        total_invalid += invalid_count
    
    print("-" * 90)
    total_all_actions = sum(total_actions.values()) if total_actions else total_invalid
    total_failure_rate = (total_invalid / total_all_actions * 100) if total_all_actions > 0 else 0
    print(f"{'TOTAL':<20} {total_invalid}/{total_all_actions:<15} {total_failure_rate:>6.1f}%")
    print("=" * 90)


def print_termination_breakdown(termination_reasons: List[str]):
    """Print episode termination breakdown."""
    print("\n" + "=" * 80)
    print("🏁 EPISODE TERMINATION BREAKDOWN")
    print("=" * 80)
    
    if not termination_reasons:
        print("No termination data available.")
        return
    
    # Count termination reasons
    reason_counts = {}
    for reason in termination_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    # Sort by count descending
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    
    total_episodes = len(termination_reasons)
    
    print(f"{'Termination Reason':<40} {'Count':<10} {'Percentage':<12} {'Description'}")
    print("-" * 80)
    
    # Descriptions for different termination types
    descriptions = {
        "max_turns_reached": "Episode completed all turns",
        "death": "Agent died (health <= 0)",
        "environment_terminated": "Environment ended episode",
        "no_actions_provided": "Agent failed to provide actions"
    }
    
    for reason, count in sorted_reasons:
        percentage = (count / total_episodes * 100) if total_episodes > 0 else 0
        
        # Parse complex reasons
        display_reason = reason
        description = "Other termination reason"
        
        if reason == "max_turns_reached":
            description = descriptions.get(reason, "Episode completed all turns")
        elif reason == "death":
            description = descriptions.get(reason, "Agent died (health <= 0)")
        elif reason == "environment_terminated":
            description = descriptions.get(reason, "Environment ended episode")
        elif reason.startswith("agent_terminate:"):
            display_reason = "agent_terminate"
            description = f"Agent chose to quit: {reason.split(':', 1)[1][:30]}"
        elif reason.startswith("http_error:"):
            display_reason = "http_error"
            description = f"API request failed: {reason.split(':', 1)[1]}"
        elif reason.startswith("exception:"):
            display_reason = "exception"
            error_detail = reason.split(':', 1)[1].strip()
            if error_detail:
                description = f"Runtime error: {error_detail[:40]}"
            else:
                description = "Runtime error (unknown cause)"
        elif reason.startswith("outer_exception:"):
            display_reason = "outer_exception"
            error_detail = reason.split(':', 1)[1].strip()
            if error_detail:
                description = f"Fatal error: {error_detail[:40]}"
            else:
                description = "Fatal error (unknown cause)"
        elif reason == "no_actions_provided":
            description = descriptions.get(reason, "Agent failed to provide actions")
        
        print(f"{display_reason:<40} {count:<10} {percentage:<11.1f}% {description}")
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_episodes:<10} {'100.0%':<11}")
    print("=" * 80)


def print_timing_analysis(results: List[Dict[str, Any]]):
    """Print comprehensive timing analysis."""
    if not results:
        return
    
    # Extract timing data from valid results
    episode_times = []
    all_step_times = []
    all_env_times = []
    all_agent_times = []
    
    for result in results:
        if not result.get("error", False) and "timing" in result:
            timing = result["timing"]
            episode_times.append(timing["episode_total_time"])
            all_step_times.extend(timing["step_times"])
            all_env_times.extend(timing["env_times"])
            all_agent_times.extend(timing["agent_times"])
    
    if not episode_times:
        print("⚠️  No timing data available for analysis")
        return
    
    print("=" * 80)
    print("⏱️  TIMING ANALYSIS")
    print("=" * 80)
    
    # Episode-level timing
    print("📊 EPISODE TIMING DISTRIBUTION")
    print("-" * 40)
    episode_times.sort()
    print(f"Total Episodes: {len(episode_times)}")
    print(f"Mean Episode Time: {sum(episode_times)/len(episode_times):.2f}s")
    print(f"Median Episode Time: {episode_times[len(episode_times)//2]:.2f}s")
    print(f"Min Episode Time: {min(episode_times):.2f}s")
    print(f"Max Episode Time: {max(episode_times):.2f}s")
    print(f"P95 Episode Time: {episode_times[int(len(episode_times)*0.95)]:.2f}s")
    print()
    
    # Step-level timing
    if all_step_times:
        print("📊 STEP TIMING DISTRIBUTION")
        print("-" * 40)
        all_step_times.sort()
        print(f"Total Steps: {len(all_step_times)}")
        print(f"Mean Step Time: {sum(all_step_times)/len(all_step_times):.2f}s")
        print(f"Median Step Time: {all_step_times[len(all_step_times)//2]:.2f}s")
        print(f"Min Step Time: {min(all_step_times):.2f}s")
        print(f"Max Step Time: {max(all_step_times):.2f}s")
        print(f"P95 Step Time: {all_step_times[int(len(all_step_times)*0.95)]:.2f}s")
        print()
    
    # Environment vs Agent timing
    if all_env_times and all_agent_times:
        print("📊 ENVIRONMENT vs AGENT TIMING")
        print("-" * 40)
        
        env_mean = sum(all_env_times)/len(all_env_times)
        agent_mean = sum(all_agent_times)/len(all_agent_times)
        
        print(f"Environment Calls: {len(all_env_times)}")
        print(f"Agent Calls: {len(all_agent_times)}")
        print(f"Mean Environment Time: {env_mean:.2f}s")
        print(f"Mean Agent Time: {agent_mean:.2f}s")
        print(f"Environment/Agent Ratio: {env_mean/agent_mean:.2f}x")
        
        # Time breakdown
        total_env_time = sum(all_env_times)
        total_agent_time = sum(all_agent_times)
        total_time = total_env_time + total_agent_time
        
        if total_time > 0:
            print(f"Environment %: {(total_env_time/total_time)*100:.1f}%")
            print(f"Agent %: {(total_agent_time/total_time)*100:.1f}%")
        
        print()
        
        # Distribution comparison
        all_env_times.sort()
        all_agent_times.sort()
        
        print("Environment Time Distribution:")
        print(f"  P50: {all_env_times[len(all_env_times)//2]:.2f}s")
        print(f"  P90: {all_env_times[int(len(all_env_times)*0.9)]:.2f}s")
        print(f"  P95: {all_env_times[int(len(all_env_times)*0.95)]:.2f}s")
        print(f"  P99: {all_env_times[int(len(all_env_times)*0.99)]:.2f}s")
        
        print("Agent Time Distribution:")
        print(f"  P50: {all_agent_times[len(all_agent_times)//2]:.2f}s")
        print(f"  P90: {all_agent_times[int(len(all_agent_times)*0.9)]:.2f}s")
        print(f"  P95: {all_agent_times[int(len(all_agent_times)*0.95)]:.2f}s")
        print(f"  P99: {all_agent_times[int(len(all_agent_times)*0.99)]:.2f}s")
    
    print("=" * 80)


def print_condensed_summary(all_achievements: Dict[str, int], invalid_actions: Dict[str, int], 
                           total_actions: Dict[str, int], termination_reasons: List[str], 
                           results: List[Dict[str, Any]], num_episodes: int):
    """Print a dense, condensed summary of all metrics in a single compact table."""
    print("\n" + "─" * 80)
    print("CRAFTER EVALUATION SUMMARY")
    print("─" * 80)
    
    # Calculate aggregated metrics
    unique_achievements = len(all_achievements)
    total_achievements = sum(all_achievements.values())
    total_invalid = sum(invalid_actions.values())
    total_acts = sum(total_actions.values())
    invalid_rate = (total_invalid / total_acts * 100) if total_acts > 0 else 0
    
    # Timing metrics
    episode_times = [r['timing_info']['episode_time'] for r in results if 'timing_info' in r and r['timing_info']]
    step_times = []
    for r in results:
        if 'timing_info' in r and r['timing_info'] and 'step_times' in r['timing_info']:
            step_times.extend(r['timing_info']['step_times'])
    
    avg_episode_time = sum(episode_times) / len(episode_times) if episode_times else 0
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    
    # Termination breakdown
    term_counts = {}
    for reason in termination_reasons:
        term_counts[reason] = term_counts.get(reason, 0) + 1
    
    # Best achievement
    best_achievement = max(all_achievements.items(), key=lambda x: x[1])[0] if all_achievements else "none"
    
    # Achievement distribution by trajectory  
    achievement_by_episode = {}
    for r in results:
        if 'num_achievements' in r:
            count = r['num_achievements']
            achievement_by_episode[count] = achievement_by_episode.get(count, 0) + 1
    
    # Format achievement distribution
    achv_dist_str = " | ".join([f"{k} achv: {v}" for k, v in sorted(achievement_by_episode.items())])
    
    # Achievement frequencies by type
    achv_freq_str = " | ".join([f"{k}: {v/total_achievements*100:.0f}%" for k, v in sorted(all_achievements.items(), key=lambda x: x[1], reverse=True)[:3]]) if total_achievements > 0 else ""
    
    # Print compact table
    print(f"Episodes: {num_episodes} | Achievements: {unique_achievements} ({total_achievements} total) | Invalid: {total_invalid}/{total_acts} ({invalid_rate:.1f}%)")
    print(f"Avg Episode: {avg_episode_time:.1f}s | Avg Step: {avg_step_time:.1f}s | Best: {best_achievement}")
    
    # Most common termination
    if term_counts:
        most_common_term = max(term_counts.items(), key=lambda x: x[1])
        print(f"Termination: {most_common_term[0]} ({most_common_term[1]}/{num_episodes})")
    
    # Achievement distributions
    if achv_dist_str:
        print(f"Achv by traj: {achv_dist_str}")
    
    # Print achievement frequencies vertically
    if all_achievements and total_achievements > 0:
        print("\nAchievement frequencies:")
        for achv, count in sorted(all_achievements.items(), key=lambda x: x[1], reverse=True):
            print(f"  {achv:<20} {count/total_achievements*100:>3.0f}%")
    
    print("─" * 80)


def analyze_trace_file(trace_file: Path):
    """Analyze a trace file and print detailed step-by-step information."""
    import time
    
    start_time = time.time()
    print(f"\n📄 Analyzing trace: {trace_file.name}")
    print("=" * 80)
    
    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load trace file: {e}")
        return
    
    # Get events and messages
    events = trace_data.get('event_history', [])
    messages = trace_data.get('message_history', [])
    
    # Group CAISEvents by turn
    cais_events = [e for e in events if e.get('system_instance_id', '').startswith('crafter-react-agent')]
    
    # Process each turn
    for i, cais_event in enumerate(cais_events):
        print(f"\n🎮 Step {i + 1}")
        print("-" * 40)
        
        # Token usage
        tokens = {
            'prompt': cais_event.get('prompt_tokens', None),
            'completion': cais_event.get('completion_tokens', None),
            'total': cais_event.get('total_tokens', None)
        }
        if any(t is not None for t in tokens.values()):
            print(f"🪙 Tokens: Prompt={tokens['prompt']}, Completion={tokens['completion']}, Total={tokens['total']}")
        else:
            # Try to get from llm_call_records
            llm_records = cais_event.get('llm_call_records', [])
            if llm_records and isinstance(llm_records[0], dict):
                response = llm_records[0].get('response', {})
                if response:
                    usage = response.get('usage', {})
                    if usage:
                        print(f"🪙 Tokens: Prompt={usage.get('prompt_tokens')}, Completion={usage.get('completion_tokens')}, Total={usage.get('total_tokens')}")
        
        # Tool calls from LLM records
        llm_records = cais_event.get('llm_call_records', [])
        if llm_records and isinstance(llm_records[0], dict):
            # First check the response for tool calls
            response = llm_records[0].get('response', {})
            if response and 'choices' in response:
                choices = response.get('choices', [])
                if choices and isinstance(choices[0], dict):
                    tool_calls = choices[0].get('message', {}).get('tool_calls', [])
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.get('function', {}).get('name', 'unknown')
                            args = json.loads(tc.get('function', {}).get('arguments', '{}'))
                            actions = args.get('actions', [])
                            reasoning = args.get('reasoning', '')
                            print(f"🔧 Tool: {tool_name}")
                            if actions:
                                print(f"   Actions: {', '.join(actions)}")
                            if reasoning:
                                print(f"   Reasoning: {reasoning[:60]}...")
        
        # Show hooks that fired for this CAISEvent
        if cais_event.get('event_metadata'):
            print("🎯 Agent Hooks Fired:")
            for meta in cais_event['event_metadata']:
                print(f"   - {meta['hook_name']}: {meta['description']}")
        
        # Find corresponding observations to track achievements and inventory
        turn_time = cais_event.get('time_record', {}).get('message_time', i)
        
        # Get observations for this turn and next turn
        turn_observations = [m for m in messages 
                           if m.get('message_type') == 'observation' 
                           and m.get('time_record', {}).get('message_time', -1) in [turn_time, turn_time + 1]]
        
        if len(turn_observations) >= 2:
            before_obs = turn_observations[0].get('content', {}).get('payload', {})
            after_obs = turn_observations[1].get('content', {}).get('payload', {})
            
            # Achievement changes
            before_achievements = before_obs.get('achievements_status', {})
            after_achievements = after_obs.get('achievements_status', {})
            
            new_achievements = []
            for ach_name, ach_status in after_achievements.items():
                if ach_status and not before_achievements.get(ach_name, False):
                    new_achievements.append(ach_name)
            
            if new_achievements:
                print(f"🏆 New Achievements: {', '.join(new_achievements)}")
            
            # Inventory changes
            before_inv = before_obs.get('inventory', {})
            after_inv = after_obs.get('inventory', {})
            
            inv_changes = []
            for item, count in after_inv.items():
                before_count = before_inv.get(item, 0)
                if count > before_count:
                    inv_changes.append(f"{item}: {before_count} → {count} (+{count - before_count})")
                elif count < before_count:
                    inv_changes.append(f"{item}: {before_count} → {count} ({count - before_count})")
            
            if inv_changes:
                print(f"📦 Inventory Changes: {', '.join(inv_changes)}")
            
            # Position change
            before_pos = before_obs.get('player_position', [0, 0])
            after_pos = after_obs.get('player_position', [0, 0])
            if before_pos != after_pos:
                print(f"📍 Position: {before_pos} → {after_pos}")
        
        # Find corresponding environment event and show its hooks
        env_events = [e for e in events 
                     if not e.get('system_instance_id', '').startswith('crafter-react-agent')
                     and e.get('time_record', {}).get('message_time', -1) == turn_time
                     and 'reward' in e]
        
        if env_events and env_events[0].get('event_metadata'):
            print("🌍 Environment Hooks Fired:")
            for meta in env_events[0]['event_metadata']:
                print(f"   - {meta['hook_name']}: {meta['description']}")
    
    print("\n" + "=" * 80)
    
    end_time = time.time()
    print(f"[DEBUG] Trace analysis completed in {end_time - start_time:.2f} seconds")


# --- Configuration Class ---
class CrafterConfig:
    """Configuration for Crafter evaluation."""

    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.model_name: Optional[str] = None  # Must be provided via config or CLI
        self.num_instances = 1  # Changed from 3 to 1
        self.max_turns = 2  # Changed to just 2 steps
        self.difficulty = "easy"
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.save_traces = True
        self.save_detailed_results = True
        self.verbose = False
        self.analyze_traces = False  # Whether to analyze traces after evaluation

        # Custom OpenAI endpoint support
        self.custom_openai_base_url = None  # e.g., "https://lora-inference-service-xyz.modal.run"
        self.custom_openai_api_key = "dummy"  # Default dummy key for custom endpoints

        # Load from TOML if provided
        if config_path and os.path.exists(config_path):
            self.load_from_toml(config_path)

        # Fail fast if no model name provided
        # Configure custom OpenAI endpoint if specified
        self._configure_custom_openai()

    def load_from_toml(self, config_path: str):
        """Load configuration from TOML file."""
        config = toml.load(config_path)

        # Extract eval settings
        eval_config = config.get("eval", {})
        self.model_name = eval_config.get("model_name", self.model_name)
        self.num_instances = eval_config.get("episodes", self.num_instances)
        self.max_turns = eval_config.get("max_steps", self.max_turns)
        self.difficulty = eval_config.get("difficulty", self.difficulty)
        self.seed = eval_config.get("seed", self.seed)

        # Extract service settings
        service_config = config.get("service", {})
        self.service_base_url = service_config.get("base_url", self.service_base_url)
        self.service_timeout = service_config.get("timeout", self.service_timeout)

        # Extract output settings
        output_config = config.get("output", {})
        self.save_traces = output_config.get("save_traces", self.save_traces)
        self.save_detailed_results = output_config.get(
            "save_detailed_results", self.save_detailed_results
        )

        # Extract custom OpenAI endpoint settings
        openai_config = config.get("openai", {})
        self.custom_openai_base_url = openai_config.get("base_url", self.custom_openai_base_url)
        self.custom_openai_api_key = openai_config.get("api_key", self.custom_openai_api_key)

    def _configure_custom_openai(self):
        """Configure environment variables for custom OpenAI endpoint if specified."""
        if self.custom_openai_base_url:
            # Ensure the base URL ends with /v1 for OpenAI compatibility
            base_url = self.custom_openai_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

            # Set environment variables for OpenAI SDK
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_KEY"] = self.custom_openai_api_key

            print(f"🔧 Configured custom OpenAI endpoint: {base_url}")
            print(f"   API Key: {self.custom_openai_api_key}")

            # Auto-detect if this looks like a fine-tuned model and add ft: regex support
            if self.model_name and (
                self.model_name.startswith("ft:") or "lora" in self.model_name.lower()
            ):
                self._add_ft_regex_support()

    def _add_ft_regex_support(self):
        """Add ft: regex pattern to OpenAI naming regexes if not already present."""
        try:
            import re
            from synth_ai.lm.core import vendor_clients

            # Check if ft: pattern already exists
            ft_pattern = re.compile(r"^ft:.*$")
            if not any(
                pattern.pattern == ft_pattern.pattern
                for pattern in vendor_clients.openai_naming_regexes
            ):
                # Add ft: pattern at the beginning to catch all fine-tuned models
                vendor_clients.openai_naming_regexes.insert(0, ft_pattern)
                print(f"✅ Added ft:* regex pattern for fine-tuned model support")
        except Exception as e:
            print(f"⚠️  Warning: Could not add ft: regex pattern: {e}")

    def set_custom_endpoint(self, base_url: str, api_key: str = "dummy"):
        """Programmatically set custom OpenAI endpoint."""
        self.custom_openai_base_url = base_url
        self.custom_openai_api_key = api_key
        self._configure_custom_openai()


# --- Global Config ---
config = CrafterConfig()


# --- Helper to build crafter semantic mapping ---
@functools.lru_cache(maxsize=1)
def get_crafter_semantic_mapping():
    """Build the crafter semantic ID to item name mapping."""
    try:
        import crafter
        import itertools

        # Create a dummy env to get ID mappings
        dummyenv = None
        try:
            dummyenv = crafter.Env()
            max_id = (
                max(
                    max(dummyenv._world._mat_ids.values()),
                    max(dummyenv._sem_view._obj_ids.values()),
                )
                + 1
            )
            id_to_item = ["void"] * max_id
            for name, ind in itertools.chain(
                dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()
            ):
                if name is None:
                    clean = "none"
                elif hasattr(name, "__name__"):
                    clean = name.__name__
                else:
                    clean = str(name)
                id_to_item[ind] = clean.lower()
            player_idx = id_to_item.index("player")
            return id_to_item, player_idx
        finally:
            if dummyenv:
                try:
                    dummyenv.close()
                except Exception:
                    pass
            del dummyenv
    except ImportError:
        # Fallback if crafter is not available
        return None, None


def format_semantic_map_view(obs_data: Dict[str, Any], view_size: int = 7) -> str:
    """Format a semantic map view around the player (ASCII)."""
    try:
        # Get mapping list
        id_to_item, _ = get_crafter_semantic_mapping()
        if id_to_item is None:
            return "Map view unavailable (crafter not installed)"

        semantic_map = obs_data.get("semantic_map")
        player_position = obs_data.get("player_position", [0, 0])

        if semantic_map is None:
            return "Map view unavailable (no semantic map data)"

        # Ensure numpy array with 2 dimensions
        sem_arr = np.asarray(semantic_map)
        if sem_arr.ndim == 1:
            # Probably flattened; try to infer square size
            size = int(np.sqrt(sem_arr.size))
            sem_arr = sem_arr.reshape(size, size)
        elif sem_arr.ndim != 2:
            return "Map view unavailable (invalid map dimensionality)"

        px, py = map(int, player_position)
        half = view_size // 2

        rows = []
        visible = set()
        for dy in range(-half, half + 1):
            row_tokens = []
            for dx in range(-half, half + 1):
                x, y = px + dx, py + dy
                if 0 <= x < sem_arr.shape[0] and 0 <= y < sem_arr.shape[1]:
                    if dx == 0 and dy == 0:
                        token = "player"
                    else:
                        idx = int(sem_arr[x, y])
                        token = id_to_item[idx] if idx < len(id_to_item) else "?"
                else:
                    token = "void"
                row_tokens.append(token)
                if token not in {"void", "player"}:
                    visible.add(token)
            rows.append(" ".join(row_tokens))

        map_view = f"\nLocal Map View ({view_size}x{view_size}):\n" + "\n".join(rows)
        if visible:
            map_view += "\nVisible items: " + ", ".join(sorted(visible))
        else:
            map_view += "\nNo special items visible (mostly grass/empty)"
        return map_view
    except Exception as e:
        return f"Map view error: {e}"


# --- Shaped Reward Configuration ---
# K-values for shaped reward calculation: reward = sum(K * log(count)) for each achievement
ACHIEVEMENT_K_VALUES = {
    "collect_coal": 3.0,
    "collect_diamond": 100.0,
    "collect_drink": 0.1,
    "collect_iron": 10.0,
    "collect_sapling": 0.1,
    "collect_stone": 1.0,
    "collect_wood": 1.0,
    "defeat_skeleton": 1.0,
    "defeat_zombie": 1.0,
    "eat_cow": 1.0,
    "eat_plant": 0.1,
    "make_iron_pickaxe": 30.0,
    "make_iron_sword": 30.0,
    "make_stone_pickaxe": 10.0,
    "make_stone_sword": 10.0,
    "make_wood_pickaxe": 3.0,
    "make_wood_sword": 3.0,
    "place_furnace": 10.0,
    "place_plant": 0.1,
    "place_stone": 1.0,
    "place_table": 3.0,
    "wake_up": 0.1,
}


# --- Tool Definitions ---
def get_openai_tools():
    """Get OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform 1-5 actions in sequence in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 1-5 action names to execute in sequence (e.g., ['move_up', 'do', 'mine_down'])"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why these actions were chosen"
                        }
                    },
                    "required": ["actions", "reasoning"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "End the episode when finished or no progress can be made.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for termination"
                        }
                    },
                    "required": ["reason"]
                }
            }
        }
    ]


# --- Shaped Reward Helper ---
def calculate_shaped_reward(achievement_counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculate shaped reward using K * log(count) for each achievement."""
    total_reward = 0.0
    reward_breakdown = {}

    for achievement, count in achievement_counts.items():
        if count > 0 and achievement in ACHIEVEMENT_K_VALUES:
            k_value = ACHIEVEMENT_K_VALUES[achievement]
            # Use log(count + 1) to handle count=0 case gracefully
            reward_contribution = k_value * math.log(count + 1)
            total_reward += reward_contribution
            reward_breakdown[achievement] = {
                "count": count,
                "k_value": k_value,
                "contribution": reward_contribution,
            }

    return {"total_shaped_reward": total_reward, "breakdown": reward_breakdown}


# --- Base ReAct Agent ---
class BaseReActAgent:
    """Base ReAct agent for environment interaction."""

    def __init__(self, model_name: str, max_turns: int = 20, verbose: bool = False, tracer: Optional[SessionTracer] = None):
        self.model_name = model_name
        self.max_turns = max_turns
        self.verbose = verbose
        self.history = []
        self.system_name = "base-react-agent"
        self.tools = get_openai_tools()
        self.tracer = tracer
        # Unique system ID for this agent instance
        import uuid
        self.system_id = uuid.uuid4()
        
        # Agent state tracking
        self.agent_state = {
            "message_history": [],  # LLM conversation history
            "steps_taken": 0,
            "steps_remaining": max_turns,
            "total_tokens_used": 0,
            "tool_calls_made": 0,
            "current_turn": 0
        }

    def decide(self, obs: str, system_message: str, turn: int) -> Dict[str, Any]:
        """Get agent decision based on observation."""
        # Update agent state
        self.agent_state["current_turn"] = turn
        self.agent_state["steps_taken"] = turn
        self.agent_state["steps_remaining"] = self.max_turns - turn
        
        # Create conversation context
        context = f"Turn {turn + 1}/{self.max_turns}\n\n{obs}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": context}
        ]
        
        # Only keep the last N messages to avoid huge state
        max_history_length = 20
        if len(self.agent_state["message_history"]) > max_history_length:
            # Keep system message and last N-1 messages
            self.agent_state["message_history"] = (
                [self.agent_state["message_history"][0]] +  # Keep first system message
                self.agent_state["message_history"][-(max_history_length-1):]
            )
        
        # Add current messages to history
        self.agent_state["message_history"].append({"role": "system", "content": system_message})
        self.agent_state["message_history"].append({"role": "user", "content": context})
        
        # Capture system state before LLM call (compress message history)
        system_state_before = self.agent_state.copy()
        # Truncate message history in the saved state to save space
        if "message_history" in system_state_before and len(system_state_before["message_history"]) > 4:
            system_state_before["message_history"] = (
                system_state_before["message_history"][:2] +  # First 2 messages
                ["... truncated ..."] +
                system_state_before["message_history"][-2:]  # Last 2 messages
            )
        
        # Use langfuse generation (langfuse 3.x API - not context manager)
        # Suppress all stdout during Langfuse operations
        import sys
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            langfuse_client = Langfuse()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # Create generation (not as context manager)
        generation = langfuse_client.generation(
            name=f"crafter_agent_turn_{turn}",
            model=self.model_name,
            input=messages,
            metadata={
                "turn": turn,
                "agent_type": self.system_name,
                "tools_available": len(self.tools)
            }
        )
        
        # Store langfuse client for cleanup
        self._langfuse_client = langfuse_client
        
        try:
            # Generate response using OpenAI client (v1.0+ API)
            prompt_size = sum(len(str(m.get('content', ''))) for m in messages)
            llm_start = time.time()
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            llm_end = time.time()
            
            # Update generation with output
            generation.update(
                output=response.choices[0].message.model_dump() if response.choices else None,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "total_tokens": response.usage.total_tokens if response.usage else None
                }
            )
            
            # End the generation
            generation.end()

            # Extract tool calls from response
            tool_calls = response.choices[0].message.tool_calls

            # Handle case where tool_calls is None or empty (graceful fallback)
            if not tool_calls:
                if self.verbose:
                    print(f"[WARNING] No tool calls returned by LLM, using default action")
                decision = {
                    "name": "interact",
                    "parameters": {
                        "actions": ["do"],
                        "reasoning": "Default action - no tool call received",
                    },
                }
            else:
                tool_call_data = tool_calls[0]
                tool_name = tool_call_data.function.name
                tool_arguments = json.loads(tool_call_data.function.arguments)
                decision = {"name": tool_name, "parameters": tool_arguments}
            
            # Update agent state with response
            if response.usage:
                self.agent_state["total_tokens_used"] += response.usage.total_tokens
            self.agent_state["tool_calls_made"] += 1
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content if response.choices else None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        },
                        "type": tc.type
                    } for tc in tool_calls
                ] if tool_calls else []
            }
            self.agent_state["message_history"].append(assistant_message)

            # Capture system state after LLM call (compress message history)
            system_state_after = self.agent_state.copy()
            # Truncate message history in the saved state to save space
            if "message_history" in system_state_after and len(system_state_after["message_history"]) > 4:
                system_state_after["message_history"] = (
                    system_state_after["message_history"][:2] +  # First 2 messages
                    ["... truncated ..."] +
                    system_state_after["message_history"][-2:]  # Last 2 messages
                )
            
            # Record LLM call as a CAISEvent (internal to agent, NOT a message)
            if self.tracer:
                try:
                    # Create LLM call record with all info needed to reproduce
                    llm_call_record = {
                        "model": self.model_name,
                        "messages": messages,
                        "tools": self.tools,
                        "tool_choice": "auto",
                        "temperature": 0.0,
                        "response": {
                            "id": response.id if response else None,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content if response.choices else None,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            },
                                            "type": tc.type
                                        } for tc in tool_calls
                                    ] if tool_calls else None
                                },
                                "finish_reason": response.choices[0].finish_reason if response.choices else None
                            }] if response.choices else [],
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            } if response.usage else None
                        }
                    }
                    
                    llm_event = CAISEvent(
                        time_record=TimeRecord(
                            event_time=datetime.now().isoformat(),
                            message_time=turn
                        ),
                        system_instance_id=f"{self.system_name}_{self.model_name}",
                        system_state_before=system_state_before,
                        system_state_after=system_state_after,
                        llm_call_records=[llm_call_record],  # Include the LLM call record
                        metadata={
                            "model_name": self.model_name,
                            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                            "completion_tokens": response.usage.completion_tokens if response.usage else None,
                            "total_tokens": response.usage.total_tokens if response.usage else None,
                            "turn": turn
                        }
                    )
                    
                    #print(f"🤖 Created CAISEvent: {llm_event.system_instance_id}, has tracer: {self.tracer is not None}")
                    
                    if hasattr(self.tracer, 'current_session') and self.tracer.current_session:
                        self.tracer.record_event(llm_event)
                        # Store the last event for progress tracking
                        self.last_cais_event = llm_event
                        if self.verbose:
                            print(f"✅ Added CAISEvent for turn {turn}")
                    else:
                        if self.verbose:
                            print(f"⚠️ No current_session in tracer")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to capture LLM event: {e}")
                        import traceback
                        traceback.print_exc()
                    # Always print errors for debugging
                    print(f"❌ Error adding CAISEvent: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            raise e

        return decision


# --- Crafter ReAct Agent ---
class CrafterReActAgent(BaseReActAgent):
    """ReAct agent for Crafter environment."""

    def __init__(self, model_name: str, max_turns: int = 20, verbose: bool = False, tracer: Optional[SessionTracer] = None):
        super().__init__(model_name, max_turns, verbose, tracer)
        self.system_name = "crafter-react-agent"

    def get_system_message(self) -> str:
        return """You are CrafterAgent playing Crafter survival environment. Your goal is to unlock as many achievements as possible while staying alive.

You will see a semantic map view showing your surroundings. Use this to navigate toward resources.

Key mechanics:
• 'do' action: collect wood from trees, stone from deposits, food from cows/plants
• 'do' does nothing on grass/water - move to find resources first
• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools
• Sleep when energy low to restore and unlock wake_up achievement
• Use semantic map view to navigate toward resources you can see

Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop

Strategy:
1. Look at the semantic map to see what's around you
2. Move toward trees to collect wood with 'do'
3. Once you have wood, place a table to enable crafting
4. Make a wood pickaxe to collect stone more efficiently
5. Progress to stone pickaxe, then iron tools
6. Eat food when health is low, sleep when energy is low

You should provide 1-5 actions in sequence for efficient gameplay. Use the semantic map view to navigate toward visible resources.

Example good action sequences:
- ['move_right', 'move_right', 'do'] (move to tree and collect wood)
- ['place_table', 'make_wood_pickaxe'] (craft progression)
- ['move_up', 'do', 'move_down', 'do'] (collect from multiple resources)

Be strategic and use the map view to find resources! Focus on unlocking achievements."""

    def format_observation(self, obs: Dict[str, Any]) -> str:
        """Format observation for Crafter with rich context."""
        # Extract key information from observation
        health = obs.get("health", 0)
        inventory = obs.get("inventory", {})

        # Extract health from inventory if not in main obs
        if health == 0 and "health" in inventory:
            health = inventory["health"]

        # Format inventory items (exclude health since we show it separately)
        inventory_items = []
        for item, count in inventory.items():
            if count > 0 and item != "health":
                inventory_items.append(f"{item}: {count}")

        inventory_str = ", ".join(inventory_items) if inventory_items else "empty"

        # Get achievements
        achievements = obs.get("achievements") or obs.get("achievements_status", {})
        unlocked_achievements = [name for name, unlocked in achievements.items() if unlocked]
        achievements_str = ", ".join(unlocked_achievements) if unlocked_achievements else "none"

        # Get position and other state
        position = obs.get("position", [0, 0])
        player_position = obs.get("player_position", position)
        player_direction = obs.get("player_direction", [0, 1])
        num_steps = obs.get("num_steps_taken", 0)

        # Check termination status
        terminated = obs.get("terminated", False)

        # Get semantic map view
        map_view = format_semantic_map_view(obs, view_size=5)

        return (
            f"Crafter Game State:\n"
            f"Step: {num_steps}\n"
            f"Health: {health}\n"
            f"Position: {player_position}\n"
            f"Direction: {player_direction}\n"
            f"Inventory: {inventory_str}\n"
            f"Achievements: {achievements_str}\n"
            f"Terminated: {terminated}\n"
            f"{map_view}\n\n"
            f"Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop\n\n"
            f"Key mechanics:\n"
            f"• 'do' action: collect wood from trees, stone from deposits, food from cows/plants\n"
            f"• 'do' does nothing on grass/water - move to find resources\n"
            f"• Craft progression: wood → table → wood_pickaxe → stone → stone_pickaxe → iron tools\n"
            f"• Sleep when energy low to restore and unlock wake_up achievement\n\n"
            f"Choose 1-5 actions to execute in sequence. Focus on exploring to find resources and crafting tools to unlock achievements."
        )


# --- Episode Runner ---
async def run_single_episode(
    client: AsyncClient, agent: CrafterReActAgent, task_instance, instance_num: int, traces_dir: str = "traces", 
    config=None, episode_progress_bars=None, all_achievements=None, all_invalid_actions=None, all_total_actions=None,
    experiment_id: Optional[str] = None
) -> Dict[str, Any]:
    # Timing tracking
    episode_start_time = time.time()
    step_times = []  # Time per step
    env_times = []   # Time for environment calls
    agent_times = [] # Time for agent LLM calls
    """Run a single Crafter episode and return episode metrics."""
    # Create session tracer for this episode with hooks and DuckDB storage
    # DuckDB will be auto-enabled based on LOCAL_SYNTH config
    tracer = SessionTracer(
        traces_dir, 
        hooks=CRAFTER_HOOKS, 
        duckdb_path="crafter_traces.duckdb",
        experiment_id=experiment_id
    )
    session_id = f"crafter_episode_{instance_num}_{task_instance.id}"
    tracer.start_session(session_id)
    
    # Progress bars already initialized in batch processing
    
    # Create system IDs
    import uuid
    runtime_id = uuid.uuid4()  # Runtime converts tool calls to actions
    environment_id = task_instance.id  # Use task instance ID for environment
    
    # Add episode metadata
    tracer.add_session_metadata("episode_config", {
        "instance_num": instance_num,
        "task_instance_id": str(task_instance.id),
        "difficulty": task_instance.metadata.difficulty,
        "max_turns": agent.max_turns,
        "model_name": agent.model_name,
        "agent_type": agent.system_name
    })
    
    # Add system ID mapping
    tracer.add_session_metadata("system_ids", {
        "agent": str(agent.system_id),
        "runtime": str(runtime_id),
        "environment": str(environment_id)
    })
    
    # Update agent with tracer
    agent.tracer = tracer
    
    try:
        # Create environment using the task instance
        create_resp = await retry_http_request(
            client, "POST", f"/env/CrafterClassic/initialize",
            json={"task_instance": await task_instance.serialize()}
        )

        if create_resp.status_code != 200:
            return {
                "eval_metric": 0.0,
                "rubric": {},
                "total_reward": 0.0,
                "num_achievements": 0,
                "terminated": False,
                "error": True,
            }

        env_id = create_resp.json()["env_id"]

        # Get initial observation
        obs = create_resp.json()["observation"]
        formatted_obs = agent.format_observation(obs)

        # DEBUG: Print initial state (minimal)
        # print(
        #     f"\n  Instance {instance_num}: Starting Crafter survival ({task_instance.metadata.difficulty}, {agent.max_turns} turns max)"
        # )

        # Track episode metrics
        total_reward = 0.0
        termination_reason = "max_turns_reached"  # Default assumption
        final_achievements = {}
        num_achievements = 0
        terminated = False
        rollout_length = 0

        # Run episode without tqdm progress bar (we use custom progress display)
        # Disable tqdm to avoid conflicts with custom progress bars
        episode_progress = range(agent.max_turns)
        
        for turn in episode_progress:
            step_start_time = time.time()
            try:
                # Create timestep for this turn (needed for record_event to work)
                if tracer:
                    tracer.start_timestep(f"turn_{turn}")
                
                # Record observation message (Environment → Runtime → Agent)
                obs_for_trace = compress_observation_for_trace(obs)
                
                obs_message = create_message(
                    content=obs_for_trace,  # Compressed observation
                    message_type="observation",
                    origin_system_id=environment_id,  # From environment
                    turn=turn
                )
                # Add message directly to session history
                if tracer.current_session:
                    tracer.current_session.add_message(obs_message)
                
                # Get agent decision
                agent_start_time = time.time()
                action = agent.decide(formatted_obs, agent.get_system_message(), turn)
                agent_end_time = time.time()
                agent_times.append(agent_end_time - agent_start_time)
                # print(f"  ✅ Agent decision received: {action}")

                # Record tool call message (Agent → Runtime)
                tool_call_message = create_message(
                    content=[{
                        "tool": action["name"],
                        "args": action["parameters"]
                    }],
                    message_type="tool_call",
                    origin_system_id=agent.system_id,  # From agent
                    turn=turn
                )
                # Add message directly to session history
                if tracer.current_session:
                    tracer.current_session.add_message(tool_call_message)

                # # DEBUG: Print agent decision with safer access
                # try:
                #     actions = action.get('parameters', {}).get('actions', action.get('arguments', {}).get('actions', []))
                #     reasoning = action.get('parameters', {}).get('reasoning', action.get('arguments', {}).get('reasoning', 'no reasoning'))
                #     #print(f"  Turn {turn+1}: Agent chose {actions} - {reasoning}")
                # except Exception as e:
                #     print(f"  Turn {turn+1}: Agent action structure: {action}")
                #     print(f"  Error parsing action: {e}")

                # Check for termination
                if action["name"] == "terminate":
                    reason = action.get("parameters", {}).get(
                        "reason", action.get("arguments", {}).get("reason", "no reason given")
                    )
                    termination_reason = f"agent_terminate: {reason}"
                    # Update tqdm progress bar for early termination
                    if episode_progress_bars is not None and instance_num in episode_progress_bars:
                        episode_progress_bars[instance_num].update(1)
                        episode_progress_bars[instance_num].set_description(
                            f"Episode {instance_num:2d} [T]"  # T for terminated
                        )
                    break

                # Execute actions in environment with safer access
                action_sequence = action.get("parameters", {}).get(
                    "actions", action.get("arguments", {}).get("actions", [])
                )
                if not action_sequence:
                    print(f"  ⚠️  No actions found in: {action}")
                    termination_reason = "no_actions_provided"
                    break

                # Convert action names to integers using the proper action map
                # Define the proper Crafter action mapping
                CRAFTER_ACTION_MAP = {
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

                action_ints = []
                for action_name in action_sequence:
                    if action_name in CRAFTER_ACTION_MAP:
                        action_int = CRAFTER_ACTION_MAP[action_name]
                    else:
                        action_int = 0  # Default to noop
                    action_ints.append(action_int)

                # Record runtime event before executing actions
                prev_obs = compress_observation_for_trace(obs)
                
                runtime_event = RuntimeEvent(
                    system_state_before={"observation": prev_obs},
                    actions=action_ints,
                    metadata={
                        "action_names": action_sequence,
                        "action_reasoning": action.get("parameters", {}).get("reasoning", ""),
                        "turn": turn
                    }
                )

                # Record action messages (Runtime → Environment)
                for action_int in action_ints:
                    action_message = create_message(
                        content={
                            "action": action_int,
                            "action_type": "crafter_action"
                        },
                        message_type="action",
                        origin_system_id=runtime_id,  # From runtime
                        turn=turn
                    )
                    if tracer.current_session:
                        tracer.current_session.add_message(action_message)
                
                # Execute each action individually (Crafter expects single actions)
                for i, action_int in enumerate(action_ints):
                    try:
                        # Time just the HTTP request
                        env_start_time = time.time()
                        step_resp = await retry_http_request(
                            client, "POST", f"/env/CrafterClassic/step",
                            json={
                                "env_id": env_id,
                                "request_id": str(uuid.uuid4()),
                                "action": {
                                    "tool_calls": [{"tool": "interact", "args": {"action": action_int}}]
                                },
                            }
                        )
                        env_end_time = time.time()
                        env_times.append(env_end_time - env_start_time)

                        if step_resp.status_code != 200:
                            print(
                                f"    ❌ Action {i + 1} failed: {step_resp.status_code}: {step_resp.text}"
                            )
                            termination_reason = f"http_error: {step_resp.status_code}"
                            break

                        # Update observation after each action
                        obs = step_resp.json()["observation"]
                        
                    except Exception as e:
                        print(f"    ❌ Action {i + 1} failed after retries: {type(e).__name__}: {str(e)[:100]}")
                        termination_reason = f"http_error: {type(e).__name__}"
                        break

                # Check if we broke out of the action loop due to an error
                if termination_reason.startswith("http_error"):
                    break

                # Convert observation to compressed format before saving
                obs_for_trace = compress_observation_for_trace(obs)
                
                # Record runtime event now that we have the final state
                runtime_event.system_state_after = {"observation": obs_for_trace}
                if tracer.current_session:
                    tracer.record_event(runtime_event)

                # Record environment event for the state transition
                # Extract public and private state from observations
                prev_public_state = {
                    k: v for k, v in prev_obs.items() 
                    if k not in ["reward_last_step", "total_reward_episode", "terminated", "truncated", "tool_error"]
                }
                prev_private_state = {
                    "reward_last_step": prev_obs.get("reward_last_step", 0.0),
                    "total_reward_episode": prev_obs.get("total_reward_episode", 0.0),
                    "terminated": prev_obs.get("terminated", False),
                    "truncated": prev_obs.get("truncated", False)
                }
                
                new_public_state = {
                    k: v for k, v in obs_for_trace.items() 
                    if k not in ["reward_last_step", "total_reward_episode", "terminated", "truncated", "tool_error"]
                }
                new_private_state = {
                    "reward_last_step": obs_for_trace.get("reward_last_step", 0.0),
                    "total_reward_episode": obs_for_trace.get("total_reward_episode", 0.0),
                    "terminated": obs_for_trace.get("terminated", False),
                    "truncated": obs_for_trace.get("truncated", False)
                }
                
                env_event = EnvironmentEvent(
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=turn
                    ),
                    system_instance_id=str(environment_id),
                    system_state_before={
                        "public_state": prev_public_state,
                        "private_state": prev_private_state
                    },
                    system_state_after={
                        "public_state": new_public_state,
                        "private_state": new_private_state
                    },
                    reward=obs.get("reward_last_step", 0.0),
                    terminated=obs.get("terminated", False),
                    metadata={
                        "actions_executed": action_sequence,
                        "turn": turn
                    }
                )
                if tracer.current_session:
                    tracer.record_event(env_event)

                # Show final state after all actions
                formatted_obs = agent.format_observation(obs)
                step_count = obs.get("num_steps_taken", 0)
                rollout_length = step_count
                position = obs.get("player_position", [0, 0])
                # print(f"  Turn {turn+1}: Actions completed - Step: {step_count}, Position: {position}")
                
                # Track step timing
                step_end_time = time.time()
                step_times.append(step_end_time - step_start_time)
                
                # Track progress for this turn
                if episode_progress_bars is not None:
                    # Check what hooks fired by looking at the events we just created
                    step_code = '#'  # Default for regular step
                    
                    # Collect all hooks that fired with their priorities
                    all_hooks = []
                    
                    # Check environment event hooks
                    if env_event and env_event.event_metadata:
                        for meta in env_event.event_metadata:
                            if 'code' in meta:
                                all_hooks.append((meta.get('priority', 0), meta.get('code', '?')))
                    
                    # Check agent hooks from the agent's recorded event
                    if hasattr(agent, 'last_cais_event') and agent.last_cais_event and agent.last_cais_event.event_metadata:
                        for meta in agent.last_cais_event.event_metadata:
                            if 'code' in meta:
                                all_hooks.append((meta.get('priority', 0), meta.get('code', '?')))
                    
                    # Select the hook with highest priority
                    if all_hooks:
                        all_hooks.sort(key=lambda x: x[0], reverse=True)  # Sort by priority descending
                        step_code = all_hooks[0][1]  # Get code of highest priority hook
                    
                    # Update tqdm progress bar for this episode
                    if instance_num in episode_progress_bars:
                        # Update the progress bar by 1 step
                        episode_progress_bars[instance_num].update(1)
                        
                        # Update the description with achievement count
                        if step_code in ['E', 'M', 'H']:  # On achievements
                            # Count total achievements for this episode
                            achv_count = len([k for k, v in all_achievements.items() if v > 0 and k.startswith(f"episode_{instance_num}_")])
                            episode_progress_bars[instance_num].set_description(
                                f"Episode {instance_num:2d} [{achv_count} achv]"
                            )

                # Update history with safer access
                reasoning = action.get("parameters", {}).get(
                    "reasoning", action.get("arguments", {}).get("reasoning", "")
                )
                agent.history.append(f"{', '.join(action_sequence)}: {reasoning[:50]}")

                # Track episode progress - Use the FINAL observation from the last action
                terminated = obs.get("terminated", False)
                step_reward = obs.get("reward", 0.0)
                total_reward += step_reward
                achievements = obs.get("achievements") or obs.get("achievements_status", {})

                # ALWAYS update final_achievements with the latest observation
                final_achievements = achievements

                num_achievements = sum(1 for v in achievements.values() if v) if achievements else 0

                # Update progress bar description with achievements
                if episode_progress_bars is not None and instance_num in episode_progress_bars:
                    episode_progress_bars[instance_num].set_description(
                        f"Episode {instance_num:2d} [{num_achievements} achv]"
                    )

                # No need to advance turn - we're tracking via message_time

                if terminated:
                    # Check if it's death or other environment termination
                    health = obs.get("health", 9)  # Default health value
                    if health <= 0:
                        termination_reason = "death"
                    else:
                        termination_reason = "environment_terminated"
                    break

            except Exception as e:
                # Error occurred
                error_msg = str(e) if str(e) else f"{type(e).__name__}"
                termination_reason = f"exception: {error_msg[:50]}"
                print(f"  ❌ Episode {instance_num} failed with exception: {error_msg}")
                import traceback
                traceback.print_exc()
                break
        
        # Close/finish the tqdm progress bar for this episode
        if episode_progress_bars is not None and instance_num in episode_progress_bars:
            # Ensure the progress bar reaches 100% if not already
            remaining = episode_progress_bars[instance_num].total - episode_progress_bars[instance_num].n
            if remaining > 0:
                episode_progress_bars[instance_num].update(remaining)
            episode_progress_bars[instance_num].close()

        # Cleanup
        await client.post(f"/env/CrafterClassic/terminate", json={"env_id": env_id})

        # Add final episode metadata to trace
        tracer.add_session_metadata("episode_results", {
            "total_reward": total_reward,
            "num_achievements": num_achievements,
            "achievements": final_achievements,
            "rollout_length": rollout_length,
            "terminated": terminated
        })
        
        # End session - only upload to DuckDB, no JSON saving
        tracer.end_session(save=False, upload_to_db=True)
        
        # Clear the tracer reference to allow garbage collection
        agent.tracer = None

        # Track achievements globally
        if all_achievements is not None and final_achievements:
            for achievement, unlocked in final_achievements.items():
                if unlocked:
                    all_achievements[achievement] += 1
        
        # Track invalid actions globally
        if all_invalid_actions is not None and tracer.hooks:
            # Find the InvalidActionHook and get its tracked actions
            for hook in tracer.hooks:
                if hasattr(hook, '__class__') and hook.__class__.__name__ == 'InvalidActionHook':
                    if hasattr(hook, 'invalid_actions'):
                        for action, count in hook.invalid_actions.items():
                            all_invalid_actions[action] += count
                    if hasattr(hook, 'total_actions') and all_total_actions is not None:
                        for action, count in hook.total_actions.items():
                            all_total_actions[action] += count
        
        # Calculate K-weighted achievement reward
        achievement_reward = 0.0
        if final_achievements:
            for achievement, unlocked in final_achievements.items():
                if unlocked and achievement in ACHIEVEMENT_K_VALUES:
                    k_value = ACHIEVEMENT_K_VALUES[achievement]
                    achievement_reward += k_value * math.log(2)  # log(1+1) for single achievement

        # Use achievement reward as the total reward
        total_reward = achievement_reward

        # Calculate eval metric and rubric
        eval_metric = float(num_achievements)  # Simple metric: number of achievements

        # Create rubric with specific achievement checks
        rubric = {}
        if final_achievements:
            rubric = {
                "collect_wood": 1.0 if final_achievements.get("collect_wood", False) else 0.0,
                "collect_stone": 1.0 if final_achievements.get("collect_stone", False) else 0.0,
                "collect_coal": 1.0 if final_achievements.get("collect_coal", False) else 0.0,
                "collect_iron": 1.0 if final_achievements.get("collect_iron", False) else 0.0,
                "collect_diamond": 1.0 if final_achievements.get("collect_diamond", False) else 0.0,
                "place_table": 1.0 if final_achievements.get("place_table", False) else 0.0,
                "place_furnace": 1.0 if final_achievements.get("place_furnace", False) else 0.0,
                "make_wood_pickaxe": 1.0
                if final_achievements.get("make_wood_pickaxe", False)
                else 0.0,
                "make_stone_pickaxe": 1.0
                if final_achievements.get("make_stone_pickaxe", False)
                else 0.0,
                "make_iron_pickaxe": 1.0
                if final_achievements.get("make_iron_pickaxe", False)
                else 0.0,
                "make_wood_sword": 1.0 if final_achievements.get("make_wood_sword", False) else 0.0,
                "make_stone_sword": 1.0
                if final_achievements.get("make_stone_sword", False)
                else 0.0,
                "make_iron_sword": 1.0 if final_achievements.get("make_iron_sword", False) else 0.0,
                "defeat_skeleton": 1.0 if final_achievements.get("defeat_skeleton", False) else 0.0,
                "defeat_zombie": 1.0 if final_achievements.get("defeat_zombie", False) else 0.0,
                "wake_up": 1.0 if final_achievements.get("wake_up", False) else 0.0,
                "eat_cow": 1.0 if final_achievements.get("eat_cow", False) else 0.0,
                "eat_plant": 1.0 if final_achievements.get("eat_plant", False) else 0.0,
            }
        else:
            # Default rubric with all zeros
            rubric = {
                "collect_wood": 0.0,
                "collect_stone": 0.0,
                "collect_coal": 0.0,
                "collect_iron": 0.0,
                "collect_diamond": 0.0,
                "place_table": 0.0,
                "place_furnace": 0.0,
                "make_wood_pickaxe": 0.0,
                "make_stone_pickaxe": 0.0,
                "make_iron_pickaxe": 0.0,
                "make_wood_sword": 0.0,
                "make_stone_sword": 0.0,
                "make_iron_sword": 0.0,
                "defeat_skeleton": 0.0,
                "defeat_zombie": 0.0,
                "wake_up": 0.0,
                "eat_cow": 0.0,
                "eat_plant": 0.0,
            }

        episode_end_time = time.time()
        episode_total_time = episode_end_time - episode_start_time
        
        return {
            "eval_metric": eval_metric,
            "rubric": rubric,
            "total_reward": total_reward,
            "num_achievements": num_achievements,
            "achievements": final_achievements,
            "rollout_length": rollout_length,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "error": False,
            "timing": {
                "episode_total_time": episode_total_time,
                "step_times": step_times,
                "env_times": env_times,
                "agent_times": agent_times,
            }
        }

    except Exception as e:
        # Save trace even on error
        try:
            tracer.add_session_metadata("error", {"error_message": str(e)})
            tracer.end_session(save=True)
        except:
            pass  # Don't let trace saving errors mask the original error
        
        error_msg = str(e) if str(e) else f"{type(e).__name__}"
        clear_progress_display()
        print(f"  ❌ Episode {instance_num} failed with outer exception: {error_msg}")
        import traceback
        traceback.print_exc()
            
        return {
            "eval_metric": 0.0,
            "rubric": {},
            "total_reward": 0.0,
            "num_achievements": 0,
            "terminated": False,
            "termination_reason": f"outer_exception: {error_msg[:50]}",
            "error": True,
        }


# --- Batch Evaluation ---
async def evaluate_crafter_batch() -> Dict[str, Any]:
    """Evaluate Crafter agent on multiple easy instances."""
    print(f"🎯 Evaluating Crafter on {config.num_instances} {config.difficulty} instances...")
    
    # Create experiment context
    with DuckDBTraceManager("crafter_traces.duckdb") as db_manager:
        experiment_context = create_experiment_context(
            db_manager,
            experiment_name=None,  # Will auto-generate pet name
            description=f"Crafter evaluation: {config.num_instances} {config.difficulty} instances with {config.model_name}",
            system_name="crafter-react-agent",
            system_description=f"ReAct agent for Crafter using {config.model_name}"
        )
    
    experiment_id = experiment_context["experiment_id"]
    
    # Print experiment header with clear formatting
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {experiment_context['experiment_name']}")
    print(f"{'='*80}")
    print(f"📋 Experiment ID: {experiment_id}")
    print(f"🤖 System: {experiment_context['system_id']}")
    print(f"📌 Version: {experiment_context['system_version_id']}")
    print(f"🌿 Git Branch: {experiment_context['git_branch']}")
    print(f"📍 Git Commit: {experiment_context['git_commit']}")
    print(f"💾 Database: crafter_traces.duckdb")
    print(f"{'='*80}\n")
    
    # Traces are now saved to DuckDB only
    script_dir = Path(__file__).parent
    traces_dir = script_dir / "traces"  # Keep for compatibility but not used
    
    # Initialize progress tracking
    all_achievements = defaultdict(int)  # Track all achievements across episodes
    all_invalid_actions = defaultdict(int)  # Track all invalid actions across episodes
    all_total_actions = defaultdict(int)  # Track all actions across episodes
    
    # Print hook legend
    print_hook_legend()
    
    # Track trace files created during this run
    initial_trace_files = set(traces_dir.glob("*.json"))
    
    # Create tqdm progress bars for all episodes
    from tqdm.asyncio import tqdm
    episode_pbar_dict = {}
    for i in range(config.num_instances):
        episode_pbar_dict[i + 1] = tqdm(
            total=config.max_turns,
            desc=f"Episode {i + 1:2d}",
            position=i,
            leave=True,
            unit="steps"
        )

    # Get easy task instances using the taskset system
    from synth_ai.environments.examples.crafter_classic.taskset import (
        CrafterTaskInstance,
        CrafterTaskInstanceMetadata,
    )
    from synth_ai.environments.tasks.core import Impetus, Intent

    easy_task_instances = []
    for seed in range(config.num_instances):
        try:
            metadata = CrafterTaskInstanceMetadata(
                difficulty=config.difficulty,
                seed=seed,
                num_trees_radius=5,  # Good for easy difficulty
                num_cows_radius=2,
                num_hostiles_radius=0,  # No hostiles for easy
            )
            task_instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions=f"Survive and unlock achievements in an {config.difficulty} environment."
                ),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            easy_task_instances.append(task_instance)
        except Exception as e:
            print(f"  ⚠️  Failed to create task instance for seed {seed}: {e}")
            continue
    
    # print(f"  📝 Generated {len(easy_task_instances)} {config.difficulty} task instances")

    # Configure connection pool to prevent ReadErrors
    limits = httpx.Limits(
        max_keepalive_connections=30,  # Increase for better concurrency
        max_connections=100,  # More connections for parallel episodes
        keepalive_expiry=60.0  # Longer keepalive
    )
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        limits=limits,
        retries=3  # Transport-level retries
    )
    async with AsyncClient(
        base_url=config.service_base_url, 
        timeout=httpx.Timeout(
            connect=5.0,
            read=HTTP_TIMEOUT,
            write=5.0,
            pool=5.0
        ),
        transport=transport,
        limits=limits
    ) as client:
        # Run ALL trajectories in parallel (no batching)
        all_tasks = []
        
        # Create all tasks at once
        for i, task_instance in enumerate(easy_task_instances):
            agent = CrafterReActAgent(config.model_name, max_turns=config.max_turns, verbose=False)
            all_tasks.append(
                run_single_episode(client, agent, task_instance, i + 1, str(traces_dir), 
                                 config, episode_pbar_dict, all_achievements, all_invalid_actions, all_total_actions,
                                 experiment_id=experiment_id)
            )
        
        # Run all episodes in parallel
        all_results = await asyncio.gather(*all_tasks)

        results = all_results
        
        # Close all progress bars
        for pbar in episode_pbar_dict.values():
            pbar.close()

        # Filter out error results and exception-terminated episodes
        valid_results = []
        all_termination_reasons = []
        excluded_count = 0
        for r in results:
            termination_reason = r.get("termination_reason", "unknown")
            all_termination_reasons.append(termination_reason)
            
            # Exclude episodes that errored or ended with exceptions
            if r.get("error", False) or "exception" in termination_reason:
                excluded_count += 1
                print(f"  ⚠️  Excluding episode from stats due to: {termination_reason}")
                continue
            valid_results.append(r)
        
        if excluded_count > 0:
            print(f"  📊 Excluded {excluded_count} episodes from aggregate statistics due to errors/exceptions")

        if not valid_results:
            return {
                "eval_metrics": [],
                "mean_eval_metric": 0.0,
                "mean_rubric": {},
                "num_episodes": 0,
            }

        # Extract eval metrics and rubrics
        eval_metrics = [r["eval_metric"] for r in valid_results]
        mean_eval_metric = sum(eval_metrics) / len(eval_metrics)

        # --- Rollout length statistics ---
        rollout_lengths = [r["rollout_length"] for r in valid_results]
        sorted_lengths = sorted(rollout_lengths)
        n_lengths = len(sorted_lengths)
        # Median (Q2)
        if n_lengths % 2 == 1:
            q2_rollout = sorted_lengths[n_lengths // 2]
        else:
            q2_rollout = (sorted_lengths[n_lengths // 2 - 1] + sorted_lengths[n_lengths // 2]) / 2
        # 90th percentile (P90)
        p90_index = int(0.9 * (n_lengths - 1))
        p90_rollout = sorted_lengths[p90_index]
        max_rollout = sorted_lengths[-1]

        # Calculate mean rubric values
        all_rubric_keys = set()
        for r in valid_results:
            all_rubric_keys.update(r["rubric"].keys())

        mean_rubric = {}
        for key in all_rubric_keys:
            values = [r["rubric"].get(key, 0.0) for r in valid_results]
            mean_rubric[key] = sum(values) / len(values)

        # Calculate shaped reward (training rubric)
        # Count total achievements across all episodes
        achievement_counts = {}
        unique_achievements_per_trajectory = []
        all_unique_achievements = set()

        for result in valid_results:
            achievements = result.get("achievements", {})
            trajectory_achievements = set()
            for achievement, unlocked in achievements.items():
                if unlocked:
                    achievement_counts[achievement] = achievement_counts.get(achievement, 0) + 1
                    trajectory_achievements.add(achievement)
                    all_unique_achievements.add(achievement)
            unique_achievements_per_trajectory.append(trajectory_achievements)

        # Calculate shaped reward using the counts
        shaped_reward_data = calculate_shaped_reward(achievement_counts)

        # Calculate unique achievements by N trajectories
        unique_achievements_by_n = {}
        for n in range(1, len(valid_results) + 1):
            unique_at_n = set()
            for i in range(n):
                unique_at_n.update(unique_achievements_per_trajectory[i])
            unique_achievements_by_n[n] = len(unique_at_n)

        # Create training rubric (normalized shaped reward components)
        training_rubric = {}
        total_episodes = len(valid_results)
        if shaped_reward_data["breakdown"]:
            for achievement, data in shaped_reward_data["breakdown"].items():
                # Normalize by number of episodes for comparison
                training_rubric[achievement] = data["contribution"] / total_episodes

        # Get newly created trace files
        final_trace_files = set(traces_dir.glob("*.json"))
        new_trace_files = sorted(list(final_trace_files - initial_trace_files))
        
        return {
            "eval_metrics": eval_metrics,
            "mean_eval_metric": mean_eval_metric,
            "mean_rubric": mean_rubric,
            "achievement_counts": achievement_counts,
            "shaped_reward_data": shaped_reward_data,
            "training_rubric": training_rubric,
            "unique_achievements_per_trajectory": unique_achievements_per_trajectory,
            "all_unique_achievements": all_unique_achievements,
            "unique_achievements_by_n": unique_achievements_by_n,
            "num_episodes": len(valid_results),
            "q2_rollout": q2_rollout,
            "p90_rollout": p90_rollout,
            "max_rollout": max_rollout,
            "new_trace_files": new_trace_files,
            "all_achievements": dict(all_achievements),  # Add the tracked achievements
            "all_invalid_actions": dict(all_invalid_actions),  # Add the tracked invalid actions
            "all_total_actions": dict(all_total_actions),  # Add the total actions
            "termination_reasons": all_termination_reasons,  # Add termination reasons
            "raw_results": all_results,  # Add raw results for timing analysis
            "experiment_context": experiment_context,  # Add experiment context for tracking
        }


async def main():
    """Run Crafter evaluation."""
    # Record start time for trace filtering
    import time
    config._run_start_time = time.time()
    
    # Configure logging to reduce verbosity
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.ERROR)
    logging.getLogger("google.generativeai").setLevel(logging.ERROR)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("google_genai.types").setLevel(logging.ERROR)
    logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
    
    # Disable synth_ai LM debug logs
    logging.getLogger("synth_ai.lm.provider_support.openai").setLevel(logging.WARNING)
    logging.getLogger("synth_ai.lm").setLevel(logging.WARNING)
    
    # Suppress worker/trajectory logs from other systems
    logging.getLogger("synth_ai").setLevel(logging.WARNING)
    logging.getLogger("synth_ai.environments").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Suppress Langfuse client warnings
    logging.getLogger("langfuse").setLevel(logging.ERROR)
    logging.getLogger("langfuse.client").setLevel(logging.ERROR)
    
    # Set root logger to WARNING to suppress debug prints
    logging.getLogger().setLevel(logging.WARNING)

    print(f"🎮 Crafter ReAct Agent Evaluation")
    print(f"Model: {config.model_name}")
    print(f"Service: {config.service_base_url}")
    print(f"Instances: {config.num_instances}")
    print(f"Max Turns: {config.max_turns}")
    print(f"Difficulty: {config.difficulty}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Test service health
    # Use same connection pool settings for health check
    limits = httpx.Limits(
        max_keepalive_connections=30,
        max_connections=100,
        keepalive_expiry=60.0
    )
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        limits=limits,
        retries=3  # Transport-level retries
    )
    async with AsyncClient(
        base_url=config.service_base_url, 
        timeout=httpx.Timeout(
            connect=5.0,
            read=HTTP_TIMEOUT,
            write=5.0,
            pool=5.0
        ),
        transport=transport,
        limits=limits
    ) as client:
        try:
            health_resp = await retry_http_request(client, "GET", "/health")
            health_data = health_resp.json()

            if "CrafterClassic" not in health_data.get("supported_environments", []):
                print("❌ CrafterClassic not available on service")
                return

            print("✅ Service health check passed")

        except Exception as e:
            print(f"❌ Service health check failed after retries: {type(e).__name__}: {str(e)[:100]}")
            return

    # Run evaluation
    try:
        results = await evaluate_crafter_batch()

        print("\n" + "=" * 80)
        print("🏆 CRAFTER EVALUATION RESULTS")
        print("=" * 80)
        
        # Check if any episodes were excluded
        total_episodes = len(results.get("termination_reasons", []))
        valid_episodes = results["num_episodes"]
        if total_episodes > valid_episodes:
            excluded = total_episodes - valid_episodes
            print(f"📝 Note: {excluded} episode(s) excluded from statistics due to errors/exceptions")
            print("=" * 80)

        # Calculate key metrics
        achievements_per_trajectory = [
            len(achievements)
            for achievements in results.get("unique_achievements_per_trajectory", [])
        ]
        avg_achievements = (
            sum(achievements_per_trajectory) / len(achievements_per_trajectory)
            if achievements_per_trajectory
            else 0.0
        )
        total_unique = len(results.get("all_unique_achievements", set()))
        shaped_reward = results.get("shaped_reward_data", {}).get("total_shaped_reward", 0.0)
        mean_k_score = (
            shaped_reward / results["num_episodes"] if results["num_episodes"] > 0 else 0.0
        )
        achievement_counts = results.get("achievement_counts", {})
        
        # Assessment
        if results["mean_eval_metric"] >= 3.0:
            assessment = "🎉 Excellent"
        elif results["mean_eval_metric"] >= 1.0:
            assessment = "✅ Good"
        elif results["mean_eval_metric"] >= 0.5:
            assessment = "⚠️  Moderate"
        else:
            assessment = "📈 Learning"

        # Create dense results table
        print(f"│ Metric                    │ Value                                     │")
        print(f"├───────────────────────────┼───────────────────────────────────────────┤")
        print(f"│ Model                     │ {config.model_name:<41} │")
        print(f"│ Episodes                  │ {results['num_episodes']:<41} │")
        print(f"│ Mean Score                │ {results['mean_eval_metric']:<41.2f} │")
        print(f"│ Avg Achievements/Episode  │ {avg_achievements:<41.2f} │")
        print(f"│ Unique Achievements       │ {total_unique:<41} │")
        print(f"│ Shaped Reward (Total)     │ {shaped_reward:<41.3f} │")
        print(f"│ Mean K-Score/Episode      │ {mean_k_score:<41.3f} │")
        print(f"│ Q2 Rollout Length         │ {results.get('q2_rollout', 0):<41} │")
        print(f"│ Max Rollout Length        │ {results.get('max_rollout', 0):<41} │")
        
        # Show unlocked achievements
        all_unique = results.get("all_unique_achievements", set())
        if all_unique:
            achievements_str = ', '.join(sorted(all_unique))
            if len(achievements_str) > 41:
                achievements_str = achievements_str[:38] + "..."
            print(f"│ Unlocked Achievements     │ {achievements_str:<41} │")
        
        print(f"└───────────────────────────┴───────────────────────────────────────────┘")
        
        # Print experiment info
        if 'experiment_context' in results:
            exp_ctx = results['experiment_context']
            print(f"\n{'='*80}")
            print(f"📊 EXPERIMENT SAVED TO DUCKDB")
            print(f"{'='*80}")
            print(f"🧪 Name: {exp_ctx['experiment_name']}")
            print(f"📋 ID: {exp_ctx['experiment_id']}")
            print(f"🌿 Git: {exp_ctx['git_branch']} @ {exp_ctx['git_commit'][:8]}")
            print(f"\n🔍 Query this experiment:")
            print(f"   python -m synth_ai.tui.cli.query_experiments -e {exp_ctx['experiment_id'][:8]}")
            print(f"\n📊 Or view all experiments:")
            print(f"   python -m synth_ai.tui.cli.query_experiments")
            print(f"{'='*80}")
        
        # Print trace file sizes for newly created files only
        new_trace_files = results.get("new_trace_files", [])
        if new_trace_files:
            total_size_mb = 0
            for trace_file in new_trace_files:
                size_bytes = trace_file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                total_size_mb += size_mb
            
            # Skip verbose trace analysis (moved to summary tables only)
        
        # Check if verbose output is requested
        if hasattr(config, 'verbose_output') and config.verbose_output:
            # Display achievements table
            all_achievements = results.get("all_achievements", {})
            if all_achievements:
                print_achievements_table(all_achievements, results['num_episodes'])
            
            # Display invalid actions table
            invalid_actions = results.get("all_invalid_actions", {})
            total_actions = results.get("all_total_actions", {})
            if invalid_actions:
                print_invalid_actions_table(invalid_actions, total_actions)
            
            # Display termination breakdown
            termination_reasons = results.get("termination_reasons", [])
            if termination_reasons:
                print_termination_breakdown(termination_reasons)
            
            # Display timing analysis
            raw_results = results.get("raw_results", [])
            if raw_results:
                print_timing_analysis(raw_results)
        else:
            # Display condensed summary (default)
            all_achievements = results.get("all_achievements", {})
            invalid_actions = results.get("all_invalid_actions", {})
            total_actions = results.get("all_total_actions", {})
            termination_reasons = results.get("termination_reasons", [])
            raw_results = results.get("raw_results", [])
            
            print_condensed_summary(all_achievements, invalid_actions, total_actions, 
                                  termination_reasons, raw_results, results['num_episodes'])
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Crafter ReAct Agent Evaluation")
    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--model", "-m", type=str, help="Model name (overrides config)")
    parser.add_argument("--episodes", "-e", type=int, help="Number of episodes (overrides config)")
    parser.add_argument(
        "--max-turns", "-t", type=int, help="Maximum turns per episode (overrides config)"
    )
    parser.add_argument("--difficulty", "-d", type=str, help="Difficulty level (overrides config)")
    parser.add_argument("--analyze-traces", action="store_true", help="Analyze trace files after evaluation")
    parser.add_argument("--evaluate-traces", action="store_true", help="Run trace evaluation scoring after episodes")
    parser.add_argument("--verbose", action="store_true", help="Use verbose output format (detailed tables and statistics)")

    # Custom OpenAI endpoint support
    parser.add_argument(
        "--openai-base-url",
        type=str,
        help="Custom OpenAI-compatible base URL (e.g., https://lora-service.modal.run)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="dummy",
        help="API key for custom endpoint (default: 'dummy')",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = CrafterConfig(args.config)
    else:
        # Try to load default config
        default_config_path = (
            Path(__file__).parent.parent.parent.parent / "evals" / "configs" / "crafter.toml"
        )
        if default_config_path.exists():
            config = CrafterConfig(str(default_config_path))
        else:
            config = CrafterConfig()

    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    if args.episodes:
        config.num_instances = args.episodes
    if args.max_turns:
        config.max_turns = args.max_turns
    if args.difficulty:
        config.difficulty = args.difficulty
    if args.analyze_traces:
        config.analyze_traces = True
    if args.evaluate_traces:
        config.evaluate_traces = True
    if args.verbose:
        config.verbose_output = True

    # Configure custom OpenAI endpoint if provided
    if args.openai_base_url:
        config.set_custom_endpoint(args.openai_base_url, args.openai_api_key)

    # Fail fast if model_name still missing
    if not config.model_name:
        raise ValueError(
            "CrafterConfig: 'model_name' must be specified in the TOML config or via --model CLI argument; no fallback default."
        )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run trace evaluation if requested
    if config.analyze_traces or hasattr(config, 'evaluate_traces') and config.evaluate_traces:
        print("\n" + "=" * 80)
        print("📊 TRACE EVALUATION - DEPRECATED")
        print("=" * 80)
        print("⚠️  JSON trace files are no longer generated.")
        print("All trace data is now stored in DuckDB (crafter_traces.duckdb)")
        print("\nTo analyze traces:")
        print("1. Use the DuckDB analytics summary shown above")
        print("2. Query the database directly using DuckDBTraceManager")
        print("3. Use filter_traces_sft_duckdb.py to extract training data")
    else:
        # Original trace evaluation code (now unreachable)
        pass
        
    # Skip the old trace evaluation code entirely
    if False:  # Never execute old trace evaluation
        try:
            # Add current directory to Python path to import trace_eval
            import sys
            from pathlib import Path
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            
            from trace_eval import evaluate_all_traces, print_evaluation_summary
            
            trace_dir = current_dir / "traces"
            if trace_dir.exists():
                # Find trace files created during this run
                recent_traces = []
                
                # Use run start time if available, otherwise fall back to last 60 seconds
                import time
                if hasattr(config, '_run_start_time'):
                    start_time = config._run_start_time
                else:
                    start_time = time.time() - 60  # Only traces from last minute
                
                for trace_file in trace_dir.glob("*.json"):
                    # Check if file was created after start time
                    if trace_file.stat().st_mtime >= start_time:
                        recent_traces.append(trace_file)
                
                if recent_traces:
                    print(f"Evaluating {len(recent_traces)} trace files from this run...")
                    results = []
                    for trace_file in recent_traces:
                        from trace_eval import evaluate_trace, print_trace_evaluation
                        result = evaluate_trace(trace_file)
                        results.append(result)
                    
                    # Sort by score
                    results.sort(key=lambda x: x['total_score'], reverse=True)
                    
                    # Check if verbose output is requested
                    if hasattr(config, 'verbose_output') and config.verbose_output:
                        # Show detailed evaluation only if not too many traces
                        if len(results) <= 5:
                            for result in results:
                                print_trace_evaluation(result)
                        
                        # Always show summary
                        print_evaluation_summary(results)
                    else:
                        # Show only condensed trace summary (default)
                        if results:
                            print("\n" + "─" * 80)
                            print("TRACE EVALUATION")
                            print("─" * 80)
                            avg_score = sum(r['total_score'] for r in results) / len(results)
                            best_score = max(r['total_score'] for r in results)
                            worst_score = min(r['total_score'] for r in results)
                            print(f"Traces: {len(results)} | Avg: {avg_score:.2f} | Best: {best_score:.2f} | Worst: {worst_score:.2f}")
                            print("─" * 80)
                    
                    # Also save to file for debugging
                    eval_file = current_dir / "last_evaluation.txt"
                    with open(eval_file, 'w') as f:
                        f.write(f"Evaluation of {len(results)} traces\n")
                        f.write("="*60 + "\n")
                        for result in results:
                            f.write(f"\nTrace: {result['trace_file']}\n")
                            f.write(f"Score: {result['total_score']:.2f}\n")
                            f.write(f"Trajectory: {result['trajectory']}\n")
                    print(f"\n📝 Evaluation saved to: {eval_file}")
                else:
                    print("No recent trace files found to evaluate.")
            else:
                print(f"Trace directory not found: {trace_dir}")
                
        except ImportError:
            print("⚠️  trace_eval module not found. Skipping trace evaluation.")
        except Exception as e:
            print(f"⚠️  Error during trace evaluation: {e}")
    
    # Show DuckDB analytics summary if available
    try:
        from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
        print("\n" + "─" * 80)
        print("DUCKDB TRACE ANALYTICS")
        print("─" * 80)
        
        with DuckDBTraceManager("crafter_traces.duckdb") as db:
            # Get model usage stats
            model_stats = db.get_model_usage()
            if not model_stats.empty:
                print("\n📊 Model Usage:")
                for _, row in model_stats.iterrows():
                    print(f"  • {row['model_name']}: {row['call_count']} calls, "
                          f"{row['total_tokens']} tokens, ${row['total_cost']:.4f}")
            
            # Get session summary
            sessions = db.get_session_summary()
            if not sessions.empty:
                print(f"\n📈 Sessions: {len(sessions)} total")
                print(f"  • Avg events per session: {sessions['num_events'].mean():.1f}")
                print(f"  • Total cost: ${sessions['total_cost'].sum():.4f}")
            
            print("─" * 80)
            print(f"💾 Trace data stored in: crafter_traces.duckdb")
    except Exception as e:
        # Silently skip if DuckDB not available or no data
        pass
    
    # Normal exit (allow cleanup and final output)
    sys.exit(0)
