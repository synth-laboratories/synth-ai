#!/usr/bin/env python3
"""
Run Crafter rollouts with Qwen models and display results in a table format
"""

import asyncio
import json
import uuid
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
import httpx
import os
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from collections import defaultdict

# Disable Langfuse 
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = "dummy"
os.environ["LANGFUSE_SECRET_KEY"] = "dummy"

# Import Crafter hooks
try:
    from synth_ai.environments.examples.crafter_classic.trace_hooks import CRAFTER_HOOKS
except ImportError:
    CRAFTER_HOOKS = []

# Service configuration
MODAL_BASE_URL = "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run"
MODAL_API_KEY = os.environ.get("MODAL_API_KEY", "sk-test-11111111111111111111111111111111")

# Model size routing based on Modal service configuration
MODEL_SIZE_ROUTING = {
    "0.5B": "small",
    "1.5B": "small", 
    "3B": "small",
    "7B": "medium",
    "14B": "medium",
    "32B": "large32",
    "72B": "large72"
}

def get_model_size_category(model_name: str) -> str:
    """Get the size category for routing."""
    for size, category in MODEL_SIZE_ROUTING.items():
        if f"-{size}-" in model_name or model_name.endswith(f"-{size}"):
            return category
    return "medium"  # Default to medium

# HTTP retry configuration
MAX_RETRIES = 3
BASE_DELAY = 0.1
MAX_DELAY = 2.0
HTTP_TIMEOUT = 120.0

console = Console()

class RolloutConfig(BaseModel):
    """Configuration for rollout evaluation."""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 512
    
    # Evaluation settings
    num_episodes: int = 10
    max_steps_per_episode: int = 100
    difficulty: str = "easy"
    seed: Optional[int] = None
    
    # Service settings
    crafter_url: str = "http://localhost:8901"
    llm_base_url: str = MODAL_BASE_URL
    llm_api_key: str = MODAL_API_KEY
    
    # Display settings
    show_live_progress: bool = True
    save_results: bool = True
    output_file: Optional[str] = None


class EpisodeStats:
    """Track statistics for an episode."""
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.steps = 0
        self.total_reward = 0.0
        self.achievements = []
        self.final_health = 0
        self.final_hunger = 0
        self.final_thirst = 0
        self.resources_collected = defaultdict(int)
        self.actions_taken = defaultdict(int)
        self.start_time = time.time()
        self.end_time = None
        self.termination_reason = None
        self.llm_response_times = []
        
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def avg_response_time(self) -> float:
        if self.llm_response_times:
            return np.mean(self.llm_response_times)
        return 0.0


async def retry_http_request(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Any:
    """Retry HTTP requests with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(BASE_DELAY * (2 ** (attempt - 1)))
            
            response = await client.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            
            if response.status_code < 500:
                return response
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise e
    
    raise Exception(f"Failed after {MAX_RETRIES} attempts")


async def warmup_model(config: RolloutConfig, max_attempts: int = 30) -> bool:
    """Warmup the model by polling until it's ready."""
    console.print(f"[yellow]Warming up {config.model_name}...[/yellow]")
    
    # First try the warmup endpoint if available
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        # Try warmup endpoint
        try:
            warmup_url = f"{config.llm_base_url}/warmup/{config.model_name}"
            response = await client.post(warmup_url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                console.print("[green]✓ Model warmup endpoint called[/green]")
        except:
            pass  # Warmup endpoint might not exist
        
        # Now poll with actual inference requests
        test_messages = [
            {"role": "user", "content": "Say 'ready' if you're loaded."}
        ]
        
        for attempt in range(max_attempts):
            try:
                start_time = time.time()
                response = await client.post(
                    f"{config.llm_base_url}/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": config.model_name,
                        "messages": test_messages,
                        "temperature": 0.1,
                        "max_tokens": 10,
                    },
                    timeout=120.0
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and data["choices"]:
                        console.print(f"[green]✓ Model ready! (response time: {elapsed:.1f}s)[/green]")
                        return True
                
                # If we get here, model is still loading
                if elapsed > 10:
                    console.print(f"[yellow]Model is loading... attempt {attempt + 1}/{max_attempts} (took {elapsed:.1f}s)[/yellow]")
                
            except httpx.TimeoutException:
                console.print(f"[yellow]Timeout waiting for model... attempt {attempt + 1}/{max_attempts}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error during warmup: {str(e)[:100]}[/yellow]")
            
            # Wait before retrying
            await asyncio.sleep(5)
        
        console.print(f"[red]Failed to warmup model after {max_attempts} attempts[/red]")
        return False


async def call_llm(messages: List[Dict[str, str]], config: RolloutConfig) -> Tuple[str, float]:
    """Call LLM and return response with timing."""
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {config.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        
        start_time = time.time()
        response = await retry_http_request(
            client, 
            "POST",
            f"{config.llm_base_url}/v1/chat/completions",
            headers=headers,
            json=payload
        )
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"], elapsed


def format_observation(obs: Dict[str, Any]) -> str:
    """Format observation into a concise prompt."""
    inv = obs.get("inventory", {})
    health = obs.get("health", 10)
    hunger = obs.get("food", 10)
    thirst = obs.get("drink", 10)
    
    # Get nearby objects in a 5x5 view
    semantic_map = obs.get("semantic_map")
    if semantic_map is not None:
        # Simple 5x5 view around player
        view = []
        for dy in range(-2, 3):
            row = []
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    row.append("P")
                else:
                    # Simplified - just show if something is there
                    row.append(".")
            view.append(" ".join(row))
        map_str = "\n".join(view)
    else:
        map_str = "Map unavailable"
    
    # Format inventory (only non-zero items)
    inv_items = [f"{k}:{v}" for k, v in inv.items() 
                 if v > 0 and k not in ["health", "food", "drink", "energy"]]
    inv_str = ", ".join(inv_items) if inv_items else "empty"
    
    return f"""Status: Health={health}/10, Hunger={hunger}/10, Thirst={thirst}/10
Inventory: {inv_str}
Nearby (5x5, P=player):
{map_str}

What action should you take? Choose one:
move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword

Action:"""


async def run_episode(
    episode_id: str,
    config: RolloutConfig,
    progress: Optional[Any] = None
) -> EpisodeStats:
    """Run a single episode."""
    stats = EpisodeStats(episode_id)
    
    async with httpx.AsyncClient() as client:
        # Create environment
        create_resp = await retry_http_request(
            client,
            "POST",
            f"{config.crafter_url}/CrafterClassic/create",
            json={
                "instance_id": episode_id,
                "render_mode": "rgb_array",
                "difficulty": config.difficulty,
                "seed": config.seed
            }
        )
        
        env_data = create_resp.json()
        instance_id = env_data["instance_id"]
        
        # Reset environment
        reset_resp = await retry_http_request(
            client,
            "POST",
            f"{config.crafter_url}/CrafterClassic/{instance_id}/reset",
            json={}
        )
        
        obs_data = reset_resp.json().get("private", {})
        
        # System message for the agent
        messages = [{
            "role": "system",
            "content": "You are playing Crafter, a survival game. Your goals are to: 1) Stay alive by maintaining health/hunger/thirst, 2) Gather resources (wood, stone, etc), 3) Craft tools and items. Respond with only the action name."
        }]
        
        # Action mapping
        action_map = {
            'noop': 0, 'move_left': 1, 'move_right': 2, 'move_up': 3,
            'move_down': 4, 'do': 5, 'sleep': 6, 'place_stone': 7,
            'place_table': 8, 'place_furnace': 9, 'place_plant': 10,
            'make_wood_pickaxe': 11, 'make_stone_pickaxe': 12,
            'make_iron_pickaxe': 13, 'make_wood_sword': 14,
            'make_stone_sword': 15, 'make_iron_sword': 16
        }
        
        # Run episode
        for step in range(config.max_steps_per_episode):
            # Create prompt
            prompt = format_observation(obs_data)
            messages.append({"role": "user", "content": prompt})
            
            # Get LLM response
            try:
                response_text, response_time = await call_llm(messages, config)
                stats.llm_response_times.append(response_time)
                
                # Parse action
                action = None
                response_lower = response_text.strip().lower()
                for action_name in action_map.keys():
                    if action_name in response_lower:
                        action = action_name
                        break
                
                if not action:
                    action = "do"  # Default
                
                stats.actions_taken[action] += 1
                action_idx = action_map[action]
                
                # Take action
                step_payload = {
                    "env_id": instance_id,
                    "request_id": f"{episode_id}_step_{step}",
                    "action": {
                        "tool_calls": [{
                            "tool": "interact",
                            "args": {"action": action_idx}
                        }]
                    }
                }
                
                step_resp = await retry_http_request(
                    client,
                    "POST",
                    f"{config.crafter_url}/env/CrafterClassic/step",
                    json=step_payload
                )
                
                step_data = step_resp.json()
                new_obs = step_data.get("private", {})
                reward = step_data.get("reward", 0) or 0
                done = step_data.get("done", False)
                
                stats.total_reward += reward
                stats.steps += 1
                
                # Track achievements
                for ach, status in new_obs.get("achievements_status", {}).items():
                    if status and ach not in stats.achievements:
                        stats.achievements.append(ach)
                
                # Track resources
                inv = new_obs.get("inventory", {})
                for item, count in inv.items():
                    if item not in ["health", "food", "drink", "energy"] and count > 0:
                        stats.resources_collected[item] = max(stats.resources_collected[item], count)
                
                # Update final stats
                stats.final_health = inv.get("health", 0)
                stats.final_hunger = inv.get("food", 0)
                stats.final_thirst = inv.get("drink", 0)
                
                # Keep conversation short
                messages = messages[-4:]  # Keep only recent context
                messages.append({"role": "assistant", "content": action})
                
                if done:
                    stats.termination_reason = step_data.get("termination_reason", "completed")
                    break
                
                obs_data = new_obs
                
                if progress:
                    progress()
                    
            except Exception as e:
                stats.termination_reason = f"error: {str(e)}"
                break
        
        # Clean up
        try:
            await client.post(f"{config.crafter_url}/CrafterClassic/{instance_id}/terminate")
        except:
            pass
        
        stats.end_time = time.time()
        return stats


def create_results_table(all_stats: List[EpisodeStats]) -> Table:
    """Create a rich table with results."""
    table = Table(title="Crafter Rollout Results", show_header=True, header_style="bold magenta")
    
    table.add_column("Episode", style="cyan", width=12)
    table.add_column("Steps", justify="right", style="green")
    table.add_column("Reward", justify="right", style="yellow")
    table.add_column("Achievements", justify="right", style="blue")
    table.add_column("Resources", justify="center", style="magenta")
    table.add_column("Final Status", justify="center")
    table.add_column("Time (s)", justify="right", style="dim")
    table.add_column("Avg LLM (s)", justify="right", style="dim")
    
    for stats in all_stats:
        # Format resources
        resources = []
        for item, count in stats.resources_collected.items():
            resources.append(f"{item}:{count}")
        resources_str = ", ".join(resources[:3]) if resources else "none"
        if len(resources) > 3:
            resources_str += "..."
        
        # Format final status
        status = f"H:{stats.final_health} F:{stats.final_hunger} T:{stats.final_thirst}"
        
        # Color code based on performance
        reward_style = "green" if stats.total_reward > 0 else "red"
        ach_style = "green" if len(stats.achievements) > 0 else "dim"
        
        table.add_row(
            stats.episode_id.split("_")[-1][:8],
            str(stats.steps),
            f"[{reward_style}]{stats.total_reward:.1f}[/{reward_style}]",
            f"[{ach_style}]{len(stats.achievements)}[/{ach_style}]",
            resources_str,
            status,
            f"{stats.duration():.1f}",
            f"{stats.avg_response_time():.1f}"
        )
    
    return table


def create_summary_panel(all_stats: List[EpisodeStats], config: RolloutConfig) -> Panel:
    """Create a summary panel."""
    total_episodes = len(all_stats)
    successful_episodes = sum(1 for s in all_stats if s.total_reward > 0)
    
    avg_reward = np.mean([s.total_reward for s in all_stats]) if all_stats else 0
    avg_steps = np.mean([s.steps for s in all_stats]) if all_stats else 0
    avg_achievements = np.mean([len(s.achievements) for s in all_stats]) if all_stats else 0
    
    # Count all achievements
    all_achievements = defaultdict(int)
    for stats in all_stats:
        for ach in stats.achievements:
            all_achievements[ach] += 1
    
    # Most common actions
    all_actions = defaultdict(int)
    for stats in all_stats:
        for action, count in stats.actions_taken.items():
            all_actions[action] += count
    
    top_actions = sorted(all_actions.items(), key=lambda x: x[1], reverse=True)[:5]
    
    summary_text = f"""[bold]Model:[/bold] {config.model_name}
[bold]Episodes:[/bold] {total_episodes} (Successful: {successful_episodes})
[bold]Average Reward:[/bold] {avg_reward:.2f}
[bold]Average Steps:[/bold] {avg_steps:.1f}
[bold]Average Achievements:[/bold] {avg_achievements:.1f}

[bold]Top Achievements:[/bold]
"""
    
    for ach, count in sorted(all_achievements.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / total_episodes) * 100
        summary_text += f"  • {ach}: {count} ({pct:.0f}%)\n"
    
    summary_text += "\n[bold]Top Actions:[/bold]\n"
    for action, count in top_actions:
        summary_text += f"  • {action}: {count}\n"
    
    return Panel(summary_text, title="Summary Statistics", border_style="green")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Crafter rollouts with Qwen models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--difficulty", type=str, default="easy",
                       choices=["easy", "normal", "hard", "peaceful"],
                       help="Game difficulty")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="LLM temperature")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--skip-warmup", action="store_true",
                       help="Skip model warmup phase")
    
    args = parser.parse_args()
    
    # Create config
    config = RolloutConfig(
        model_name=args.model,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        difficulty=args.difficulty,
        seed=args.seed,
        temperature=args.temperature,
        save_results=args.save,
        output_file=args.output
    )
    
    # Set up logging - suppress httpx INFO logs
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    console.print(f"[bold green]🎮 Crafter Rollouts with {config.model_name}[/bold green]")
    console.print(f"Episodes: {config.num_episodes}, Max steps: {config.max_steps_per_episode}")
    console.print(f"Difficulty: {config.difficulty}, Temperature: {config.temperature}")
    
    # Show expected routing
    expected_category = get_model_size_category(config.model_name)
    console.print(f"[dim]Expected Modal container: base_model_{expected_category}_generate[/dim]")
    console.print()
    
    # Warmup the model first
    if not args.skip_warmup:
        warmup_success = await warmup_model(config)
        if not warmup_success:
            console.print("[red]Failed to warmup model. Continue anyway? (y/n)[/red]")
            response = input().strip().lower()
            if response != 'y':
                return
    else:
        console.print("[yellow]Skipping model warmup (--skip-warmup specified)[/yellow]")
    
    console.print()
    all_stats = []
    
    # Run episodes with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        total_steps = config.num_episodes * config.max_steps_per_episode
        task = progress.add_task(f"Running {config.num_episodes} episodes...", total=total_steps)
        
        # Run episodes concurrently
        tasks = []
        for i in range(config.num_episodes):
            episode_id = f"qwen_{i}_{uuid.uuid4().hex[:8]}"
            task_coro = run_episode(episode_id, config, lambda: progress.update(task, advance=1))
            tasks.append(task_coro)
        
        # Limit concurrency to avoid overwhelming the services
        sem = asyncio.Semaphore(3)
        async def run_with_semaphore(coro):
            async with sem:
                return await coro
        
        results = await asyncio.gather(*[run_with_semaphore(t) for t in tasks], return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                console.print(f"[red]Episode {i} failed: {result}[/red]")
            else:
                all_stats.append(result)
    
    # Display results
    console.print()
    
    if all_stats:
        # Show results table
        table = create_results_table(all_stats)
        console.print(table)
        console.print()
        
        # Show summary
        summary = create_summary_panel(all_stats, config)
        console.print(summary)
        
        # Save results if requested
        if config.save_results:
            output_file = config.output_file or f"qwen_rollouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            results_data = {
                "config": config.dict(),
                "timestamp": datetime.now().isoformat(),
                "episodes": [
                    {
                        "episode_id": s.episode_id,
                        "steps": s.steps,
                        "total_reward": s.total_reward,
                        "achievements": s.achievements,
                        "resources_collected": dict(s.resources_collected),
                        "actions_taken": dict(s.actions_taken),
                        "final_health": s.final_health,
                        "final_hunger": s.final_hunger,
                        "final_thirst": s.final_thirst,
                        "duration": s.duration(),
                        "avg_response_time": s.avg_response_time(),
                        "termination_reason": s.termination_reason
                    }
                    for s in all_stats
                ]
            }
            
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2)
            
            console.print(f"\n[green]Results saved to: {output_file}[/green]")
    else:
        console.print("[red]No successful episodes completed![/red]")


if __name__ == "__main__":
    asyncio.run(main())