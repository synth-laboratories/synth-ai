#!/usr/bin/env python3
"""Evaluate GPT-5-nano policy on Pokemon Red Pallet Town progression.

Runs 10 parallel rollouts and reports rewards in a table.
"""
import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from tabulate import tabulate


# Load environment variables
load_dotenv()

# Configuration
TASK_APP_URL = "http://127.0.0.1:8913"
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 10  # 10 policy calls per episode (each may return 5-10 actions)
MODEL = "gpt-5-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def run_single_rollout(
    client: httpx.AsyncClient,
    episode_id: int,
) -> dict[str, Any]:
    """Run a single rollout with policy-driven actions."""
    
    # Build rollout request with policy actions
    # Send "policy" for each step to trigger LLM inference
    rollout_request = {
        "run_id": f"eval_episode_{episode_id:03d}",
        "env": {"instance_id": f"pallet_town_{episode_id:03d}"},
        "ops": ["policy"] * MAX_STEPS_PER_EPISODE,  # Let policy drive all actions
        "policy": {
            "type": "llm",
            "model": MODEL,
            "config": {
                "model": MODEL,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        },
    }
    
    try:
        response = await client.post(
            f"{TASK_APP_URL}/rollout",
            json=rollout_request,
            timeout=300.0,  # 5 minutes per rollout
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract metrics
        trajectories = result.get("trajectories", [])
        if not trajectories:
            return {
                "episode_id": episode_id,
                "status": "error",
                "error": "No trajectories returned",
            }
        
        trajectory = trajectories[0]
        steps = trajectory.get("steps", [])
        num_steps = len(steps) - 1  # Subtract initial observation
        
        # Get metrics
        metrics = result.get("metrics", {})
        total_reward = metrics.get("episode_returns", [0.0])[0]
        outcome_score = metrics.get("outcome_score", 0.0)
        details = metrics.get("details", {})
        
        # Extract milestone info
        reward_components = details.get("reward_components", [])
        milestone_events = details.get("milestone_events", [])
        final_map = details.get("final_map", -1)
        party_count = details.get("party_count", 0)
        badges = details.get("badges", 0)
        
        return {
            "episode_id": episode_id,
            "status": "success",
            "total_reward": total_reward,
            "outcome_score": outcome_score,
            "num_steps": num_steps,
            "final_map": final_map,
            "party_count": party_count,
            "badges": badges,
            "num_milestones": len(milestone_events),
            "reward_components": reward_components,
            "milestone_events": milestone_events,
        }
        
    except httpx.TimeoutException:
        return {
            "episode_id": episode_id,
            "status": "timeout",
            "error": "Rollout timed out after 5 minutes",
        }
    except Exception as e:
        return {
            "episode_id": episode_id,
            "status": "error",
            "error": str(e),
        }


async def main():
    print("=" * 80)
    print("POKÉMON RED - POLICY EVALUATION")
    print("=" * 80)
    print()
    print(f"Task: Pallet Town Progression")
    print(f"Policy: {MODEL}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Server: {TASK_APP_URL}")
    print()
    
    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{TASK_APP_URL}/health", timeout=5.0)
            response.raise_for_status()
            print("✓ Server is healthy")
        except Exception as e:
            print(f"❌ Server not responding: {e}")
            print(f"   Start it with: uv run -m synth_ai task-app deploy --runtime uvicorn pokemon_red --port 8913")
            return
        
        # Check API key
        if not OPENAI_API_KEY:
            print("❌ OPENAI_API_KEY not found in environment")
            print("   Make sure .env file contains OPENAI_API_KEY")
            return
        print(f"✓ API key loaded (sk_env...{OPENAI_API_KEY[-4:]})")
        print()
        
        # Run rollouts in parallel
        print(f"🎮 Running {NUM_EPISODES} episodes in parallel...")
        print()
        
        tasks = [
            run_single_rollout(client, episode_id)
            for episode_id in range(1, NUM_EPISODES + 1)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Separate successful and failed results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]
        
        # Print summary table
        print()
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()
        
        if successful:
            table_data = []
            for r in successful:
                table_data.append([
                    r["episode_id"],
                    f"{r['total_reward']:.1f}",
                    r["num_steps"],
                    f"Map{r['final_map']}",
                    r["party_count"],
                    r["badges"],
                    r["num_milestones"],
                    f"{r['outcome_score']:.3f}",
                ])
            
            headers = [
                "Episode",
                "Reward",
                "Steps",
                "Final Map",
                "Party",
                "Badges",
                "Milestones",
                "Outcome Score",
            ]
            
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
            
            # Print statistics
            rewards = [r["total_reward"] for r in successful]
            steps = [r["num_steps"] for r in successful]
            outcome_scores = [r["outcome_score"] for r in successful]
            
            print("Statistics:")
            print(f"  Mean reward: {sum(rewards) / len(rewards):.2f}")
            print(f"  Max reward: {max(rewards):.2f}")
            print(f"  Min reward: {min(rewards):.2f}")
            print(f"  Mean steps: {sum(steps) / len(steps):.1f}")
            print(f"  Mean outcome score: {sum(outcome_scores) / len(outcome_scores):.4f}")
            print()
            
            # Print milestone breakdown for best episode
            best_episode = max(successful, key=lambda r: r["total_reward"])
            print(f"Best Episode (#{best_episode['episode_id']}):")
            print(f"  Total reward: {best_episode['total_reward']:.1f}")
            print(f"  Steps taken: {best_episode['num_steps']}")
            print(f"  Milestones achieved:")
            for milestone in best_episode["milestone_events"]:
                print(f"    Step {milestone['step']}: {milestone['description']} (+{milestone['reward']:.1f})")
            print()
        
        if failed:
            print(f"Failed episodes: {len(failed)}")
            for r in failed:
                print(f"  Episode {r['episode_id']}: {r.get('error', 'Unknown error')}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
