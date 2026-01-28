#!/usr/bin/env python3
"""
Run Crafter ReAct MIPRO Online Optimization with Ontology Backend

Simple online MIPRO demo for Crafter:
1. Create MIPRO job -> get proxy URL
2. Run rollouts using proxy URL for LLM calls
3. Report rewards to backend
4. Print learned ontology at end

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai && \
    export SYNTH_API_KEY=sk_live_... && \
    RUST_BACKEND_URL=https://api.usesynth.ai uv run python demos/mipro_crafter_ontology/run_online_demo.py \
        --rollouts 20 --train-size 10 --min-proposal-rollouts 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos" / "gepa_crafter_vlm"))

from crafter_logic import (
    ACTION_STRING_TO_INT,
    CRAFTER_ALLOWED_ACTIONS,
    CrafterEnvironmentWrapper,
    CrafterScorer,
    normalize_action_name,
)

from synth_ai.core.utils.urls import BACKEND_URL_BASE


# =============================================================================
# Ontology Client - Tracks learned concepts in-memory
# =============================================================================

@dataclass
class OntologyClient:
    """Tracks learned concepts from rollouts."""
    
    org_id: str = "crafter_mipro_demo"
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    properties: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_action_outcome(self, action: str, reward: float, context: Dict[str, Any]) -> None:
        """Record an action outcome."""
        action_name = f"action_{action}"
        if action_name not in self.nodes:
            self.nodes[action_name] = {
                "name": action_name,
                "type": "action",
                "description": f"Crafter action: {action}",
                "relevance": 0.5,
            }
        
        if action_name not in self.properties:
            self.properties[action_name] = []
        
        # Find or create stats property
        stats_prop = next(
            (p for p in self.properties[action_name] if p["predicate"] == "avg_reward"),
            None,
        )
        if stats_prop:
            old_avg = float(stats_prop["value"].split(":")[0])
            old_count = int(stats_prop["value"].split(":")[1])
            new_count = old_count + 1
            new_avg = (old_avg * old_count + reward) / new_count
            stats_prop["value"] = f"{new_avg:.3f}:{new_count}"
            stats_prop["confidence"] = min(new_count / 10.0, 1.0)
        else:
            self.properties[action_name].append({
                "predicate": "avg_reward",
                "value": f"{reward:.3f}:1",
                "confidence": 0.1,
            })
        
        # Track achievements
        for achievement in context.get("achievements", []):
            ach_name = f"achievement_{achievement}"
            if ach_name not in self.nodes:
                self.nodes[ach_name] = {
                    "name": ach_name,
                    "type": "achievement",
                    "description": f"Crafter achievement: {achievement}",
                    "relevance": 0.8,
                }
                self.relationships.append({
                    "from": action_name,
                    "to": ach_name,
                    "relation_type": "can_trigger",
                    "value": "observed",
                })
    
    def record_candidate_performance(self, candidate_id: str, avg_reward: float, count: int) -> None:
        """Record prompt candidate performance."""
        name = f"candidate_{candidate_id[:16]}"
        self.nodes[name] = {
            "name": name,
            "type": "prompt_candidate",
            "description": f"MIPRO candidate {candidate_id[:16]}",
            "relevance": avg_reward,
        }
        self.properties[name] = [
            {"predicate": "avg_reward", "value": f"{avg_reward:.3f}", "confidence": min(count/20, 1.0)},
            {"predicate": "rollout_count", "value": str(count), "confidence": 1.0},
        ]
    
    def print_ontology(self) -> None:
        """Print the learned ontology."""
        print("\n" + "=" * 70)
        print("LEARNED ONTOLOGY")
        print("=" * 70)
        
        # Group by type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for name, node in self.nodes.items():
            t = node.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(node)
        
        # Print actions with stats
        if "action" in by_type:
            print("\n--- Actions (sorted by avg reward) ---")
            actions = sorted(by_type["action"], key=lambda n: self._get_avg_reward(n["name"]), reverse=True)
            for action in actions:
                name = action["name"]
                avg, count = self._parse_stats(name)
                bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
                print(f"  {name[7:]:20s} [{bar}] {avg:.1%} (n={count})")
        
        # Print achievements
        if "achievement" in by_type:
            print("\n--- Discovered Achievements ---")
            for ach in by_type["achievement"]:
                print(f"  - {ach['name'][12:]}")
        
        # Print candidates
        if "prompt_candidate" in by_type:
            print("\n--- Prompt Candidates ---")
            candidates = sorted(by_type["prompt_candidate"], key=lambda n: n.get("relevance", 0), reverse=True)
            for cand in candidates[:5]:
                props = self.properties.get(cand["name"], [])
                avg_r = next((p["value"] for p in props if p["predicate"] == "avg_reward"), "?")
                count = next((p["value"] for p in props if p["predicate"] == "rollout_count"), "?")
                print(f"  {cand['name']}: reward={avg_r}, rollouts={count}")
        
        print("\n" + "=" * 70)
        print(f"Total: {len(self.nodes)} nodes, {sum(len(v) for v in self.properties.values())} properties")
        print("=" * 70)
    
    def _get_avg_reward(self, name: str) -> float:
        props = self.properties.get(name, [])
        stat = next((p for p in props if p["predicate"] == "avg_reward"), None)
        if stat:
            return float(stat["value"].split(":")[0])
        return 0.0
    
    def _parse_stats(self, name: str) -> tuple[float, int]:
        props = self.properties.get(name, [])
        stat = next((p for p in props if p["predicate"] == "avg_reward"), None)
        if stat:
            parts = stat["value"].split(":")
            return float(parts[0]), int(parts[1])
        return 0.0, 0


# =============================================================================
# Backend API Helpers
# =============================================================================

def resolve_backend_url() -> str:
    """Resolve backend URL from environment."""
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        url = (os.environ.get(env_var) or "").strip()
        if url:
            return url.rstrip("/")
    return BACKEND_URL_BASE.rstrip("/")


async def create_mipro_job(backend_url: str, api_key: str, config: Dict[str, Any]) -> str:
    """Create MIPRO online job."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{backend_url}/api/prompt-learning/online/jobs",
            json={"algorithm": "mipro", "config_body": config},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create job: {response.status_code} - {response.text}")
        return response.json().get("job_id")


async def get_job_metadata(backend_url: str, api_key: str, job_id: str) -> Dict[str, Any]:
    """Get job metadata including proxy URL."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{backend_url}/api/prompt-learning/online/jobs/{job_id}",
            params={"include_metadata": True},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json().get("metadata", {})


async def get_system_state(backend_url: str, api_key: str, system_id: str) -> Dict[str, Any]:
    """Get MIPRO system state."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return response.json()


async def push_status(backend_url: str, api_key: str, system_id: str, rollout_id: str, reward: float) -> str:
    """Push reward and done status to backend. Returns candidate_id."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Push reward
        response = await client.post(
            f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
            json={"rollout_id": rollout_id, "status": "reward", "reward": reward},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        candidate_id = response.json().get("candidate_id", "unknown")
        
        # Push done
        await client.post(
            f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
            json={"rollout_id": rollout_id, "status": "done"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return candidate_id


# =============================================================================
# LLM Call via Proxy
# =============================================================================

CRAFTER_TOOL = {
    "type": "function",
    "function": {
        "name": "crafter_interact",
        "description": "Execute actions in Crafter",
        "parameters": {
            "type": "object",
            "properties": {
                "actions_list": {
                    "type": "array",
                    "items": {"type": "string", "enum": CRAFTER_ALLOWED_ACTIONS},
                    "minItems": 2,
                    "maxItems": 5,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["actions_list"],
        },
    },
}


async def call_llm_via_proxy(
    proxy_url: str,
    rollout_id: str,
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
) -> tuple[List[str], Dict[str, Any]]:
    """Call LLM via MIPRO proxy. Returns (actions_list, raw_response)."""
    url = f"{proxy_url}/{rollout_id}/chat/completions"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": messages,
                "tools": [CRAFTER_TOOL],
                "tool_choice": "required",
                "max_tokens": 512,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM call failed: {response.status_code} - {response.text[:200]}")
        
        data = response.json()
    
    # Parse tool call
    try:
        tool_call = data["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tool_call["function"]["arguments"])
        actions = args.get("actions_list", ["noop"])
    except (KeyError, IndexError, json.JSONDecodeError):
        actions = ["noop"]
    
    return actions, data


# =============================================================================
# Rollout Execution
# =============================================================================

def build_system_prompt() -> str:
    """Build the baseline system prompt."""
    allowed = ", ".join(CRAFTER_ALLOWED_ACTIONS)
    return (
        "You are an agent playing Crafter, a survival crafting game. "
        "Your goal is to survive and unlock achievements by exploring, crafting, and building. "
        "You see the game through images. Analyze each image to understand your surroundings, "
        "inventory, health, and resources. Use the crafter_interact tool to execute actions. "
        "Key mechanics: use 'do' only when adjacent to a resource (tree, stone, cow, plant). "
        "Craft progression: wood -> table -> wood_pickaxe -> stone -> stone_pickaxe -> iron tools. "
        f"Available actions: {allowed}. "
        "Return 2-5 actions per decision."
    )


def build_messages(system_prompt: str, observation: Dict[str, Any], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build messages for LLM call."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    
    # Add current observation as image
    image_url = observation.get("observation_image_data_url")
    if image_url:
        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
    
    return messages


async def run_single_rollout(
    *,
    proxy_url: str,
    rollout_id: str,
    seed: int,
    model: str,
    api_key: str,
    system_prompt: str,
    max_steps: int = 100,
    max_turns: int = 25,
) -> tuple[float, Dict[str, Any]]:
    """Run a single Crafter rollout. Returns (reward, details)."""
    env = CrafterEnvironmentWrapper(seed=seed, max_steps=max_steps)
    observation = await env.reset()
    
    history: List[Dict[str, Any]] = []
    episode_rewards: List[float] = []
    action_sequence: List[str] = []
    
    for turn in range(max_turns):
        # Build messages: system + history + current observation
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        
        # Add current observation image
        image_url = observation.get("observation_image_data_url")
        if image_url:
            messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}]
            })
        
        try:
            actions, response_data = await call_llm_via_proxy(
                proxy_url, rollout_id, messages, model, api_key
            )
        except Exception as e:
            print(f"    LLM error: {e}")
            actions = ["noop"]
            response_data = {}
        
        # Extract tool_calls from response (with real IDs)
        tool_calls = []
        try:
            tool_calls = response_data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        except (KeyError, IndexError):
            pass
        
        # Execute actions and collect results
        next_observation = observation
        tool_responses: List[Dict[str, Any]] = []
        
        if tool_calls:
            for tc in tool_calls:
                tool_call_id = tc.get("id") or tc.get("tool_call_id")
                
                # Parse actions from this tool call
                actions_for_tc = []
                try:
                    args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                    raw_actions = args.get("actions_list", [])
                    actions_for_tc = [str(a) for a in raw_actions if str(a).strip()][:5]
                except Exception:
                    pass
                if not actions_for_tc:
                    actions_for_tc = ["noop"]
                
                # Execute actions
                action_results = []
                for action_str in actions_for_tc:
                    normalized = normalize_action_name(action_str) or "noop"
                    action_sequence.append(normalized)
                    action_int = ACTION_STRING_TO_INT.get(normalized, 0)
                    next_observation = await env.step(action_int)
                    reward = next_observation.get("reward", 0.0)
                    episode_rewards.append(float(reward))
                    action_results.append({
                        "action": normalized,
                        "reward": reward,
                        "terminated": next_observation.get("terminated"),
                        "truncated": next_observation.get("truncated"),
                    })
                    if next_observation.get("terminated") or next_observation.get("truncated"):
                        break
                
                if tool_call_id:
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "actions": [r["action"] for r in action_results],
                        "results": action_results,
                    })
                
                if next_observation.get("terminated") or next_observation.get("truncated"):
                    break
        else:
            # No tool calls - just step with noop
            next_observation = await env.step(0)
            episode_rewards.append(float(next_observation.get("reward", 0.0)))
            action_sequence.append("noop")
        
        # Update history with assistant message (only if tool_calls present)
        if tool_calls:
            history.append({
                "role": "assistant",
                "content": response_data.get("choices", [{}])[0].get("message", {}).get("content") or "",
                "tool_calls": tool_calls,
            })
            # Add tool responses
            for tr in tool_responses:
                history.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": json.dumps({
                        "actions": tr["actions"],
                        "results": tr["results"],
                        "terminated": next_observation.get("terminated"),
                        "truncated": next_observation.get("truncated"),
                    }),
                })
        
        observation = next_observation
        if observation.get("terminated") or observation.get("truncated"):
            break
    
    # Score episode
    outcome_reward, details = CrafterScorer.score_episode(observation, len(episode_rewards), max_steps)
    details["action_sequence"] = action_sequence
    details["achievements"] = observation.get("achievements", [])
    
    return outcome_reward, details


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(description="Crafter MIPRO Online with Ontology")
    parser.add_argument("--rollouts", type=int, default=20, help="Number of rollouts")
    parser.add_argument("--train-size", type=int, default=10, help="Training seeds")
    parser.add_argument("--val-size", type=int, default=3, help="Validation seeds")
    parser.add_argument("--min-proposal-rollouts", type=int, default=10, help="Min rollouts before proposals")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model for inference")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--max-turns", type=int, default=25, help="Max turns per rollout")
    args = parser.parse_args()

    backend_url = resolve_backend_url()
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY required")
    
    print(f"Backend: {backend_url}")
    print(f"API Key: {api_key[:20]}...")
    print(f"Model: {args.model}")

    # Build config
    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))
    system_prompt = build_system_prompt()
    
    config = {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "crafter_react",
            "task_app_url": "http://unused",  # Not used for online
            "initial_prompt": {
                "id": "crafter_react",
                "name": "Crafter ReAct",
                "messages": [{"role": "system", "order": 0, "pattern": system_prompt}],
                "wildcards": {},
            },
            "policy": {
                "model": args.model,
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 512,
            },
            "mipro": {
                "mode": "online",
                "bootstrap_train_seeds": train_seeds,
                "val_seeds": val_seeds,
                "online_pool": train_seeds,
                "online_proposer_mode": "inline",
                "online_proposer_min_rollouts": args.min_proposal_rollouts,
                "online_rollouts_per_candidate": 5,
                "proposer": {
                    "mode": "instruction_only",
                    "model": "gpt-4.1-mini",
                    "provider": "openai",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "generate_at_iterations": [0, 1, 2],
                    "instructions_per_batch": 2,
                },
            },
        },
    }

    # Create job
    print("\nCreating MIPRO online job...")
    job_id = await create_mipro_job(backend_url, api_key, config)
    print(f"Job ID: {job_id}")
    
    metadata = await get_job_metadata(backend_url, api_key, job_id)
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")
    
    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing system_id or proxy_url: {metadata}")
    
    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    # Initialize ontology
    ontology = OntologyClient()

    # Run rollouts
    print(f"\n{'='*60}")
    print(f"Running {args.rollouts} rollouts")
    print(f"{'='*60}")

    results = []
    total_reward = 0.0
    candidate_stats: Dict[str, Dict[str, Any]] = {}
    start_time = time.time()

    for i in range(args.rollouts):
        seed = i % len(train_seeds)
        rollout_id = f"crafter_{i}_{uuid.uuid4().hex[:6]}"
        
        try:
            reward, details = await run_single_rollout(
                proxy_url=proxy_url,
                rollout_id=rollout_id,
                seed=seed,
                model=args.model,
                api_key=api_key,
                system_prompt=system_prompt,
                max_steps=args.max_steps,
                max_turns=args.max_turns,
            )
            
            total_reward += reward
            results.append({"rollout_id": rollout_id, "seed": seed, "reward": reward})
            
            # Update ontology with action outcomes
            for action in details.get("action_sequence", []):
                ontology.record_action_outcome(action, reward / max(len(details.get("action_sequence", [1])), 1), details)
            
            # Push status to backend
            candidate_id = await push_status(backend_url, api_key, system_id, rollout_id, reward)
            
            # Track candidate stats
            if candidate_id not in candidate_stats:
                candidate_stats[candidate_id] = {"count": 0, "total_reward": 0.0}
            candidate_stats[candidate_id]["count"] += 1
            candidate_stats[candidate_id]["total_reward"] += reward
            
            if (i + 1) % 5 == 0:
                avg = total_reward / (i + 1)
                print(f"  [{i+1}/{args.rollouts}] Avg reward: {avg:.3f} | Candidate: {candidate_id[:20]}...")
                
        except Exception as e:
            print(f"  Error on rollout {i}: {e}")
            results.append({"rollout_id": rollout_id, "seed": seed, "error": str(e)})

    elapsed = time.time() - start_time

    # Get final state
    print("\nFetching final system state...")
    state = await get_system_state(backend_url, api_key, system_id)
    
    # Update ontology with candidate stats
    for cid, stats in candidate_stats.items():
        avg = stats["total_reward"] / max(stats["count"], 1)
        ontology.record_candidate_performance(cid, avg, stats["count"])

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Rollouts: {len(results)}")
    print(f"Avg reward: {total_reward / max(len(results), 1):.3f}")
    print(f"Candidates explored: {len(state.get('candidates', {}))}")
    print(f"Time: {elapsed:.1f}s ({len(results)/elapsed:.2f} rollouts/sec)")
    
    print("\nCandidate Performance:")
    for cid, stats in sorted(candidate_stats.items(), key=lambda x: x[1]["total_reward"]/max(x[1]["count"],1), reverse=True):
        avg = stats["total_reward"] / max(stats["count"], 1)
        print(f"  {cid[:30]}...: {stats['count']} rollouts, avg={avg:.3f}")

    # Print ontology
    ontology.print_ontology()

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"crafter_mipro_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "job_id": job_id,
            "system_id": system_id,
            "elapsed": elapsed,
            "results": results,
            "candidate_stats": candidate_stats,
            "ontology": {"nodes": ontology.nodes, "properties": ontology.properties},
        }, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
