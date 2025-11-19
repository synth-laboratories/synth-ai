#!/usr/bin/env python3
"""Quick test to measure rollout timing and get baseline scores for agentic task apps."""

import asyncio
import time
import os
import sys
from pathlib import Path
from typing import Dict, List

import httpx
from dotenv import load_dotenv

# Load .env from synth-ai root
synth_ai_root = Path(__file__).resolve().parents[5]
env_file = synth_ai_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # Fallback to current directory

TASK_APPS = {
    "crafter": {
        "url": "http://127.0.0.1:8116",
        "seeds": list(range(0, 20)),  # 20 samples
    },
    "sokoban": {
        "url": "http://127.0.0.1:8117",
        "seeds": list(range(0, 20)),  # 20 samples
    },
    "verilog": {
        "url": "http://127.0.0.1:8118",
        "seeds": list(range(0, 20)),  # 20 samples
    },
}

API_KEY = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY") or "sk_env_30c78a78"
# Get API key from environment (from synth-ai/.env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. LLM calls will fail.", flush=True)
    print(f"Looking for .env at: {env_file}", flush=True)

POLICY_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "provider": "groq",
    "temperature": 0.0,
    "max_completion_tokens": 512,
    "inference_url": "https://api.groq.com/openai/v1",  # Explicit URL
    "api_key": GROQ_API_KEY,  # Pass API key directly
    "max_steps": 15,  # Multi-step rollouts
}


async def run_rollout(task_app_name: str, task_app_url: str, seed: int) -> Dict:
    """Run a single rollout and return timing + score."""
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    
    payload = {
        "run_id": f"test_{task_app_name}_{seed}",
        "env": {
            "env_name": task_app_name,
            "seed": seed,
            "config": {},
        },
        "policy": {
            "policy_id": f"test_{seed}",
            "config": POLICY_CONFIG,
        },
        "ops": [],  # Empty ops list - task app will execute LLM call internally
        "mode": "eval",  # Required: RL or EVAL mode
    }
    
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{task_app_url}/rollout",
                json=payload,
                headers=headers,
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get("metrics", {})
                score = metrics.get("mean_return", 0.0)
                num_steps = metrics.get("num_steps", 0)
                
                return {
                    "success": True,
                    "elapsed": elapsed,
                    "score": score,
                    "num_steps": num_steps,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "elapsed": elapsed,
                    "score": None,
                    "num_steps": 0,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "elapsed": elapsed,
                "score": None,
                "num_steps": 0,
                "error": str(e),
            }


async def test_task_app(task_app_name: str, config: Dict) -> Dict:
    """Test a task app with multiple rollouts."""
    print(f"\n{'='*80}")
    print(f"Testing {task_app_name.upper()}")
    print(f"{'='*80}")
    print(f"URL: {config['url']}")
    print(f"Seeds: {config['seeds']}")
    print()
    
    results = []
    for seed in config['seeds']:
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = await run_rollout(task_app_name, config['url'], seed)
        results.append(result)
        
        if result['success']:
            print(f"✅ {result['elapsed']:.2f}s | Score: {result['score']:.3f} | Steps: {result['num_steps']}")
        else:
            print(f"❌ {result['elapsed']:.2f}s | Error: {result['error']}")
    
    # Calculate averages
    successful = [r for r in results if r['success']]
    if successful:
        avg_time = sum(r['elapsed'] for r in successful) / len(successful)
        avg_score = sum(r['score'] for r in successful) / len(successful)
        avg_steps = sum(r['num_steps'] for r in successful) / len(successful)
        
        print()
        print(f"  Summary ({len(successful)}/{len(results)} successful):")
        print(f"    Average time: {avg_time:.2f}s")
        print(f"    Average score: {avg_score:.3f}")
        print(f"    Average steps: {avg_steps:.1f}")
        
        return {
            "task_app": task_app_name,
            "success_rate": len(successful) / len(results),
            "avg_time": avg_time,
            "avg_score": avg_score,
            "avg_steps": avg_steps,
            "results": results,
        }
    else:
        print()
        print(f"  ❌ All rollouts failed!")
        return {
            "task_app": task_app_name,
            "success_rate": 0.0,
            "avg_time": None,
            "avg_score": None,
            "avg_steps": None,
            "results": results,
        }


async def main():
    print("="*80)
    print("ROLLOUT TIMING & BASELINE SCORES TEST")
    print("="*80)
    print(f"Policy: {POLICY_CONFIG['model']} ({POLICY_CONFIG['provider']})")
    print(f"API Key: {API_KEY[:15]}...")
    print(f"Sample size: 20 rollouts per task app")
    print()
    
    all_results = []
    
    for task_app_name, config in TASK_APPS.items():
        result = await test_task_app(task_app_name, config)
        all_results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Task App':<15} {'Success':<10} {'Avg Time':<12} {'Avg Score':<12} {'Avg Steps':<12}")
    print("-"*80)
    
    for result in all_results:
        if result['avg_time'] is not None:
            print(
                f"{result['task_app']:<15} "
                f"{result['success_rate']*100:>5.1f}%    "
                f"{result['avg_time']:>6.2f}s     "
                f"{result['avg_score']:>6.3f}      "
                f"{result['avg_steps']:>6.1f}"
            )
        else:
            print(f"{result['task_app']:<15} {'FAILED':<10}")
    
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)

