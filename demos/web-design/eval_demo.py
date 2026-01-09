#!/usr/bin/env python3
"""Quick evaluation of web design image generation - just test baseline and report metrics."""

import asyncio
import os
import time
from pathlib import Path

import httpx

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError as e:
    raise ImportError(
        "Failed to import synth_ai. Run `uv sync` or `pip install -e .` first."
    ) from e

# Load .env
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

SYNTH_API_KEY = os.getenv("SYNTH_API_KEY")
ENVIRONMENT_API_KEY = os.getenv("ENVIRONMENT_API_KEY")
if not SYNTH_API_KEY:
    raise RuntimeError("SYNTH_API_KEY not set")
if not ENVIRONMENT_API_KEY:
    raise RuntimeError("ENVIRONMENT_API_KEY not set")

# Baseline prompt to test
BASELINE_STYLE_PROMPT = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""


async def main():
    print("=" * 80)
    print("WEB DESIGN EVAL - BASELINE TEST")
    print("=" * 80)
    
    # Import from run_demo
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_demo import create_web_design_local_api
    sys.path.pop(0)
    
    print("Starting local API...")
    local_api = create_web_design_local_api(BASELINE_STYLE_PROMPT)
    port = 8103
    
    # Start server (returns a thread)
    server_thread = run_server_background(local_api, port=port)
    
    # Wait for server to be ready
    local_api_url = f"http://localhost:{port}"
    auth_headers = {
        "X-API-Key": ENVIRONMENT_API_KEY,
        "Content-Type": "application/json",
    }
    
    for i in range(30):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{local_api_url}/health",
                    headers=auth_headers
                )
                if response.status_code == 200:
                    break
        except Exception:
            if i == 0:  # Only print on first attempt
                pass
        await asyncio.sleep(1)
    else:
        raise RuntimeError("Local API failed to start after 30s")
    
    print(f"✓ Local API running on port {port}")
    print()
    
    # Submit eval job to backend
    test_seeds = [0, 1, 2]
    total_start = time.time()
    
    print(f"Submitting eval job for {len(test_seeds)} seeds...")
    
    backend_url = "http://localhost:8000"
    
    eval_payload = {
        "job_type": "eval",
        "task_app_url": local_api_url,
        "task_app_api_key": ENVIRONMENT_API_KEY,
        "seeds": test_seeds,
        "policy_model": "gemini-2.5-flash-image",
        "policy_provider": "google",
        "inference_mode": "synth_hosted",
        "verifier_enabled": True,
        "verifier_backend_base": backend_url,
        "verifier_backend_provider": "google",
        "verifier_backend_model": "gemini-2.5-flash",
        "verifier_graph_id": "zero_shot_verifier_rubric_single",
        "verifier_backend_outcome_enabled": True,
        "verifier_backend_event_enabled": False,
        "verifier_weight_env": 0.0,
        "verifier_weight_event": 0.0,
        "verifier_weight_outcome": 1.0,
        "verifier_timeout": 240.0,
    }
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        # Submit job
        response = await client.post(
            f"{backend_url}/api/eval/jobs",
            json=eval_payload,
            headers={
                "Authorization": f"Bearer {SYNTH_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        
        if response.status_code != 200:
            print(f"✗ Failed to submit job: {response.status_code}")
            print(response.text[:500])
            return
        
        job_data = response.json()
        job_id = job_data.get("job_id")
        print(f"✓ Job submitted: {job_id}")
        print()
        
        # Poll for completion
        print("Polling for completion...")
        for _ in range(120):  # 10 min timeout
            await asyncio.sleep(5)
            
            status_response = await client.get(
                f"{backend_url}/api/eval/jobs/{job_id}",
                headers={"Authorization": f"Bearer {SYNTH_API_KEY}"},
            )
            
            if status_response.status_code != 200:
                print(f"✗ Failed to get status: {status_response.status_code}")
                break
            
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            
            elapsed = time.time() - total_start
            mins, secs = divmod(int(elapsed), 60)
            
            if status in ["succeeded", "failed", "cancelled"]:
                print(f"\r[{mins:02d}:{secs:02d}] {status}")
                break
            else:
                results_info = status_data.get("results", {})
                completed = results_info.get("completed", 0)
                total = results_info.get("total", len(test_seeds))
                print(f"\r[{mins:02d}:{secs:02d}] {status} | {completed}/{total} completed", end="", flush=True)
        
        total_time = time.time() - total_start
        
        # Get final results
        results_response = await client.get(
            f"{backend_url}/api/eval/jobs/{job_id}/results",
            headers={"Authorization": f"Bearer {SYNTH_API_KEY}"},
        )
        
        if results_response.status_code != 200:
            print(f"\n✗ Failed to get results: {results_response.status_code}")
            return
        
        results_data = results_response.json()
    
    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    summary = results_data.get("summary", {})
    results_list = results_data.get("results", [])
    
    print(f"Status: {status_data.get('status', 'unknown')}")
    print(f"Mean Score: {summary.get('mean_score', 'N/A')}")
    print(f"Total Cost: ${summary.get('total_cost_usd', 0.0):.4f}")
    print(f"Total Time: {total_time:.1f}s")
    print()
    
    if results_list:
        print("Per-Example Results:")
        for r in results_list:
            seed = r.get("seed", "?")
            score = r.get("score", "N/A")
            cost = r.get("cost_usd", 0.0)
            print(f"  Seed {seed}: score={score} cost=${cost:.4f}")
    
    print()
    print("=" * 80)
    
    # Note: Server thread will clean up automatically


if __name__ == "__main__":
    asyncio.run(main())

