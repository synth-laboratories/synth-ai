#!/usr/bin/env python3
"""
In-Process GEPA Demo
=====================

This script runs GEPA optimization with a task app started entirely 
in-process - no separate terminals or manual process management needed!

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai/examples/blog_posts/langprobe/task_specific/banking77
    source ../../../../.env
    uv run python run_gepa_in_process.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - cloudflared binary (will auto-install if missing)
    - Dev backend running (default: https://synth-backend-dev-docker.onrender.com)
    
Configuration:
    Default: Uses dev backend (synth-backend-dev-docker.onrender.com)
    Override: Set BACKEND_BASE_URL env var to use different backend
    
    The script automatically matches tunnel mode:
    - If BACKEND_BASE_URL is localhost → both backend and task app use localhost (local/local)
    - If BACKEND_BASE_URL is a tunnel URL → both backend and task app use tunnels (tunnel/tunnel)
    
    For local backend:
    - Set BACKEND_BASE_URL=http://localhost:8000
    - Both backend and task app will use localhost (no tunnels)
    
    For tunnel mode:
    - Start backend tunnel first: cd monorepo && bash scripts/run_backend_tunnel.sh
    - Set BACKEND_BASE_URL to the tunnel URL: export BACKEND_BASE_URL=https://backend-local.usesynth.ai
    - Then run this script
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment from repo root
# Script is at: examples/blog_posts/langprobe/task_specific/banking77/run_gepa_in_process.py
# Need to go up 6 levels: banking77 -> task_specific -> langprobe -> blog_posts -> examples -> synth-ai (root)
env_path = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp


async def main():
    """Run GEPA optimization with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process GEPA Demo")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env (for policy model)")
        sys.exit(1)

    # Configuration
    config_path = Path(__file__).parent / "banking77_gepa.toml"

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    # Default to dev backend, allow override via BACKEND_BASE_URL env var
    backend_url = os.getenv("BACKEND_BASE_URL", "https://synth-backend-dev-docker.onrender.com")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    # Determine tunnel mode based on backend URL
    # Rule: Both backend and task app must use same mode (local/local or tunnel/tunnel)
    is_backend_localhost = (
        backend_url.startswith("http://localhost") 
        or backend_url.startswith("http://127.0.0.1")
    )
    
    if is_backend_localhost:
        # Backend is localhost → use local mode for task app (no tunnel)
        # Both backend and task app will use localhost (consistent configuration)
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        use_local_mode = True
        print("ℹ️  Configuration: local/local")
        print("   Backend: localhost:8000")
        print("   Task App: localhost (no tunnel)")
    else:
        # Backend is tunneled → use tunnel mode for task app
        # Both backend and task app will use tunnels (consistent configuration)
        os.environ["SYNTH_TUNNEL_MODE"] = "quick"
        use_local_mode = False
        # Set EXTERNAL_BACKEND_URL so backend knows its public URL for interceptor
        # This is critical: backend needs to know its tunnel URL to tell task apps
        os.environ["EXTERNAL_BACKEND_URL"] = backend_url.rstrip("/")
        print("ℹ️  Configuration: tunnel/tunnel")
        print(f"   Backend tunnel: {backend_url}")
        print(f"   Task app: will create its own tunnel")
        print(f"   Note: Ensure backend is running with tunnel (use run_backend_tunnel.sh)")

    print("Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: Starting in-process...")
    print()

    # Find task app path (banking77 task app)
    task_app_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        / "examples"
        / "task_apps"
        / "banking77"
        / "banking77_task_app.py"
    )

    if not task_app_path.exists():
        print(f"❌ Error: Task app not found: {task_app_path}")
        sys.exit(1)

    # Run GEPA with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=task_app_path,
            port=8114,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            if use_local_mode:
                print("✅ Running in local mode (no tunnel)\n")
            else:
                print("✅ Cloudflare tunnel active\n")
            print("=" * 80)
            print("Running GEPA Optimization")
            print("=" * 80 + "\n")

            # Load and modify config before creating job
            import toml

            config = toml.load(config_path)
            config["prompt_learning"]["task_app_url"] = task_app.url
            
            # Keep original iterations - don't reduce for demo
            if "gepa" in config["prompt_learning"]:
                gepa_config = config["prompt_learning"]["gepa"]
                if "population" in gepa_config:
                    num_generations = gepa_config["population"].get("num_generations", 1)
                    children_per_gen = gepa_config["population"].get("children_per_generation", 5)
                    print(f"📊 Running {num_generations} generations with {children_per_gen} children per generation")
                    print(f"   (Total: {num_generations * children_per_gen} prompt candidates)\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                toml.dump(config, f)
                temp_config_path = f.name

            try:
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    backend_url=backend_url,
                    api_key=api_key,
                    task_app_api_key=task_app_api_key,
                )

                print(f"Task app URL: {task_app.url}")
                print(f"Backend URL: {backend_url}\n")
                print(f"Submitting job...\n")

                try:
                    job_id = job.submit()
                    print(f"✅ Job submitted: {job_id}\n")
                except Exception as e:
                    print(f"\n❌ Error submitting job:")
                    print(f"   Type: {type(e).__name__}")
                    print(f"   Message: {str(e)}")
                    print(f"\n   Full error details:")
                    import traceback
                    traceback.print_exc()
                    raise
            finally:
                # Clean up temp config file
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)

            # Poll for completion
            start_time = time.time()
            last_status = None

            def on_status(status):
                nonlocal last_status
                elapsed = time.time() - start_time
                state = status.get("status", "unknown")

                # Only print if status changed or every 10 seconds
                if state != last_status or int(elapsed) % 10 == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    progress = status.get("progress", {})
                    best_score = status.get("best_score")
                    
                    if progress:
                        completed = progress.get("completed", 0)
                        total = progress.get("total", 0)
                        if total > 0:
                            pct = (completed / total) * 100
                            score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                            print(
                                f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%){score_str}"
                            )
                        else:
                            score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    else:
                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    last_status = state

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: job.poll_until_complete(
                    timeout=3600.0,
                    interval=5.0,
                    on_status=on_status,
                ),
            )

            total_time = time.time() - start_time
            print(f"\n✅ GEPA optimization complete in {total_time:.1f}s\n")

            # Get results - try multiple sources like MIPRO does
            import httpx
            from synth_ai.learning.prompt_learning_client import PromptLearningClient
            from synth_ai.api.train.utils import ensure_api_base

            client = PromptLearningClient(
                ensure_api_base(backend_url),
                api_key,
            )
            
            # First, get job status/metadata directly (like MIPRO does)
            job_status_url = f"{ensure_api_base(backend_url)}/prompt-learning/online/jobs/{job._job_id}"
            async with httpx.AsyncClient() as http_client:
                status_resp = await http_client.get(
                    job_status_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                status_resp.raise_for_status()
                job_data = status_resp.json()
            
            print("=" * 80)
            print("Results")
            print("=" * 80 + "\n")
            
            # Extract from job metadata (primary source)
            metadata = job_data.get("metadata", {}) or {}
            job_metadata = job_data.get("job_metadata", {}) or {}
            combined_metadata = {**metadata, **job_metadata}  # job_metadata takes precedence
            
            # Debug: print what we found
            print(f"🔍 Debug: Job status = {job_data.get('status')}")
            print(f"🔍 Debug: Metadata keys = {list(metadata.keys())}")
            print(f"🔍 Debug: Job_metadata keys = {list(job_metadata.keys())}")
            
            # Try to get attempted_candidates from metadata
            attempted_candidates = combined_metadata.get("attempted_candidates")
            
            # ASSERTION: Show what we got from backend
            print(f"🔍 Debug: attempted_candidates type: {type(attempted_candidates)}")
            if attempted_candidates is not None:
                if isinstance(attempted_candidates, list):
                    print(f"🔍 Debug: attempted_candidates length: {len(attempted_candidates)}")
                    if len(attempted_candidates) > 0:
                        print(f"🔍 Debug: First candidate type: {type(attempted_candidates[0])}")
                        if isinstance(attempted_candidates[0], dict):
                            print(f"🔍 Debug: First candidate keys: {list(attempted_candidates[0].keys())}")
                            # Show sample of first candidate structure
                            first_candidate = attempted_candidates[0]
                            print(f"🔍 Debug: First candidate sample:")
                            for key in list(first_candidate.keys())[:10]:  # First 10 keys
                                value = first_candidate[key]
                                if isinstance(value, (dict, list)):
                                    print(f"  {key}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                                else:
                                    value_str = str(value)[:100] if value else "None"
                                    print(f"  {key}: {value_str}")
                else:
                    print(f"🔍 Debug: attempted_candidates is not a list: {type(attempted_candidates)}")
            
            if attempted_candidates is None:
                # Fallback: try from events via get_prompts
                print("🔍 Debug: attempted_candidates is None, trying get_prompts()...")
                prompt_results = await client.get_prompts(job._job_id)
                attempted_candidates = prompt_results.attempted_candidates
                print(f"🔍 Debug: Got attempted_candidates from events: {len(attempted_candidates) if attempted_candidates else 0}")
                if attempted_candidates and len(attempted_candidates) > 0:
                    print(f"🔍 Debug: First candidate from events type: {type(attempted_candidates[0])}")
                    if isinstance(attempted_candidates[0], dict):
                        print(f"🔍 Debug: First candidate from events keys: {list(attempted_candidates[0].keys())}")
            else:
                print(f"🔍 Debug: Got attempted_candidates from metadata: {len(attempted_candidates) if isinstance(attempted_candidates, list) else 'not a list'}")
            
            # ASSERTION: We should have candidates if job succeeded
            assert attempted_candidates is not None, "attempted_candidates should not be None after fallback"
            assert isinstance(attempted_candidates, list), f"attempted_candidates should be a list, got {type(attempted_candidates)}"
            
            # Best score
            best_score = (
                combined_metadata.get("best_score") 
                or combined_metadata.get("prompt_best_score")
                or job_data.get("prompt_best_score")
            )
            if best_score is not None:
                print(f"Best score: {best_score:.2%}")
            else:
                print("Best score: N/A (job may have failed)")
            
            # Parse and display candidates info
            candidates = attempted_candidates
            if candidates is not None:
                if isinstance(candidates, list):
                    if len(candidates) > 0:
                        # Extract useful stats from candidates (handle both typed and raw)
                        from synth_ai.learning.prompt_learning_types import OptimizedCandidate, AttemptedCandidate
                        
                        accuracies = []
                        for idx, c in enumerate(candidates):
                            # ASSERTION: Show structure of each candidate
                            if idx < 3:  # First 3 only
                                print(f"🔍 Debug Candidate #{idx+1} for accuracy extraction:")
                                print(f"  Type: {type(c)}")
                                if isinstance(c, dict):
                                    print(f"  Keys: {list(c.keys())}")
                                    if "score" in c:
                                        print(f"  score: {c['score']} (type: {type(c['score'])})")
                                    if "accuracy" in c:
                                        print(f"  accuracy (top-level): {c['accuracy']}")
                            
                            if isinstance(c, OptimizedCandidate):
                                assert c.score is not None, f"Candidate #{idx+1}: OptimizedCandidate.score is None"
                                accuracies.append(c.score.accuracy)
                            elif isinstance(c, AttemptedCandidate):
                                accuracies.append(c.accuracy)
                            elif isinstance(c, dict):
                                score = c.get("score")
                                if isinstance(score, dict):
                                    accuracy = score.get("accuracy")
                                    assert accuracy is not None, f"Candidate #{idx+1}: score.accuracy is None. score keys: {list(score.keys())}"
                                    accuracies.append(accuracy)
                                else:
                                    accuracy = c.get("accuracy")
                                    assert accuracy is not None, f"Candidate #{idx+1}: accuracy is None. candidate keys: {list(c.keys())}"
                                    accuracies.append(accuracy)
                            else:
                                raise TypeError(f"Candidate #{idx+1}: Unexpected type {type(c)}")
                        
                        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                        max_accuracy = max(accuracies) if accuracies else 0.0
                        min_accuracy = min(accuracies) if accuracies else 0.0
                        
                        print(f"\nTotal candidates: {len(candidates)}")
                        if accuracies:
                            print(f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})")
                        
                        # Print all resulting prompts
                        print("\n" + "=" * 80)
                        print("All Resulting Prompts")
                        print("=" * 80 + "\n")
                        
                        for idx, candidate in enumerate(candidates, 1):
                            # Handle both typed dataclasses and raw dicts (backward compatibility)
                            from synth_ai.learning.prompt_learning_types import OptimizedCandidate, AttemptedCandidate
                            
                            # Extract stats first
                            accuracy = None
                            prompt_length = None
                            tool_call_rate = None
                            
                            # Debug: print candidate structure for first few
                            if idx <= 3 and isinstance(candidate, dict):
                                print(f"🔍 Debug Candidate #{idx} structure: keys={list(candidate.keys())}")
                                if "score" in candidate:
                                    print(f"  score type: {type(candidate['score'])}, value: {candidate['score']}")
                            
                            if isinstance(candidate, OptimizedCandidate):
                                if candidate.score:
                                    accuracy = candidate.score.accuracy
                                    prompt_length = candidate.score.prompt_length
                                    tool_call_rate = candidate.score.tool_call_rate
                            elif isinstance(candidate, AttemptedCandidate):
                                accuracy = candidate.accuracy
                                prompt_length = candidate.prompt_length
                                tool_call_rate = candidate.tool_call_rate
                            elif isinstance(candidate, dict):
                                # Handle raw dict - check nested score first, then top-level
                                score = candidate.get("score")
                                if isinstance(score, dict):
                                    accuracy = score.get("accuracy")
                                    prompt_length = score.get("prompt_length")
                                    tool_call_rate = score.get("tool_call_rate")
                                # Fallback to top-level fields (for AttemptedCandidate-style flat structure)
                                if accuracy is None:
                                    accuracy = candidate.get("accuracy")
                                if prompt_length is None:
                                    prompt_length = candidate.get("prompt_length")
                                if tool_call_rate is None:
                                    tool_call_rate = candidate.get("tool_call_rate")
                                
                                # Debug: show what we found
                                if idx <= 3:
                                    print(f"  Extracted stats: accuracy={accuracy}, length={prompt_length}, tool_call_rate={tool_call_rate}")
                            
                            # Extract prompt using SDK abstraction (works with both typed and raw)
                            from synth_ai.learning.prompt_extraction import PromptExtractor
                            
                            extracted = PromptExtractor.extract_from_candidate(candidate)
                            
                            # If prompt_length is missing but we have extracted prompt, compute it
                            if prompt_length is None or prompt_length == 0:
                                if extracted and extracted.text:
                                    # Rough token estimate: ~4 chars per token
                                    prompt_length = len(extracted.text) // 4
                                else:
                                    prompt_length = 0
                            
                            # Default to 0.0 if still None
                            accuracy = accuracy if accuracy is not None else 0.0
                            prompt_length = prompt_length if prompt_length is not None else 0
                            tool_call_rate = tool_call_rate if tool_call_rate is not None else 0.0
                            
                            print(f"--- Candidate #{idx} (Accuracy: {accuracy:.2%}, Length: {prompt_length}, Tool Call Rate: {tool_call_rate:.2%}) ---")
                            
                            if extracted:
                                print(extracted.to_formatted_string())
                                print()
                            else:
                                print("(No prompt content available)")
                                # Debug info
                                if isinstance(candidate, OptimizedCandidate):
                                    payload_kind = candidate.payload_kind
                                    obj_keys = list(candidate.object.keys()) if candidate.object else []
                                elif isinstance(candidate, AttemptedCandidate):
                                    payload_kind = candidate.payload_kind
                                    obj_keys = list(candidate.object.keys()) if candidate.object else []
                                else:
                                    obj = candidate.get("object", {})
                                    payload_kind = candidate.get("payload_kind", candidate.get("type", ""))
                                    obj_keys = list(obj.keys()) if isinstance(obj, dict) else []
                                print(f"  Debug: payload_kind={payload_kind}, object_keys={obj_keys}")
                                print()
                            
                            print()
                    else:
                        print(f"\nTotal candidates: 0 (no candidates evaluated)")
                else:
                    print(f"\nTotal candidates: {candidates} (not a list)")
            else:
                print("\nTotal candidates: N/A")
            print()

            # Best prompt
            best_prompt = combined_metadata.get("best_prompt") or job_data.get("best_prompt")
            if best_prompt:
                print("=" * 80)
                print("Best Prompt")
                print("=" * 80)
                # Extract prompt text
                if isinstance(best_prompt, dict):
                    if "prompt_sections" in best_prompt:
                        sections = best_prompt["prompt_sections"]
                        prompt_text = "\n\n".join(
                            [s.get("content", "") for s in sections if s.get("content")]
                        )
                        print(prompt_text)
                    elif "sections" in best_prompt:
                        sections = best_prompt["sections"]
                        prompt_text = "\n\n".join(
                            [s.get("content", "") for s in sections if s.get("content")]
                        )
                        print(prompt_text)
                    else:
                        print(best_prompt)
                else:
                    print(best_prompt)
                print()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✅ In-process GEPA demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

