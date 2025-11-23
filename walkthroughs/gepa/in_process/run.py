#!/usr/bin/env python3
"""
In-Process GEPA Optimization for Banking77
===========================================

This script runs GEPA optimization with a task app started entirely 
in-process - no separate terminals or manual process management needed!

Usage:
    uv run python /Users/joshpurtell/Documents/GitHub/synth-ai/walkthroughs/gepa/in_process/run.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - SYNTH_API_KEY in .env (for backend authentication)
    - ENVIRONMENT_API_KEY in .env (for task app authentication)
    - Backend URL (default: https://agent-learning.onrender.com)
    
Configuration:
    Default: Uses production backend (https://agent-learning.onrender.com)
    Override: Set BACKEND_BASE_URL env var to use different backend
    
    The script automatically matches tunnel mode:
    - If BACKEND_BASE_URL is localhost → both backend and task app use localhost (local/local)
    - If BACKEND_BASE_URL is a tunnel URL → both backend and task app use tunnels (tunnel/tunnel)
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
# Script is at: walkthroughs/gepa/in_process/run.py
# Need to go up 4 levels: in_process -> gepa -> walkthroughs -> synth-ai (root)
repo_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = repo_root / ".env"
load_dotenv(env_path)

# Add repo root to path for imports
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.cli.lib.task_app_env import preflight_env_key
from synth_ai.learning.rl.secrets import mint_environment_api_key
from synth_ai.task import InProcessTaskApp


async def main():
    """Run GEPA optimization with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process GEPA Optimization: Banking77")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env (for policy model)")
        sys.exit(1)
    
    if not os.getenv("SYNTH_API_KEY"):
        print("❌ Error: SYNTH_API_KEY required in .env (for backend authentication)")
        sys.exit(1)

    # Configuration - use config from walkthroughs/gepa/
    config_path = repo_root / "walkthroughs" / "gepa" / "banking77_gepa.toml"

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    # Default to production backend, allow override via BACKEND_BASE_URL env var
    backend_url = os.getenv("BACKEND_BASE_URL", "https://agent-learning.onrender.com")
    api_key = os.getenv("SYNTH_API_KEY")
    
    # Generate and register ENVIRONMENT_API_KEY if not set
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY")
    if not task_app_api_key:
        print("ℹ️  ENVIRONMENT_API_KEY not set, generating and registering with backend...")
        task_app_api_key = mint_environment_api_key()
        os.environ["ENVIRONMENT_API_KEY"] = task_app_api_key
        
        # Create temporary env file for registration
        temp_env_file = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False)
        temp_env_file.write(f"ENVIRONMENT_API_KEY={task_app_api_key}\n")
        temp_env_file.close()
        
        try:
            # Register with backend
            from pathlib import Path
            preflight_env_key([Path(temp_env_file.name)], crash_on_failure=False)
            print(f"✅ ENVIRONMENT_API_KEY generated and registered: {task_app_api_key[:20]}...")
        except Exception as e:
            print(f"⚠️  Key generated but registration failed: {e}")
            print(f"   Continuing with key: {task_app_api_key[:20]}...")
        finally:
            # Clean up temp file
            if os.path.exists(temp_env_file.name):
                os.unlink(temp_env_file.name)
    else:
        print(f"ℹ️  Using existing ENVIRONMENT_API_KEY: {task_app_api_key[:20]}...")

    # Determine tunnel mode based on backend URL
    # Rule: Both backend and task app must use same mode (local/local or tunnel/tunnel)
    is_backend_localhost = (
        backend_url.startswith("http://localhost") 
        or backend_url.startswith("http://127.0.0.1")
    )
    
    if is_backend_localhost:
        # Backend is localhost → use local mode for task app (no tunnel)
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        use_local_mode = True
        print("ℹ️  Configuration: local/local")
        print("   Backend: localhost")
        print("   Task App: localhost (no tunnel)")
    else:
        # Backend is tunneled → use tunnel mode for task app
        os.environ["SYNTH_TUNNEL_MODE"] = "quick"
        use_local_mode = False
        # Set EXTERNAL_BACKEND_URL so backend knows its public URL for interceptor
        os.environ["EXTERNAL_BACKEND_URL"] = backend_url.rstrip("/")
        print("ℹ️  Configuration: tunnel/tunnel")
        print(f"   Backend: {backend_url}")
        print("   Task App: will create its own tunnel")

    print("\nConfiguration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print("  Task App: Starting in-process...")
    print()

    # Find task app path (from walkthroughs/gepa/task_app/)
    task_app_path = repo_root / "walkthroughs" / "gepa" / "task_app" / "banking77_task_app.py"

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

            try:
                config = toml.load(config_path)
                config["prompt_learning"]["task_app_url"] = task_app.url
            except Exception as e:
                print("\n❌ Error loading/modifying config:")
                print(f"   Type: {type(e).__name__}")
                print(f"   Message: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Show configuration summary
            if "gepa" in config["prompt_learning"]:
                gepa_config = config["prompt_learning"]["gepa"]
                if "population" in gepa_config:
                    num_generations = gepa_config["population"].get("num_generations", 1)
                    children_per_gen = gepa_config["population"].get("children_per_generation", 5)
                    print(f"📊 Running {num_generations} generations with {children_per_gen} children per generation")
                    print(f"   (Total: {num_generations * children_per_gen} prompt candidates)\n")
                
                if "rollout" in gepa_config:
                    budget = gepa_config["rollout"].get("budget", 200)
                    print(f"📊 Rollout budget: {budget}")

            # Write modified config to temp file
            print(f"\nTask app URL: {task_app.url}")
            print(f"Backend URL: {backend_url}\n")
            print("Writing config to temp file...")
            
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                    toml.dump(config, f)
                    temp_config_path = f.name
                print(f"✅ Config written to: {temp_config_path}\n")
            except Exception as e:
                print("\n❌ Error writing config file:")
                print(f"   Type: {type(e).__name__}")
                print(f"   Message: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

            job = None
            job_id = None
            try:
                print("Creating job from config...\n")
                
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    backend_url=backend_url,
                    api_key=api_key,
                    task_app_api_key=task_app_api_key,
                )
                print("✅ Job created successfully\n")

                print("Submitting job...\n")
                job_id = job.submit()
                print(f"✅ Job submitted: {job_id}\n")
                
            except Exception as e:
                print("\n❌ Error during job creation/submission:")
                print(f"   Type: {type(e).__name__}")
                print(f"   Message: {str(e)}")
                print("\n   Full error details:")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to exit context manager
            finally:
                # Clean up temp config file
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)
            
            # Only poll if job was successfully submitted
            if job_id is None:
                print("\n⚠️  Job submission failed, cannot poll for results")
                return
            
            print("=" * 80)
            print("Streaming Results")
            print("=" * 80 + "\n")

            # Poll for completion with streaming updates
            start_time = time.time()
            last_status = None
            
            # Custom polling loop for streaming results
            from synth_ai.api.train.utils import ensure_api_base, http_get, sleep
            
            def poll_with_streaming():
                nonlocal last_status
                elapsed = 0.0
                timeout = 3600.0
                interval = 5.0
                base_url = ensure_api_base(backend_url)
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                path = f"/prompt-learning/online/jobs/{job_id}"
                
                # Normalize URL
                if base_url.endswith("/api") and path.startswith("/api"):
                    path_normalized = path[4:].lstrip("/")
                    url = f"{base_url}/{path_normalized}"
                else:
                    path_clean = path.lstrip("/")
                    url = f"{base_url}/{path_clean}"
                
                while elapsed <= timeout:
                    try:
                        resp = http_get(url, headers=headers)
                        info = (
                            resp.json()
                            if resp.headers.get("content-type", "").startswith("application/json")
                            else {}
                        )
                        status = (info.get("status") or info.get("state") or "").lower()
                        
                        # Call on_status callback for streaming
                        elapsed_real = time.time() - start_time
                        timestamp = time.strftime("%H:%M:%S")
                        progress = info.get("progress", {})
                        best_score = info.get("best_score")
                        
                        # Always print status updates for streaming
                        if progress:
                            completed = progress.get("completed", 0)
                            total = progress.get("total", 0)
                            if total > 0:
                                pct = (completed / total) * 100
                                score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                                print(
                                    f"[{timestamp}] {elapsed_real:6.1f}s  Status: {status} ({completed}/{total} = {pct:.1f}%){score_str}"
                                )
                            else:
                                score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                                print(f"[{timestamp}] {elapsed_real:6.1f}s  Status: {status}{score_str}")
                        else:
                            score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                            print(f"[{timestamp}] {elapsed_real:6.1f}s  Status: {status}{score_str}")
                        
                        last_status = status
                        
                        # Check if terminal state
                        if status in {"succeeded", "failed", "cancelled", "canceled", "completed"}:
                            break
                    except Exception as exc:
                        print(f"[poll] error: {exc}")
                    
                    sleep(interval)
                    elapsed += interval
                else:
                    print(f"[poll] timeout after {timeout}s")
                
                return info

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                poll_with_streaming,
            )

            total_time = time.time() - start_time
            print(f"\n✅ GEPA optimization complete in {total_time:.1f}s\n")

            # Get results
            import httpx
            from synth_ai.learning.prompt_learning_client import PromptLearningClient

            client = PromptLearningClient(
                ensure_api_base(backend_url),
                api_key,
            )
            
            # Get job status/metadata
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
            
            # Extract metadata
            metadata = job_data.get("metadata", {}) or {}
            job_metadata = job_data.get("job_metadata", {}) or {}
            combined_metadata = {**metadata, **job_metadata}
            
            # Get attempted candidates
            attempted_candidates = combined_metadata.get("attempted_candidates")
            
            if attempted_candidates is None:
                # Fallback: try from events via get_prompts
                prompt_results = await client.get_prompts(job._job_id)
                attempted_candidates = prompt_results.attempted_candidates
            
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
            if attempted_candidates and isinstance(attempted_candidates, list):
                print(f"\nTotal candidates: {len(attempted_candidates)}")
                
                # Extract accuracies
                accuracies = []
                for c in attempted_candidates:
                    if isinstance(c, dict):
                        score = c.get("score")
                        if isinstance(score, dict):
                            accuracy = score.get("accuracy")
                            if accuracy is not None:
                                accuracies.append(accuracy)
                        else:
                            accuracy = c.get("accuracy")
                            if accuracy is not None:
                                accuracies.append(accuracy)
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    max_accuracy = max(accuracies)
                    min_accuracy = min(accuracies)
                    print(f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})")
                
                # Show best prompt
                best_prompt = combined_metadata.get("best_prompt") or job_data.get("best_prompt")
                if best_prompt:
                    print("\n" + "=" * 80)
                    print("Best Prompt")
                    print("=" * 80)
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
    print("✅ In-process GEPA optimization complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

