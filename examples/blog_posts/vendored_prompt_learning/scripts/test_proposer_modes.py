#!/usr/bin/env python3
"""
Test script for GEPA and MIPRO proposer modes (dspy, synth, gepa-ai).

This script tests all proposer modes on single-file implementations:
- GEPA with dspy-like proposer
- GEPA with synth-like proposer
- GEPA with gepa-ai proposer
- MIPRO with dspy-like proposer
- MIPRO with synth-like proposer
- MIPRO with gepa-ai proposer

Usage:
    python test_proposer_modes.py [--gepa-only] [--mipro-only] [--mode dspy|synth|gepa-ai]

Requirements:
    - GROQ_API_KEY environment variable set
    - synth-ai backend running (localhost:8000)
    - synth-ai package installed
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
examples_dir = script_dir.parent.parent
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp

# Load environment variables
load_dotenv()


async def verify_tunnel_dns(
    tunnel_url: str,
    name: str = "tunnel",
    timeout_seconds: float = 60.0,
) -> None:
    """Verify that a tunnel URL's hostname can be resolved via DNS.
    
    Args:
        tunnel_url: The tunnel URL to verify
        name: Human-readable name for logging
    
    Raises:
        RuntimeError: If DNS resolution fails after multiple attempts
    """
    parsed = urlparse(tunnel_url)
    hostname = parsed.hostname
    if not hostname:
        print(f"⚠️  No hostname in {name} tunnel URL: {tunnel_url}")
        return
    
    # Skip DNS check for localhost
    if hostname in ("localhost", "127.0.0.1"):
        print(f"✓ Skipping DNS check for localhost {name}")
        return
    
    max_delay = 3.0
    delay = 0.5
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    attempt = 0
    
    print(f"  Verifying DNS resolution for {name}: {hostname} (timeout {timeout_seconds:.0f}s)...")
    
    while True:
        attempt += 1
        try:
            # DNS lookup (blocking I/O)
            resolved = await loop.run_in_executor(
                None,
                socket.gethostbyname,
                hostname
            )
            print(f"  ✓ DNS resolution successful (attempt {attempt}): {hostname} -> {resolved}")
            
            # Also verify HTTP connectivity
            import httpx
            try:
                test_url = f"{parsed.scheme}://{hostname}/health"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(test_url)
                    if resp.status_code in (200, 404, 405):  # Any response means tunnel is up
                        print(f"  ✓ HTTP connectivity verified: {test_url} -> {resp.status_code}")
                        return
                    else:
                        print(f"  ⚠️  HTTP check returned unexpected status: {resp.status_code}")
            except Exception as http_exc:
                print(f"  ⚠️  HTTP connectivity check failed (attempt {attempt}): {http_exc}")
                # Continue to next attempt
            
            # DNS resolved, but HTTP check failed - wait and retry
            now = loop.time()
            if now >= deadline:
                break
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            print(f"  Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
            
        except socket.gaierror as e:
            print(f"  ⚠️  DNS resolution failed (attempt {attempt}): {e}")
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"DNS resolution failed for {name} tunnel hostname {hostname} after {timeout_seconds:.0f}s. "
                    f"This usually means the Cloudflare tunnel DNS has not propagated yet. "
                    f"Tunnel URL: {tunnel_url}. "
                    f"Error: {e}"
                ) from e
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            print(f"  Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)
        except Exception as e:
            print(f"  ❌ Unexpected error during DNS verification (attempt {attempt}): {e}")
            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"DNS verification failed for {hostname} after {timeout_seconds:.0f}s: {e}"
                ) from e
            delay = min(delay * 2 if attempt > 1 else delay, max_delay)
            sleep_for = min(delay, max(0.0, deadline - now))
            print(f"  Waiting {sleep_for:.1f}s before retry...")
            await asyncio.sleep(sleep_for)

    raise RuntimeError(
        f"DNS verification did not complete for {hostname} after {timeout_seconds:.0f}s"
    )


class TestRunner:
    """Runner for testing proposer modes."""

    def __init__(self):
        self.task_app: Optional[InProcessTaskApp] = None
        self.results: Dict[str, Any] = {}

    async def test_config(
        self,
        config_path: Path,
        test_name: str,
        task_app_url: str,
        timeout: float = 1800.0,
    ) -> Dict[str, Any]:
        """Test a single config file."""
        print(f"\n{'='*80}")
        print(f"Testing: {test_name}")
        print(f"{'='*80}\n")
        print(f"Config: {config_path.name}")
        print(f"Task app URL: {task_app_url}")

        if not config_path.exists():
            return {
                "test_name": test_name,
                "config_path": str(config_path),
                "status": "skipped",
                "error": f"Config file not found: {config_path}",
            }

        start_time = time.time()
        try:
            # Load config and override task_app_url
            import toml
            config = toml.load(config_path)
            config["prompt_learning"]["task_app_url"] = task_app_url
            
            # Create a temporary config file with updated URL
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)
            tmp_config_path = Path(tmp_file.name)
            toml.dump(config, tmp_file)
            tmp_file.close()
            
            # Verify backend route is available before submitting
            backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
            test_url = f"{backend_url.rstrip('/')}/api/prompt-learning/online/jobs"
            import httpx
            try:
                test_resp = httpx.get(f"{backend_url.rstrip('/')}/api/health", timeout=2.0)
                if test_resp.status_code != 200:
                    raise RuntimeError(f"Backend health check failed: {test_resp.status_code}")
            except Exception as e:
                raise RuntimeError(
                    f"Backend not reachable at {backend_url}. "
                    f"Is the backend server running? Error: {e}"
                ) from e
            
            try:
                job = PromptLearningJob.from_config(
                    config_path=tmp_config_path,
                    backend_url=backend_url,
                    api_key=os.getenv("SYNTH_API_KEY", "test"),
                    task_app_api_key=os.getenv("ENVIRONMENT_API_KEY", "test"),
                )
                
                job_id = job.submit()
                print(f"✓ Job submitted: {job_id}")

                # Create client for event checking (reused later)
                from synth_ai.learning.prompt_learning_client import PromptLearningClient
                from synth_ai.api.train.utils import ensure_api_base
                client = PromptLearningClient(
                    ensure_api_base(job.config.backend_url),
                    job.config.api_key,
                )

                # Verify interceptor tunnel is created when task app is remote
                if "trycloudflare.com" in task_app_url or "ngrok" in task_app_url or "modal.run" in task_app_url:
                    print(f"\n🔍 Verifying interceptor tunnel creation...")
                    print(f"  Task app is remote: {task_app_url}")
                    
                    # Wait a bit for interceptor to start and tunnel to be created
                    await asyncio.sleep(3)
                    
                    # Check job events for tunnel-related messages
                    events = await client.get_events(job_id, since_seq=0, limit=1000)
                    
                    # Look for interceptor tunnel messages in events
                    tunnel_events = [
                        e for e in events
                        if isinstance(e, dict) and (
                            "interceptor" in str(e.get("message", "")).lower() and "tunnel" in str(e.get("message", "")).lower()
                            or "tunnel" in str(e.get("message", "")).lower() and "interceptor" in str(e.get("message", "")).lower()
                            or "trycloudflare.com" in str(e.get("message", ""))
                        )
                    ]
                    
                    if tunnel_events:
                        print(f"  ✓ Found {len(tunnel_events)} tunnel-related events")
                        for event in tunnel_events[:3]:  # Show first 3
                            msg = event.get("message", "")
                            if msg:
                                print(f"    - {msg[:100]}")
                    else:
                        print(f"  ⚠️  No tunnel-related events found (may be in logs)")
                    
                    # Assert: Check that job doesn't fail with tunnel error
                    # The backend should create tunnel automatically, so we shouldn't see
                    # "CRITICAL: Task app is remote but interceptor public URL is local"
                    tunnel_error_events = [
                        e for e in events
                        if isinstance(e, dict) and (
                            "CRITICAL" in str(e.get("message", ""))
                            and "interceptor" in str(e.get("message", "")).lower()
                            and "localhost" in str(e.get("message", "")).lower()
                        )
                    ]
                    
                    assert not tunnel_error_events, (
                        f"Interceptor tunnel setup failed! Found error events: {tunnel_error_events}. "
                        f"This means the backend did not create a tunnel for the interceptor when task app is remote."
                    )
                    print(f"  ✓ No tunnel setup errors detected")

                result = job.poll_until_complete(
                    timeout=timeout,
                    interval=5.0,
                    on_status=lambda status: print(f"  Status: {status.get('status', 'unknown')}"),
                )

                elapsed_time = time.time() - start_time
                final_status = result.get("status", "unknown")

                print(f"✓ Job complete! Status: {final_status}, Time: {elapsed_time:.1f}s")

                # Final verification: Check that job completed successfully without tunnel errors
                if "trycloudflare.com" in task_app_url or "ngrok" in task_app_url or "modal.run" in task_app_url:
                    print(f"\n🔍 Final tunnel verification...")
                    all_events = await client.get_events(job_id, since_seq=0, limit=5000)
                    
                    # Assert: Job should not have failed due to tunnel issues
                    tunnel_failure_events = [
                        e for e in all_events
                        if isinstance(e, dict) and (
                            ("CRITICAL" in str(e.get("message", "")) or e.get("level") == "error")
                            and "interceptor" in str(e.get("message", "")).lower()
                            and ("localhost" in str(e.get("message", "")).lower() or "tunnel" in str(e.get("message", "")).lower())
                        )
                    ]
                    
                    if tunnel_failure_events:
                        print(f"  ❌ Found tunnel-related errors:")
                        for event in tunnel_failure_events:
                            print(f"    - {event.get('message', '')[:200]}")
                    
                    # Only assert if job failed - if it succeeded, tunnel must have worked
                    if final_status == "failed":
                        assert not tunnel_failure_events, (
                            f"Job failed and tunnel errors were found. "
                            f"This suggests interceptor tunnel was not created properly. "
                            f"Errors: {[e.get('message', '') for e in tunnel_failure_events]}"
                        )
                    elif final_status == "completed":
                        print(f"  ✓ Job completed successfully - tunnel must have worked correctly")
                    
                    # Assert: If job completed, there should be no tunnel setup errors
                    if final_status == "completed":
                        assert not tunnel_failure_events, (
                            f"Job completed but tunnel errors were found in events. "
                            f"This is unexpected. Errors: {[e.get('message', '') for e in tunnel_failure_events]}"
                        )
                
                results_obj = await client.get_prompts(job_id)
                results = {
                    "best_prompt": results_obj.best_prompt,
                    "best_score": results_obj.best_score,
                    "top_prompts": results_obj.top_prompts,
                    "optimized_candidates": results_obj.optimized_candidates,
                    "attempted_candidates": results_obj.attempted_candidates,
                    "validation_results": results_obj.validation_results,
                }
                best_score = results.get("best_score")
                
                if best_score is not None:
                    print(f"✓ Best validation score: {best_score:.4f} ({best_score*100:.2f}%)")
                else:
                    print(f"✓ Best validation score: N/A (job may have terminated early)")
                    best_score = 0.0  # Default for results dict

                return {
                    "test_name": test_name,
                    "config_path": str(config_path),
                    "job_id": job_id,
                    "status": final_status,
                    "best_score": best_score,
                    "elapsed_time": elapsed_time,
                    "success": final_status == "completed",
                }
            finally:
                # Clean up temp file
                if tmp_config_path.exists():
                    tmp_config_path.unlink()


        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "test_name": test_name,
                "config_path": str(config_path),
                "status": "error",
                "error": str(e),
                "elapsed_time": elapsed_time,
                "success": False,
            }

    async def run_all_tests(
        self,
        gepa_only: bool = False,
        mipro_only: bool = False,
        mode_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all proposer mode tests."""
        configs_dir = Path(__file__).parent.parent / "configs"

        # Define all test configs
        test_configs = []

        if not mipro_only:
            # GEPA tests
            gepa_configs = [
                ("GEPA + DSPy", "heartdisease_gepa_dspy.toml"),
                ("GEPA + Synth", "heartdisease_gepa_synth.toml"),
                ("GEPA + GEPA-AI", "heartdisease_gepa_gepa_ai.toml"),
            ]
            test_configs.extend(gepa_configs)

        if not gepa_only:
            # MIPRO tests
            mipro_configs = [
                ("MIPRO + DSPy", "heartdisease_mipro_dspy.toml"),
                ("MIPRO + Synth", "heartdisease_mipro_synth.toml"),
                ("MIPRO + GEPA-AI", "heartdisease_mipro_gepa_ai.toml"),
            ]
            test_configs.extend(mipro_configs)

        # Filter by mode if specified
        if mode_filter:
            mode_filter = mode_filter.lower()
            test_configs = [
                (name, config) for name, config in test_configs
                if mode_filter in name.lower()
            ]

        print(f"\n{'='*80}")
        print("PROPOSER MODE TEST SUITE")
        print(f"{'='*80}\n")
        print(f"Total tests: {len(test_configs)}")
        print(f"Tests to run:")
        for name, _ in test_configs:
            print(f"  - {name}")
        print()

        # Start in-process task app using InProcessTaskApp (like run_synth_gepa_in_process.py)
        # Find the task app file
        script_dir = Path(__file__).resolve().parent
        examples_dir = script_dir.parent.parent
        task_app_path = examples_dir / "task_apps" / "other_langprobe_benchmarks" / "heartdisease_task_app.py"
        
        if not task_app_path.exists():
            raise FileNotFoundError(f"Task app not found: {task_app_path}")

        print(f"\n{'='*80}")
        print("Starting In-Process Task App")
        print(f"{'='*80}\n")
        print(f"Task app: {task_app_path.name}")

        # Use InProcessTaskApp context manager (like run_synth_gepa_in_process.py)
        task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")
        
        async with InProcessTaskApp(
            task_app_path=task_app_path,
            port=8114,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✓ Task app running at: {task_app.url}")
            print(f"✓ Cloudflare tunnel active and verified")
            
            # Note: DNS verification is now handled internally by InProcessTaskApp
            # No need for duplicate verification here
            
            print()

            # Run all tests
            all_results = []
            for test_name, config_file in test_configs:
                config_path = configs_dir / config_file
                result = await self.test_config(
                    config_path, test_name, task_app.url, timeout=600.0
                )
                all_results.append(result)
                self.results[test_name] = result
                await asyncio.sleep(2)
            
            # Print summary
            self.print_summary(all_results)
            
            # Save results
            output_dir = Path(__file__).parent / "results" / "proposer_modes_test"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "results.json"
            with open(output_file, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "tests": all_results,
                }, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
            
            return {"tests": all_results, "summary": self._compute_summary(all_results)}

    def print_summary(self, results: list[Dict[str, Any]]) -> None:
        """Print test summary."""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}\n")

        # Table header
        print("┌" + "─"*78 + "┐")
        print("│" + " Test Name".ljust(30) + "│" + " Status".center(15) + "│" + " Score".center(15) + "│" + " Time".center(16) + "│")
        print("├" + "─"*30 + "┼" + "─"*15 + "┼" + "─"*15 + "┼" + "─"*16 + "┤")

        for result in results:
            test_name = result.get("test_name", "unknown")
            status = result.get("status", "unknown")
            best_score = result.get("best_score", 0.0)
            elapsed_time = result.get("elapsed_time", 0.0)

            # Truncate test name if too long
            if len(test_name) > 28:
                test_name = test_name[:25] + "..."

            status_str = "✓" if result.get("success") else "✗"
            score_str = f"{best_score:.4f}" if best_score > 0 else "N/A"
            time_str = f"{elapsed_time:.1f}s"

            print(f"│ {test_name.ljust(28)} │ {status_str.center(13)} │ {score_str.center(13)} │ {time_str.center(14)} │")

        print("└" + "─"*30 + "┴" + "─"*15 + "┴" + "─"*15 + "┴" + "─"*16 + "┘")

        # Summary stats
        summary = self._compute_summary(results)
        print(f"\nTotal tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        if summary['passed'] > 0:
            avg_score = summary['avg_score']
            print(f"Average score: {avg_score:.4f} ({avg_score*100:.2f}%)")

    def _compute_summary(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics."""
        total = len(results)
        passed = sum(1 for r in results if r.get("success"))
        failed = sum(1 for r in results if not r.get("success") and r.get("status") != "skipped")
        skipped = sum(1 for r in results if r.get("status") == "skipped")

        scores = [r.get("best_score", 0.0) for r in results if r.get("best_score", 0.0) > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "avg_score": avg_score,
        }


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test GEPA and MIPRO proposer modes")
    parser.add_argument("--gepa-only", action="store_true", help="Test only GEPA algorithms")
    parser.add_argument("--mipro-only", action="store_true", help="Test only MIPRO algorithms")
    parser.add_argument("--mode", choices=["dspy", "synth", "gepa-ai"], help="Test only specific proposer mode")
    args = parser.parse_args()

    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required")
        sys.exit(1)

    runner = TestRunner()
    try:
        await runner.run_all_tests(
            gepa_only=args.gepa_only,
            mipro_only=args.mipro_only,
            mode_filter=args.mode,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

