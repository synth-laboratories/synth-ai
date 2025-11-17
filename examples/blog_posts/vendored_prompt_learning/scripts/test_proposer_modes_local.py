#!/usr/bin/env python3
"""
Local-only test script for GEPA and MIPRO proposer modes (SYNTH DEVELOPERS ONLY).

⚠️  WARNING: This test suite is designed EXCLUSIVELY for synth developers working
    on the synth-ai codebase. It tests proposer modes using localhost only, without
    requiring Cloudflare tunnels or external services.

This script tests all proposer modes on single-file implementations:
- GEPA with dspy-like proposer
- GEPA with synth-like proposer
- GEPA with gepa-ai proposer
- MIPRO with dspy-like proposer
- MIPRO with synth-like proposer
- MIPRO with gepa-ai proposer

Usage:
    python test_proposer_modes_local.py [--gepa-only] [--mipro-only] [--mode dspy|synth|gepa-ai]

Requirements:
    - GROQ_API_KEY environment variable set
    - synth-ai backend running (localhost:8000)
    - synth-ai package installed

DO NOT USE THIS FOR PRODUCTION TESTING OR CUSTOMER-FACING FUNCTIONALITY.
This is for internal development and debugging only.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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


class TestRunner:
    """Runner for testing proposer modes (local-only)."""

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
        print(f"Task app URL: {task_app_url} (LOCAL ONLY - no tunnels)")

        if not config_path.exists():
            return {
                "test_name": test_name,
                "config_path": str(config_path),
                "status": "skipped",
                "error": f"Config file not found: {config_path}",
            }

        start_time = time.time()
        try:
            # Load config and override task_app_url and policy settings
            import toml
            config = toml.load(config_path)
            config["prompt_learning"]["task_app_url"] = task_app_url
            
            # Override policy to use llama-3.1-8b-instant via Groq API
            if "policy" not in config["prompt_learning"]:
                config["prompt_learning"]["policy"] = {}
            original_model = config["prompt_learning"]["policy"].get("model", "unknown")
            original_provider = config["prompt_learning"]["policy"].get("provider", "unknown")
            config["prompt_learning"]["policy"]["model"] = "llama-3.1-8b-instant"
            config["prompt_learning"]["policy"]["provider"] = "groq"
            config["prompt_learning"]["policy"]["inference_mode"] = "synth_hosted"
            config["prompt_learning"]["policy"].setdefault("temperature", 0.0)
            config["prompt_learning"]["policy"].setdefault("max_completion_tokens", 512)
            print(f"ℹ️  Policy overridden: {original_provider}/{original_model} → groq/llama-3.1-8b-instant")
            
            # Create a temporary config file with updated settings
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

                # Create client for event checking
                from synth_ai.learning.prompt_learning_client import PromptLearningClient
                from synth_ai.api.train.utils import ensure_api_base
                client = PromptLearningClient(
                    ensure_api_base(job.config.backend_url),
                    job.config.api_key,
                )

                # Note: No tunnel verification needed for local-only testing
                print(f"\n✓ Using localhost task app - no tunnel verification needed")

                result = job.poll_until_complete(
                    timeout=timeout,
                    interval=5.0,
                    on_status=lambda status: print(f"  Status: {status.get('status', 'unknown')}"),
                )

                elapsed_time = time.time() - start_time
                final_status = result.get("status", "unknown")

                print(f"✓ Job complete! Status: {final_status}, Time: {elapsed_time:.1f}s")

                # For local-only testing, we don't check for tunnel errors
                # since we're using localhost directly
                print(f"\n✓ Local-only test completed (no tunnel checks)")

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
                early_termination = (best_score is None and final_status in ("completed", "succeeded"))
                
                if best_score is None:
                    best_score = 0.0
                    if early_termination:
                        print(f"ℹ️  Best validation score: None (optimization terminated early - acceptable for local testing)")
                    else:
                        print(f"⚠️  Best validation score: None (optimization may have terminated early)")
                else:
                    print(f"✓ Best validation score: {best_score:.4f} ({best_score*100:.2f}%)")

                # Early termination is acceptable - treat completed/succeeded as success regardless of best_score
                is_success = final_status in ("completed", "succeeded")

                return {
                    "test_name": test_name,
                    "config_path": str(config_path),
                    "job_id": job_id,
                    "status": final_status,
                    "best_score": best_score,
                    "elapsed_time": elapsed_time,
                    "success": is_success,
                    "early_termination": early_termination,
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
        """Run all proposer mode tests (local-only)."""
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
        print("PROPOSER MODE TEST SUITE (LOCAL-ONLY - SYNTH DEVELOPERS ONLY)")
        print(f"{'='*80}\n")
        print("⚠️  WARNING: This is a local-only test suite.")
        print("   It does NOT use Cloudflare tunnels or external services.")
        print("   For production testing, use test_proposer_modes.py\n")
        print(f"Total tests: {len(test_configs)}")
        print(f"Tests to run:")
        for name, _ in test_configs:
            print(f"  - {name}")
        print()

        # Start in-process task app using InProcessTaskApp (LOCAL ONLY)
        # Find the task app file
        script_dir = Path(__file__).resolve().parent
        examples_dir = script_dir.parent.parent
        task_app_path = examples_dir / "task_apps" / "other_langprobe_benchmarks" / "heartdisease_task_app.py"
        
        if not task_app_path.exists():
            raise FileNotFoundError(f"Task app not found: {task_app_path}")

        print(f"\n{'='*80}")
        print("Starting In-Process Task App (LOCAL ONLY)")
        print(f"{'='*80}\n")
        print(f"Task app: {task_app_path.name}")
        print(f"⚠️  Using localhost only - NO Cloudflare tunnels")

        # Use InProcessTaskApp context manager with localhost mode
        task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")
        
        # Set environment variable to force local mode (no tunnels)
        # InProcessTaskApp checks SYNTH_TUNNEL_MODE env var and uses localhost if set to "local"
        original_tunnel_mode = os.environ.get("SYNTH_TUNNEL_MODE")
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        
        try:
            async with InProcessTaskApp(
                task_app_path=task_app_path,
                port=8114,
                api_key=task_app_api_key,
            ) as task_app:
                task_app_url = task_app.url
                
                # Verify it's using localhost (not a tunnel)
                if "localhost" not in task_app_url and "127.0.0.1" not in task_app_url:
                    print(f"⚠️  WARNING: Task app URL is not localhost: {task_app_url}")
                    print(f"   This local-only test expects localhost URLs only.")
                    print(f"   Consider using test_proposer_modes.py for tunnel-based testing.")
                else:
                    print(f"✓ Task app running at: {task_app_url} (LOCAL - NO TUNNELS)")
                
                print()

                # Run all tests
                all_results = []
                for test_name, config_file in test_configs:
                    config_path = configs_dir / config_file
                    result = await self.test_config(
                        config_path, test_name, task_app_url, timeout=600.0
                    )
                    all_results.append(result)
                    self.results[test_name] = result
                    await asyncio.sleep(2)
                
                # Print summary
                self.print_summary(all_results)
                
                # Save results
                output_dir = Path(__file__).parent / "results" / "proposer_modes_test_local"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "results.json"
                with open(output_file, "w") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "tests": all_results,
                        "note": "Local-only test results (no Cloudflare tunnels)",
                    }, f, indent=2)
                print(f"\n✓ Results saved to: {output_file}")
                
                # Force cleanup: ensure all async tasks complete before exiting
                import gc
                gc.collect()
                
                return {"tests": all_results, "summary": self._compute_summary(all_results)}
        finally:
            # Restore original tunnel mode if it was set
            if original_tunnel_mode is not None:
                os.environ["SYNTH_TUNNEL_MODE"] = original_tunnel_mode
            elif "SYNTH_TUNNEL_MODE" in os.environ:
                del os.environ["SYNTH_TUNNEL_MODE"]

    def print_summary(self, results: list[Dict[str, Any]]) -> None:
        """Print test summary."""
        print(f"\n{'='*80}")
        print("TEST SUMMARY (LOCAL-ONLY)")
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

    parser = argparse.ArgumentParser(
        description="Test GEPA and MIPRO proposer modes (LOCAL-ONLY - SYNTH DEVELOPERS ONLY)"
    )
    parser.add_argument("--gepa-only", action="store_true", help="Test only GEPA algorithms")
    parser.add_argument("--mipro-only", action="store_true", help="Test only MIPRO algorithms")
    parser.add_argument("--mode", choices=["dspy", "synth", "gepa-ai"], help="Test only specific proposer mode")
    args = parser.parse_args()

    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required")
        print("   This test uses llama-3.1-8b-instant via the Groq API")
        sys.exit(1)

    print("\n⚠️  WARNING: This is a LOCAL-ONLY test suite for synth developers.")
    print("   It does NOT use Cloudflare tunnels or external services.")
    print("   For production testing with tunnels, use test_proposer_modes.py\n")

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

