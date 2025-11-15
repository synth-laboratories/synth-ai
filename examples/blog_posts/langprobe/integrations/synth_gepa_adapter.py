"""Synth GEPA adapter for blog post comparisons."""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from enum import Enum
from tqdm import tqdm

from .learning_curve_tracker import LearningCurveTracker
from .task_app_client import TaskAppClient

# Load environment variables from .env file
load_dotenv()


class JobStatus(str, Enum):
    """Job status values from backend API."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"  # Some endpoints use "completed" instead of "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLED_ALT = "canceled"  # Alternative spelling
    
    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is a terminal state (job is done)."""
        status_lower = status.lower() if isinstance(status, str) else str(status).lower()
        return status_lower in (
            cls.SUCCEEDED.value,
            cls.COMPLETED.value,
            "done",
            "finished",
            "success",
            cls.FAILED.value,
            "error",
            "crashed",
            cls.CANCELLED.value,
            cls.CANCELLED_ALT.value,
        )
    
    @classmethod
    def is_success(cls, status: str) -> bool:
        """Check if status indicates success."""
        status_lower = status.lower() if isinstance(status, str) else str(status).lower()
        return status_lower in (
            cls.SUCCEEDED.value,
            cls.COMPLETED.value,
            "done",
            "finished",
            "success",
        )
    
    @classmethod
    def is_failure(cls, status: str) -> bool:
        """Check if status indicates failure."""
        status_lower = status.lower() if isinstance(status, str) else str(status).lower()
        return status_lower in (
            cls.FAILED.value,
            "error",
            "crashed",
        )

# Try to import tunnel SDK - optional dependency
try:
    from synth_ai.tunnel import open_quick_tunnel, stop_tunnel
    TUNNEL_AVAILABLE = True
except ImportError:
    TUNNEL_AVAILABLE = False
    open_quick_tunnel = None  # type: ignore
    stop_tunnel = None  # type: ignore


class SynthGEPAAdapter:
    """Adapter for Synth GEPA that uses HTTP API calls to backend."""

    def __init__(
        self,
        backend_url: str,
        task_app_url: str,
        task_app_id: str,
        initial_prompt_messages: list[dict[str, Any]],
        rollout_budget: int = 400,
        tunnel_url: str | None = None,
        auto_tunnel: bool = False,
    ):
        """Initialize Synth GEPA adapter.

        Args:
            backend_url: Backend API URL (e.g., "http://localhost:8000")
            task_app_url: Task app URL (e.g., "http://127.0.0.1:8115")
            task_app_id: Task app ID (e.g., "iris")
            initial_prompt_messages: Initial prompt messages
            rollout_budget: Total rollout budget (~400)
            tunnel_url: Public tunnel URL for task app (required for Modal).
                        If None, will try to detect from TUNNEL_URL env var.
                        Examples: "https://abc123.ngrok.io" or "https://xyz.loca.lt"
            auto_tunnel: If True and task_app_url is localhost, automatically create
                        a Cloudflare tunnel. Requires synth_ai.tunnel to be available.
        """
        self.backend_url = backend_url
        self.original_task_app_url = task_app_url
        self.task_app_id = task_app_id
        self.initial_prompt_messages = initial_prompt_messages
        self.rollout_budget = rollout_budget
        self.auto_tunnel = auto_tunnel
        self._tunnel_proc = None  # Store tunnel process handle for cleanup
        
        # Determine final task app URL - FAIL FAST if localhost without tunnel
        is_localhost = ("127.0.0.1" in task_app_url or "localhost" in task_app_url)
        
        if tunnel_url:
            self.task_app_url = tunnel_url
        elif os.getenv("TUNNEL_URL"):
            self.task_app_url = os.getenv("TUNNEL_URL")
        elif auto_tunnel and is_localhost:
            # Extract port from localhost URL and create tunnel
            port_match = re.search(r":(\d+)", task_app_url)
            if port_match:
                port = int(port_match.group(1))
                if TUNNEL_AVAILABLE and open_quick_tunnel:
                    print(f"🌐 Creating Cloudflare tunnel for localhost:{port}...")
                    try:
                        tunnel_url_result, tunnel_proc = open_quick_tunnel(port)
                        self.task_app_url = tunnel_url_result
                        self._tunnel_proc = tunnel_proc
                        print(f"✅ Tunnel created: {self.task_app_url}")
                        print(f"   Tunnel process PID: {tunnel_proc.pid}")
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to create tunnel: {e}\n"
                            "Install cloudflared: brew install cloudflare/cloudflare/cloudflared"
                        ) from e
                else:
                    raise RuntimeError(
                        "auto_tunnel=True but synth_ai.tunnel not available. "
                        "Install synth-ai package or set TUNNEL_URL manually."
                    )
            else:
                raise ValueError(f"Cannot extract port from task_app_url: {task_app_url}")
        elif is_localhost:
            # FAIL FAST: localhost URL without tunnel
            raise ValueError(
                f"❌ Invalid task_app_url: '{task_app_url}' is localhost. "
                "Modal backend cannot reach localhost URLs.\n\n"
                "Options:\n"
                "  1. Use --tunnel-url <public-url> (e.g., https://abc123.trycloudflare.com)\n"
                "  2. Use --auto-tunnel to automatically create a Cloudflare tunnel\n"
                "  3. Set TUNNEL_URL environment variable\n"
                "  4. Deploy task app to a publicly accessible URL\n\n"
                "Example: --auto-tunnel (will create tunnel automatically)"
            )
        else:
            self.task_app_url = task_app_url

        # Learning curve tracker
        self.learning_curve = LearningCurveTracker(
            framework="synth_gepa",
            benchmark=task_app_id,
            total_budget=rollout_budget,
        )
    
    def _get_initial_population_size(self) -> int:
        """Auto-scale initial population size based on rollout budget."""
        if self.rollout_budget < 10:
            # Ultra-minimal smoke test mode
            return 1
        elif self.rollout_budget < 50:
            return 2
        elif self.rollout_budget < 100:
            return 5
        elif self.rollout_budget < 200:
            return 10
        else:
            return 20
    
    def _get_num_generations(self) -> int:
        """Auto-scale number of generations based on rollout budget."""
        if self.rollout_budget < 10:
            # Ultra-minimal smoke test mode
            return 1
        elif self.rollout_budget < 50:
            return 2
        elif self.rollout_budget < 100:
            return 5
        elif self.rollout_budget < 200:
            return 10
        else:
            return 15
    
    def _get_minibatch_size(self) -> int:
        """Auto-scale minibatch size based on rollout budget."""
        if self.rollout_budget < 10:
            # Ultra-minimal smoke test mode
            return 1
        elif self.rollout_budget < 50:
            return 1
        elif self.rollout_budget < 100:
            return 2
        elif self.rollout_budget < 200:
            return 4
        else:
            return 8
    
    def _get_children_per_generation(self) -> int:
        """Auto-scale children per generation based on rollout budget."""
        if self.rollout_budget < 10:
            # Ultra-minimal smoke test mode
            return 1
        elif self.rollout_budget < 50:
            return 1
        elif self.rollout_budget < 100:
            return 4
        elif self.rollout_budget < 200:
            return 8
        else:
            return 12

    async def optimize(self, train_seeds: list[int]) -> dict[str, Any]:
        """Run optimization via HTTP API and return results with learning curve.

        Args:
            train_seeds: List of seed IDs for training

        Returns:
            Dictionary with best_prompt, best_score, learning_curve, etc.
        """
        import httpx

        # Build TOML config - backend expects config_body as dict
        # We'll parse the TOML string to dict
        toml_str = self._build_toml_config(train_seeds)
        try:
            import tomli
            config_dict = tomli.loads(toml_str)
        except ImportError:
            # Fallback: parse manually or use tomllib (Python 3.11+)
            import tomllib
            config_dict = tomllib.loads(toml_str.encode('utf-8'))

        # Call backend API
        async with httpx.AsyncClient(timeout=7200.0) as client:
            # Load API key from .env (already loaded at module level, but ensure it's available)
            api_key = os.getenv("SYNTH_API_KEY") or os.getenv("ENVIRONMENT_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "SYNTH_API_KEY or ENVIRONMENT_API_KEY required. "
                    "Make sure .env file exists and contains one of these keys."
                )

            # Use online jobs endpoint
            # Backend accepts both Authorization: Bearer <key> and X-API-Key: <key>
            response = await client.post(
                f"{self.backend_url}/api/prompt-learning/online/jobs",
                json={
                    "algorithm": "gepa",
                    "config_body": config_dict,  # Pass as dict, not string
                    "overrides": {
                        "task_app_url": self.task_app_url,
                    },
                    "auto_start": True,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-API-Key": api_key,  # Also send as X-API-Key header for compatibility
                    "Content-Type": "application/json",
                },
            )

            if response.status_code not in (200, 201):
                raise RuntimeError(
                    f"Job submission failed: {response.status_code} - {response.text[:500]}"
                )

            job_data = response.json()
            job_id = job_data.get("job_id") or job_data.get("id")

            if not job_id:
                raise RuntimeError("Response missing job ID")

            print(f"✓ Job submitted: {job_id}")
            
            # Poll until complete with progress bar
            max_wait = 7200  # 2 hours
            start_time = time.time()
            poll_interval = 5.0  # Poll every 5 seconds
            last_event_seq = 0
            last_metric_step = -1
            status = "unknown"
            best_score = None
            is_done = False  # Track if job completed
            
            # Create progress bar
            with tqdm(
                total=self.rollout_budget,
                desc=f"Optimizing ({job_id[:8]}...)",
                unit="rollout",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            ) as pbar:
                while time.time() - start_time < max_wait:
                    # Get job status
                    status_response = await client.get(
                        f"{self.backend_url}/api/prompt-learning/online/jobs/{job_id}",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "X-API-Key": api_key,
                        },
                    )

                    if status_response.status_code != 200:
                        raise RuntimeError(
                            f"Failed to get job status: {status_response.status_code}"
                        )

                    status_data = status_response.json()
                    status = status_data.get("status", "unknown")
                    best_score = status_data.get("best_train_score") or status_data.get("best_score")
                    
                    # Check tunnel health periodically (every 10 polls = ~50 seconds)
                    if int((time.time() - start_time) / poll_interval) % 10 == 0 and self._tunnel_proc:
                        if self._tunnel_proc.poll() is not None:
                            raise RuntimeError(
                                f"Tunnel process died with exit code {self._tunnel_proc.returncode}. "
                                "The tunnel must stay alive for the entire job duration. "
                                "Try using a managed tunnel or ensure cloudflared is stable."
                            )
                    
                    # Get events for progress tracking
                    try:
                        events_response = await client.get(
                            f"{self.backend_url}/api/prompt-learning/online/jobs/{job_id}/events",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "X-API-Key": api_key,
                            },
                            params={"limit": 100},
                        )
                        if events_response.status_code == 200:
                            events = events_response.json()
                            if isinstance(events, list):
                                for event in events:
                                    seq = event.get("seq", 0)
                                    if seq > last_event_seq:
                                        last_event_seq = seq
                                        # Extract rollout count from events if available
                                        msg = event.get("message", "") or event.get("msg", "")
                                        if "rollout" in msg.lower() or "evaluation" in msg.lower():
                                            # Try to extract rollout number
                                            rollout_match = re.search(r'(\d+)\s*(?:rollout|eval)', msg.lower())
                                            if rollout_match:
                                                current_rollouts = int(rollout_match.group(1))
                                                pbar.n = min(current_rollouts, self.rollout_budget)
                                                pbar.refresh()
                    except Exception:
                        pass  # Ignore event fetch errors
                    
                    # Get metrics for checkpoint tracking
                    try:
                        metrics_response = await client.get(
                            f"{self.backend_url}/api/prompt-learning/online/jobs/{job_id}/metrics",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "X-API-Key": api_key,
                            },
                            params={"after_step": last_metric_step, "limit": 50},
                        )
                        if metrics_response.status_code == 200:
                            metrics_data = metrics_response.json()
                            points = metrics_data.get("points", [])
                            if isinstance(points, list):
                                for point in points:
                                    step = point.get("step", -1)
                                    if step > last_metric_step:
                                        last_metric_step = step
                                        # Track checkpoints from metrics
                                        metric_name = point.get("name", "")
                                        metric_value = point.get("value", 0.0)
                                        
                                        # Record checkpoint if it's a performance metric
                                        if "score" in metric_name.lower() or "performance" in metric_name.lower():
                                            checkpoint_pct = step / self.rollout_budget if self.rollout_budget > 0 else 0.0
                                            if checkpoint_pct <= 1.0:
                                                self.learning_curve.curve.record(
                                                    rollout_count=step,
                                                    performance=float(metric_value),
                                                    checkpoint_pct=checkpoint_pct,
                                                )
                    except Exception:
                        pass  # Ignore metric fetch errors
                    
                    # Update progress bar
                    status_display = status.upper()
                    score_display = f"best={best_score:.4f}" if best_score is not None else "best=N/A"
                    pbar.set_postfix_str(f"{status_display} | {score_display}")
                    
                    # Update progress based on status
                    if status.lower() == JobStatus.RUNNING.value:
                        # Estimate progress if we don't have exact rollout count
                        elapsed = time.time() - start_time
                        # Rough estimate: assume ~1 rollout per 2 seconds
                        estimated_rollouts = min(int(elapsed / 2), self.rollout_budget)
                        if pbar.n < estimated_rollouts:
                            pbar.n = estimated_rollouts
                            pbar.refresh()

                    # Check if job is done using enum
                    is_done = JobStatus.is_terminal(status)
                    
                    if is_done:
                        pbar.n = self.rollout_budget  # Set to max on completion
                        pbar.refresh()
                        if JobStatus.is_failure(status):
                            # Try to get error details from events
                            try:
                                events_response = await client.get(
                                    f"{self.backend_url}/api/prompt-learning/online/jobs/{job_id}/events",
                                    headers={
                                        "Authorization": f"Bearer {api_key}",
                                        "X-API-Key": api_key,
                                    },
                                    params={"limit": 10},
                                )
                                if events_response.status_code == 200:
                                    events = events_response.json()
                                    if isinstance(events, list):
                                        error_events = [e for e in events if "error" in str(e.get("message", "")).lower() or "failed" in str(e.get("message", "")).lower()]
                                        if error_events:
                                            error_msg = error_events[0].get("message", "Unknown error")
                                            print(f"\n⚠️  Job failed: {error_msg}")
                            except Exception:
                                pass
                        break  # Exit the while loop

                    await asyncio.sleep(poll_interval)
                
                # Handle timeout case
                if not is_done and time.time() - start_time >= max_wait:
                    print(f"\n⚠️  Polling timeout after {max_wait}s. Job may still be running.")
                    print(f"   Final status: {status}")
                    print(f"   Check job status manually: {self.backend_url}/api/prompt-learning/online/jobs/{job_id}")
                
                # Final update
                pbar.set_postfix_str(f"{status.upper()} | best={best_score:.4f}" if best_score else f"{status.upper()}")
                pbar.close()

            # IMPORTANT: Keep tunnel alive until job completes
            # Don't cleanup tunnel here - it's needed for the entire job duration
            # Cleanup will happen in adapter.cleanup() after results are retrieved

            # Get final results from job detail endpoint (includes best_snapshot)
            try:
                final_response = await client.get(
                    f"{self.backend_url}/api/prompt-learning/online/jobs/{job_id}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-API-Key": api_key,
                    },
                )

                if final_response.status_code == 200:
                    job_detail = final_response.json()
                    final_best_score = (
                        job_detail.get("best_train_score")
                        or job_detail.get("best_score")
                        or job_detail.get("best_validation_score")
                        or 0.0
                    )
                    best_snapshot = job_detail.get("best_snapshot", {})
                    best_snapshot_id = job_detail.get("best_snapshot_id")

                    # Record final checkpoint
                    self.learning_curve.curve.record(
                        rollout_count=self.rollout_budget,
                        performance=final_best_score,
                        checkpoint_pct=1.0,
                    )

                    return {
                        "job_id": job_id,
                        "best_score": final_best_score,
                        "best_prompt": best_snapshot,
                        "best_snapshot_id": best_snapshot_id,
                        "learning_curve": self.learning_curve.curve.to_dict(),
                        "total_rollouts": self.rollout_budget,
                        "status": status or job_detail.get("status", "unknown"),
                    }
                else:
                    # Fallback: use status data from polling
                    final_best_score = best_score or 0.0
                    self.learning_curve.curve.record(
                        rollout_count=self.rollout_budget,
                        performance=final_best_score,
                        checkpoint_pct=1.0,
                    )
                    return {
                        "job_id": job_id,
                        "best_score": final_best_score,
                        "learning_curve": self.learning_curve.curve.to_dict(),
                        "total_rollouts": self.rollout_budget,
                        "status": status or "unknown",
                    }
            except Exception as e:
                print(f"\n⚠️  Error fetching final results: {e}")
                # Fallback: use status data from polling
                final_best_score = best_score or 0.0
                self.learning_curve.curve.record(
                    rollout_count=self.rollout_budget,
                    performance=final_best_score,
                    checkpoint_pct=1.0,
                )
                return {
                    "job_id": job_id,
                    "best_score": final_best_score,
                    "learning_curve": self.learning_curve.curve.to_dict(),
                    "total_rollouts": self.rollout_budget,
                    "status": status or "unknown",
                }

    def _build_toml_config(self, train_seeds: list[int]) -> str:
        """Build TOML config string for backend API with all required GEPA sections."""
        # Escape quotes and newlines in patterns for TOML multiline strings
        system_pattern = (
            self.initial_prompt_messages[0]["pattern"]
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        user_pattern = (
            self.initial_prompt_messages[1]["pattern"]
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        
        # Get API key for task app
        task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "")
        
        # Determine validation seeds (held-out set)
        # Use seeds 100-149 for validation (assuming train_seeds are 0-99 or smaller)
        max_train_seed = max(train_seeds) if train_seeds else 99
        val_seeds = list(range(max_train_seed + 1, max_train_seed + 51))  # 50 validation seeds
        
        return f"""
[prompt_learning]
algorithm = "gepa"
task_app_url = "{self.task_app_url}"
task_app_api_key = "{task_app_api_key}"
task_app_id = "{self.task_app_id}"

[prompt_learning.initial_prompt]
id = "{self.task_app_id}_classification"
name = "{self.task_app_id.title()} Classification"

[[prompt_learning.initial_prompt.messages]]
role = "{self.initial_prompt_messages[0]['role']}"
pattern = "{system_pattern}"
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "{self.initial_prompt_messages[1]['role']}"
pattern = "{user_pattern}"
order = 1

[prompt_learning.initial_prompt.wildcards]
features = "REQUIRED"

[prompt_learning.policy]
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 1.0
max_completion_tokens = 512
policy_name = "{self.task_app_id}-gepa"

[prompt_learning.gepa]
env_name = "{self.task_app_id}"

# Rollout configuration
[prompt_learning.gepa.rollout]
budget = {self.rollout_budget}
max_concurrent = 20
minibatch_size = {self._get_minibatch_size()}

# Evaluation configuration
[prompt_learning.gepa.evaluation]
seeds = {train_seeds}
validation_seeds = {val_seeds}
validation_pool = "train"  # Iris dataset only has "train" split
validation_top_k = 3

# Mutation configuration
[prompt_learning.gepa.mutation]
rate = 0.3
llm_model = "openai/gpt-oss-20b"
llm_provider = "groq"

# Population configuration
[prompt_learning.gepa.population]
initial_size = {self._get_initial_population_size()}
num_generations = {self._get_num_generations()}
children_per_generation = {self._get_children_per_generation()}
crossover_rate = 0.5
selection_pressure = 1.0
patience_generations = 5

# Archive configuration
[prompt_learning.gepa.archive]
size = 64
pareto_set_size = 64
pareto_eps = 1e-6
feedback_fraction = 0.5

# Token configuration
[prompt_learning.gepa.token]
counting_model = "gpt-4"
enforce_pattern_limit = true
"""

    def save_results(self, output_dir: Path) -> None:
        """Save results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_curve.save(output_dir)
    
    def cleanup(self) -> None:
        """Clean up resources (e.g., stop tunnel if auto-created)."""
        if self._tunnel_proc and TUNNEL_AVAILABLE and stop_tunnel:
            print("🛑 Stopping tunnel...")
            stop_tunnel(self._tunnel_proc)
            self._tunnel_proc = None
            print("✅ Tunnel stopped")


async def run_synth_gepa_iris(
    backend_url: str = "http://localhost:8000",
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: list[int] | None = None,
    rollout_budget: int = 400,
    output_dir: Path | None = None,
    tunnel_url: str | None = None,
    auto_tunnel: bool = False,
) -> dict[str, Any]:
    """Run Synth GEPA on Iris benchmark.

    Args:
        backend_url: Backend API URL (default: http://localhost:8000)
        task_app_url: Task app URL (local)
        train_seeds: Training seeds (default: 0-99)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory (default: results/synth_gepa/)
        tunnel_url: Public tunnel URL for task app (required for Modal).
                    If None, checks TUNNEL_URL env var.
        auto_tunnel: If True, automatically create Cloudflare tunnel for localhost URLs.

    Returns:
        Results dictionary
    """
    if train_seeds is None:
        # Auto-scale number of seeds based on rollout budget for smoke tests
        if rollout_budget < 10:
            train_seeds = list(range(5))  # Ultra-minimal: 5 seeds
        elif rollout_budget < 50:
            train_seeds = list(range(10))  # Minimal: 10 seeds
        elif rollout_budget < 100:
            train_seeds = list(range(20))  # Small: 20 seeds
        else:
            train_seeds = list(range(100))  # Full: 100 seeds

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "synth_gepa"

    # Initial prompt messages (from iris_task_app.py)
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a botany classification assistant. Based on the flower's measurements, "
                "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Flower Measurements:\n{features}\n\n"
                "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
    ]

    # Create adapter
    adapter = SynthGEPAAdapter(
        backend_url=backend_url,
        task_app_url=task_app_url,
        task_app_id="iris",
        initial_prompt_messages=initial_prompt_messages,
        rollout_budget=rollout_budget,
        tunnel_url=tunnel_url,
        auto_tunnel=auto_tunnel,
    )

    try:
        # Run optimization
        results = await adapter.optimize(train_seeds=train_seeds)

        # Save results
        adapter.save_results(output_dir)

        return results
    finally:
        # Cleanup tunnel if auto-created
        adapter.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Synth GEPA on Iris")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL",
    )
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8115",
        help="Task app URL. If localhost, MUST use --auto-tunnel or --tunnel-url (Modal cannot reach localhost)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=400,
        help="Rollout budget (for smoke test, use 5-10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--tunnel-url",
        type=str,
        default=None,
        help="Public tunnel URL for task app (required for Modal). "
             "Example: https://abc123.ngrok.io. "
             "Can also set TUNNEL_URL env var.",
    )
    parser.add_argument(
        "--auto-tunnel",
        action="store_true",
        help="Automatically create a Cloudflare tunnel if task_app_url is localhost. "
             "Requires cloudflared to be installed.",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_synth_gepa_iris(
            backend_url=args.backend_url,
            task_app_url=args.task_app_url,
            rollout_budget=args.rollout_budget,
            output_dir=args.output_dir,
            tunnel_url=args.tunnel_url,
            auto_tunnel=args.auto_tunnel,
        )
    )

    print(f"Best score: {results['best_score']}")
    print(f"Total rollouts: {results['total_rollouts']}")

