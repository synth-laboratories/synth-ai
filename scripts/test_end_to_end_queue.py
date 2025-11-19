#!/usr/bin/env python3
"""End-to-end test script for experiment queue with full logging.

This script tests the complete flow:
1. Verify Redis is available
2. Check/kill existing workers
3. Start queue worker with Beat
4. Submit experiment to queue
5. Wait for job completion
6. Verify success

Run with: python scripts/test_end_to_end_queue.py
Or: uv run python scripts/test_end_to_end_queue.py
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Configure logging BEFORE any imports
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv

# Load .env file if it exists
env_file = repo_root / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)
    logger.info(f"Loaded .env file from {env_file}")
else:
    logger.warning(f".env file not found at {env_file}")

# Set a temporary DB path before imports to avoid module-level errors
# This will be overridden later with the actual test DB path
os.environ.setdefault("EXPERIMENT_QUEUE_DB_PATH", str(repo_root / "traces" / "v3" / "synth_ai.db"))
os.environ.setdefault("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")

# Now import project modules
from synth_ai.experiment_queue import celery_app as queue_celery
from synth_ai.experiment_queue import config as queue_config
from synth_ai.experiment_queue import database as queue_db
from synth_ai.experiment_queue import models as queue_models
from synth_ai.experiment_queue import service as queue_service
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest


def check_redis() -> bool:
    """Check if Redis is available."""
    logger.info("🔍 Checking Redis availability...")
    try:
        import redis
        redis_client = redis.Redis.from_url("redis://localhost:6379/0", socket_timeout=2)
        redis_client.ping()
        logger.info("✓ Redis broker is available")
        return True
    except Exception as e:
        logger.error(f"❌ Redis not available: {e}")
        logger.error("Start Redis with: brew services start redis")
        return False


def kill_existing_celery_workers() -> int:
    """Kill any existing Celery workers to avoid conflicts."""
    logger.info("🔍 Checking for existing Celery workers...")
    killed = 0
    
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if cmdline and any("celery" in str(arg).lower() for arg in cmdline):
                    if any("synth_ai.experiment_queue" in str(arg) for arg in cmdline):
                        logger.info(f"  Killing existing worker PID {proc.info['pid']}")
                        proc.kill()
                        killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # Fallback to pgrep/pkill
        try:
            result = subprocess.run(
                ["pgrep", "-f", "synth_ai.experiment_queue.celery_app"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        logger.info(f"  Killing existing worker PID {pid}")
                        os.kill(int(pid), signal.SIGTERM)
                        killed += 1
        except Exception as e:
            logger.warning(f"Could not check for existing workers: {e}")
    
    if killed > 0:
        logger.info(f"✓ Killed {killed} existing worker(s)")
        time.sleep(2)  # Give processes time to fully terminate
    else:
        logger.info("✓ No existing workers found")
    
    return killed


def wait_for_service(url: str, timeout: int = 30, api_key: str | None = None) -> bool:
    """Wait for a service to become available."""
    import urllib.request
    import urllib.error
    
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as response:
                # Consider 4xx/5xx as "service running" (just not healthy)
                logger.debug(f"Service responded with status {response.status}")
                return True
        except urllib.error.URLError:
            pass
        except Exception as e:
            logger.debug(f"Service check error: {e}")
        time.sleep(0.5)
    return False


def check_task_app() -> tuple[bool, str | None]:
    """Check if task app is running."""
    logger.info("🔍 Checking for task app on port 8102...")
    task_app_url = "http://127.0.0.1:8102"
    api_key = os.environ.get("ENVIRONMENT_API_KEY")
    
    if api_key:
        logger.info(f"  Using ENVIRONMENT_API_KEY (prefix: {api_key[:15]}...)")
    else:
        logger.warning("  ⚠ ENVIRONMENT_API_KEY not found in environment")
    
    task_app_running = wait_for_service(task_app_url, timeout=2, api_key=api_key)
    
    if task_app_running:
        logger.info(f"✓ Task app is running at {task_app_url}")
        return True, api_key
    else:
        logger.warning(f"⚠ Task app not running on port 8102")
        logger.warning("Start it with: uv run synth-ai deploy --task-app examples/task_apps/banking77/banking77_task_app.py --runtime local --port 8102 --env .env")
        return False, api_key


def wait_for_experiment_completion(
    experiment_id: str,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> queue_models.Experiment | None:
    """Wait for experiment to complete."""
    logger.info(f"⏳ Waiting for experiment {experiment_id} to complete (timeout: {timeout}s / {timeout/60:.1f} minutes)...")
    start_time = time.time()
    last_status = None
    last_log_time = start_time
    
    while time.time() - start_time < timeout:
        with queue_db.session_scope() as session:
            experiment = session.get(queue_models.Experiment, experiment_id)
            if experiment:
                status = experiment.status
                elapsed = time.time() - start_time
                
                # Log status changes or every 30 seconds
                if status != last_status or (time.time() - last_log_time) >= 30:
                    logger.info(f"  Status: {status.value} (elapsed: {int(elapsed)}s / {elapsed/60:.1f}m)")
                    
                    # Show job progress
                    for job in experiment.jobs:
                        job_status = job.status.value if job.status else "unknown"
                        if job.celery_task_id:
                            logger.debug(f"    Job {job.job_id[:8]}...: {job_status}")
                    
                    last_status = status
                    last_log_time = time.time()
                
                if status in [queue_models.ExperimentStatus.COMPLETED, queue_models.ExperimentStatus.FAILED]:
                    elapsed = time.time() - start_time
                    logger.info(f"✓ Experiment {status.value} after {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
                    return experiment
        
        time.sleep(poll_interval)
    
    logger.error(f"❌ Experiment did not complete within {timeout}s ({timeout/60:.1f} minutes)")
    return None


def main() -> int:
    """Run end-to-end test."""
    logger.info("=" * 80)
    logger.info("END-TO-END EXPERIMENT QUEUE TEST")
    logger.info("=" * 80)
    
    # Step 1: Verify Redis
    if not check_redis():
        logger.error("❌ Redis check failed. Aborting.")
        return 1
    
    # Step 2: Check task app
    task_app_running, api_key = check_task_app()
    if not task_app_running:
        logger.warning("⚠ Task app not running, but continuing anyway...")
        logger.warning("  The test will fail if task app is required for job execution")
    
    # Step 3: Kill existing workers
    kill_existing_celery_workers()
    
    # Step 4: Set up database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "queue.db"
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db_path)
        os.environ["EXPERIMENT_QUEUE_BROKER_URL"] = "redis://localhost:6379/0"
        os.environ["EXPERIMENT_QUEUE_RESULT_BACKEND_URL"] = "redis://localhost:6379/1"
        
        logger.info(f"📁 Using database: {db_path}")
        logger.info(f"📁 EXPERIMENT_QUEUE_DB_PATH: {os.environ['EXPERIMENT_QUEUE_DB_PATH']}")
        
        # Reset config cache and reload modules
        queue_config.reset_config_cache()
        modules = [
            queue_config,
            queue_db,
            queue_models,
            queue_celery,
            queue_service,
        ]
        for module in modules:
            importlib.reload(module)
        
        queue_db.init_db()
        logger.info("✓ Database initialized")
        
        # Step 5: Find config file
        banking77_config = repo_root / "examples" / "blog_posts" / "langprobe" / "task_specific" / "banking77" / "banking77_gepa.toml"
        if not banking77_config.exists():
            logger.error(f"❌ Banking77 config not found: {banking77_config}")
            return 1
        
        logger.info(f"📄 Using config: {banking77_config}")
        
        # Step 6: Start queue worker
        logger.info("🚀 Starting queue worker with Beat...")
        import shutil
        celery_bin = shutil.which("celery")
        if not celery_bin:
            queue_cmd = [
                "uv", "run", "celery",
                "-A", "synth_ai.experiment_queue.celery_app",
                "worker", "--beat",
                "--pool", "solo",
                "--concurrency", "1",
                "--loglevel", "info",  # Full logging
            ]
        else:
            queue_cmd = [
                celery_bin,
                "-A", "synth_ai.experiment_queue.celery_app",
                "worker", "--beat",
                "--pool", "solo",
                "--concurrency", "1",
                "--loglevel", "info",  # Full logging
            ]
        
        queue_env = os.environ.copy()
        queue_env["PYTHONUNBUFFERED"] = "1"
        
        logger.info(f"  Command: {' '.join(queue_cmd)}")
        logger.info(f"  DB path: {db_path}")
        
        queue_proc = subprocess.Popen(
            queue_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            env=queue_env,
            start_new_session=True,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        logger.info(f"  Worker PID: {queue_proc.pid}")
        logger.info("  Waiting for worker to initialize...")
        time.sleep(2)
        
        if queue_proc.poll() is not None:
            stdout, _ = queue_proc.communicate(timeout=1)
            logger.error(f"❌ Worker process exited immediately!")
            if stdout:
                logger.error(f"Output:\n{stdout[:2000]}")
            return 1
        
        logger.info("✓ Worker started successfully")
        
        # Start a thread to log worker output
        import threading
        
        def log_worker_output():
            if queue_proc.stdout:
                for line in iter(queue_proc.stdout.readline, ""):
                    if line:
                        logger.info(f"[WORKER] {line.rstrip()}")
        
        worker_log_thread = threading.Thread(target=log_worker_output, daemon=True)
        worker_log_thread.start()
        
        try:
            # Step 7: Submit experiment with REAL GEPA parameters
            logger.info("📝 Submitting experiment with REAL GEPA run (200 rollouts)...")
            request = ExperimentSubmitRequest.model_validate({
                "name": "E2E Real GEPA Test - Banking77",
                "description": "End-to-end test with full GEPA run (200 rollouts)",
                "parallelism": 1,
                "jobs": [
                    {
                        "job_type": "gepa",
                        "config_path": str(banking77_config),
                        "config_overrides": {
                            # Real GEPA parameters - nested under prompt_learning sections
                            # Config already has budget=200, but we ensure it and increase cost/time limits
                            "prompt_learning": {
                                "gepa": {
                                    "rollout": {
                                        "budget": 200,  # Full 200 rollouts (already in config)
                                    },
                                },
                                "termination_config": {
                                    "max_cost_usd": 20.0,  # Higher cost limit for real run (config has 1.0)
                                    "time_limit_seconds": 7200,  # 2 hours max for real run
                                },
                            },
                        },
                    },
                ],
            })
            
            experiment = queue_service.create_experiment(request)
            assert experiment is not None, "Experiment should be created"
            assert experiment.status == queue_models.ExperimentStatus.QUEUED, f"Expected QUEUED, got {experiment.status}"
            
            logger.info(f"✓ Experiment created: {experiment.experiment_id}")
            logger.info(f"  Status: {experiment.status}")
            logger.info(f"  Jobs: {len(experiment.jobs)}")
            
            # Check if job was dispatched and show config_overrides
            time.sleep(2)
            with queue_db.session_scope() as session:
                job = session.get(queue_models.ExperimentJob, experiment.jobs[0].job_id)
                if job:
                    logger.info(f"  Job status: {job.status}")
                    logger.info(f"  Job celery_task_id: {job.celery_task_id}")
                    logger.info(f"  Config path: {job.config_path}")
                    logger.info(f"  Config overrides: {job.config_overrides}")
                    if job.celery_task_id:
                        logger.info("✓ Job was dispatched to Celery")
                    else:
                        logger.warning("⚠ Job not yet dispatched (will be picked up by periodic task)")
            
            # Step 8: Wait for completion (longer timeout for real GEPA run)
            logger.info("⏳ This will take a while - GEPA with 200 rollouts...")
            logger.info("   You can monitor progress in the worker logs above")
            completed_experiment = wait_for_experiment_completion(
                experiment.experiment_id,
                timeout=7200,  # 2 hours for real GEPA run
                poll_interval=10.0,  # Check every 10 seconds
            )
            
            if not completed_experiment:
                logger.error("❌ Experiment did not complete")
                return 1
            
            # Step 9: Verify success
            logger.info("🔍 Verifying experiment results...")
            with queue_db.session_scope() as session:
                experiment = session.get(queue_models.Experiment, experiment.experiment_id)
                if experiment:
                    logger.info(f"  Final status: {experiment.status}")
                    logger.info(f"  Jobs: {len(experiment.jobs)}")
                    
                    for job in experiment.jobs:
                        logger.info(f"    Job {job.job_id}: {job.status}")
                        if job.error:
                            logger.warning(f"      Error: {job.error}")
                        if job.result:
                            result = job.result
                            logger.info(f"      Return code: {result.get('returncode', 'N/A')}")
                            logger.info(f"      Total rollouts: {result.get('total_rollouts', 'N/A')}")
                            logger.info(f"      Best score: {result.get('best_score', 'N/A')}")
                            if result.get('stderr'):
                                logger.warning(f"      Stderr (last 500 chars): {result['stderr'][-500:]}")
                            if result.get('stdout'):
                                logger.info(f"      Stdout (last 500 chars): {result['stdout'][-500:]}")
                    
                    if experiment.status == queue_models.ExperimentStatus.COMPLETED:
                        logger.info("=" * 80)
                        logger.info("✅ END-TO-END TEST PASSED")
                        logger.info("=" * 80)
                        return 0
                    else:
                        logger.error("=" * 80)
                        logger.error(f"❌ END-TO-END TEST FAILED: Experiment status is {experiment.status}")
                        logger.error("=" * 80)
                        return 1
            
            logger.error("❌ Could not fetch experiment results")
            return 1
            
        finally:
            # Cleanup
            logger.info("🧹 Cleaning up...")
            try:
                queue_proc.terminate()
                queue_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("  Worker did not terminate gracefully, killing...")
                queue_proc.kill()
                queue_proc.wait()
            except Exception as e:
                logger.warning(f"  Error stopping worker: {e}")


if __name__ == "__main__":
    sys.exit(main())

