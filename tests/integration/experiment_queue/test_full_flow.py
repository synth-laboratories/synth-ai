"""Full integration tests for experiment queue with task app deployment.

These tests verify the complete flow:
1. Start task app (via deploy)
2. Start queue worker with Beat
3. Submit experiment to queue
4. Wait for GEPA completion
5. Verify success
"""

from __future__ import annotations

import importlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from synth_ai.experiment_queue import celery_app as queue_celery
from synth_ai.experiment_queue import config as queue_config
from synth_ai.experiment_queue import database as queue_db
from synth_ai.experiment_queue import models as queue_models
from synth_ai.experiment_queue import service as queue_service
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest


@pytest.fixture(autouse=True)
def queue_env(tmp_path, monkeypatch):
    """Provide an isolated SQLite database for tests and Redis broker."""
    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    # Use Redis for Celery broker (defaults to localhost:6379)
    monkeypatch.setenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")
    queue_config.load_config.cache_clear()

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
    yield
    queue_config.load_config.cache_clear()


@pytest.fixture
def test_task_app_dir(tmp_path) -> Path:
    """Create a minimal task app for testing."""
    app_dir = tmp_path / "task_app"
    app_dir.mkdir()
    
    # Create a simple task app Python file
    task_app_py = app_dir / "task_app.py"
    task_app_py.write_text("""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HealthResponse(BaseModel):
    status: str = "ok"

@app.get("/health")
def health():
    return HealthResponse()

@app.post("/rollout")
def rollout():
    return {"status": "success", "reward": 0.5}
""")
    
    return app_dir


@pytest.fixture
def test_env_file(tmp_path) -> Path:
    """Create a test .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("""ENVIRONMENT_API_KEY=test_key_123
SYNTH_API_KEY=test_synth_key_456
""")
    return env_file


@pytest.fixture
def test_gepa_config(tmp_path, test_env_file) -> Path:
    """Create a minimal GEPA config TOML."""
    config_file = tmp_path / "gepa_config.toml"
    config_file.write_text(f"""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://127.0.0.1:8114"
task_app_api_key = "${{ENVIRONMENT_API_KEY}}"
env_file_path = "{test_env_file}"
results_folder = "results"

[prompt_learning.initial_prompt]
id = "test_pattern"
name = "Test Pattern"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a test assistant."
order = 0

[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "llama-3.1-8b-instant"
provider = "groq"
temperature = 0.0

[prompt_learning.gepa]
env_name = "test"

[prompt_learning.gepa.evaluation]
train_seeds = [0, 1]
val_seeds = [2, 3]
validation_pool = "train"

[prompt_learning.gepa.rollout]
budget = 10
max_concurrent = 2

[prompt_learning.gepa.population]
initial_size = 2
num_generations = 1
children_per_generation = 2

[prompt_learning.termination_config]
max_cost_usd = 0.1
max_trials = 5
rollout_limit = 5

[display]
local_backend = true
tui = false
""")
    return config_file


def kill_existing_celery_workers() -> int:
    """Kill any existing Celery workers to avoid conflicts.
    
    Returns the number of workers killed.
    """
    killed = 0
    try:
        # Try using psutil first (more reliable)
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue
                # Check if this is a Celery worker for our experiment queue
                cmdline_str = ' '.join(cmdline)
                if 'celery' in cmdline_str.lower() and 'synth_ai.experiment_queue' in cmdline_str:
                    print(f"  Killing existing Celery worker (PID: {proc.info['pid']})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except ImportError:
        # Fallback to using pgrep/pkill if psutil not available
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'celery.*synth_ai.experiment_queue'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        print(f"  Killing existing Celery worker (PID: {pid})")
                        try:
                            subprocess.run(['kill', '-TERM', pid], timeout=3)
                            time.sleep(1)
                            # Force kill if still running
                            subprocess.run(['kill', '-9', pid], timeout=1)
                            killed += 1
                        except Exception:
                            pass
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠ Could not check for existing workers: {e}")
    if killed > 0:
        time.sleep(0.5)  # FAIL FAST: Reduced wait
    return killed


class ProcessManager:
    """Helper to manage background processes."""
    
    def __init__(self):
        self.processes: list[subprocess.Popen[Any]] = []
    
    def start(self, cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        """Start a process and track it."""
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs
        )
        self.processes.append(proc)
        return proc
    
    def stop_all(self):
        """Stop all tracked processes."""
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        self.processes.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.stop_all()


def wait_for_service(url: str, timeout: int = 5, api_key: str | None = None) -> bool:  # FAIL FAST: Reduced timeout
    """Wait for a service to become available."""
    import urllib.request
    import urllib.error
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health")
            if api_key:
                # Send API key header (case-insensitive, but use standard case)
                req.add_header("X-API-Key", api_key)
            with urllib.request.urlopen(req, timeout=2) as response:
                # 200 means healthy
                if response.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            # 200 means healthy
            if e.code == 200:
                return True
            # For 4xx/5xx, service is running and responding (even if auth failed)
            # This means the service is up, just needs proper auth
            if 400 <= e.code < 600:
                return True
        except Exception as e:
            # For connection errors, keep trying
            if "timeout" not in str(e).lower() and "refused" not in str(e).lower():
                # Unexpected error - log but continue
                pass
        time.sleep(0.1)  # FAIL FAST: Minimal wait
    return False


def wait_for_experiment_completion(
    experiment_id: str,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> queue_models.Experiment | None:
    """Wait for an experiment to complete."""
    start = time.time()
    while time.time() - start < timeout:
        with queue_db.session_scope() as session:
            experiment = session.get(queue_models.Experiment, experiment_id)
            if experiment and experiment.status in [
                queue_models.ExperimentStatus.COMPLETED,
                queue_models.ExperimentStatus.FAILED,
                queue_models.ExperimentStatus.CANCELED,
            ]:
                return experiment
        time.sleep(poll_interval)
    return None


# DELETED: Slow integration test - use scripts/test_end_to_end_queue.py instead


@pytest.mark.integration
def test_experiment_submission_validates_files(tmp_path, test_gepa_config, test_env_file):
    """Test that experiment submission validates required files exist."""
    
    # Test with missing config file
    with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
        request = ExperimentSubmitRequest.model_validate({
            "name": "Test",
            "jobs": [
                {
                    "job_type": "gepa",
                    "config_path": str(tmp_path / "missing.toml"),
                },
            ],
        })
        queue_service.create_experiment(request)
    
    # Test with valid config file
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "jobs": [
            {
                "job_type": "gepa",
                "config_path": str(test_gepa_config),
            },
        ],
    })
    
    # Should succeed if config file exists
    experiment = queue_service.create_experiment(request)
    assert experiment is not None


def test_config_file_references_env_file(tmp_path, test_env_file):
    """Test that config file can reference env_file_path."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(f"""
[prompt_learning]
env_file_path = "{test_env_file}"
task_app_url = "http://127.0.0.1:8114"
""")
    
    from synth_ai.experiment_queue.config_utils import prepare_config_file
    
    prepared = prepare_config_file(config_file)
    assert prepared.path.exists()
    
    # Verify env_file_path was normalized
    try:
        import tomllib  # Python 3.11+
        with open(prepared.path, "rb") as f:
            config = tomllib.load(f)
    except ImportError:
        import tomli
        with open(prepared.path, "rb") as f:
            config = tomli.load(f)
    
    env_path = config.get("prompt_learning", {}).get("env_file_path")
    assert env_path is not None
    assert Path(env_path).is_absolute(), "env_file_path should be absolute"
    assert Path(env_path).exists(), "env_file_path should exist"

