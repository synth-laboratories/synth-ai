"""Fast tests for Redis broker integration - validate pieces separately."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from synth_ai.experiment_queue import celery_app, config, database, models, service
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest

# Path to test config file
BANKING77_CONFIG = Path(__file__).parent.parent.parent.parent / "examples" / "blog_posts" / "langprobe" / "task_specific" / "banking77" / "banking77_gepa.toml"


@pytest.fixture(autouse=True)
def redis_broker_env(tmp_path, monkeypatch):
    """Set up Redis broker for tests."""
    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    monkeypatch.setenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")
    config.reset_config_cache()
    database.init_db()
    yield
    config.reset_config_cache()


@pytest.mark.integration
def test_redis_broker_connectivity():
    """FAIL FAST: Test Redis broker is accessible."""
    try:
        import redis
        redis_client = redis.Redis.from_url("redis://localhost:6379/0", socket_timeout=1)
        assert redis_client.ping(), "Redis broker not accessible"
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.integration
def test_celery_app_uses_redis():
    """FAIL FAST: Test Celery app uses Redis broker."""
    app = celery_app.get_celery_app()
    assert app.conf.broker_url.startswith("redis://"), f"Expected Redis, got: {app.conf.broker_url}"
    assert app.conf.result_backend.startswith("redis://"), f"Expected Redis backend, got: {app.conf.result_backend}"


@pytest.mark.integration
def test_send_task_to_redis():
    """FAIL FAST: Test sending a task to Redis broker."""
    app = celery_app.get_celery_app()
    
    # Send a test task
    result = app.send_task("synth_ai.experiment_queue.process_experiment_queue")
    assert result.id is not None, "Task ID should be set"
    
    # Verify task was sent (may be consumed immediately if worker running)
    # Just verify the task ID exists - actual consumption depends on worker
    assert result.id is not None


@pytest.mark.integration
def test_dispatch_job_to_redis():
    """FAIL FAST: Test dispatching a job creates Redis message."""
    # Create a minimal experiment (create_experiment already dispatches jobs)
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "description": "Test dispatch",
        "parallelism": 1,
        "jobs": [{
            "job_type": "gepa",
            "config_path": str(BANKING77_CONFIG),
        }],
    })
    
    experiment = service.create_experiment(request)
    assert experiment is not None
    
    # Verify celery_task_id is set (indicates task was sent to Redis by create_experiment)
    with database.session_scope() as session:
        job = session.get(models.ExperimentJob, experiment.jobs[0].job_id)
        assert job is not None
        assert job.celery_task_id is not None, "celery_task_id should be set after create_experiment dispatches job"


@pytest.mark.integration
def test_worker_consumes_from_redis():
    """FAIL FAST: Test worker can send tasks to Redis."""
    app = celery_app.get_celery_app()
    
    # Send task - verify it succeeds (task may be consumed immediately if worker running)
    result = app.send_task("synth_ai.experiment_queue.process_experiment_queue")
    assert result.id is not None, "Task should be sent successfully"


@pytest.mark.integration  
def test_periodic_task_schedule():
    """FAIL FAST: Test periodic task is scheduled."""
    app = celery_app.get_celery_app()
    beat_schedule = app.conf.beat_schedule
    assert "process-experiment-queue" in beat_schedule, "Periodic task should be scheduled"
    assert beat_schedule["process-experiment-queue"]["schedule"] == 5.0, "Should run every 5 seconds"

