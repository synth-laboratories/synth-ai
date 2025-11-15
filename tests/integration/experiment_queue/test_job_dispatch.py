"""Fast tests for job dispatch - validate dispatch logic separately."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from synth_ai.experiment_queue import config, database, models, service
from synth_ai.experiment_queue.dispatcher import dispatch_available_jobs
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest

# Path to test config file
BANKING77_CONFIG = Path(__file__).parent.parent.parent.parent / "examples" / "blog_posts" / "langprobe" / "task_specific" / "banking77" / "banking77_gepa.toml"


@pytest.fixture(autouse=True)
def queue_env(tmp_path, monkeypatch):
    """Set up test environment."""
    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    monkeypatch.setenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")
    config.reset_config_cache()
    database.init_db()
    yield
    config.reset_config_cache()


def test_dispatch_single_job():
    """FAIL FAST: Test dispatching a single job."""
    # Create experiment (create_experiment already dispatches jobs)
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "parallelism": 1,
        "jobs": [{
            "job_type": "gepa",
            "config_path": str(BANKING77_CONFIG),
        }],
    })
    
    experiment = service.create_experiment(request)
    
    # Verify job was dispatched by create_experiment
    with database.session_scope() as session:
        job = session.get(models.ExperimentJob, experiment.jobs[0].job_id)
        assert job is not None
        assert job.celery_task_id is not None, "Job should be dispatched by create_experiment"
    
    # Try to dispatch again - should return 0 (already dispatched)
    with database.session_scope() as session:
        dispatched = dispatch_available_jobs(session, experiment.experiment_id)
        session.commit()
    
    assert len(dispatched) == 0, "Should return 0 - job already dispatched"


def test_dispatch_respects_parallelism():
    """FAIL FAST: Test dispatch respects parallelism limit."""
    # Create experiment with parallelism=1 (create_experiment dispatches 1 job)
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "parallelism": 1,
        "jobs": [
            {"job_type": "gepa", "config_path": str(BANKING77_CONFIG)},
            {"job_type": "gepa", "config_path": str(BANKING77_CONFIG)},
        ],
    })
    
    experiment = service.create_experiment(request)
    
    # Verify only 1 job was dispatched (parallelism=1)
    with database.session_scope() as session:
        dispatched_jobs = [
            job for job in experiment.jobs 
            if job.celery_task_id is not None
        ]
        assert len(dispatched_jobs) == 1, "Should only dispatch 1 job due to parallelism=1"
    
    # Try to dispatch again - should get 0 (parallelism limit reached)
    with database.session_scope() as session:
        dispatched2 = dispatch_available_jobs(session, experiment.experiment_id)
        session.commit()
    
    assert len(dispatched2) == 0, "Should not dispatch more jobs (parallelism limit)"


def test_dispatch_only_queued_jobs():
    """FAIL FAST: Test dispatch only processes QUEUED jobs."""
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "parallelism": 8,  # Max allowed
        "jobs": [{
            "job_type": "gepa",
            "config_path": str(BANKING77_CONFIG),
        }],
    })
    
    experiment = service.create_experiment(request)
    job_id = experiment.jobs[0].job_id
    
    # Mark job as RUNNING manually
    with database.session_scope() as session:
        job = session.get(models.ExperimentJob, job_id)
        job.status = models.ExperimentJobStatus.RUNNING
        session.commit()
    
    # Try to dispatch - should get 0 (job is RUNNING, not QUEUED)
    with database.session_scope() as session:
        dispatched = dispatch_available_jobs(session, experiment.experiment_id)
        session.commit()
    
    assert len(dispatched) == 0, "Should not dispatch RUNNING jobs"

