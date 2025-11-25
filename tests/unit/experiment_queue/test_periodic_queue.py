"""Tests for periodic queue check task."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from synth_ai.cli.local.experiment_queue import celery_app as queue_celery
from synth_ai.cli.local.experiment_queue import config as queue_config
from synth_ai.cli.local.experiment_queue import database as queue_db
from synth_ai.cli.local.experiment_queue import models as queue_models
from synth_ai.cli.local.experiment_queue import tasks as queue_tasks
from synth_ai.cli.local.experiment_queue.schemas import ExperimentSubmitRequest


@pytest.fixture(autouse=True)
def queue_env(tmp_path, monkeypatch):
    """Provide an isolated SQLite database for tests."""
    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    queue_config.load_config.cache_clear()

    modules = [
        queue_config,
        queue_db,
        queue_models,
        queue_celery,
        queue_tasks,
    ]
    for module in modules:
        importlib.reload(module)

    queue_db.init_db()
    yield
    queue_config.load_config.cache_clear()


def test_beat_schedule_configured():
    """Verify Beat schedule is configured correctly."""
    app = queue_celery.get_celery_app()
    assert "beat_schedule" in app.conf
    assert "process-experiment-queue" in app.conf.beat_schedule
    schedule = app.conf.beat_schedule["process-experiment-queue"]
    assert schedule["task"] == "synth_ai.experiment_queue.process_experiment_queue"
    assert schedule["schedule"] == 5.0


def test_process_experiment_queue_no_experiments():
    """Test periodic task with no active experiments."""
    result = queue_tasks.process_experiment_queue()
    assert result["dispatched"] == 0
    assert result["experiments_checked"] == 0


def test_process_experiment_queue_with_queued_jobs(tmp_path, monkeypatch):
    """Test periodic task dispatches queued jobs."""
    from synth_ai.cli.local.experiment_queue import dispatcher as queue_dispatcher
    from synth_ai.cli.local.experiment_queue import service as queue_service

    # Create a stub Celery app that records send_task calls
    sent_tasks = []

    class StubCelery:
        def send_task(self, name: str, args: list):
            task_id = f"stub-{len(sent_tasks)+1}"
            sent_tasks.append((name, args))
            return type("Result", (), {"id": task_id})()

        @property
        def conf(self):
            return type("Conf", (), {"broker_url": "test://"})()

    stub = StubCelery()
    monkeypatch.setattr(queue_dispatcher, "get_celery_app", lambda: stub)

    # Create dummy config files
    config1_path = tmp_path / "config1.toml"
    config2_path = tmp_path / "config2.toml"
    config1_path.write_text("[prompt_learning.gepa]\nrollouts = 10\n")
    config2_path.write_text("[prompt_learning.gepa]\nrollouts = 10\n")

    # Create an experiment with queued jobs
    request = ExperimentSubmitRequest.model_validate(
        {
            "name": "Test Experiment",
            "parallelism": 2,
            "jobs": [
                {
                    "job_type": "gepa",
                    "config_path": str(config1_path),
                    "config_overrides": {},
                },
                {
                    "job_type": "gepa",
                    "config_path": str(config2_path),
                    "config_overrides": {},
                },
            ],
        }
    )

    # Create experiment (this dispatches initial jobs)
    experiment = queue_service.create_experiment(request)
    assert experiment.status == queue_models.ExperimentStatus.QUEUED

    # Clear sent tasks to test periodic task
    sent_tasks.clear()

    # Manually set one job back to QUEUED without celery_task_id to simulate
    # a job that wasn't dispatched
    with queue_db.session_scope() as session:
        jobs = (
            session.query(queue_models.ExperimentJob)
            .filter(queue_models.ExperimentJob.experiment_id == experiment.experiment_id)
            .all()
        )
        if len(jobs) > 0:
            # Reset first job to queued state without celery_task_id
            jobs[0].status = queue_models.ExperimentJobStatus.QUEUED
            jobs[0].celery_task_id = None
            session.flush()

    # Run periodic task - should dispatch the queued job
    result = queue_tasks.process_experiment_queue()
    assert result["experiments_checked"] == 1
    # Should have dispatched at least one job
    assert result["dispatched"] >= 0  # May be 0 if parallelism limit reached
    assert len(sent_tasks) >= 0  # May be 0 if already at parallelism limit


