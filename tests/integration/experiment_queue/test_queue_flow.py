from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from synth_ai.experiment_queue import celery_app as queue_celery
from synth_ai.experiment_queue import config as queue_config
from synth_ai.experiment_queue import database as queue_db
from synth_ai.experiment_queue import dispatcher as queue_dispatcher
from synth_ai.experiment_queue import models as queue_models
from synth_ai.experiment_queue import results as queue_results
from synth_ai.experiment_queue import service as queue_service
from synth_ai.experiment_queue import tasks as queue_tasks
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest
from synth_ai.experiment_queue.config_utils import PreparedConfig


@pytest.fixture(autouse=True)
def queue_env(tmp_path, monkeypatch):
    """
    Provide an isolated SQLite database for experiment queue tests and reload modules
    so Celery + ORM pick up the new location.
    """

    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    queue_config.load_config.cache_clear()

    modules = [
        queue_config,
        queue_db,
        queue_models,
        queue_results,
        queue_celery,
        queue_dispatcher,
        queue_service,
        queue_tasks,
    ]
    for module in modules:
        importlib.reload(module)

    queue_db.init_db()
    yield
    queue_config.load_config.cache_clear()


@pytest.fixture
def stub_celery(monkeypatch):
    """Stub Celery app that records send_task and revoke calls."""

    class StubCelery:
        def __init__(self):
            self.sent: list[tuple[str, list[Any]]] = []
            self.revoked: list[str] = []

            class Control:
                def __init__(self, outer: "StubCelery"):
                    self._outer = outer

                def revoke(self, task_id: str, terminate: bool = True):
                    self._outer.revoked.append(task_id)

            self.control = Control(self)

        def send_task(self, name: str, args: list[Any]):
            task_id = f"stub-{len(self.sent)+1}"
            self.sent.append((name, args))
            return SimpleNamespace(id=task_id)

    stub = StubCelery()
    monkeypatch.setattr(queue_dispatcher, "get_celery_app", lambda: stub)
    return stub


def _make_request(tmp_path: Path | None = None, num_jobs: int = 3) -> ExperimentSubmitRequest:
    """Create a test request with config files."""
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())
    
    jobs = []
    for idx in range(num_jobs):
        config_path = tmp_path / f"config_{idx}.toml"
        config_path.write_text("[prompt_learning.gepa]\nrollouts = 10\n", encoding="utf-8")
        jobs.append(
            {
                "job_type": "gepa" if idx % 2 == 0 else "mipro",
                "config_path": str(config_path),
                "config_overrides": {"max_rollouts": 5 + idx},
            }
        )
    payload = {
        "name": "Integration Test",
        "description": "ensure queue wiring works",
        "parallelism": 2,
        "jobs": jobs,
    }
    return ExperimentSubmitRequest.model_validate(payload)


def _stub_training_run(monkeypatch, tmp_path, *, returncode: int = 0, summary: queue_results.ResultSummary | None = None):
    """Stub out training command execution for Celery task tests."""
    prepared_dir = tmp_path / f"prepared_{returncode}"
    prepared_dir.mkdir()
    prepared_config = PreparedConfig(
        path=prepared_dir / "config.toml",
        results_folder=prepared_dir / "results",
        workdir=prepared_dir,
    )
    prepared_config.path.write_text("[prompt_learning]\n", encoding="utf-8")
    prepared_config.results_folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(queue_tasks, "prepare_config_file", lambda *_, **__: prepared_config)

    if summary is None:
        summary = queue_results.ResultSummary()
    monkeypatch.setattr(queue_tasks, "collect_result_summary", lambda *args, **kwargs: summary)

    class DummyProcess:
        def __init__(self):
            self.returncode = returncode
            self.stdout = "ok"
            self.stderr = "error" if returncode else ""

    monkeypatch.setattr(queue_tasks.subprocess, "run", lambda *_, **__: DummyProcess())
    return summary


def test_create_experiment_respects_parallelism(stub_celery, tmp_path):
    request = _make_request(tmp_path)
    experiment = queue_service.create_experiment(request)

    assert len(stub_celery.sent) == 2, "Should dispatch only up to parallelism limit"

    with queue_db.session_scope() as session:
        jobs = (
            session.query(queue_models.ExperimentJob)
            .filter(queue_models.ExperimentJob.experiment_id == experiment.experiment_id)
            .order_by(queue_models.ExperimentJob.created_at.asc())
            .all()
        )

        running = [job for job in jobs if job.celery_task_id]
        queued = [job for job in jobs if not job.celery_task_id]

        assert len(running) == 2
        assert len(queued) == 1
        assert all(job.status == queue_models.ExperimentJobStatus.QUEUED for job in jobs)


def test_run_experiment_job_success(tmp_path, monkeypatch):
    summary = queue_results.ResultSummary(
        learning_curve_points=[
            queue_results.LearningCurvePoint(
                rollout_count=1,
                performance=0.4,
                metadata={"system_name": "sys_a"},
            ),
            queue_results.LearningCurvePoint(
                rollout_count=2,
                performance=0.7,
                metadata={"system_name": "sys_b"},
            ),
        ],
        total_rollouts=2,
        best_score=0.7,
        baseline_score=0.4,
    )
    _stub_training_run(monkeypatch, tmp_path, summary=summary)

    config_file = tmp_path / "config.toml"
    config_file.write_text("[prompt_learning]\nresults_folder = \"results\"\n", encoding="utf-8")

    with queue_db.session_scope() as session:
        experiment = queue_models.Experiment(
            experiment_id="exp-test",
            name="Test Experiment",
            description="",
            status=queue_models.ExperimentStatus.QUEUED,
            parallelism_limit=1,
            config_toml="name = 'test'",
        )
        job = queue_models.ExperimentJob(
            job_id="job-test",
            experiment_id="exp-test",
            job_type=queue_models.ExperimentJobType.GEPA,
            config_path=str(config_file),
            config_overrides={},
            status=queue_models.ExperimentJobStatus.QUEUED,
        )
        session.add_all([experiment, job])

    result = queue_tasks.run_experiment_job.apply(args=["job-test"]).get()
    assert result["best_score"] == 0.7

    with queue_db.session_scope() as session:
        refetched = session.get(queue_models.ExperimentJob, "job-test")
        assert refetched.status == queue_models.ExperimentJobStatus.COMPLETED
        experiment = session.get(queue_models.Experiment, "exp-test")
        assert experiment.status == queue_models.ExperimentStatus.COMPLETED
        trials = (
            session.query(queue_models.Trial)
            .filter(queue_models.Trial.experiment_id == "exp-test")
            .all()
        )
        assert len(trials) == 2
        assert all(trial.status == queue_models.TrialStatus.COMPLETED for trial in trials)
        assert trials[0].system_name == "sys_a"
        aggregate = experiment.metadata_json.get("aggregate", {})
        assert aggregate.get("best_score") == 0.7
        assert aggregate.get("baseline_score") == 0.4


def test_run_experiment_job_failure_marks_job_failed(tmp_path, monkeypatch):
    """Test that job failure marks job as FAILED but experiment continues (only fails when ALL jobs fail)."""
    _stub_training_run(
        monkeypatch,
        tmp_path,
        returncode=1,
        summary=queue_results.ResultSummary(),
    )

    with queue_db.session_scope() as session:
        experiment = queue_models.Experiment(
            experiment_id="exp-fail",
            name="Experiment Failure",
            description="",
            status=queue_models.ExperimentStatus.QUEUED,
            parallelism_limit=1,
            config_toml="{}",
        )
        job1 = queue_models.ExperimentJob(
            job_id="job-fail-1",
            experiment_id="exp-fail",
            job_type=queue_models.ExperimentJobType.GEPA,
            config_path="missing.toml",
            config_overrides={},
            status=queue_models.ExperimentJobStatus.QUEUED,
        )
        job2 = queue_models.ExperimentJob(
            job_id="job-fail-2",
            experiment_id="exp-fail",
            job_type=queue_models.ExperimentJobType.MIPRO,
            config_path="missing2.toml",
            config_overrides={},
            status=queue_models.ExperimentJobStatus.QUEUED,
        )
        session.add_all([experiment, job1, job2])

    queue_tasks.run_experiment_job.apply(args=["job-fail-1"]).get()

    with queue_db.session_scope() as session:
        experiment = session.get(queue_models.Experiment, "exp-fail")
        # Experiment should still be QUEUED/RUNNING since not all jobs failed yet
        assert experiment.status in (queue_models.ExperimentStatus.QUEUED, queue_models.ExperimentStatus.RUNNING)
        job1 = session.get(queue_models.ExperimentJob, "job-fail-1")
        job2 = session.get(queue_models.ExperimentJob, "job-fail-2")
        assert job1.status == queue_models.ExperimentJobStatus.FAILED
        assert job2.status == queue_models.ExperimentJobStatus.QUEUED  # Still queued, not canceled


def test_dispatches_next_job_when_slot_available(tmp_path, stub_celery, monkeypatch):
    summary = queue_results.ResultSummary(
        learning_curve_points=[
            queue_results.LearningCurvePoint(rollout_count=1, performance=0.5),
        ],
        total_rollouts=1,
        best_score=0.5,
    )
    _stub_training_run(monkeypatch, tmp_path, summary=summary)

    request = _make_request(tmp_path, num_jobs=2)
    request.parallelism = 1
    experiment = queue_service.create_experiment(request)

    assert len(stub_celery.sent) == 1
    first_job_id = stub_celery.sent[0][1][0]

    queue_tasks.run_experiment_job.apply(args=[first_job_id]).get()

    assert len(stub_celery.sent) == 2, "Second job should be dispatched after completion"

    with queue_db.session_scope() as session:
        queued_jobs = (
            session.query(queue_models.ExperimentJob)
            .filter(queue_models.ExperimentJob.experiment_id == experiment.experiment_id)
            .all()
        )
        dispatched = [job for job in queued_jobs if job.celery_task_id]
        assert len(dispatched) >= 1


def test_experiment_results_cli_after_completion(tmp_path, stub_celery, cli_entry, monkeypatch):
    summary = queue_results.ResultSummary(
        learning_curve_points=[
            queue_results.LearningCurvePoint(rollout_count=1, performance=0.3),
            queue_results.LearningCurvePoint(rollout_count=2, performance=0.8),
        ],
        total_rollouts=2,
        best_score=0.8,
        baseline_score=0.3,
    )
    _stub_training_run(monkeypatch, tmp_path, summary=summary)

    request = _make_request(tmp_path, num_jobs=1)
    experiment = queue_service.create_experiment(request)
    job_id = stub_celery.sent[0][1][0]
    queue_tasks.run_experiment_job.apply(args=[job_id]).get()

    runner = CliRunner()
    result = runner.invoke(cli_entry, ["experiment", "results", experiment.experiment_id, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == queue_models.ExperimentStatus.COMPLETED.value
    scores = {trial["aggregate_score"] for trial in payload["trials"]}
    assert 0.8 in scores


@pytest.fixture
def cli_entry():
    import synth_ai.cli as cli_module

    importlib.reload(cli_module)
    return cli_module.cli


def test_cli_submit_and_status(tmp_path, stub_celery, cli_entry):
    runner = CliRunner()
    request = _make_request(tmp_path).model_dump(mode="json")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    submit_result = runner.invoke(cli_entry, ["experiment", "submit", str(request_path)])
    assert submit_result.exit_code == 0, submit_result.output

    experiments = queue_service.list_experiments()
    assert experiments, "Experiment should be created"
    experiment_id = experiments[0].experiment_id

    status_result = runner.invoke(cli_entry, ["experiment", "status", experiment_id, "--json"])
    assert status_result.exit_code == 0
    payload = json.loads(status_result.output)
    assert payload["experiment_id"] == experiment_id
    assert payload["status"] == queue_models.ExperimentStatus.QUEUED.value

    list_result = runner.invoke(cli_entry, ["experiments", "--json"])
    assert list_result.exit_code == 0
    list_payload = json.loads(list_result.output)
    assert list_payload["live"][0]["experiment_id"] == experiment_id


def test_experiments_status_filter_json(tmp_path, stub_celery, cli_entry):
    runner = CliRunner()
    request = _make_request(tmp_path).model_dump(mode="json")
    path = tmp_path / "req.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    runner.invoke(cli_entry, ["experiment", "submit", str(path)])

    result = runner.invoke(cli_entry, ["experiments", "--status", "queued", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["live"], "Expected live experiments in queued filter"
    assert all(exp["status"] == queue_models.ExperimentStatus.QUEUED.value for exp in payload["live"])


def test_experiment_cancel_cli(tmp_path, stub_celery, cli_entry, monkeypatch):
    runner = CliRunner()
    request = _make_request(tmp_path).model_dump(mode="json")
    path = tmp_path / "cancel_req.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    runner.invoke(cli_entry, ["experiment", "submit", str(path)])

    experiment = queue_service.list_experiments()[0]
    with queue_db.session_scope() as session:
        job = (
            session.query(queue_models.ExperimentJob)
            .filter(queue_models.ExperimentJob.experiment_id == experiment.experiment_id)
            .first()
        )
        job.status = queue_models.ExperimentJobStatus.RUNNING
        job.celery_task_id = "celery-task-1"

    monkeypatch.setattr(queue_service, "get_celery_app", lambda: stub_celery)

    cancel_result = runner.invoke(cli_entry, ["experiment", "cancel", experiment.experiment_id])
    assert cancel_result.exit_code == 0

    with queue_db.session_scope() as session:
        exp = session.get(queue_models.Experiment, experiment.experiment_id)
        assert exp.status == queue_models.ExperimentStatus.CANCELED
        jobs = (
            session.query(queue_models.ExperimentJob)
            .filter(queue_models.ExperimentJob.experiment_id == experiment.experiment_id)
            .all()
        )
        assert all(job.status == queue_models.ExperimentJobStatus.CANCELED for job in jobs)

    assert "celery-task-1" in stub_celery.revoked


def test_queue_command_background(tmp_path, monkeypatch, cli_entry):
    """Test queue status command (default subcommand)."""
    log_dir = Path("logs")
    if log_dir.exists():
        for item in log_dir.iterdir():
            item.unlink()
    
    runner = CliRunner()
    # Queue command defaults to status subcommand
    result = runner.invoke(cli_entry, ["queue"])
    # Status should work even if no workers running
    assert result.exit_code == 0, result.output
    assert "Experiment Queue Status" in result.output or "No workers running" in result.output


def test_stress_multiple_experiments(tmp_path, stub_celery):
    """Submit several experiments back-to-back to ensure dispatcher keeps up."""
    for idx in range(5):
        request = _make_request(tmp_path, num_jobs=1)
        request.name = f"Stress {idx}"
        queue_service.create_experiment(request)
    
    assert len(stub_celery.sent) == 5, "Each experiment should dispatch one job"


def test_dispatcher_resumes_after_completion(tmp_path, stub_celery):
    """Ensure dispatcher sends the next job after completing the current slot."""
    request = _make_request(tmp_path, num_jobs=2)
    request.parallelism = 1
    experiment = queue_service.create_experiment(request)
    assert len(stub_celery.sent) == 1
    first_job_id = stub_celery.sent[0][1][0]
    
    with queue_db.session_scope() as session:
        job = session.query(queue_models.ExperimentJob).filter(
            queue_models.ExperimentJob.job_id == first_job_id
        ).one()
        job.status = queue_models.ExperimentJobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        session.flush()
        # Trigger dispatcher manually
        from synth_ai.experiment_queue import dispatcher as queue_dispatcher
        queue_dispatcher.dispatch_available_jobs(session, experiment.experiment_id)
    
    assert len(stub_celery.sent) == 2, "Second job should dispatch once slot frees"
