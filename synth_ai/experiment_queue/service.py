"""High-level orchestration helpers for experiment queue CLI/API."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Sequence
from uuid import uuid4

import tomli_w
from sqlalchemy.orm import selectinload

# Clear config cache if env vars are set (must happen before other imports)
if os.getenv("EXPERIMENT_QUEUE_DB_PATH") or os.getenv("EXPERIMENT_QUEUE_TRAIN_CMD"):
    from . import config as queue_config

    queue_config.reset_config_cache()

from .celery_app import get_celery_app
from .database import get_session, init_db, session_scope
from .dispatcher import dispatch_available_jobs
from .models import (
    Experiment,
    ExperimentJob,
    ExperimentJobStatus,
    ExperimentStatus,
)
from .schemas import ExperimentJobSpec, ExperimentSubmitRequest


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _normalize_statuses(values: Iterable[str | ExperimentStatus] | None) -> list[ExperimentStatus] | None:
    if not values:
        return None
    normalized: list[ExperimentStatus] = []
    for value in values:
        if isinstance(value, ExperimentStatus):
            normalized.append(value)
        else:
            raw = value if isinstance(value, str) else str(value)
            if raw.startswith("ExperimentStatus."):
                raw = raw.split(".", 1)[-1].lower()
            normalized.append(ExperimentStatus(raw))
    return normalized


def validate_job_spec(job_spec: ExperimentJobSpec) -> None:
    """Validate that a job spec has all required files.
    
    Raises:
        FileNotFoundError: If config file or referenced env file doesn't exist
        ValueError: If config file is invalid
    """
    from pathlib import Path
    
    config_path = Path(job_spec.config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not config_path.suffix == ".toml":
        raise ValueError(f"Config file must be a TOML file: {config_path}")
    
    # Check if config references an env file
    try:
        from .config_utils import _load_toml
        
        config_data = _load_toml(config_path)
        env_file_path = (
            config_data.get("prompt_learning", {}).get("env_file_path")
            or config_data.get("env_file_path")
        )
        
        if env_file_path:
            env_path = Path(env_file_path)
            if not env_path.is_absolute():
                # Resolve relative to config file directory
                env_path = (config_path.parent / env_path).resolve()
            else:
                env_path = env_path.expanduser().resolve()
            
            if not env_path.exists():
                raise FileNotFoundError(
                    f"Env file referenced in config not found: {env_path}\n"
                    f"  Config: {config_path}\n"
                    f"  Referenced as: {env_file_path}"
                )
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        # If we can't parse the config, that's okay - it will fail later during execution
        pass


def create_experiment(request: ExperimentSubmitRequest) -> Experiment:
    """Persist a new experiment and enqueue initial jobs.
    
    Validates that all job specs have required files (config TOML and env files).
    """
    init_db()
    
    # Validate all job specs before creating experiment
    for job_spec in request.jobs:
        validate_job_spec(job_spec)
    
    experiment_id = _generate_id("exp")
    # Exclude None values for TOML serialization (TOML doesn't support None)
    request_dict = request.model_dump(mode="json", exclude_none=True)
    config_text = tomli_w.dumps(request_dict)

    with session_scope() as session:
        experiment = Experiment(
            experiment_id=experiment_id,
            name=request.name,
            description=request.description,
            status=ExperimentStatus.QUEUED,
            parallelism_limit=request.parallelism,
            config_toml=config_text,
            metadata_json=request.metadata or {},
        )
        session.add(experiment)

        for job_spec in request.jobs:
            job = ExperimentJob(
                job_id=_generate_id("job"),
                experiment_id=experiment_id,
                job_type=job_spec.job_type,
                config_path=job_spec.config_path,
                config_overrides=job_spec.config_overrides or {},
                status=ExperimentJobStatus.QUEUED,
            )
            session.add(job)

        session.flush()
        dispatch_available_jobs(session, experiment_id)  # type: ignore[return-value]

    return fetch_experiment(experiment_id)  # type: ignore[return-value]


def fetch_experiment(experiment_id: str) -> Experiment | None:
    """Load an experiment with jobs and trials."""
    init_db()
    session = get_session()
    try:
        experiment = (
            session.query(Experiment)
            .options(
                selectinload(Experiment.jobs),
                selectinload(Experiment.trials),
            )
            .filter(Experiment.experiment_id == experiment_id)
            .first()
        )
        if experiment:
            session.expunge(experiment)
        return experiment
    finally:
        session.close()


def list_experiments(
    *,
    status: Iterable[str | ExperimentStatus] | None = None,
    limit: int | None = None,
    include_live: bool = False,
) -> list[Experiment]:
    """Return experiments filtered by status."""
    init_db()
    session = get_session()
    try:
        query = session.query(Experiment).options(
            selectinload(Experiment.jobs),
            selectinload(Experiment.trials),
        )
        statuses = _normalize_statuses(status)
        if statuses:
            query = query.filter(Experiment.status.in_(statuses))
        if not include_live:
            query = query.order_by(Experiment.created_at.desc())
        else:
            query = query.order_by(Experiment.started_at.desc().nullslast())
        if limit:
            query = query.limit(limit)
        experiments = query.all()
        for exp in experiments:
            session.expunge(exp)
        return experiments
    finally:
        session.close()


def cancel_experiment(experiment_id: str) -> Experiment | None:
    """Mark experiment and outstanding jobs as canceled."""
    init_db()
    app = get_celery_app()
    with session_scope() as session:
        experiment = (
            session.query(Experiment)
            .options(selectinload(Experiment.jobs))
            .filter(Experiment.experiment_id == experiment_id)
            .first()
        )
        if not experiment:
            return None

        experiment.status = ExperimentStatus.CANCELED
        experiment.completed_at = datetime.utcnow()

        for job in experiment.jobs:
            if job.status in {
                ExperimentJobStatus.QUEUED,
                ExperimentJobStatus.RUNNING,
            }:
                job.status = ExperimentJobStatus.CANCELED
                job.completed_at = datetime.utcnow()
                if job.celery_task_id:
                    from contextlib import suppress
                    with suppress(Exception):
                        app.control.revoke(job.celery_task_id, terminate=True)

        session.flush()
        session.expunge(experiment)
        return experiment


def collect_dashboard_data(
    *,
    status_filter: Sequence[str | ExperimentStatus] | None,
    recent_limit: int,
) -> tuple[list[Experiment], list[Experiment]]:
    """Return (live, recent) experiment lists."""
    init_db()
    session = get_session()
    try:
        live_statuses = _normalize_statuses(status_filter) or [
            ExperimentStatus.RUNNING,
            ExperimentStatus.QUEUED,
        ]
        live_query = (
            session.query(Experiment)
            .options(
                selectinload(Experiment.jobs),
                selectinload(Experiment.trials),
            )
            .filter(Experiment.status.in_(live_statuses))
        )
        live = (
            live_query.order_by(Experiment.started_at.desc().nullslast()).limit(50).all()
        )

        default_recent_statuses = [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELED,
        ]
        recent_statuses = _normalize_statuses(status_filter) or default_recent_statuses
        recent_query = (
            session.query(Experiment)
            .options(
                selectinload(Experiment.jobs),
                selectinload(Experiment.trials),
            )
            .filter(Experiment.status.in_(recent_statuses))
        )
        recent = (
            recent_query.order_by(Experiment.completed_at.desc().nullslast())
            .limit(recent_limit)
            .all()
        )

        return live, recent
    finally:
        session.close()
