"""Utility helpers to enqueue Celery tasks while respecting parallelism limits."""

from __future__ import annotations

import logging
import os
from typing import List

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

# Clear config cache if env vars are set (must happen before other imports)
if os.getenv("EXPERIMENT_QUEUE_DB_PATH") or os.getenv("EXPERIMENT_QUEUE_TRAIN_CMD"):
    from . import config as queue_config

    queue_config.reset_config_cache()

from .celery_app import get_celery_app
from .models import Experiment, ExperimentJob, ExperimentJobStatus

logger = logging.getLogger(__name__)

TASK_NAME = "synth_ai.cli.local.experiment_queue.run_experiment_job"


def _active_jobs_query(experiment_id: str) -> Select[tuple[int]]:
    """Build a SQLAlchemy query to count active jobs for an experiment.
    
    Counts jobs that are:
    - QUEUED or RUNNING status
    - Have a Celery task ID assigned (dispatched)
    - Not yet completed (completed_at is None)
    
    Args:
        experiment_id: Experiment identifier
        
    Returns:
        SQLAlchemy Select query that returns a count
    """
    return (
        select(func.count(ExperimentJob.job_id))
        .where(
            ExperimentJob.experiment_id == experiment_id,
            ExperimentJob.status.in_(
                [ExperimentJobStatus.RUNNING, ExperimentJobStatus.QUEUED]
            ),
            ExperimentJob.celery_task_id.is_not(None),
            ExperimentJob.completed_at.is_(None),
        )
    )


def dispatch_available_jobs(session: Session, experiment_id: str) -> list[str]:
    """Send ready jobs to Celery while honoring experiment parallelism."""
    experiment = session.get(Experiment, experiment_id)
    if not experiment:
        return []

    limit = max(experiment.parallelism_limit or 1, 1)
    active = session.execute(_active_jobs_query(experiment_id)).scalar_one()
    remaining_slots = max(limit - active, 0)
    if remaining_slots <= 0:
        return []

    ready_jobs: List[ExperimentJob] = (
        session.query(ExperimentJob)
        .filter(
            ExperimentJob.experiment_id == experiment_id,
            ExperimentJob.status == ExperimentJobStatus.QUEUED,
            ExperimentJob.celery_task_id.is_(None),
        )
        .order_by(ExperimentJob.created_at.asc())
        .limit(remaining_slots)
        .all()
    )
    if not ready_jobs:
        return []

    app = get_celery_app()
    # Get broker URL for logging (may not be available in test stubs)
    try:
        broker_url = app.conf.broker_url
    except AttributeError:
        broker_url = "unknown"
    logger.info(
        "Dispatching %d jobs using Celery broker: %s",
        len(ready_jobs),
        broker_url,
    )
    
    # Extract job IDs before committing session
    job_ids = [job.job_id for job in ready_jobs]
    
    # Commit session before calling send_task
    # Redis broker handles concurrency natively, no locking issues
    session.flush()
    session.commit()
    
    dispatched: list[str] = []
    for job_id in job_ids:
        # send_task sends message to Redis broker
        result = app.send_task(TASK_NAME, args=[job_id])
        
        # Update celery_task_id in a new transaction
        job = session.query(ExperimentJob).filter(ExperimentJob.job_id == job_id).first()
        if job:
            job.celery_task_id = result.id
            logger.debug(
                "Dispatched job %s with task_id %s",
                job_id,
                result.id,
            )
            dispatched.append(job_id)
    
    # Flush to persist celery_task_id updates
    session.flush()
    # Note: Task message is sent to Redis broker
    # The final commit happens when session_scope exits in create_experiment()
    return dispatched
