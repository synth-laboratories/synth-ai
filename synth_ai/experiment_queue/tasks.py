"""Celery task definitions for running experiment jobs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celery.utils.log import get_task_logger

from .celery_app import celery_app
from .config import load_config
from .config_utils import PreparedConfig, prepare_config_file
from .database import session_scope
from .dispatcher import dispatch_available_jobs
from .models import (
    Experiment,
    ExperimentJob,
    ExperimentJobStatus,
    ExperimentStatus,
)
from .results import ResultSummary, collect_result_summary
from .api_schemas import BackendEventsResponse, BackendJobEvent
from .status import ExperimentStatusTracker
from .status_tracker import extract_config_info, update_status_from_output
from .trace_storage import persist_trials_from_summary, update_experiment_metadata

logger = get_task_logger(__name__)


TRAIN_COMMAND_ENV = "EXPERIMENT_QUEUE_TRAIN_CMD"


def _find_venv_python() -> str:
    """Find the venv Python executable to avoid uv cache permission issues.
    
    Checks in order:
    1. sys.executable if already in a venv
    2. .venv/bin/python relative to current working directory
    3. .venv/bin/python relative to repo root (if synth_ai package is installed)
    4. Falls back to 'python' if venv not found
    """
    # If we're already running in a venv, use that
    if sys.executable and ("venv" in sys.executable or ".venv" in sys.executable):
        return sys.executable
    
    # Check .venv/bin/python relative to current working directory
    cwd_venv = Path.cwd() / ".venv" / "bin" / "python"
    if cwd_venv.exists() and os.access(cwd_venv, os.X_OK):
        return str(cwd_venv)
    
    # Check .venv/bin/python relative to synth_ai package location
    try:
        import synth_ai
        
        package_path = Path(synth_ai.__file__).parent.parent.parent
        pkg_venv = package_path / ".venv" / "bin" / "python"
        if pkg_venv.exists() and os.access(pkg_venv, os.X_OK):
            return str(pkg_venv)
    except Exception:
        pass
    
    # Fallback to system python
    return "python"


def _get_default_train_cmd() -> str:
    """Get the default training command, evaluating venv path lazily.
    
    This is called when building the command, not at module import time,
    so it can properly detect the venv based on the current working directory.
    """
    return f"{_find_venv_python()} -m synth_ai.cli train"


def _extract_backend_job_id(output: str) -> str | None:
    """Extract backend job ID from subprocess output.
    
    Looks for patterns like:
    - JSON: "job_id": "pl_xxxxx"
    - Pattern: pl_[a-f0-9]+
    
    Args:
        output: Subprocess stdout/stderr output
        
    Returns:
        Backend job ID if found, None otherwise
        
    Raises:
        AssertionError: If extracted ID doesn't match expected format
    """
    if not output:
        return None
    
    # Assert output is a string
    assert isinstance(output, str), f"Expected str, got {type(output).__name__}"
    
    # Look for job_id in JSON response
    match = re.search(r'"job_id"\s*:\s*"([^"]+)"', output)
    if match:
        job_id = match.group(1)
        # Validate format
        assert job_id.startswith("pl_"), f"Extracted job_id doesn't match expected format 'pl_*': {job_id}"
        assert len(job_id) > 3, f"Extracted job_id too short: {job_id}"
        return job_id
    
    # Try pattern pl_xxxxx
    match = re.search(r'pl_[a-f0-9]+', output)
    if match:
        job_id = match.group(0)
        # Validate format
        assert job_id.startswith("pl_"), f"Extracted job_id doesn't match expected format 'pl_*': {job_id}"
        assert len(job_id) > 3, f"Extracted job_id too short: {job_id}"
        return job_id
    
    return None


def _poll_backend_progress(
    backend_job_id: str,
    status_tracker: ExperimentStatusTracker,
    policy: str | None,
    environment: str | None,
    backend_url: str,
    api_key: str,
    stop_event: threading.Event,
) -> None:
    """Poll backend API for progress events and update status_json.
    
    Polls the backend API endpoint `/prompt-learning/online/jobs/{backend_job_id}/events`
    every 5 seconds to fetch `prompt.learning.progress` events containing rollouts,
    ETA, and best score information. Updates the experiment status_json in real-time.
    
    Backend URL Configuration:
    - Default: Production (https://agent-learning.onrender.com/api)
    - Local: Set EXPERIMENT_QUEUE_LOCAL=true or use --local flag (http://localhost:8000/api)
    - Custom: Set EXPERIMENT_QUEUE_BACKEND_URL env var
    
    Args:
        backend_job_id: Backend job ID to poll (e.g., "pl_xxxxx")
        status_tracker: ExperimentStatusTracker instance for updating status_json
        policy: Policy model name (e.g., "gpt-4", "llama-3.1-8b-instant")
        environment: Environment name (e.g., "heartdisease", "hotpotqa")
        backend_url: Backend API base URL (from config.backend_url)
        api_key: API key for authentication (from SYNTH_API_KEY env var)
        stop_event: Threading event to signal when to stop polling
    """
    import logging
    import os
    import requests
    
    # Import BackendJobEvent locally to ensure it's available in this function's scope
    from .api_schemas import BackendJobEvent
    
    # Get logger for this thread (logger from parent thread may not work correctly)
    poller_logger = logging.getLogger(f"synth_ai.experiment_queue.poller.{backend_job_id}")
    
    # Set log level from environment variable if set (allows --loglevel flag to control verbosity)
    # Use Celery's logger hierarchy instead of creating our own handler to avoid duplicates
    log_level_env = os.getenv("EXPERIMENT_QUEUE_LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_env)
        poller_logger.setLevel(log_level)
        # Don't create handlers - let Celery's logging handle it
        # Just propagate to parent logger (Celery's task logger)
        poller_logger.propagate = True
    except (AttributeError, ValueError):
        # Invalid log level, use default
        pass
    
    # Validate inputs with assertions
    assert backend_job_id, "backend_job_id cannot be empty"
    assert backend_job_id.startswith("pl_"), f"Invalid backend_job_id format: expected 'pl_*', got '{backend_job_id}'"
    assert backend_url, "backend_url cannot be empty"
    assert backend_url.startswith(("http://", "https://")), f"Invalid backend_url format: {backend_url}"
    assert api_key, "api_key cannot be empty"
    assert status_tracker is not None, "status_tracker cannot be None"
    assert stop_event is not None, "stop_event cannot be None"
    
    url = f"{backend_url.rstrip('/')}/prompt-learning/online/jobs/{backend_job_id}/events"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_seq = 0
    
    poller_logger.info("📡 Starting progress poller for backend job %s (URL: %s)", backend_job_id, url)
    
    while not stop_event.is_set():
        events_received = 0
        try:
            # Assert URL is valid before making request
            assert url.startswith(("http://", "https://")), f"Invalid URL format: {url}"
            
            poller_logger.info("Polling backend API: %s (since_seq: %d)", url, last_seq)
            
            resp = requests.get(
                url,
                headers=headers,
                params={"since_seq": last_seq, "limit": 100},
                timeout=10,
            )
            
            # Assert we got a response object
            assert resp is not None, "requests.get() returned None"
            
            poller_logger.info("API response: status=%d, content_length=%d", resp.status_code, len(resp.content))
            
            if resp.status_code == 200:
                # Parse and validate API response using Pydantic models
                try:
                    raw_data = resp.json()
                    # Assert response is not None
                    assert raw_data is not None, "API returned None response"
                    
                    # Parse response with validation
                    assert isinstance(raw_data, (dict, list)), (
                        f"API response must be dict or list, got {type(raw_data).__name__}: {raw_data}"
                    )
                    
                    events_response = BackendEventsResponse.parse_response(raw_data)
                    assert isinstance(events_response, BackendEventsResponse), (
                        f"parse_response returned wrong type: {type(events_response).__name__}"
                    )
                    assert isinstance(events_response.events, list), (
                        f"events_response.events must be list, got {type(events_response.events).__name__}"
                    )
                    
                    events_received = len(events_response.events)
                    assert events_received >= 0, (
                        f"events_received must be >= 0, got {events_received}"
                    )
                    
                    # Process each event
                    event_types_seen: dict[str, int] = {}
                    for idx, event in enumerate(events_response.events):
                        # Assert event is BackendJobEvent instance
                        assert isinstance(event, BackendJobEvent), (
                            f"Event at index {idx} must be BackendJobEvent, got {type(event).__name__}"
                        )
                        # Assert event has required fields
                        assert event.seq >= 0, f"Invalid seq: {event.seq}"
                        assert event.type, f"Event missing type field: {event}"
                        assert event.message, f"Event missing message field: {event}"
                        
                        # Track event types for debugging
                        event_types_seen[event.type] = event_types_seen.get(event.type, 0) + 1
                        
                        # Check if this is a progress event
                        if event.type == "prompt.learning.progress":
                            poller_logger.info(
                                "Found progress event seq=%d: %s",
                                event.seq,
                                event.message[:100],
                            )
                            # Extract progress data with validation
                            progress_data = event.get_progress_data()
                            if progress_data is None:
                                poller_logger.warning(
                                    "Progress event seq=%d has no parseable data. Event data: %s",
                                    event.seq,
                                    event.data,
                                )
                                continue
                            
                            poller_logger.debug(
                                "Progress event seq=%d data: rollouts_completed=%s, rollouts_total=%s, best_score=%s, eta=%s",
                                event.seq,
                                progress_data.rollouts_completed,
                                progress_data.effective_rollouts_total,
                                progress_data.effective_best_score,
                                progress_data.eta_seconds,
                            )
                            
                            # Use effective getters that handle field name variations
                            rollouts_completed = progress_data.rollouts_completed
                            rollouts_total = progress_data.effective_rollouts_total
                            eta_seconds = progress_data.eta_seconds
                            # percent_rollouts from backend is 0-1, convert to 0-100 for display
                            progress_pct = None
                            if progress_data.percent_rollouts is not None:
                                progress_pct = progress_data.percent_rollouts * 100.0
                            elif progress_data.percent_overall is not None:
                                # Fallback to percent_overall if percent_rollouts not available
                                progress_pct = progress_data.percent_overall * 100.0
                            best_score = progress_data.effective_best_score
                            
                            # Assert data types and ranges
                            if rollouts_completed is not None:
                                assert isinstance(rollouts_completed, int), (
                                    f"rollouts_completed must be int, got {type(rollouts_completed).__name__}: {rollouts_completed}"
                                )
                                assert rollouts_completed >= 0, (
                                    f"rollouts_completed must be >= 0, got {rollouts_completed}"
                                )
                            
                            if rollouts_total is not None:
                                assert isinstance(rollouts_total, int), (
                                    f"rollouts_total must be int, got {type(rollouts_total).__name__}: {rollouts_total}"
                                )
                                assert rollouts_total > 0, (
                                    f"rollouts_total must be > 0, got {rollouts_total}"
                                )
                            
                            if eta_seconds is not None:
                                assert isinstance(eta_seconds, (int, float)), (
                                    f"eta_seconds must be int | float, got {type(eta_seconds).__name__}: {eta_seconds}"
                                )
                                assert eta_seconds >= 0, (
                                    f"eta_seconds must be >= 0, got {eta_seconds}"
                                )
                            
                            if best_score is not None:
                                assert isinstance(best_score, (int, float)), (
                                    f"best_score must be int | float, got {type(best_score).__name__}: {best_score}"
                                )
                                assert 0 <= best_score <= 1, (
                                    f"best_score must be in [0, 1], got {best_score}"
                                )
                            
                            if progress_pct is not None:
                                assert isinstance(progress_pct, (int, float)), (
                                    f"progress_pct must be int | float, got {type(progress_pct).__name__}: {progress_pct}"
                                )
                                assert 0 <= progress_pct <= 100, (
                                    f"progress_pct must be in [0, 100], got {progress_pct}"
                                )
                            
                            # Assert consistency: rollouts_completed <= rollouts_total
                            if rollouts_completed is not None and rollouts_total is not None:
                                assert rollouts_completed <= rollouts_total, (
                                    f"rollouts_completed ({rollouts_completed}) > rollouts_total ({rollouts_total})"
                                )
                            
                            # Assert we have meaningful progress data
                            has_progress = (
                                rollouts_completed is not None
                                or best_score is not None
                                or rollouts_total is not None
                            )
                            
                            if has_progress:
                                # Validate status_tracker before update
                                assert status_tracker is not None, "status_tracker is None"
                                assert hasattr(status_tracker, "update"), "status_tracker missing update method"
                                assert hasattr(status_tracker, "job_id"), "status_tracker missing job_id"
                                
                                status_tracker.update(
                                    policy=policy,
                                    environment=environment,
                                    rollouts_completed=rollouts_completed,
                                    total_rollouts=rollouts_total,
                                    eta_seconds=eta_seconds,
                                    progress_pct=progress_pct,
                                    best_score=best_score,
                                )
                                poller_logger.info(
                                    "📊 Progress update for job %s: %s/%s rollouts, ETA: %s, Best: %s",
                                    backend_job_id,
                                    rollouts_completed,
                                    rollouts_total,
                                    eta_seconds,
                                    best_score,
                                )
                            
                            # Update last_seq (always update, even if no progress data)
                            last_seq = max(last_seq, event.seq)
                        else:
                            # Non-progress event - just update seq
                            last_seq = max(last_seq, event.seq)
                    
                    # Log summary of events processed
                    if events_received == 0:
                        poller_logger.info("Progress poller heartbeat for job %s (no new events, last_seq=%d)", backend_job_id, last_seq)
                    else:
                        event_types_str = ", ".join(f"{k}:{v}" for k, v in sorted(event_types_seen.items()))
                        poller_logger.info(
                            "Processed %d events (types: %s), updated last_seq to %d",
                            events_received,
                            event_types_str,
                            last_seq,
                        )
                        # Log if we're not seeing progress events
                        if "prompt.learning.progress" not in event_types_seen:
                            poller_logger.debug(
                                "No progress events in this batch (last_seq=%d). Event types seen: %s",
                                last_seq,
                                event_types_str,
                            )
                
                except AssertionError as e:
                    poller_logger.error(
                        "❌ Assertion failed while parsing events for job %s: %s. Response: %s",
                        backend_job_id,
                        e,
                        resp.text[:500] if resp else "No response",
                    )
                    # Continue polling - don't stop on validation errors
                except ValueError as e:
                    poller_logger.error(
                        "❌ Invalid API response format for job %s: %s. Response: %s",
                        backend_job_id,
                        e,
                        resp.text[:500] if resp else "No response",
                    )
                    # Continue polling - don't stop on validation errors
                except Exception as e:
                    poller_logger.error(
                        "❌ Unexpected error parsing events for job %s: %s. Response: %s",
                        backend_job_id,
                        e,
                        resp.text[:500] if resp else "No response",
                        exc_info=True,
                    )
                    # Continue polling - don't stop on parsing errors
            elif resp.status_code == 404:
                # Job not found yet or doesn't exist - stop polling
                poller_logger.warning("Backend job %s not found (404), stopping poller", backend_job_id)
                break
            elif resp.status_code != 200:
                poller_logger.warning(
                    "Backend API returned status %d for job %s: %s",
                    resp.status_code,
                    backend_job_id,
                    resp.text[:200],
                )
        except requests.exceptions.RequestException as e:
            poller_logger.error("Network error polling job %s: %s", backend_job_id, e, exc_info=True)
        except Exception as e:
            poller_logger.error("Progress poller error for job %s: %s", backend_job_id, e, exc_info=True)
        
        # Poll every 5 seconds
        stop_event.wait(timeout=5)
    
    poller_logger.info("📡 Stopped progress poller for backend job %s", backend_job_id)


def _truncate(text: str, limit: int = 4000) -> str:
    """Truncate text to a maximum length, keeping the end portion.
    
    Args:
        text: Text to truncate
        limit: Maximum length in characters (default: 4000)
        
    Returns:
        Truncated text (last `limit` characters if text exceeds limit)
    """
    if len(text) <= limit:
        return text
    return text[-limit:]


def _build_train_command(config_path: str) -> list[str]:
    """Build the training command for running a prompt learning job.
    
    Constructs a command list suitable for subprocess execution by:
    1. Getting the base command from EXPERIMENT_QUEUE_TRAIN_CMD env var or default
    2. Parsing the base command into segments
    3. Appending prompt learning specific flags (--type, --config, --poll, etc.)
    
    Args:
        config_path: Path to the TOML config file for the experiment
        
    Returns:
        List of command segments ready for subprocess execution
        
    Note:
        The base command defaults to `python -m synth_ai.cli train` if
        EXPERIMENT_QUEUE_TRAIN_CMD is not set. The command always includes
        --type prompt_learning, --config, --poll, and --stream-format cli flags.
    """
    # Get command from env var or use default (lazily evaluated)
    base_cmd = os.getenv(TRAIN_COMMAND_ENV)
    if base_cmd:
        logger.debug("Using training command from EXPERIMENT_QUEUE_TRAIN_CMD: %s", base_cmd)
    else:
        base_cmd = _get_default_train_cmd()
        logger.debug("Using default training command: %s", base_cmd)
    
    segments: list[str] = []
    for part in shlex.split(base_cmd):
        if part:
            segments.append(part)

    segments.extend(
        [
            "--type",
            "prompt_learning",
            "--config",
            config_path,
            "--poll",
            "--stream-format",
            "cli",
        ]
    )
    return segments


def _mark_job_running(job_id: str, task_id: str | None) -> ExperimentJob | None:
    """Mark a job as running and update its status in the database.
    
    Updates the job status to RUNNING, sets the started_at timestamp, and
    optionally associates a Celery task ID. If the parent experiment is
    QUEUED, it is also marked as RUNNING.
    
    Args:
        job_id: Job identifier
        task_id: Optional Celery task ID to associate with the job
        
    Returns:
        ExperimentJob instance if found, None otherwise
        
    Note:
        The job is expunged from the session so it can be safely used outside
        the session scope. The session is committed automatically by session_scope.
    """
    with session_scope() as session:
        job = session.get(ExperimentJob, job_id)
        if not job:
            logger.warning("Job %s missing from database", job_id)
            return None
        job.status = ExperimentJobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        if task_id:
            job.celery_task_id = task_id
        experiment = job.experiment
        if experiment and experiment.status == ExperimentStatus.QUEUED:
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now(UTC)
        session.flush()
        # Expunge so job can be safely used outside session scope
        session.expunge(job)
        return job


def _jobs_remaining(session, experiment_id: str) -> int:
    """Count remaining jobs (QUEUED or RUNNING) for an experiment.
    
    Args:
        session: SQLAlchemy session
        experiment_id: Experiment identifier
        
    Returns:
        Number of jobs that are still QUEUED or RUNNING (not completed/failed)
    """
    return (
        session.query(ExperimentJob)
        .filter(
            ExperimentJob.experiment_id == experiment_id,
            ExperimentJob.status.in_(
                [
                    ExperimentJobStatus.QUEUED,
                    ExperimentJobStatus.RUNNING,
                ]
            ),
        )
        .count()
    )


def _finalize_job(
    job_id: str,
    *,
    summary: ResultSummary,
    success: bool,
    error_message: str | None = None,
) -> dict[str, Any] | None:
    """Finalize a job by updating its status and persisting results.
    
    Updates the job status to COMPLETED or FAILED based on success flag,
    persists trial data if successful, and updates experiment status when
    all jobs are done. If the experiment has remaining jobs, dispatches them.
    
    Args:
        job_id: Job identifier
        summary: Result summary containing stdout, stderr, metrics, etc.
        success: Whether the job completed successfully
        error_message: Optional error message if job failed
        
    Returns:
        Summary dictionary if job found, None otherwise
        
    Note:
        - If successful: Job status set to COMPLETED, trials persisted
        - If failed: Job status set to FAILED, error message stored
        - Experiment status updated to COMPLETED/FAILED only when all jobs done
        - Remaining jobs are dispatched if experiment still has queued jobs
    """
    with session_scope() as session:
        job = session.get(ExperimentJob, job_id)
        if not job:
            logger.warning("Job %s missing during finalize", job_id)
            return None

        job.completed_at = datetime.now(UTC)
        job.result = summary.to_dict()
        experiment = job.experiment

        if success:
            job.status = ExperimentJobStatus.COMPLETED
            persist_trials_from_summary(session, job, summary)
            if experiment:
                update_experiment_metadata(experiment, summary)
        else:
            job.status = ExperimentJobStatus.FAILED
            job.error = error_message or summary.stderr or "Job failed"
            if experiment:
                # Don't immediately mark experiment as failed - let remaining jobs continue
                # The experiment will be marked as failed only if all jobs fail
                logger.warning(
                    "Job %s failed for experiment %s, but allowing remaining jobs to continue",
                    job_id,
                    experiment.experiment_id,
                )

        session.flush()

        if experiment:
            remaining = _jobs_remaining(session, experiment.experiment_id)
            if remaining == 0:
                # All jobs completed - check if experiment succeeded or failed
                all_jobs = (
                    session.query(ExperimentJob)
                    .filter(ExperimentJob.experiment_id == experiment.experiment_id)
                    .all()
                )
                all_failed = all(
                    job.status == ExperimentJobStatus.FAILED for job in all_jobs
                )
                if all_failed:
                    experiment.status = ExperimentStatus.FAILED
                    experiment.error = (
                        all_jobs[0].error if all_jobs else "All jobs failed"
                    )
                else:
                    experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_at = datetime.now(UTC)
            else:
                # Dispatch remaining jobs (periodic task will also handle this as backup)
                dispatch_available_jobs(session, experiment.experiment_id)

        return summary.to_dict()


@celery_app.task(bind=True, name="synth_ai.experiment_queue.run_experiment_job")
def run_experiment_job(self, job_id: str) -> dict[str, Any] | None:
    """Celery task entrypoint for running a prompt learning experiment job.
    
    This is the main Celery task that executes prompt learning jobs. It:
    1. Marks the job as RUNNING
    2. Prepares the config file (applies overrides)
    3. Builds and executes the training command via subprocess
    4. Collects results (stdout, stderr, metrics, artifacts)
    5. Finalizes the job (updates status, persists results)
    
    Args:
        self: Celery task instance (bound task)
        job_id: Job identifier from the experiment queue database
        
    Returns:
        Result summary dictionary if successful, None if job not found
        
    Raises:
        AssertionError: If inputs are invalid (should not happen in production)
        
    Note:
        The task runs the training command (`synth-ai train --type prompt_learning`)
        as a subprocess and captures stdout/stderr. Health check failures and
        authentication errors are detected and cause job failure even if returncode is 0.
    """
    # Validate input
    assert isinstance(job_id, str), (
        f"job_id must be str, got {type(job_id).__name__}: {job_id}"
    )
    assert job_id, "job_id cannot be empty"
    
    job = _mark_job_running(job_id, getattr(self.request, "id", None))
    if not job:
        logger.warning("Job %s not found or could not be marked as running", job_id)
        return None
    
    # Validate job object
    assert isinstance(job, ExperimentJob), (
        f"_mark_job_running must return ExperimentJob, got {type(job).__name__}"
    )
    assert job.job_id == job_id, (
        f"Job ID mismatch: expected {job_id}, got {job.job_id}"
    )
    assert job.status == ExperimentJobStatus.RUNNING, (
        f"Job status must be RUNNING, got {job.status}"
    )

    summary = ResultSummary()
    prepared: PreparedConfig | None = None
    success = False
    error_message: str | None = None  # Will be set if training fails

    # Initialize status tracker
    assert job.job_id, "job.job_id cannot be empty"
    status_tracker = ExperimentStatusTracker(job.job_id)
    assert status_tracker.job_id == job.job_id, (
        f"Status tracker job_id mismatch: expected {job.job_id}, got {status_tracker.job_id}"
    )
    
    job_start_time = time.time()
    assert job_start_time > 0, f"job_start_time must be > 0, got {job_start_time}"
    
    policy: str | None = None
    environment: str | None = None

    try:
        # Validate config_path
        assert job.config_path, "job.config_path cannot be empty"
        assert isinstance(job.config_path, str), (
            f"job.config_path must be str, got {type(job.config_path).__name__}"
        )
        
        # Validate config_overrides
        if job.config_overrides is not None:
            assert isinstance(job.config_overrides, dict), (
                f"job.config_overrides must be dict, got {type(job.config_overrides).__name__}"
            )
        
        prepared = prepare_config_file(job.config_path, job.config_overrides or {})
        assert prepared is not None, "prepare_config_file returned None"
        assert isinstance(prepared, PreparedConfig), (
            f"prepare_config_file must return PreparedConfig, got {type(prepared).__name__}"
        )
        assert prepared.path.exists(), (
            f"Prepared config file must exist: {prepared.path}"
        )
        
        # Extract policy and environment from config
        policy, environment = extract_config_info(prepared.path)
        assert isinstance(policy, (str, type(None))), (
            f"policy must be str | None, got {type(policy).__name__}: {policy}"
        )
        assert isinstance(environment, (str, type(None))), (
            f"environment must be str | None, got {type(environment).__name__}: {environment}"
        )
        
        # ASSERT: Verify overrides were applied by checking the prepared config
        if job.config_overrides:
            model_override = job.config_overrides.get("prompt_learning.policy.model")
            provider_override = job.config_overrides.get("prompt_learning.policy.provider")
            rollout_budget_override = job.config_overrides.get("prompt_learning.gepa.rollout.budget")
            max_rollouts_override = job.config_overrides.get("prompt_learning.termination_config.max_rollouts")
            
            # Assert model override matches extracted policy
            if model_override:
                assert policy == model_override, (
                    f"CRITICAL: Policy model mismatch for job {job.job_id}: "
                    f"override={model_override!r} but extracted={policy!r}. "
                    f"This indicates the override wasn't applied correctly to the prepared config. "
                    f"Config path: {prepared.path}"
                )
                logger.info(
                    "✅ Config override verified for job %s: model=%s matches extracted policy",
                    job.job_id,
                    model_override,
                )
            
            # Assert provider override if specified
            if provider_override:
                # Extract provider from prepared config
                import tomllib
                with open(prepared.path, "rb") as f:
                    prepared_config = tomllib.load(f)
                pl_section = prepared_config.get("prompt_learning", {})
                policy_section = pl_section.get("policy", {})
                extracted_provider = policy_section.get("provider") if isinstance(policy_section, dict) else None
                if extracted_provider:
                    assert extracted_provider == provider_override, (
                        f"CRITICAL: Provider mismatch for job {job.job_id}: "
                        f"override={provider_override!r} but extracted={extracted_provider!r}. "
                        f"Config path: {prepared.path}"
                    )
            
            # Assert rollout budget override if specified
            if rollout_budget_override is not None:
                import tomllib
                with open(prepared.path, "rb") as f:
                    prepared_config = tomllib.load(f)
                pl_section = prepared_config.get("prompt_learning", {})
                gepa_section = pl_section.get("gepa", {})
                rollout_section = gepa_section.get("rollout", {}) if isinstance(gepa_section, dict) else {}
                extracted_budget = rollout_section.get("budget") if isinstance(rollout_section, dict) else None
                if extracted_budget is not None:
                    assert extracted_budget == rollout_budget_override, (
                        f"CRITICAL: Rollout budget mismatch for job {job.job_id}: "
                        f"override={rollout_budget_override} but extracted={extracted_budget}. "
                        f"Config path: {prepared.path}"
                    )
            
            # Assert max_rollouts override if specified
            if max_rollouts_override is not None:
                import tomllib
                with open(prepared.path, "rb") as f:
                    prepared_config = tomllib.load(f)
                pl_section = prepared_config.get("prompt_learning", {})
                termination_section = pl_section.get("termination_config", {})
                extracted_max_rollouts = termination_section.get("max_rollouts") if isinstance(termination_section, dict) else None
                if extracted_max_rollouts is not None:
                    assert extracted_max_rollouts == max_rollouts_override, (
                        f"CRITICAL: Max rollouts mismatch for job {job.job_id}: "
                        f"override={max_rollouts_override} but extracted={extracted_max_rollouts}. "
                        f"Config path: {prepared.path}"
                    )
        
        if policy or environment:
            status_tracker.update(policy=policy, environment=environment)
            logger.info(
                "📊 Experiment config for job %s: policy=%s, environment=%s",
                job.job_id,
                policy or "unknown",
                environment or "unknown",
            )
        
        cmd = _build_train_command(str(prepared.path))
        assert isinstance(cmd, list), (
            f"_build_train_command must return list, got {type(cmd).__name__}"
        )
        assert len(cmd) > 0, "Command list cannot be empty"
        assert all(isinstance(arg, str) for arg in cmd), (
            f"All command arguments must be str, got types: {[type(arg).__name__ for arg in cmd]}"
        )
        logger.info("Executing job %s via command: %s", job.job_id, " ".join(cmd))
        
        # Run command with unbuffered output to see errors immediately
        env = os.environ.copy()
        assert isinstance(env, dict), (
            f"os.environ.copy() must return dict, got {type(env).__name__}"
        )
        env["PYTHONUNBUFFERED"] = "1"
        
        # Log authentication status BEFORE running command
        synth_key = env.get("SYNTH_API_KEY")
        env_key = env.get("ENVIRONMENT_API_KEY")
        logger.info(
            "🔐 Authentication status for job %s:\n"
            "  SYNTH_API_KEY: %s\n"
            "  ENVIRONMENT_API_KEY: %s",
            job.job_id,
            f"{synth_key[:8]}...{synth_key[-4:]}" if synth_key and len(synth_key) > 12 else "(NOT SET)",
            f"{env_key[:8]}...{env_key[-4:]}" if env_key and len(env_key) > 12 else "(NOT SET)",
        )
        
        logger.info(
            "🚀 Starting subprocess for job %s:\n"
            "  Command: %s\n"
            "  Working directory: %s\n"
            "  Python executable: %s\n"
            "  Environment keys: %s",
            job.job_id,
            " ".join(cmd),
            os.getcwd(),
            env.get("PYTHON", "python"),
            ", ".join(sorted([k for k in env if "API" in k or "KEY" in k])),
        )
        
        # Get backend URL and API key for progress polling
        config = load_config()
        assert config is not None, "load_config() returned None"
        backend_url = config.backend_url
        assert isinstance(backend_url, str), (
            f"config.backend_url must be str, got {type(backend_url).__name__}"
        )
        assert backend_url.startswith(("http://", "https://")), (
            f"backend_url must start with http:// or https://, got {backend_url}"
        )
        
        # Get API key from worker environment (not subprocess env)
        # This is needed for the poller thread, which runs in the worker process
        api_key = os.getenv("SYNTH_API_KEY") or env.get("SYNTH_API_KEY")
        # api_key may be None (poller is optional)
        
        # Start background progress poller (will be started once we have backend_job_id)
        poller_stop = threading.Event()
        assert poller_stop is not None, "threading.Event() returned None"
        poller_thread: threading.Thread | None = None
        backend_job_id: str | None = None
        
        try:
            # Stream subprocess output line-by-line to extract backend_job_id and parse progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,  # Line buffered
            )
            assert process is not None, "subprocess.Popen() returned None"
            assert process.stdout is not None, "process.stdout is None"
            
            stdout_lines: list[str] = []
            accumulated_output = ""  # Accumulate output for better pattern matching
            last_status_update_time = job_start_time
            STATUS_UPDATE_INTERVAL = 5.0  # Update status_json every 5 seconds even without progress
            assert STATUS_UPDATE_INTERVAL > 0, (
                f"STATUS_UPDATE_INTERVAL must be > 0, got {STATUS_UPDATE_INTERVAL}"
            )
            
            # Read output line-by-line
            for line in process.stdout:
                assert isinstance(line, str), (
                    f"process.stdout line must be str, got {type(line).__name__}"
                )
                stdout_lines.append(line)
                assert isinstance(accumulated_output, str), (
                    f"accumulated_output must be str, got {type(accumulated_output).__name__}"
                )
                accumulated_output += line
                assert len(accumulated_output) >= len(line), (
                    f"accumulated_output length should increase, got {len(accumulated_output)} < {len(line)}"
                )
                
                # Try to extract backend_job_id from output
                if not backend_job_id:
                    extracted_id = _extract_backend_job_id(line)
                    if extracted_id:
                        # Assert extracted ID is valid before using it
                        assert extracted_id.startswith("pl_"), (
                            f"Invalid backend_job_id format: {extracted_id}"
                        )
                        assert len(extracted_id) > 3, (
                            f"Backend job ID too short: {extracted_id}"
                        )
                        
                        backend_job_id = extracted_id
                        logger.info("📋 Extracted backend job ID: %s", backend_job_id)
                        
                        # Update job with backend_job_id
                        with session_scope() as session:
                            db_job = session.get(ExperimentJob, job.job_id)
                            if db_job:
                                db_job.backend_job_id = backend_job_id
                                session.commit()
                        
                        # Start progress poller now that we have backend_job_id
                        # Validate inputs before starting poller (poller is optional - don't fail job if missing)
                        if not api_key:
                            logger.warning(
                                "⚠️  Cannot start progress poller for job %s: SYNTH_API_KEY not set in worker environment. "
                                "Progress updates will not be available, but job will continue.",
                                job.job_id,
                            )
                        elif not backend_url:
                            logger.warning(
                                "⚠️  Cannot start progress poller for job %s: backend_url not configured. "
                                "Progress updates will not be available, but job will continue.",
                                job.job_id,
                            )
                        elif not backend_job_id.startswith("pl_"):
                            logger.warning(
                                "⚠️  Cannot start progress poller for job %s: invalid backend_job_id format: %s. "
                                "Progress updates will not be available, but job will continue.",
                                job.job_id,
                                backend_job_id,
                            )
                        
                        if api_key and backend_url and backend_job_id.startswith("pl_"):
                            # Validate all inputs before starting thread
                            assert isinstance(backend_job_id, str), (
                                f"backend_job_id must be str, got {type(backend_job_id).__name__}"
                            )
                            assert isinstance(status_tracker, ExperimentStatusTracker), (
                                f"status_tracker must be ExperimentStatusTracker, got {type(status_tracker).__name__}"
                            )
                            assert isinstance(backend_url, str), (
                                f"backend_url must be str, got {type(backend_url).__name__}"
                            )
                            assert isinstance(api_key, str), (
                                f"api_key must be str, got {type(api_key).__name__}"
                            )
                            assert poller_stop is not None, "poller_stop cannot be None"
                            
                            poller_thread = threading.Thread(
                                target=_poll_backend_progress,
                                args=(
                                    backend_job_id,
                                    status_tracker,
                                    policy,
                                    environment,
                                    backend_url,
                                    api_key,
                                    poller_stop,
                                ),
                                daemon=True,
                            )
                            assert poller_thread is not None, "threading.Thread() returned None"
                            poller_thread.start()
                            assert poller_thread.is_alive() or not poller_thread.is_alive(), (
                                "Thread should be startable"
                            )
                            logger.info("📡 Started progress poller for backend job %s", backend_job_id)
                        else:
                            logger.warning(
                                "Cannot start progress poller: missing API key or backend URL"
                            )
                
                # Parse accumulated output for progress updates (fallback if API polling fails)
                # Use accumulated output (not just current line) for better pattern matching
                # Update status_json periodically even without progress data to show elapsed time
                current_time = time.time()
                assert current_time >= job_start_time, (
                    f"current_time ({current_time}) < job_start_time ({job_start_time})"
                )
                assert isinstance(accumulated_output, str), (
                    f"accumulated_output must be str, got {type(accumulated_output).__name__}"
                )
                
                should_update = (
                    # Update if we find progress patterns
                    "rollouts=" in line.lower() or
                    "progress:" in line.lower() or
                    "gepa progress:" in line.lower() or
                    # Or update periodically (every 5 seconds) to show elapsed time
                    (current_time - last_status_update_time) >= STATUS_UPDATE_INTERVAL
                )
                assert isinstance(should_update, bool), (
                    f"should_update must be bool, got {type(should_update).__name__}"
                )
                
                if should_update:
                    # Validate accumulated_output before parsing
                    assert len(accumulated_output) > 0, "accumulated_output cannot be empty"
                    output_to_parse = accumulated_output[-5000:]  # Last 5KB to avoid parsing huge outputs
                    assert isinstance(output_to_parse, str), (
                        f"output_to_parse must be str, got {type(output_to_parse).__name__}"
                    )
                    assert len(output_to_parse) <= len(accumulated_output), (
                        f"output_to_parse length ({len(output_to_parse)}) > accumulated_output length ({len(accumulated_output)})"
                    )
                    
                    update_status_from_output(
                        status_tracker,
                        output_to_parse,
                        policy=policy,
                        environment=environment,
                        start_time=job_start_time,
                    )
                    last_status_update_time = current_time
                    assert last_status_update_time >= job_start_time, (
                        f"last_status_update_time ({last_status_update_time}) < job_start_time ({job_start_time})"
                    )
            
            # Wait for process to complete
            assert process is not None, "process is None before wait()"
            returncode = process.wait()
            assert isinstance(returncode, int), (
                f"process.wait() must return int, got {type(returncode).__name__}: {returncode}"
            )
            
            # Combine output
            assert isinstance(stdout_lines, list), (
                f"stdout_lines must be list, got {type(stdout_lines).__name__}"
            )
            assert all(isinstance(line, str) for line in stdout_lines), (
                f"All stdout_lines must be str, got types: {[type(line).__name__ for line in stdout_lines[:5]]}"
            )
            
            stdout = "".join(stdout_lines)
            assert isinstance(stdout, str), (
                f"stdout must be str, got {type(stdout).__name__}"
            )
            stderr = ""  # stderr is redirected to stdout
            assert isinstance(stderr, str), (
                f"stderr must be str, got {type(stderr).__name__}"
            )
            
            # Create CompletedProcess-like object for compatibility
            class CompletedProcess:
                def __init__(self, returncode: int, stdout: str, stderr: str):
                    assert isinstance(returncode, int), (
                        f"returncode must be int, got {type(returncode).__name__}"
                    )
                    assert isinstance(stdout, str), (
                        f"stdout must be str, got {type(stdout).__name__}"
                    )
                    assert isinstance(stderr, str), (
                        f"stderr must be str, got {type(stderr).__name__}"
                    )
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            completed = CompletedProcess(returncode, stdout, stderr)
            assert isinstance(completed, CompletedProcess), (
                f"CompletedProcess() must return CompletedProcess, got {type(completed).__name__}"
            )
            
            logger.info(
                "✅ Subprocess completed for job %s:\n"
                "  Return code: %s\n"
                "  Stdout length: %d chars\n"
                "  Stderr length: %d chars",
                job.job_id,
                completed.returncode,
                len(completed.stdout) if completed.stdout else 0,
                len(completed.stderr) if completed.stderr else 0,
            )
            
            # Final status update from complete output
            assert isinstance(completed.stdout, str), (
                f"completed.stdout must be str before final update, got {type(completed.stdout).__name__}"
            )
            assert len(completed.stdout) > 0 or len(accumulated_output) > 0, (
                "Must have some output for final status update"
            )
            
            # Use accumulated_output if available (more complete), otherwise stdout
            final_output = accumulated_output if accumulated_output else completed.stdout
            assert isinstance(final_output, str), (
                f"final_output must be str, got {type(final_output).__name__}"
            )
            
            update_status_from_output(
                status_tracker,
                final_output,
                policy=policy,
                environment=environment,
                start_time=job_start_time,
            )
        except subprocess.TimeoutExpired as e:
            logger.error("⏱️ Subprocess TIMEOUT for job %s after %s seconds", job.job_id, e.timeout)
            raise
        except Exception as e:
            logger.error(
                "❌ Subprocess EXCEPTION for job %s:\n"
                "  Type: %s\n"
                "  Message: %s",
                job.job_id,
                type(e).__name__,
                str(e),
                exc_info=True,
            )
            raise
        finally:
            # Stop progress poller
            if poller_thread and poller_thread.is_alive():
                poller_stop.set()
                poller_thread.join(timeout=5)
                logger.info("📡 Stopped progress poller for job %s", job.job_id)
        
        # Log full output for debugging - prioritize auth errors
        logger.info("Training command returncode: %s", completed.returncode)
        
        # Check for critical errors FIRST - these should cause failure even if returncode is 0
        stdout_lower = (completed.stdout or "").lower()
        stderr_lower = (completed.stderr or "").lower()
        combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        combined_lower = combined_output.lower()
        
        # Check for health check failures (common cause of silent failures)
        health_check_failures = []
        health_check_details = []
        if "health check failed" in combined_lower or "aborting due to failing health check" in combined_lower:
            # Extract full context around health check failure - look for error patterns
            for source_name, source_text in [("STDOUT", completed.stdout), ("STDERR", completed.stderr)]:
                if not source_text:
                    continue
                source_lower = source_text.lower()
                if "health check" in source_lower:
                    # Find health check failure message
                    idx = source_lower.find("health check")
                    start = max(0, idx - 200)
                    end = min(len(source_text), idx + 500)
                    health_check_failures.append(f"{source_name} (health check context):\n{source_text[start:end]}")
                    
                    # Also look for error patterns that might explain WHY it failed
                    # Look for HTTP status codes, error messages, exceptions
                    if "500" in source_text or "internal server error" in source_lower:
                        # Find the 500 error context
                        error_idx = source_lower.find("500") if "500" in source_text else source_lower.find("internal server error")
                        if error_idx >= 0:
                            error_start = max(0, error_idx - 100)
                            error_end = min(len(source_text), error_idx + 800)
                            health_check_details.append(f"{source_name} (500 error details):\n{source_text[error_start:error_end]}")
                    
                    # Look for tracebacks or exception messages
                    if "traceback" in source_lower or "exception" in source_lower or "error:" in source_lower:
                        # Find traceback/exception
                        tb_idx = source_lower.find("traceback") if "traceback" in source_lower else (
                            source_lower.find("exception") if "exception" in source_lower else source_lower.find("error:")
                        )
                        if tb_idx >= 0:
                            tb_start = max(0, tb_idx - 50)
                            tb_end = min(len(source_text), tb_idx + 1500)  # Get more context for tracebacks
                            health_check_details.append(f"{source_name} (exception/traceback):\n{source_text[tb_start:tb_end]}")
                    
                    # Look for specific error messages like "ModuleNotFoundError", "RuntimeError", etc.
                    error_patterns = [
                        r"(ModuleNotFoundError|ImportError|RuntimeError|ValueError|KeyError|AttributeError)[^\n]*",
                        r"Failed to [^\n]+",
                        r"Unable to [^\n]+",
                        r"Missing [^\n]+",
                    ]
                    for pattern in error_patterns:
                        matches = re.finditer(pattern, source_text, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            match_start = max(0, match.start() - 100)
                            match_end = min(len(source_text), match.end() + 300)
                            health_check_details.append(f"{source_name} (error pattern '{pattern[:30]}...'):\n{source_text[match_start:match_end]}")
        
        if health_check_failures:
            success = False
            # Build informative error message
            error_parts = [
                "Training command failed health check. Task app endpoint returned error.",
            ]
            if health_check_details:
                error_parts.append("See details below for root cause.")
            else:
                error_parts.append("Check task app logs and ensure /task_info endpoint is working.")
            
            error_message = " ".join(error_parts)
            
            logger.error(
                "🚨 HEALTH CHECK FAILURE for job %s:\n%s",
                job.job_id,
                "\n".join(health_check_failures),
            )
            
            if health_check_details:
                logger.error(
                    "🔍 ROOT CAUSE ANALYSIS for job %s:\n%s",
                    job.job_id,
                    "\n" + "="*80 + "\n".join(health_check_details) + "\n" + "="*80,
                )
        
        # Check for authentication-related errors
        auth_keywords = [
            "authentication",
            "authorization",
            "api key",
            "api_key",
            "missing api",
            "invalid api",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "missing",
            "not set",
            "required",
        ]
        
        auth_errors = []
        for keyword in auth_keywords:
            if keyword in stdout_lower:
                # Extract context around the keyword
                idx = stdout_lower.find(keyword)
                start = max(0, idx - 100)
                end = min(len(completed.stdout), idx + 200)
                auth_errors.append(f"STDOUT: ...{completed.stdout[start:end]}...")
            if keyword in stderr_lower:
                idx = stderr_lower.find(keyword)
                start = max(0, idx - 100)
                end = min(len(completed.stderr), idx + 200)
                auth_errors.append(f"STDERR: ...{completed.stderr[start:end]}...")
        
        if auth_errors:
            logger.error(
                "🚨 AUTHENTICATION ERRORS DETECTED for job %s:\n%s",
                job.job_id,
                "\n".join(auth_errors),
            )
        
        # Log full output
        if completed.stdout:
            logger.info("Training command stdout (last 2000 chars):\n%s", completed.stdout[-2000:])
        else:
            logger.warning("Training command stdout is EMPTY - command may have exited before producing output")
            
        if completed.stderr:
            logger.warning("Training command stderr (last 2000 chars):\n%s", completed.stderr[-2000:])
        else:
            logger.info("Training command stderr is empty")
        # Validate inputs before collecting results
        assert prepared is not None, "prepared cannot be None"
        assert isinstance(prepared, PreparedConfig), (
            f"prepared must be PreparedConfig, got {type(prepared).__name__}"
        )
        assert isinstance(prepared.results_folder, Path), (
            f"prepared.results_folder must be Path, got {type(prepared.results_folder).__name__}"
        )
        assert isinstance(completed.stdout, str), (
            f"completed.stdout must be str, got {type(completed.stdout).__name__}"
        )
        assert isinstance(completed.stderr, str), (
            f"completed.stderr must be str, got {type(completed.stderr).__name__}"
        )
        
        artifact_summary = collect_result_summary(
            prepared.results_folder,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        assert isinstance(artifact_summary, ResultSummary), (
            f"collect_result_summary must return ResultSummary, got {type(artifact_summary).__name__}"
        )
        
        artifact_summary.stdout = _truncate(completed.stdout)
        assert isinstance(artifact_summary.stdout, str), (
            f"artifact_summary.stdout must be str after truncate, got {type(artifact_summary.stdout).__name__}"
        )
        artifact_summary.stderr = _truncate(completed.stderr)
        assert isinstance(artifact_summary.stderr, str), (
            f"artifact_summary.stderr must be str after truncate, got {type(artifact_summary.stderr).__name__}"
        )
        artifact_summary.returncode = completed.returncode
        assert isinstance(artifact_summary.returncode, int), (
            f"artifact_summary.returncode must be int, got {type(artifact_summary.returncode).__name__}"
        )
        summary = artifact_summary
        assert isinstance(summary, ResultSummary), (
            f"summary must be ResultSummary, got {type(summary).__name__}"
        )
        
        # Check if training actually ran - for prompt learning (GEPA/MIPRO), we expect results
        # Note: success may have been set to False above if health check failed
        if not error_message:  # Only check returncode if we haven't already detected a failure
            success = completed.returncode == 0
        if success and job.job_type == "gepa":
            # GEPA should produce rollouts - that's the primary indicator of success
            # If returncode is 0 but no rollouts were produced, it failed silently
            if summary.total_rollouts is None or summary.total_rollouts == 0:
                success = False
                error_message = (
                    "Training command exited with returncode 0 but produced no rollouts. "
                    "This indicates GEPA did not actually run. "
                    f"Check stdout/stderr for errors. "
                    f"Results folder: {prepared.results_folder}"
                )
                logger.error(
                    "Job %s failed silently: %s\nStdout tail:\n%s\nStderr tail:\n%s",
                    job.job_id,
                    error_message,
                    summary.stdout[-1000:] if summary.stdout else "(empty)",
                    summary.stderr[-1000:] if summary.stderr else "(empty)",
                )
            else:
                # We have rollouts - that's sufficient evidence GEPA ran successfully
                # Learning curve and stats are nice-to-have but not required
                logger.info(
                    "Job %s completed successfully with %d rollouts (best_score=%s, learning_curve_points=%d, stats=%s)",
                    job.job_id,
                    summary.total_rollouts,
                    summary.best_score,
                    len(summary.learning_curve_points),
                    "yes" if summary.stats else "no",
                )
        
        if not success and not error_message:
            error_message = f"Training command exited with {completed.returncode}"
            logger.error(
                "Job %s failed: %s\nStdout tail:\n%s\nStderr tail:\n%s",
                job.job_id,
                error_message,
                summary.stdout[-1000:] if summary.stdout else "(empty)",
                summary.stderr[-1000:] if summary.stderr else "(empty)",
            )
    except Exception as exc:
        error_message = str(exc)
        summary.stderr = _truncate((summary.stderr or "") + f"\n{error_message}")
        logger.exception("Job %s encountered error: %s", job.job_id, error_message)
    finally:
        if prepared:
            prepared.cleanup()

    return _finalize_job(job.job_id, summary=summary, success=success, error_message=error_message)


@celery_app.task(name="synth_ai.experiment_queue.process_experiment_queue")
def process_experiment_queue() -> dict[str, Any]:
    """Periodic task that checks for queued jobs and dispatches them.
    
    This task runs every 5 seconds (via Celery Beat) to ensure queued jobs
    are dispatched even if:
    - Previous dispatch attempts failed
    - Jobs were queued while other jobs were running
    - Worker restarted and missed dispatch events
    
    Returns a summary of dispatched jobs.
    """
    # Verify we're using the correct database
    from .config import load_config
    config = load_config()
    env_db_path = os.getenv("EXPERIMENT_QUEUE_DB_PATH")
    if env_db_path:
        from pathlib import Path
        env_db_path_resolved = Path(env_db_path).expanduser().resolve()
        if config.sqlite_path != env_db_path_resolved:
            logger.error(
                "Database path mismatch in periodic task! ENV: %s != CONFIG: %s",
                env_db_path_resolved,
                config.sqlite_path,
            )
    
    logger.debug("Processing experiment queue for queued jobs (database: %s)", config.sqlite_path)
    dispatched_count = 0
    experiments_checked = 0
    
    with session_scope() as session:
        # Find all running or queued experiments that might have jobs to dispatch
        active_experiments = (
            session.query(Experiment)
            .filter(
                Experiment.status.in_([ExperimentStatus.QUEUED, ExperimentStatus.RUNNING])
            )
            .all()
        )
        
        for experiment in active_experiments:
            experiments_checked += 1
            # Check if there are any queued jobs without celery_task_id
            queued_jobs = (
                session.query(ExperimentJob)
                .filter(
                    ExperimentJob.experiment_id == experiment.experiment_id,
                    ExperimentJob.status == ExperimentJobStatus.QUEUED,
                    ExperimentJob.celery_task_id.is_(None),
                )
                .count()
            )
            
            if queued_jobs > 0:
                logger.debug(
                    "Found %d queued jobs for experiment %s, attempting dispatch",
                    queued_jobs,
                    experiment.experiment_id,
                )
                dispatched = dispatch_available_jobs(session, experiment.experiment_id)
                dispatched_count += len(dispatched)
                if dispatched:
                    logger.info(
                        "Dispatched %d jobs for experiment %s",
                        len(dispatched),
                        experiment.experiment_id,
                    )
    
    result = {
        "dispatched": dispatched_count,
        "experiments_checked": experiments_checked,
    }
    logger.debug("Queue check completed: %s", result)
    return result
