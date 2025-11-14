"""Celery task definitions for running experiment jobs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from celery.utils.log import get_task_logger

from .celery_app import celery_app
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


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _build_train_command(config_path: str) -> list[str]:
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
    with session_scope() as session:
        job = session.get(ExperimentJob, job_id)
        if not job:
            logger.warning("Job %s missing from database", job_id)
            return None
        job.status = ExperimentJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        if task_id:
            job.celery_task_id = task_id
        experiment = job.experiment
        if experiment and experiment.status == ExperimentStatus.QUEUED:
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.utcnow()
        session.flush()
        # Expunge so job can be safely used outside session scope
        session.expunge(job)
        return job


def _jobs_remaining(session, experiment_id: str) -> int:
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
    with session_scope() as session:
        job = session.get(ExperimentJob, job_id)
        if not job:
            logger.warning("Job %s missing during finalize", job_id)
            return None

        job.completed_at = datetime.utcnow()
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
                experiment.completed_at = datetime.utcnow()
            else:
                # Dispatch remaining jobs (periodic task will also handle this as backup)
                dispatch_available_jobs(session, experiment.experiment_id)

        return summary.to_dict()


@celery_app.task(bind=True, name="synth_ai.experiment_queue.run_experiment_job")
def run_experiment_job(self, job_id: str) -> dict[str, Any] | None:
    """Celery task entrypoint."""
    job = _mark_job_running(job_id, getattr(self.request, "id", None))
    if not job:
        return None

    summary = ResultSummary()
    prepared: PreparedConfig | None = None
    success = False
    error_message: str | None = None  # Will be set if training fails

    try:
        prepared = prepare_config_file(job.config_path, job.config_overrides or {})
        cmd = _build_train_command(str(prepared.path))
        logger.info("Executing job %s via command: %s", job.job_id, " ".join(cmd))
        
        # Run command with unbuffered output to see errors immediately
        env = os.environ.copy()
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
        
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=None,  # No timeout - let the training command handle its own timeouts
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
        artifact_summary = collect_result_summary(
            prepared.results_folder,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        artifact_summary.stdout = _truncate(completed.stdout)
        artifact_summary.stderr = _truncate(completed.stderr)
        artifact_summary.returncode = completed.returncode
        summary = artifact_summary
        
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
