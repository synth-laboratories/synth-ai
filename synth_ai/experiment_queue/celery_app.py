"""Celery application configuration for the experiment queue."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ALWAYS clear config cache to ensure fresh config on worker startup
# This is critical for workers that may have cached config from previous runs
from . import config as queue_config

queue_config.reset_config_cache()

# Log the database path that will be used (before any imports that might cache it)
db_path_env = os.getenv("EXPERIMENT_QUEUE_DB_PATH")
if db_path_env:
    db_path_resolved = Path(db_path_env).expanduser().resolve()
    print(f"[celery_app] EXPERIMENT_QUEUE_DB_PATH={db_path_resolved}", file=sys.stderr, flush=True)
    # Verify the path exists or can be created
    db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    print(f"[celery_app] Database file will be: {db_path_resolved}", file=sys.stderr, flush=True)
else:
    print("[celery_app] EXPERIMENT_QUEUE_DB_PATH not set, will use default", file=sys.stderr, flush=True)

from celery import Celery  # noqa: E402

from .config import load_config, reset_config_cache  # noqa: E402
from .database import init_db  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level celery app instance and its config (for change detection)
_celery_app_instance: Celery | None = None
_celery_app_broker_url: str | None = None


def _create_celery_app() -> Celery:
    """Instantiate the Celery application with shared configuration."""
    # CRITICAL: EXPERIMENT_QUEUE_DB_PATH is REQUIRED - fail if not set
    env_db_path = os.getenv("EXPERIMENT_QUEUE_DB_PATH")
    if not env_db_path:
        error_msg = (
            "EXPERIMENT_QUEUE_DB_PATH environment variable is REQUIRED. "
            "This ensures all workers use the same database path. "
            "Set it before starting the worker."
        )
        print(f"[celery_app] ERROR: {error_msg}", file=sys.stderr, flush=True)
        raise RuntimeError(error_msg)
    
    # CRITICAL: Load config AFTER verifying env var is set
    # This ensures config uses the env var, not a cached default
    reset_config_cache()
    config = load_config()
    
    # CRITICAL: Verify database path matches environment variable
    env_db_path_resolved = Path(env_db_path).expanduser().resolve()
    config_db_path_resolved = config.sqlite_path.resolve()
    
    if config_db_path_resolved != env_db_path_resolved:
        error_msg = (
            f"Database path mismatch! "
            f"ENV: {env_db_path_resolved} != CONFIG: {config_db_path_resolved}. "
            f"This will cause workers to use different databases. "
            f"Clear config cache and restart worker."
        )
        print(f"[celery_app] ERROR: {error_msg}", file=sys.stderr, flush=True)
        raise RuntimeError(error_msg)
    
    # CRITICAL: Ensure database path is unique and enforce single instance
    # Verify no other process is using a different database path
    db_file = str(config.sqlite_path)
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize database (for our application DB only - Celery uses Redis broker)
    init_db()
    
    # Log database path for debugging (especially in worker subprocess)
    # Use print to stderr to ensure it's visible even if logging isn't configured
    print(
        f"[celery_app] Initializing with database: {config.sqlite_path} (broker: {config.broker_url})",
        file=sys.stderr,
        flush=True,
    )
    logger.info(
        "Celery app initializing with database: %s (broker: %s)",
        config.sqlite_path,
        config.broker_url,
    )

    app = Celery(
        "synth_ai.experiment_queue",
        broker=config.broker_url,
        backend=config.result_backend_url,
        include=["synth_ai.experiment_queue.tasks"],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=25,
        task_acks_late=True,
        broker_pool_limit=4,
        # Celery Beat schedule for periodic queue processing
        beat_schedule={
            "process-experiment-queue": {
                "task": "synth_ai.experiment_queue.process_experiment_queue",
                "schedule": 5.0,  # Run every 5 seconds
            },
        },
    )
    
    # Redis broker - no need to pre-create tables or configure WAL mode
    # Redis handles concurrency natively, no SQLite locking issues
    
    return app


def get_celery_app() -> Celery:
    """Return the shared Celery app, recreating it if config has changed.
    
    This function checks if the config cache was cleared (by comparing
    current broker URL to cached one) and recreates the app if needed.
    This ensures CLI and worker use the same database when env vars change.
    
    CRITICAL: EXPERIMENT_QUEUE_DB_PATH must be set before calling this.
    """
    global _celery_app_instance, _celery_app_broker_url
    
    # Check if config has changed by comparing broker URLs
    # Note: load_config() will use default if EXPERIMENT_QUEUE_DB_PATH not set,
    # but _create_celery_app() will fail if it's not set
    current_config = load_config()
    current_broker_url = current_config.broker_url
    
    if _celery_app_instance is None or _celery_app_broker_url != current_broker_url:
        # Config changed or app not initialized - recreate it
        logger.debug(
            "Recreating Celery app: broker URL changed from %s to %s",
            _celery_app_broker_url,
            current_broker_url,
        )
        _celery_app_instance = _create_celery_app()
        _celery_app_broker_url = current_broker_url
    
    return _celery_app_instance


# Create app at module import time (for worker entrypoints and decorators)
# This ensures workers get the correct config at startup
# CRITICAL: This will fail if EXPERIMENT_QUEUE_DB_PATH is not set
# For tests, set the env var before importing this module
try:
    celery_app = get_celery_app()
except RuntimeError as e:
    # If EXPERIMENT_QUEUE_DB_PATH is not set, create a placeholder that will fail on use
    # This allows the module to be imported but will fail when celery_app is actually used
    class _LazyCeleryApp:
        """Placeholder that raises error when accessed without EXPERIMENT_QUEUE_DB_PATH."""
        def __init__(self, original_error: RuntimeError):
            self._original_error = original_error
        
        def __getattr__(self, name):
            # Special handling for 'task' attribute used in decorators
            # This allows @celery_app.task decorators to be defined without immediate error
            if name == "task":
                # Return a decorator factory that will raise error when task is actually called
                def task_decorator(*args, **kwargs):
                    def decorator(func):
                        # Store the function, but wrap it to check env var on call
                        def wrapper(*func_args, **func_kwargs):
                            # Try to get real celery app now
                            try:
                                real_app = get_celery_app()
                                # Re-register the task with the real app
                                real_task = real_app.task(*args, **kwargs)(func)
                                return real_task(*func_args, **func_kwargs)
                            except RuntimeError as runtime_err:
                                raise RuntimeError(
                                    "EXPERIMENT_QUEUE_DB_PATH environment variable is REQUIRED. "
                                    "Set it before using the Celery app. "
                                    f"Original error: {self._original_error}"
                                ) from runtime_err
                        return wrapper
                    return decorator
                return task_decorator
            raise RuntimeError(
                "EXPERIMENT_QUEUE_DB_PATH environment variable is REQUIRED. "
                "Set it before using the Celery app. "
                f"Original error: {self._original_error}"
            )
        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "EXPERIMENT_QUEUE_DB_PATH environment variable is REQUIRED. "
                "Set it before using the Celery app."
            )
        def __getitem__(self, key):
            raise RuntimeError(
                "EXPERIMENT_QUEUE_DB_PATH environment variable is REQUIRED. "
                "Set it before using the Celery app."
            )
    celery_app = _LazyCeleryApp(e)  # type: ignore[assignment]
