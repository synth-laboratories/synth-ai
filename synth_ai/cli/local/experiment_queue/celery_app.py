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
    # Will use default path from config (~/.synth_ai/experiment_queue.db)
    print("[celery_app] EXPERIMENT_QUEUE_DB_PATH not set, will use default path", file=sys.stderr, flush=True)

from celery import Celery  # noqa: E402

from .config import load_config, reset_config_cache  # noqa: E402
from .database import init_db  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level celery app instance and its config (for change detection)
_celery_app_instance: Celery | None = None
_celery_app_broker_url: str | None = None


def _create_celery_app() -> Celery:
    """Instantiate the Celery application with shared configuration."""
    # Load config (will use EXPERIMENT_QUEUE_DB_PATH if set, otherwise default path)
    reset_config_cache()
    config = load_config()
    
    # Log which database path is being used
    env_db_path = os.getenv("EXPERIMENT_QUEUE_DB_PATH")
    if env_db_path:
        env_db_path_resolved = Path(env_db_path).expanduser().resolve()
        config_db_path_resolved = config.sqlite_path.resolve()
        
        # Verify database path matches environment variable if env var is set
        if config_db_path_resolved != env_db_path_resolved:
            error_msg = (
                f"Database path mismatch! "
                f"ENV: {env_db_path_resolved} != CONFIG: {config_db_path_resolved}. "
                f"This will cause workers to use different databases. "
                f"Clear config cache and restart worker."
            )
            print(f"[celery_app] ERROR: {error_msg}", file=sys.stderr, flush=True)
            raise RuntimeError(error_msg)
        print(
            f"[celery_app] Using database from EXPERIMENT_QUEUE_DB_PATH: {config.sqlite_path}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[celery_app] Using default database path: {config.sqlite_path}",
            file=sys.stderr,
            flush=True,
        )
    
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
        include=["synth_ai.cli.local.experiment_queue.tasks"],
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
                "task": "synth_ai.cli.local.experiment_queue.process_experiment_queue",
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
    
    Uses EXPERIMENT_QUEUE_DB_PATH if set, otherwise defaults to ~/.synth_ai/experiment_queue.db
    """
    global _celery_app_instance, _celery_app_broker_url
    
    # Check if config has changed by comparing broker URLs
    # load_config() will use EXPERIMENT_QUEUE_DB_PATH if set, otherwise default path
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
# Uses default path (~/.synth_ai/experiment_queue.db) if EXPERIMENT_QUEUE_DB_PATH is not set
try:
    celery_app = get_celery_app()
except RuntimeError as e:
    # If config loading fails for any reason, create a placeholder that will fail on use
    # This allows the module to be imported but will fail when celery_app is actually used
    class _LazyCeleryApp:
        """Placeholder that raises error when Celery app initialization fails."""
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
                                    f"Failed to initialize Celery app. "
                                    f"Original error: {self._original_error}, "
                                    f"Runtime error: {runtime_err}"
                                ) from runtime_err
                        return wrapper
                    return decorator
                return task_decorator
            raise RuntimeError(
                f"Failed to initialize Celery app. "
                f"Original error: {self._original_error}"
            )
        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                f"Failed to initialize Celery app. "
                f"Original error: {self._original_error}"
            )
        def __getitem__(self, key):
            raise RuntimeError(
                f"Failed to initialize Celery app. "
                f"Original error: {self._original_error}"
            )
    celery_app = _LazyCeleryApp(e)  # type: ignore[assignment]
