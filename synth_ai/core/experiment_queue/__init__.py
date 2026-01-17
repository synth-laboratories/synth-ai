"""Experiment queue service primitives (database, Celery integration, CLI helpers)."""

# Import submodules for re-export (relative imports)
# NOTE: celery_app, tasks, service, and dispatcher are lazy-loaded via __getattr__
# to avoid ~2s startup penalty when importing this package. These modules import
# the Celery app which connects to Redis and initializes the database - only needed
# for CLI queue commands, not for SDK/backend usage.
from . import (
    api_schemas,
    config,
    config_utils,
    database,
    models,
    progress_info,
    results,
    schemas,
    status,
    status_tracker,
    trace_storage,
    validation,
)

# Re-export main items
from .database import (
    Base,
    get_engine,
    get_session,
    init_db,
    session_scope,
)
from .models import (
    Experiment,
    ExperimentJob,
    ExperimentJobStatus,
    ExperimentStatus,
    Trial,
    TrialStatus,
)

__all__ = [
    # Submodules
    "api_schemas",
    "celery_app",  # lazy-loaded via __getattr__
    "config",
    "config_utils",
    "database",
    "dispatcher",
    "models",
    "progress_info",
    "results",
    "schemas",
    "service",
    "status",
    "status_tracker",
    "tasks",
    "trace_storage",
    "validation",
    # Main exports
    "Base",
    "Experiment",
    "ExperimentJob",
    "ExperimentJobStatus",
    "ExperimentStatus",
    "Trial",
    "TrialStatus",
    "get_engine",
    "get_session",
    "init_db",
    "session_scope",
]


# Modules that depend on celery_app and should be lazy-loaded
_LAZY_MODULES = {"celery_app", "tasks", "service", "dispatcher"}


def __getattr__(name: str):
    """Lazy-load celery-dependent modules to avoid ~2s startup penalty.
    
    The Celery app initialization connects to Redis and initializes the database,
    which is only needed for CLI queue commands, not for SDK/backend usage.
    Modules that import celery_app (tasks, service, dispatcher) are also lazy-loaded.
    """
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache for subsequent access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
