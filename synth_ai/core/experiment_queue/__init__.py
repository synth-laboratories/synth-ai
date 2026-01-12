"""Experiment queue service primitives (database, Celery integration, CLI helpers)."""

# Import submodules for re-export (relative imports)
from . import (
    api_schemas,
    config,
    database,
    models,
    progress_info,
    results,
    schemas,
    status,
    trace_storage,
    validation,
)

# Optional imports - these require extra dependencies (celery, tomli_w, etc.)
try:
    from . import celery_app
except Exception:
    celery_app = None  # type: ignore[assignment]

try:
    from . import dispatcher
except Exception:
    dispatcher = None  # type: ignore[assignment]

try:
    from . import service
except Exception:
    service = None  # type: ignore[assignment]

try:
    from . import config_utils
except Exception:
    config_utils = None  # type: ignore[assignment]

try:
    from . import status_tracker
except Exception:
    status_tracker = None  # type: ignore[assignment]

try:
    from . import tasks
except Exception:
    tasks = None  # type: ignore[assignment]

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
    "config",
    "database",
    "models",
    "progress_info",
    "results",
    "schemas",
    "status",
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

# Add optional modules to __all__ only if they were successfully imported
if celery_app is not None:
    __all__.append("celery_app")
if dispatcher is not None:
    __all__.append("dispatcher")
if service is not None:
    __all__.append("service")
if config_utils is not None:
    __all__.append("config_utils")
if status_tracker is not None:
    __all__.append("status_tracker")
if tasks is not None:
    __all__.append("tasks")
