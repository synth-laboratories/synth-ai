from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_ai.sdk.task import task_app_health, validate_task_app_url

from .client import LearningClient
from .context_learning_client import (
    ContextLearningClient,
)
from .context_learning_client import (
    create_job as create_context_learning_job,
)
from .context_learning_client import (
    get_best_script as get_context_learning_best_script,
)
from .context_learning_client import (
    get_job_status as get_context_learning_status,
)
from .context_learning_client import (
    run_job as run_context_learning_job,
)
from .context_learning_types import (
    AlgorithmConfig,
    BestScriptResult,
    ContextLearningEvent,
    ContextLearningJobConfig,
    ContextLearningJobStatus,
    ContextLearningMetric,
    ContextLearningResults,
    EnvironmentConfig,
)
from .health import backend_health, balance_autumn_normalized, pricing_preflight
from .jobs import JobHandle, JobsApiResolver
from .pattern_discovery import (
    PatternDiscoveryClient,
    PatternDiscoveryRequest,
    get_eval_patterns,
)
from .prompt_learning_client import (
    PromptLearningClient,
    get_prompt_text,
    get_prompts,
    get_scoring_summary,
)
from .sse import stream_events as stream_job_events
from .validators import validate_trainer_cfg_rl, validate_training_jsonl

__all__ = [
    # Learning clients
    "LearningClient",
    # Prompt Learning
    "PromptLearningClient",
    "get_prompts",
    "get_prompt_text",
    "get_scoring_summary",
    "PatternDiscoveryClient",
    "PatternDiscoveryRequest",
    "get_eval_patterns",
    # Context Learning
    "ContextLearningClient",
    "ContextLearningJobConfig",
    "ContextLearningJobStatus",
    "ContextLearningEvent",
    "ContextLearningMetric",
    "ContextLearningResults",
    "BestScriptResult",
    "EnvironmentConfig",
    "AlgorithmConfig",
    "create_context_learning_job",
    "get_context_learning_status",
    "get_context_learning_best_script",
    "run_context_learning_job",
    # Utilities
    "validate_training_jsonl",
    "validate_trainer_cfg_rl",
    "validate_task_app_url",
    "backend_health",
    "task_app_health",
    "pricing_preflight",
    "balance_autumn_normalized",
    "stream_job_events",
    "JobHandle",
    "JobsApiResolver",
]

_TASK_IMPORTS = {"task_app_health", "validate_task_app_url"}


def __getattr__(name: str):
    if name in _TASK_IMPORTS:
        from synth_ai.sdk import task

        return getattr(task, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
