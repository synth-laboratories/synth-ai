from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synth_ai.sdk.container._impl import container_health, validate_container_url

from .client import LearningClient
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
    # Utilities
    "validate_training_jsonl",
    "validate_trainer_cfg_rl",
    "validate_container_url",
    "backend_health",
    "container_health",
    "pricing_preflight",
    "balance_autumn_normalized",
    "stream_job_events",
    "JobHandle",
    "JobsApiResolver",
]

_TASK_IMPORTS = {"container_health", "validate_container_url"}


def __getattr__(name: str):
    if name in _TASK_IMPORTS:
        from synth_ai.sdk import task

        return getattr(task, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
