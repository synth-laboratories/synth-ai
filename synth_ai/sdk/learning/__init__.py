from synth_ai.sdk.task import task_app_health, validate_task_app_url

from .client import LearningClient
from .context_learning_client import (
    ContextLearningClient,
    create_job as create_context_learning_job,
    get_best_script as get_context_learning_best_script,
    get_job_status as get_context_learning_status,
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
from .prompt_learning_client import (
    PromptLearningClient,
    get_prompt_text,
    get_prompts,
    get_scoring_summary,
)
from .rl import (
    MAX_ENVIRONMENT_API_KEY_BYTES,
    RlClient,
    RLJobConfig,
    RolloutEnvSpec,
    RolloutMetrics,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutSafetyConfig,
    encrypt_for_backend,
    mint_environment_api_key,
    setup_environment_api_key,
)
from .sft import FtClient
from .sft.config import SFTJobConfig, prepare_sft_job_payload
from .sse import stream_events as stream_job_events
from .validators import validate_trainer_cfg_rl, validate_training_jsonl

__all__ = [
    # Learning clients
    "LearningClient",
    "RlClient",
    "RLJobConfig",
    "FtClient",
    "SFTJobConfig",
    "prepare_sft_job_payload",
    # Prompt Learning
    "PromptLearningClient",
    "get_prompts",
    "get_prompt_text",
    "get_scoring_summary",
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
    # RL types
    "RolloutEnvSpec",
    "RolloutPolicySpec",
    "RolloutRecordConfig",
    "RolloutSafetyConfig",
    "RolloutRequest",
    "RolloutMetrics",
    "RolloutResponse",
    "mint_environment_api_key",
    "encrypt_for_backend",
    "setup_environment_api_key",
    "MAX_ENVIRONMENT_API_KEY_BYTES",
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
