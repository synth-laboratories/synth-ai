from synth_ai.sdk.task import task_app_health, validate_task_app_url

from .client import LearningClient
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
    RolloutStep,
    RolloutTrajectory,
    encrypt_for_backend,
    mint_environment_api_key,
    setup_environment_api_key,
)
from .sft import FtClient
from .sft.config import SFTJobConfig, prepare_sft_job_payload
from .sse import stream_events as stream_job_events
from .validators import validate_trainer_cfg_rl, validate_training_jsonl

__all__ = [
    "LearningClient",
    "RlClient",
    "RLJobConfig",
    "FtClient",
    "SFTJobConfig",
    "prepare_sft_job_payload",
    "PromptLearningClient",
    "get_prompts",
    "get_prompt_text",
    "get_scoring_summary",
    "RolloutEnvSpec",
    "RolloutPolicySpec",
    "RolloutRecordConfig",
    "RolloutSafetyConfig",
    "RolloutRequest",
    "RolloutStep",
    "RolloutTrajectory",
    "RolloutMetrics",
    "RolloutResponse",
    "mint_environment_api_key",
    "encrypt_for_backend",
    "setup_environment_api_key",
    "MAX_ENVIRONMENT_API_KEY_BYTES",
    # convenience re-export for typing
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
