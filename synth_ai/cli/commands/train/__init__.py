from .core import register, train_command
from .errors import (
    InvalidJudgeConfigError,
    InvalidRubricConfigError,
    TrainCliError,
)
from .judge_schemas import (
    JudgeConfig,
    JudgeOptionsConfig,
    JudgeRequestPayload,
    RubricConfig,
    RubricWeightsConfig,
    build_judge_http_options,
)
from .judge_validation import (
    check_for_deprecated_fields,
    extract_and_validate_judge_rubric,
    validate_judge_config,
    validate_rubric_config,
)
from .validation import (
    load_and_validate_rl,
    load_and_validate_sft,
    validate_rl_config,
    validate_sft_config,
)

__all__ = [
    # Core
    "register",
    "train_command",
    # Errors
    "TrainCliError",
    "InvalidJudgeConfigError",
    "InvalidRubricConfigError",
    # SFT/RL validation
    "validate_sft_config",
    "validate_rl_config",
    "load_and_validate_sft",
    "load_and_validate_rl",
    # Judge/Rubric schemas
    "RubricWeightsConfig",
    "RubricConfig",
    "JudgeOptionsConfig",
    "JudgeConfig",
    "JudgeRequestPayload",
    "build_judge_http_options",
    # Judge/Rubric validation
    "validate_rubric_config",
    "validate_judge_config",
    "extract_and_validate_judge_rubric",
    "check_for_deprecated_fields",
]
