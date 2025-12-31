from .core import register, train_command
from .errors import (
    InvalidRubricConfigError,
    InvalidVerifierConfigError,
    TrainCliError,
)
from .verifier_schemas import (
    RubricConfig,
    RubricWeightsConfig,
    VerifierConfig,
    VerifierOptionsConfig,
    VerifierRequestPayload,
    build_verifier_http_options,
)
from .verifier_validation import (
    extract_and_validate_verifier_rubric,
    validate_rubric_config,
    validate_verifier_config,
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
    "InvalidVerifierConfigError",
    "InvalidRubricConfigError",
    # SFT/RL validation
    "validate_sft_config",
    "validate_rl_config",
    "load_and_validate_sft",
    "load_and_validate_rl",
    # Verifier/Rubric schemas
    "RubricWeightsConfig",
    "RubricConfig",
    "VerifierOptionsConfig",
    "VerifierConfig",
    "VerifierRequestPayload",
    "build_verifier_http_options",
    # Verifier/Rubric validation
    "validate_rubric_config",
    "validate_verifier_config",
    "extract_and_validate_verifier_rubric",
]
