"""Shared training config validation helpers."""

from .errors import (
    InvalidRLConfigError,
    InvalidRubricConfigError,
    InvalidSFTConfigError,
    InvalidVerifierConfigError,
    MissingAlgorithmError,
    MissingComputeError,
    MissingDatasetError,
    MissingModelError,
    TomlParseError,
    TrainCliError,
    UnsupportedAlgorithmError,
)
from .prompt_learning_validation import validate_prompt_learning_config
from .validation import (
    load_and_validate_rl,
    load_and_validate_sft,
    validate_rl_config,
    validate_sft_config,
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

__all__ = [
    "TrainCliError",
    "InvalidVerifierConfigError",
    "InvalidRubricConfigError",
    "InvalidSFTConfigError",
    "InvalidRLConfigError",
    "MissingAlgorithmError",
    "MissingComputeError",
    "MissingDatasetError",
    "MissingModelError",
    "TomlParseError",
    "UnsupportedAlgorithmError",
    "validate_prompt_learning_config",
    "validate_sft_config",
    "validate_rl_config",
    "load_and_validate_sft",
    "load_and_validate_rl",
    "RubricWeightsConfig",
    "RubricConfig",
    "VerifierOptionsConfig",
    "VerifierConfig",
    "VerifierRequestPayload",
    "build_verifier_http_options",
    "validate_rubric_config",
    "validate_verifier_config",
    "extract_and_validate_verifier_rubric",
]
