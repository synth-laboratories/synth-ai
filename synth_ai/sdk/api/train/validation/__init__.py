"""Shared training config validation helpers.

Note: RL and SFT validation has been moved to the research repo.
Those functions are no longer available in synth-ai.
"""

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


# RL/SFT validation moved to research repo - provide stubs for backwards compatibility
def validate_rl_config(*args, **kwargs):
    """RL config validation has been moved to the research repo."""
    raise NotImplementedError(
        "RL config validation has been moved to the research repo. "
        "Use the research repo for RL training workflows."
    )


def validate_sft_config(*args, **kwargs):
    """SFT config validation has been moved to the research repo."""
    raise NotImplementedError(
        "SFT config validation has been moved to the research repo. "
        "Use the research repo for SFT training workflows."
    )


def load_and_validate_rl(*args, **kwargs):
    """RL config loading has been moved to the research repo."""
    raise NotImplementedError(
        "RL config loading has been moved to the research repo. "
        "Use the research repo for RL training workflows."
    )


def load_and_validate_sft(*args, **kwargs):
    """SFT config loading has been moved to the research repo."""
    raise NotImplementedError(
        "SFT config loading has been moved to the research repo. "
        "Use the research repo for SFT training workflows."
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
