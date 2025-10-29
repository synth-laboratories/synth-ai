from dataclasses import dataclass


class TrainCliError(RuntimeError):
    """Base exception for train CLI failures."""


@dataclass(slots=True)
class TomlParseError(TrainCliError):
    """Raised when TOML file cannot be parsed."""
    path: str
    detail: str


@dataclass(slots=True)
class ConfigNotFoundError(TrainCliError):
    """Raised when config file is not found."""
    path: str


@dataclass(slots=True)
class InvalidSFTConfigError(TrainCliError):
    """Raised when SFT configuration is invalid."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class InvalidRLConfigError(TrainCliError):
    """Raised when RL configuration is invalid."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class MissingAlgorithmError(TrainCliError):
    """Raised when [algorithm] section is missing or invalid."""
    detail: str


@dataclass(slots=True)
class MissingModelError(TrainCliError):
    """Raised when model specification is missing."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class MissingDatasetError(TrainCliError):
    """Raised when dataset path is missing for SFT."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class MissingComputeError(TrainCliError):
    """Raised when compute configuration is missing or incomplete."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class UnsupportedAlgorithmError(TrainCliError):
    """Raised when algorithm type is not supported."""
    algorithm_type: str
    expected: str
    hint: str | None = None


@dataclass(slots=True)
class InvalidHyperparametersError(TrainCliError):
    """Raised when hyperparameters are invalid."""
    detail: str
    parameter: str | None = None


@dataclass(slots=True)
class InvalidTopologyError(TrainCliError):
    """Raised when topology configuration is invalid."""
    detail: str
    hint: str | None = None


@dataclass(slots=True)
class InvalidJudgeConfigError(TrainCliError):
    """Raised when judge configuration validation fails."""
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(slots=True)
class InvalidRubricConfigError(TrainCliError):
    """Raised when rubric configuration validation fails."""
    detail: str

    def __str__(self) -> str:
        return self.detail


__all__ = [
    "TrainCliError",
    "TomlParseError",
    "ConfigNotFoundError",
    "InvalidSFTConfigError",
    "InvalidRLConfigError",
    "MissingAlgorithmError",
    "MissingModelError",
    "MissingDatasetError",
    "MissingComputeError",
    "UnsupportedAlgorithmError",
    "InvalidHyperparametersError",
    "InvalidTopologyError",
    "InvalidJudgeConfigError",
    "InvalidRubricConfigError",
]
