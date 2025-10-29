from __future__ import annotations

from dataclasses import dataclass


class EvalCliError(RuntimeError):
    """Base exception for eval CLI failures."""


@dataclass(slots=True)
class TomlUnavailableError(EvalCliError):
    hint: str | None = None


@dataclass(slots=True)
class EvalConfigNotFoundError(EvalCliError):
    path: str


@dataclass(slots=True)
class EvalConfigParseError(EvalCliError):
    path: str
    detail: str


@dataclass(slots=True)
class MissingEvalTableError(EvalCliError):
    """Raised when the eval config lacks an [eval] table."""


@dataclass(slots=True)
class InvalidEvalConfigError(EvalCliError):
    detail: str


@dataclass(slots=True)
class SeedParseError(EvalCliError):
    value: str


@dataclass(slots=True)
class MetadataFilterFormatError(EvalCliError):
    entry: str


@dataclass(slots=True)
class TaskInfoUnavailableError(EvalCliError):
    """Raised when metadata filters require task info but the task app does not expose it."""


@dataclass(slots=True)
class NoSeedsMatchedError(EvalCliError):
    hint: str | None = None


@dataclass(slots=True)
class MetadataSQLExecutionError(EvalCliError):
    query: str
    detail: str


@dataclass(slots=True)
class MetadataSQLResultError(EvalCliError):
    query: str
    detail: str


__all__ = [
    "EvalCliError",
    "TomlUnavailableError",
    "EvalConfigNotFoundError",
    "EvalConfigParseError",
    "MissingEvalTableError",
    "InvalidEvalConfigError",
    "SeedParseError",
    "MetadataFilterFormatError",
    "TaskInfoUnavailableError",
    "NoSeedsMatchedError",
    "MetadataSQLExecutionError",
    "MetadataSQLResultError",
]
