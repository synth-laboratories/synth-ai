from __future__ import annotations

from dataclasses import dataclass


class FilterCliError(RuntimeError):
    """Base exception for filter CLI failures."""


@dataclass(slots=True)
class TomlUnavailableError(FilterCliError):
    hint: str | None = None


@dataclass(slots=True)
class FilterConfigNotFoundError(FilterCliError):
    path: str


@dataclass(slots=True)
class FilterConfigParseError(FilterCliError):
    path: str
    detail: str


@dataclass(slots=True)
class MissingFilterTableError(FilterCliError):
    """Raised when the filter config lacks a [filter] table."""


@dataclass(slots=True)
class InvalidFilterConfigError(FilterCliError):
    detail: str


@dataclass(slots=True)
class NoTracesFoundError(FilterCliError):
    db_url: str


@dataclass(slots=True)
class NoSessionsMatchedError(FilterCliError):
    hint: str | None = None


__all__ = [
    "FilterCliError",
    "TomlUnavailableError",
    "FilterConfigNotFoundError",
    "FilterConfigParseError",
    "MissingFilterTableError",
    "InvalidFilterConfigError",
    "NoTracesFoundError",
    "NoSessionsMatchedError",
]
