"""Backwards-compatible Turso CLI entry point.

This module re-exports the infra.turso helpers so existing imports
(`synth_ai.cli.turso`) continue to work for tests and downstream users.
It also proxies infra functions so monkeypatching works in tests.
"""

from __future__ import annotations

import synth_ai.cli.infra.turso as _infra_turso
from synth_ai.cli.infra.turso import turso  # noqa: F401
from synth_ai.cli.root import SQLD_VERSION, find_sqld_binary, install_sqld  # noqa: F401

_get_sqld_version_inner = _infra_turso._get_sqld_version


def _get_sqld_version(binary: str) -> str | None:  # noqa: F401
    """Delegates to the original infra implementation (monkeypatchable)."""
    return _get_sqld_version_inner(binary)


def _proxy_get_sqld_version(binary: str) -> str | None:
    # Dynamic lookup so monkeypatching synth_ai.cli.turso._get_sqld_version is honored
    import synth_ai.cli.turso as mod

    return mod._get_sqld_version(binary)


def _proxy_find_sqld_binary() -> str | None:
    import synth_ai.cli.turso as mod

    return mod.find_sqld_binary()


def _proxy_install_sqld():
    import synth_ai.cli.turso as mod

    return mod.install_sqld()


# Point infra at proxies so tests that monkeypatch this module influence the CLI command.
_infra_turso._get_sqld_version = _proxy_get_sqld_version  # type: ignore[assignment]
_infra_turso.find_sqld_binary = _proxy_find_sqld_binary  # type: ignore[assignment]
_infra_turso.install_sqld = _proxy_install_sqld  # type: ignore[assignment]

__all__ = [
    "turso",
    "_get_sqld_version",
    "find_sqld_binary",
    "install_sqld",
    "SQLD_VERSION",
]
