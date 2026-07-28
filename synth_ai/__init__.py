"""Python-only Synth SDK surface."""

from __future__ import annotations

import importlib as _importlib
import warnings as _warnings
from importlib import metadata as _metadata
from pathlib import Path as _Path
from typing import Any as _Any

try:
    from synth_ai.core.utils.log_filter import install_log_filter as _install_log_filter

    _install_log_filter()
except Exception:
    pass

try:
    __version__ = _metadata.version("synth-ai")
except _metadata.PackageNotFoundError:
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as _toml  # type: ignore[no-redef]  # ty: ignore[unresolved-import]

    try:
        _pyproject_path = _Path(__file__).resolve().parents[1] / "pyproject.toml"
        with _pyproject_path.open("rb") as _fh:
            _pyproject = _toml.load(_fh)
        __version__ = str(_pyproject["project"]["version"])
    except Exception:
        __version__ = "0.0.0.dev0"


__all__ = [
    "AsyncContainerPoolsClient",
    "AsyncContainersClient",
    "AsyncHorizonsPrivateClient",
    "AsyncManagedAgentsAnthropicClient",
    "AsyncOpenAIAgentsSdkClient",
    "AsyncSynthClient",
    "AsyncSynthManagedAgents",
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "HorizonsPrivateClient",
    "ResearchApiError",
    "ManagedAgentsAnthropicClient",
    "ManagedAgentRun",
    "OpenAIAgentsSdkClient",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProjectCreateRequest",
    "ResearchSwarmLaunchRequest",
    "ResearchSwarmState",
    "SynthClient",
    "SynthManagedAgents",
    "TunnelsClient",
    "function_tool",
    "mcp_tool",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ResearchApiError": ("synth_ai.core.research.errors", "ResearchApiError"),
    "ResearchConcurrentRunLimitExceededError": (
        "synth_ai.core.research.errors",
        "ResearchConcurrentRunLimitExceededError",
    ),
    "ResearchInsufficientCreditsError": (
        "synth_ai.core.research.errors",
        "ResearchInsufficientCreditsError",
    ),
    "ResearchLimitExceededError": ("synth_ai.core.research.errors", "ResearchLimitExceededError"),
    "ResearchClient": ("synth_ai.core.research.facade", "ResearchClient"),
    "ResearchProjectCreateRequest": (
        "synth_ai.core.research.contracts",
        "ResearchProjectCreateRequest",
    ),
    "ResearchSwarmLaunchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmLaunchRequest",
    ),
    "ResearchSwarmState": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmState",
    ),
    "SynthClient": ("synth_ai.client", "SynthClient"),
    "AsyncSynthClient": ("synth_ai.client", "AsyncSynthClient"),
    "ContainersClient": ("synth_ai.sdk.containers", "ContainersClient"),
    "AsyncContainersClient": ("synth_ai.sdk.containers", "AsyncContainersClient"),
    "ContainerPoolsClient": ("synth_ai.sdk.pools", "ContainerPoolsClient"),
    "AsyncContainerPoolsClient": ("synth_ai.sdk.pools", "AsyncContainerPoolsClient"),
    "HorizonsPrivateClient": ("synth_ai.sdk.horizons_private", "HorizonsPrivateClient"),
    "AsyncHorizonsPrivateClient": (
        "synth_ai.sdk.horizons_private",
        "AsyncHorizonsPrivateClient",
    ),
    "ManagedAgentsAnthropicClient": (
        "synth_ai.sdk.managed_agents_anthropic",
        "ManagedAgentsAnthropicClient",
    ),
    "ManagedAgentRun": (
        "synth_ai.sdk.managed_agents_anthropic",
        "ManagedAgentRun",
    ),
    "AsyncManagedAgentsAnthropicClient": (
        "synth_ai.sdk.managed_agents_anthropic",
        "AsyncManagedAgentsAnthropicClient",
    ),
    "SynthManagedAgents": (
        "synth_ai.sdk.managed_agents",
        "SynthManagedAgents",
    ),
    "AsyncSynthManagedAgents": (
        "synth_ai.sdk.managed_agents",
        "AsyncSynthManagedAgents",
    ),
    "OpenAIAgentsSdkClient": (
        "synth_ai.sdk.openai_agents_sdk",
        "OpenAIAgentsSdkClient",
    ),
    "AsyncOpenAIAgentsSdkClient": (
        "synth_ai.sdk.openai_agents_sdk",
        "AsyncOpenAIAgentsSdkClient",
    ),
    "TunnelsClient": ("synth_ai.sdk.tunnels", "TunnelsClient"),
    "AsyncTunnelsClient": ("synth_ai.sdk.tunnels", "AsyncTunnelsClient"),
    "function_tool": ("synth_ai.sdk.openai_tools", "function_tool"),
    "mcp_tool": ("synth_ai.sdk.openai_tools", "mcp_tool"),
}


# Renamed when the SDK dropped the `Smr` prefix.  Reachable, so they get a window rather
# than a deletion; they are deliberately absent from `__all__`.
_DEPRECATED_ALIASES: dict[str, str] = {
    "SmrApiError": "ResearchApiError",
    "SmrConcurrentRunLimitExceededError": "ResearchConcurrentRunLimitExceededError",
    "SmrInsufficientCreditsError": "ResearchInsufficientCreditsError",
    "SmrLimitExceededError": "ResearchLimitExceededError",
}


def __getattr__(name: str) -> _Any:
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is not None:
        _warnings.warn(
            f"synth_ai.{name} is deprecated since synth-ai 0.18.0 and will be removed in "
            f"0.20.0 no earlier than 2026-10-01; use synth_ai.{canonical}.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = canonical

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = _importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)
