"""Python-only Synth SDK surface."""

from __future__ import annotations

import importlib
from importlib import metadata as _metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

try:
    from synth_ai.core.utils.log_filter import install_log_filter

    install_log_filter()
except Exception:
    pass

try:
    __version__ = _metadata.version("synth-ai")
except PackageNotFoundError:
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as _toml  # type: ignore[no-redef]

    try:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as fh:
            _pyproject = _toml.load(fh)
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
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "HorizonsPrivateClient",
    "ManagedAgentsAnthropicClient",
    "OpenAIAgentsSdkClient",
    "SynthClient",
    "TunnelsClient",
    "function_tool",
    "mcp_tool",
]

_EXPORTS: dict[str, tuple[str, str]] = {
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
    "AsyncManagedAgentsAnthropicClient": (
        "synth_ai.sdk.managed_agents_anthropic",
        "AsyncManagedAgentsAnthropicClient",
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


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
