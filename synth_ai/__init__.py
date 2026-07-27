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
        import tomli as _toml  # type: ignore[no-redef]  # ty: ignore[unresolved-import]

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
    "AsyncSynthManagedAgents",
    "AsyncTunnelsClient",
    "ContainerPoolsClient",
    "ContainersClient",
    "DataBindingCreateRequest",
    "DatasetRevisionCreateRequest",
    "HorizonsPrivateClient",
    "ResearchApiError",
    "ManagedAgentsAnthropicClient",
    "ManagedAgentRun",
    "OpenAIAgentsSdkClient",
    "MagiDecisionRequest",
    "MagiMode",
    "ProjectComputerProvisionRequest",
    "ProjectComputerReplaceRequest",
    "ResearchClient",
    "ResearchConcurrentRunLimitExceededError",
    "ResearchInsufficientCreditsError",
    "ResearchLimitExceededError",
    "ResearchProjectCreateRequest",
    "ResearchInternProvisionRequest",
    "ResearchInternResponse",
    "ResearchInternStatus",
    "ResearchSwarmLaunchRequest",
    "ResearchSwarmState",
    "ResearchVisual",
    "ResearchVisualPage",
    "ResearchVisualPatchRequest",
    "ResearchVisualPromotionRequest",
    "ResearchVisualVersions",
    "SynthClient",
    "SynthManagedAgents",
    "TunnelsClient",
    "function_tool",
    "mcp_tool",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DataBindingCreateRequest": (
        "synth_ai.core.research.contracts",
        "DataBindingCreateRequest",
    ),
    "DatasetRevisionCreateRequest": (
        "synth_ai.core.research.contracts",
        "DatasetRevisionCreateRequest",
    ),
    "MagiDecisionRequest": (
        "synth_ai.core.research.contracts",
        "MagiDecisionRequest",
    ),
    "MagiMode": ("synth_ai.core.research.contracts", "MagiMode"),
    "ProjectComputerProvisionRequest": (
        "synth_ai.core.research.contracts",
        "ProjectComputerProvisionRequest",
    ),
    "ProjectComputerReplaceRequest": (
        "synth_ai.core.research.contracts",
        "ProjectComputerReplaceRequest",
    ),
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
    "SmrApiError": ("synth_ai.core.research.errors", "ResearchApiError"),
    "SmrConcurrentRunLimitExceededError": (
        "synth_ai.core.research.errors",
        "ResearchConcurrentRunLimitExceededError",
    ),
    "SmrInsufficientCreditsError": (
        "synth_ai.core.research.errors",
        "ResearchInsufficientCreditsError",
    ),
    "SmrLimitExceededError": ("synth_ai.core.research.errors", "ResearchLimitExceededError"),
    "ResearchClient": ("synth_ai.core.research.facade", "ResearchClient"),
    "ResearchProjectCreateRequest": (
        "synth_ai.core.research.contracts",
        "ResearchProjectCreateRequest",
    ),
    "ResearchInternProvisionRequest": (
        "synth_ai.core.research.contracts",
        "ResearchInternProvisionRequest",
    ),
    "ResearchInternResponse": (
        "synth_ai.core.research.contracts",
        "ResearchInternResponse",
    ),
    "ResearchInternStatus": (
        "synth_ai.core.research.contracts",
        "ResearchInternStatus",
    ),
    "ResearchSwarmLaunchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmLaunchRequest",
    ),
    "ResearchSwarmState": (
        "synth_ai.core.research.contracts",
        "ResearchSwarmState",
    ),
    "ResearchVisual": ("synth_ai.core.research.contracts", "ResearchVisual"),
    "ResearchVisualPage": ("synth_ai.core.research.contracts", "ResearchVisualPage"),
    "ResearchVisualPatchRequest": (
        "synth_ai.core.research.contracts",
        "ResearchVisualPatchRequest",
    ),
    "ResearchVisualPromotionRequest": (
        "synth_ai.core.research.contracts",
        "ResearchVisualPromotionRequest",
    ),
    "ResearchVisualVersions": (
        "synth_ai.core.research.contracts",
        "ResearchVisualVersions",
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


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
