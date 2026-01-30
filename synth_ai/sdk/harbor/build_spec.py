"""HarborBuildSpec - User-facing abstraction for Harbor deployment uploads.

This module defines the primary user-facing abstraction for uploading deployments
to Harbor. Users define a HarborBuildSpec with their Dockerfile, context, and
configuration, then use the SDK to package and upload it.

Example:
    >>> from synth_ai.sdk.harbor import HarborBuildSpec, upload_harbor_deployment
    >>>
    >>> spec = HarborBuildSpec(
    ...     name="enginebench-v1",
    ...     dockerfile_path="./Dockerfile",
    ...     context_dir=".",
    ...     entrypoint="run_rollout --input /tmp/rollout.json --output /tmp/result.json",
    ...     limits={"timeout_s": 600, "cpu_cores": 4, "memory_mb": 8192},
    ...     metadata={"agent_type": "opencode", "benchmark": "engine-bench"},
    ... )
    >>>
    >>> deployment = upload_harbor_deployment(spec, api_key="...")
    >>> print(deployment.deployment_id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HarborLimits:
    """Resource limits for Harbor deployment execution.

    Attributes:
        timeout_s: Maximum execution time in seconds (30-3600, default: 300)
        cpu_cores: Number of CPU cores (1-8, default: 2)
        memory_mb: Memory limit in MB (512-32768, default: 4096)
        disk_mb: Disk space limit in MB (1024-102400, default: 10240)
    """

    timeout_s: int = 300
    cpu_cores: int = 2
    memory_mb: int = 4096
    disk_mb: int = 10240

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for API requests."""
        return {
            "timeout_s": self.timeout_s,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "disk_mb": self.disk_mb,
        }


@dataclass
class HarborBuildSpec:
    """User-facing spec for Harbor deployment upload.

    This is the primary abstraction users interact with. Define your deployment
    configuration here, then use `upload_harbor_deployment()` to package and upload.

    Attributes:
        name: Deployment name (org-unique, 1-128 chars)
        dockerfile_path: Path to Dockerfile (relative or absolute)
        context_dir: Directory to package as build context
        entrypoint: Command to run (default: standard rollout runner)
        entrypoint_mode: "file" for JSON I/O or "stdio" for long-running commands
        description: Optional human-readable description
        env_vars: Environment variables (no LLM API keys allowed)
        limits: Resource limits (timeout, CPU, memory, disk)
        metadata: Additional metadata (agent_type, benchmark, version, etc.)
        include_globs: File patterns to include (default: all files)
        exclude_globs: File patterns to exclude (default: .git, __pycache__)

    Example:
        >>> spec = HarborBuildSpec(
        ...     name="my-agent-v1",
        ...     dockerfile_path="./Dockerfile",
        ...     context_dir=".",
        ...     entrypoint="python run_rollout.py --input /tmp/rollout.json --output /tmp/result.json",
        ...     limits=HarborLimits(timeout_s=600, memory_mb=8192),
        ...     metadata={"agent_type": "codex", "version": "1.0.0"},
        ... )
    """

    # Required fields
    name: str
    dockerfile_path: str | Path
    context_dir: str | Path

    # Entrypoint configuration
    entrypoint: str = "run_rollout --input /tmp/rollout.json --output /tmp/result.json"
    entrypoint_mode: str = "file"  # "file" or "stdio" (command alias supported)

    # Optional metadata
    description: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    limits: HarborLimits | dict[str, Any] = field(default_factory=HarborLimits)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Packaging options
    include_globs: list[str] = field(default_factory=lambda: ["**/*", "*"])
    exclude_globs: list[str] = field(
        default_factory=lambda: [
            ".git/**",
            ".git",
            "__pycache__/**",
            "__pycache__",
            "*.pyc",
            ".env",
            ".env.*",
            "*.egg-info/**",
            ".venv/**",
            "venv/**",
            "node_modules/**",
            ".pytest_cache/**",
            ".mypy_cache/**",
            ".ruff_cache/**",
        ]
    )

    def __post_init__(self) -> None:
        """Validate and normalize fields after initialization."""
        # Normalize paths
        self.dockerfile_path = Path(self.dockerfile_path)
        self.context_dir = Path(self.context_dir)

        # Normalize limits
        if isinstance(self.limits, dict):
            self.limits = HarborLimits(**self.limits)

        # Validate name
        if not self.name or len(self.name) > 128:
            raise ValueError("name must be 1-128 characters")

        # Normalize/validate entrypoint_mode
        if self.entrypoint_mode == "command":
            # Backward-compatible alias; backend expects "stdio".
            self.entrypoint_mode = "stdio"
        if self.entrypoint_mode not in ("file", "stdio"):
            raise ValueError("entrypoint_mode must be 'file' or 'stdio'")

        # Validate no LLM API keys in env_vars
        forbidden_keys = {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "HUGGINGFACE_API_KEY",
            "SYNTH_API_KEY",
        }
        for key in self.env_vars:
            key_upper = key.upper()
            if any(forbidden in key_upper for forbidden in forbidden_keys):
                raise ValueError(
                    f"env_vars cannot contain LLM API keys (found: {key}). "
                    "API keys are injected automatically via the interceptor."
                )

    def validate_paths(self) -> None:
        """Validate that Dockerfile and context directory exist.

        Call this before packaging to get early errors.

        Raises:
            FileNotFoundError: If Dockerfile or context_dir doesn't exist
        """
        dockerfile = Path(self.context_dir) / self.dockerfile_path
        if not dockerfile.exists() and not Path(self.dockerfile_path).exists():
            raise FileNotFoundError(
                f"Dockerfile not found: {self.dockerfile_path} (also checked: {dockerfile})"
            )

        if not self.context_dir.exists():
            raise FileNotFoundError(f"Context directory not found: {self.context_dir}")

        if not self.context_dir.is_dir():
            raise ValueError(f"context_dir must be a directory: {self.context_dir}")

    def get_dockerfile_content(self) -> str:
        """Read and return the Dockerfile content.

        Returns:
            Dockerfile content as string

        Raises:
            FileNotFoundError: If Dockerfile doesn't exist
        """
        # Try relative to context_dir first
        dockerfile = Path(self.context_dir) / self.dockerfile_path
        if dockerfile.exists():
            return dockerfile.read_text()

        # Try as absolute path
        if Path(self.dockerfile_path).exists():
            return Path(self.dockerfile_path).read_text()

        raise FileNotFoundError(f"Dockerfile not found: {self.dockerfile_path}")

    def to_api_request(self, context_tar_base64: str) -> dict[str, Any]:
        """Convert to Harbor API deployment creation request format.

        Args:
            context_tar_base64: Base64-encoded tar.gz of the build context

        Returns:
            Dictionary suitable for POST /api/harbor/deployments
        """
        limits = (
            self.limits if isinstance(self.limits, HarborLimits) else HarborLimits(**self.limits)
        )

        return {
            "name": self.name,
            "description": self.description,
            "dockerfile": self.get_dockerfile_content(),
            "context_tar_base64": context_tar_base64,
            "entrypoint": self.entrypoint,
            "entrypoint_mode": self.entrypoint_mode,
            "limits": limits.to_dict(),
            "env_vars": self.env_vars,
            "metadata": self.metadata,
        }


@dataclass
class HarborDeploymentRef:
    """Reference to an existing Harbor deployment.

    Used when configuring LocalAPIConfig to use Harbor as the execution backend.
    The deployment must already exist and be in READY state.

    Attributes:
        deployment_id: UUID of the existing deployment
        backend_url: Synth backend URL (default: from SYNTH_BACKEND_URL env)
        api_key: Synth API key (default: from SYNTH_API_KEY env)

    Example:
        >>> ref = HarborDeploymentRef(
        ...     deployment_id="abc-123-def",
        ...     backend_url="https://api-dev.usesynth.ai",
        ...     api_key=os.getenv("SYNTH_API_KEY"),
        ... )
    """

    deployment_id: str
    backend_url: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        """Resolve defaults from environment."""
        if self.backend_url is None:
            self.backend_url = os.getenv("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
        if self.api_key is None:
            self.api_key = os.getenv("SYNTH_API_KEY")

    @property
    def rollout_url(self) -> str:
        """Get the rollout endpoint URL for this deployment."""
        return f"{self.backend_url}/api/harbor/deployments/{self.deployment_id}/rollout"

    @property
    def status_url(self) -> str:
        """Get the status endpoint URL for this deployment."""
        return f"{self.backend_url}/api/harbor/deployments/{self.deployment_id}/status"


@dataclass
class HarborDeploymentResult:
    """Result of a Harbor deployment upload operation.

    Returned by `upload_harbor_deployment()` after successful upload.

    Attributes:
        deployment_id: UUID of the created deployment
        build_id: UUID of the triggered build (if auto_build=True)
        name: Deployment name
        status: Current deployment status
        snapshot_id: Daytona snapshot ID (set after build completes)
    """

    deployment_id: str
    build_id: str | None
    name: str
    status: str
    snapshot_id: str | None = None
    deployment_name: str | None = None

    def to_ref(
        self, backend_url: str | None = None, api_key: str | None = None
    ) -> HarborDeploymentRef:
        """Convert to a HarborDeploymentRef for use with LocalAPIConfig.

        Args:
            backend_url: Override backend URL (default: from environment)
            api_key: Override API key (default: from environment)

        Returns:
            HarborDeploymentRef for this deployment
        """
        deployment_key = self.deployment_name or self.deployment_id
        return HarborDeploymentRef(
            deployment_id=deployment_key,
            backend_url=backend_url,
            api_key=api_key,
        )
