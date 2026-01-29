"""HarborInstanceProvider - Provide task instances from Harbor deployments.

This module provides an InstanceProvider that returns TaskInfo instances
based on Harbor deployment configuration. It maps seeds to Harbor instances
when seed/instance-based deployments are used.

Example:
    >>> from synth_ai.sdk.localapi import TaskAppConfig
    >>> from synth_ai.sdk.localapi.harbor_provider import HarborInstanceProvider
    >>> from synth_ai.sdk.harbor import HarborDeploymentRef
    >>>
    >>> provider = HarborInstanceProvider(
    ...     deployment_ref=HarborDeploymentRef(deployment_id="abc-123"),
    ...     task_id="enginebench",
    ...     task_name="Engine Bench",
    ... )
    >>>
    >>> config = TaskAppConfig(
    ...     app_id="enginebench",
    ...     name="Engine Bench",
    ...     description="...",
    ...     provide_taskset_description=lambda: provider.get_taskset_description(),
    ...     provide_task_instances=provider,  # Use Harbor provider
    ...     rollout=harbor_backend,
    ... )
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import httpx

from ._impl.contracts import (
    DatasetInfo,
    InferenceInfo,
    LimitsInfo,
    TaskDescriptor,
    TaskInfo,
)

if TYPE_CHECKING:
    from synth_ai.sdk.harbor import HarborDeploymentRef


class HarborInstanceProvider:
    """Provide TaskInfo instances from Harbor deployment metadata.

    This class implements the InstanceProvider callable interface,
    allowing it to be used as the `provide_task_instances` parameter
    in TaskAppConfig. It can either:

    1. Return static TaskInfo based on deployment metadata (simple mode)
    2. Query Harbor for instance-specific info when seed/instance
       deployments are used (instance mode)

    Attributes:
        deployment_ref: Reference to the Harbor deployment
        task_id: Task identifier
        task_name: Human-readable task name
        task_description: Task description
        dataset_info: Dataset metadata
        inference_info: Inference configuration
        limits_info: Operational limits

    Example:
        >>> from synth_ai.sdk.harbor import HarborDeploymentRef
        >>> from synth_ai.sdk.localapi.harbor_provider import HarborInstanceProvider
        >>>
        >>> provider = HarborInstanceProvider(
        ...     deployment_ref=HarborDeploymentRef(deployment_id="abc-123"),
        ...     task_id="my_task",
        ...     task_name="My Task",
        ... )
        >>>
        >>> # Get task info for seeds
        >>> instances = list(provider([0, 1, 2]))
    """

    def __init__(
        self,
        deployment_ref: HarborDeploymentRef,
        task_id: str,
        task_name: str,
        task_description: str | None = None,
        task_version: str | None = None,
        dataset_info: DatasetInfo | dict[str, Any] | None = None,
        inference_info: InferenceInfo | dict[str, Any] | None = None,
        limits_info: LimitsInfo | dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
        use_instance_api: bool = False,
    ) -> None:
        """Initialize Harbor instance provider.

        Args:
            deployment_ref: Reference to Harbor deployment
            task_id: Unique task identifier
            task_name: Human-readable task name
            task_description: Optional task description
            task_version: Optional task version
            dataset_info: Dataset metadata (or dict to construct DatasetInfo)
            inference_info: Inference configuration (or dict)
            limits_info: Operational limits (or dict)
            task_metadata: Additional task-specific metadata
            use_instance_api: Whether to query Harbor for per-seed instances
        """
        self.deployment_ref = deployment_ref
        self.task_id = task_id
        self.task_name = task_name
        self.task_description = task_description
        self.task_version = task_version
        self.task_metadata = task_metadata or {}
        self.use_instance_api = use_instance_api

        # Normalize info objects
        if isinstance(dataset_info, dict):
            self.dataset_info = DatasetInfo(**dataset_info)
        else:
            self.dataset_info = dataset_info or DatasetInfo(id=task_id, name=task_name)

        if isinstance(inference_info, dict):
            self.inference_info = InferenceInfo(**inference_info)
        else:
            self.inference_info = inference_info or InferenceInfo()

        if isinstance(limits_info, dict):
            self.limits_info = LimitsInfo(**limits_info)
        else:
            self.limits_info = limits_info or LimitsInfo()

        # Cache for deployment metadata
        self._deployment_metadata: dict[str, Any] | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Harbor API requests."""
        return {
            "Authorization": f"Bearer {self.deployment_ref.api_key}",
            "Content-Type": "application/json",
        }

    async def _fetch_deployment_metadata_async(self) -> dict[str, Any]:
        """Fetch deployment metadata from Harbor API."""
        if self._deployment_metadata is not None:
            return self._deployment_metadata

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.deployment_ref.status_url,
                headers=self._get_headers(),
            )

            if response.status_code >= 400:
                # Return empty metadata on error - don't fail instance provision
                return {}

            self._deployment_metadata = response.json()
            return self._deployment_metadata

    def _build_task_info(self, seed: int) -> TaskInfo:
        """Build a TaskInfo for a given seed.

        Args:
            seed: The seed value for this instance

        Returns:
            TaskInfo with Harbor deployment context
        """
        # Build task metadata with Harbor context
        metadata = {
            **self.task_metadata,
            "harbor_deployment_id": self.deployment_ref.deployment_id,
            "seed": seed,
        }

        return TaskInfo(
            task=TaskDescriptor(
                id=self.task_id,
                name=self.task_name,
                description=self.task_description,
                version=self.task_version,
            ),
            dataset=self.dataset_info,
            inference=self.inference_info,
            limits=self.limits_info,
            task_metadata=metadata,
        )

    async def _fetch_instances_async(self, seeds: Sequence[int]) -> list[TaskInfo]:
        """Fetch instance info from Harbor API for seed/instance deployments.

        This is used when use_instance_api=True and the deployment has
        pre-created instances for each seed.

        Args:
            seeds: List of seed values

        Returns:
            List of TaskInfo instances
        """
        # TODO: Implement when Harbor instance API is available
        # For now, fall back to static TaskInfo
        return [self._build_task_info(seed) for seed in seeds]

    async def provide_async(self, seeds: Sequence[int]) -> Iterable[TaskInfo]:
        """Provide TaskInfo instances asynchronously.

        Args:
            seeds: List of seed values to provide instances for

        Returns:
            Iterable of TaskInfo instances
        """
        if self.use_instance_api:
            return await self._fetch_instances_async(seeds)

        return [self._build_task_info(seed) for seed in seeds]

    def provide(self, seeds: Sequence[int]) -> Iterable[TaskInfo]:
        """Provide TaskInfo instances synchronously.

        Args:
            seeds: List of seed values to provide instances for

        Returns:
            Iterable of TaskInfo instances
        """
        # For simple mode, no async needed
        if not self.use_instance_api:
            return [self._build_task_info(seed) for seed in seeds]

        return asyncio.run(self.provide_async(seeds))

    def __call__(self, seeds: Sequence[int]) -> Iterable[TaskInfo]:
        """Make the provider callable as an InstanceProvider.

        This allows HarborInstanceProvider to be used directly as the
        `provide_task_instances` parameter in TaskAppConfig.

        Args:
            seeds: List of seed values

        Returns:
            Iterable of TaskInfo instances
        """
        return self.provide(seeds)

    def get_taskset_description(self) -> dict[str, Any]:
        """Get taskset description for use with provide_taskset_description.

        Returns:
            Dictionary with taskset metadata
        """
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "description": self.task_description,
            "harbor_deployment_id": self.deployment_ref.deployment_id,
            "splits": self.dataset_info.splits or ["default"],
            "default_split": self.dataset_info.default_split or "default",
        }

    def get_base_task_info(self) -> TaskInfo:
        """Get base TaskInfo for use in TaskAppConfig.

        Returns:
            TaskInfo with deployment defaults (no seed-specific info)
        """
        return TaskInfo(
            task=TaskDescriptor(
                id=self.task_id,
                name=self.task_name,
                description=self.task_description,
                version=self.task_version,
            ),
            dataset=self.dataset_info,
            inference=self.inference_info,
            limits=self.limits_info,
            task_metadata={
                **self.task_metadata,
                "harbor_deployment_id": self.deployment_ref.deployment_id,
            },
        )


def create_harbor_instance_provider(
    deployment_id: str,
    task_id: str,
    task_name: str,
    backend_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> HarborInstanceProvider:
    """Create a Harbor instance provider.

    Convenience function to create a HarborInstanceProvider with
    common defaults.

    Args:
        deployment_id: UUID of the Harbor deployment
        task_id: Unique task identifier
        task_name: Human-readable task name
        backend_url: Synth backend URL (default: from SYNTH_BACKEND_URL env)
        api_key: Synth API key (default: from SYNTH_API_KEY env)
        **kwargs: Additional arguments passed to HarborInstanceProvider

    Returns:
        HarborInstanceProvider ready for use

    Example:
        >>> from synth_ai.sdk.localapi.harbor_provider import create_harbor_instance_provider
        >>>
        >>> provider = create_harbor_instance_provider(
        ...     deployment_id="abc-123",
        ...     task_id="enginebench",
        ...     task_name="Engine Bench",
        ... )
    """
    from synth_ai.sdk.harbor import HarborDeploymentRef

    ref = HarborDeploymentRef(
        deployment_id=deployment_id,
        backend_url=backend_url,
        api_key=api_key,
    )

    return HarborInstanceProvider(
        deployment_ref=ref,
        task_id=task_id,
        task_name=task_name,
        **kwargs,
    )
