"""Container template utilities (stub).

This module provides template building utilities for Container containers.
Currently a minimal stub - full implementation pending.
"""

from __future__ import annotations

from typing import Any

# Import directly from sdk.task to avoid circular import with container.__init__
from synth_ai.sdk.container._impl import ContainerConfig
from synth_ai.sdk.container._impl import create_container as create_container


def build_template_config(
    app_id: str = "template",
    name: str = "Template Container",
    description: str = "A template container.",
    **kwargs: Any,
) -> ContainerConfig:
    """Build a minimal ContainerConfig for testing/scaffolding.

    This is a placeholder - real containers should build their own config.
    """
    from synth_ai.sdk.container._impl.contracts import (
        RolloutMetrics,
        RolloutRequest,
        RolloutResponse,
    )

    async def stub_rollout(request: RolloutRequest, http_request: Any) -> RolloutResponse:
        """Stub rollout that returns empty metrics."""
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=0.0),
            trace={"event_history": [], "metadata": {}},
        )

    return ContainerConfig(
        app_id=app_id,
        name=name,
        description=description,
        provide_taskset_description=lambda: {"id": app_id, "splits": ["default"]},
        provide_task_instances=lambda seeds: [],
        rollout=stub_rollout,
        # base_task_info is auto-derived from app_id/name
        **kwargs,
    )


def create_template_app(**kwargs: Any):
    """Create a template FastAPI app for testing/scaffolding."""
    config = build_template_config(**kwargs)
    return create_container(config)
