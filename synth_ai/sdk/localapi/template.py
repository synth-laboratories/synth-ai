"""LocalAPI template utilities (stub).

This module provides template building utilities for LocalAPI task apps.
Currently a minimal stub - full implementation pending.
"""

from __future__ import annotations

from typing import Any, Callable

# Import directly from sdk.task to avoid circular import with localapi.__init__
from synth_ai.sdk.task import LocalAPIConfig, create_task_app as create_local_api


def build_template_config(
    app_id: str = "template",
    name: str = "Template Task App",
    description: str = "A template task app.",
    **kwargs: Any,
) -> LocalAPIConfig:
    """Build a minimal LocalAPIConfig for testing/scaffolding.
    
    This is a placeholder - real task apps should build their own config.
    """
    from synth_ai.sdk.task.contracts import RolloutRequest, RolloutResponse, RolloutMetrics

    async def stub_rollout(request: RolloutRequest, http_request: Any) -> RolloutResponse:
        """Stub rollout that returns empty metrics."""
        return RolloutResponse(
            run_id=request.run_id,
            metrics=RolloutMetrics(outcome_reward=0.0),
            trace={"event_history": [], "metadata": {}},
        )

    return LocalAPIConfig(
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
    return create_local_api(config)
