"""LocalAPI template utilities (stub).

This module provides template building utilities for LocalAPI task apps.
Currently a minimal stub - full implementation pending.
"""

from __future__ import annotations

from typing import Any, Callable

from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api


def build_template_config(
    app_id: str = "template",
    name: str = "Template Task App",
    description: str = "A template task app.",
    **kwargs: Any,
) -> LocalAPIConfig:
    """Build a minimal LocalAPIConfig for testing/scaffolding.
    
    This is a placeholder - real task apps should build their own config.
    """
    from synth_ai.sdk.task.contracts import TaskInfo, RolloutRequest, RolloutResponse, RolloutMetrics
    
    base_task_info = TaskInfo(
        task={"id": app_id, "name": name, "version": "1.0.0"},
        environment=app_id,
        dataset={"id": app_id, "splits": ["default"]},
        rubric={"version": "1"},
        inference={"supports_proxy": False},
        limits={"max_turns": 1},
        task_metadata={},
    )
    
    async def stub_rollout(request: RolloutRequest, http_request: Any) -> RolloutResponse:
        """Stub rollout that returns empty metrics."""
        return RolloutResponse(
            run_id=request.run_id,
            branches={},
            metrics=RolloutMetrics(
                episode_returns=[0.0],
                mean_return=0.0,
                num_steps=0,
                num_episodes=1,
            ),
            aborted=False,
            trace={
                "event_history": [],
                "markov_blanket_message_history": [],
                "metadata": {},
            },
        )
    
    return LocalAPIConfig(
        app_id=app_id,
        name=name,
        description=description,
        base_task_info=base_task_info,
        describe_taskset=lambda: {"id": app_id, "splits": ["default"]},
        provide_task_instances=lambda seeds: [],
        rollout=stub_rollout,
        **kwargs,
    )


def create_template_app(**kwargs: Any):
    """Create a template FastAPI app for testing/scaffolding."""
    config = build_template_config(**kwargs)
    return create_local_api(config)
