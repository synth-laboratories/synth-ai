"""Helper utilities for building RolloutResponse with proper trace correlation."""

from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None

from synth_ai.sdk.container._impl.contracts import RolloutMetrics, RolloutRequest, RolloutResponse


def build_rollout_response(
    request: RolloutRequest,
    outcome_reward: float,
    inference_url: str | None = None,
    trace: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
    artifact: list[Any] | None = None,
    success_status: Any | None = None,
    status_detail: str | None = None,
    **kwargs,
) -> RolloutResponse:
    """Build a RolloutResponse from a RolloutRequest."""

    def _with_optional_metrics(resp: RolloutResponse) -> RolloutResponse:
        if not kwargs:
            return resp

        updates: dict[str, Any] = {}
        for key in (
            "event_rewards",
            "outcome_objectives",
            "event_objectives",
            "instance_objectives",
            "details",
        ):
            if key not in kwargs or kwargs[key] is None:
                continue

            current = getattr(resp.reward_info, key, None)
            provided = kwargs[key]

            if key == "details":
                if isinstance(provided, dict):
                    merged = dict(current or {})
                    for detail_key, detail_value in provided.items():
                        merged.setdefault(detail_key, detail_value)
                    if merged != current:
                        updates[key] = merged
                elif not current:
                    updates[key] = provided
                continue

            if current is None:
                updates[key] = provided

        if not updates:
            return resp

        reward_info = resp.reward_info.model_copy(update=updates)
        return resp.model_copy(update={"reward_info": reward_info})

    # The Rust extension can't serialize ContextOverride model instances.
    # Strip them from the request since they aren't needed for the response.
    if request.context_overrides is not None:
        request = request.model_copy(update={"context_overrides": None})

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_build_rollout_response"):
        # `value_from_pyobject` in the Rust extension expects plain Python JSON types,
        # not pydantic models.
        request_payload = request.model_dump()
        try:
            payload = synth_ai_py.container_build_rollout_response(
                request_payload,
                outcome_reward,
                inference_url,
                trace,
                policy_config,
                artifact,
                success_status,
                status_detail,
                kwargs if kwargs else None,
            )
            # Ensure we return the contract type
            if isinstance(payload, RolloutResponse):
                return _with_optional_metrics(payload)
            return _with_optional_metrics(RolloutResponse(**payload))
        except Exception:
            # Fall back to pure-Python construction when Rust parsing is too strict
            # (e.g. upstream sends floats for integer fields).
            pass

    trace_correlation_id = request.trace_correlation_id

    # Preserve optional metrics payloads even when we fall back to the pure-Python
    # construction path (e.g. when the Rust extension is unavailable or rejects
    # upstream payload types).
    metrics_kwargs: dict[str, Any] = {}
    for key in (
        "event_rewards",
        "outcome_objectives",
        "event_objectives",
        "instance_objectives",
        "details",
    ):
        if key in kwargs and kwargs[key] is not None:
            metrics_kwargs[key] = kwargs[key]

    reward_info = RolloutMetrics(outcome_reward=float(outcome_reward), **metrics_kwargs)
    trace_payload = trace
    if isinstance(trace_payload, dict) and trace_correlation_id:
        trace_payload = dict(trace_payload)
        metadata = trace_payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("trace_correlation_id", trace_correlation_id)
        trace_payload["metadata"] = metadata

    response = RolloutResponse(
        trace_correlation_id=trace_correlation_id,
        reward_info=reward_info,
        trace=trace_payload,
        inference_url=inference_url,
        artifact=artifact,
        success_status=success_status,
        status_detail=status_detail,
        override_application_results=kwargs.get("override_application_results"),
    )
    return _with_optional_metrics(response)
