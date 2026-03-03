"""Canonical optimization route construction utilities.

Prefers Rust core (``synth_ai_py``) when available, with Python fallbacks for
mixed-version local environments.
"""

from __future__ import annotations

from typing import Literal, cast

ApiVersion = Literal["v1", "v2"]

try:
    import synth_ai_py as _rs  # type: ignore
except Exception:  # pragma: no cover - optional in dev envs
    _rs = None  # type: ignore[assignment]


def _as_api_version(value: str) -> ApiVersion:
    normalized = value.strip().lower()
    if normalized in {"v1", "v2"}:
        return cast(ApiVersion, normalized)
    raise ValueError(f"Unsupported optimization API version from synth_ai_py: {value!r}")


def _rs_or_fallback(name: str, fallback: str, *args: object) -> str:
    fn = getattr(_rs, name, None) if _rs is not None else None
    if callable(fn):
        try:
            value = fn(*args)
            if isinstance(value, str):
                return value
        except Exception:
            pass
    return fallback


def _resolve_api_version(name: str, default: ApiVersion) -> ApiVersion:
    raw = _rs_or_fallback(name, default)
    try:
        return _as_api_version(raw)
    except Exception:
        return default


def _api_prefix(api_version: ApiVersion) -> str:
    return f"/{normalize_api_version(api_version)}"


GEPA_API_VERSION: ApiVersion = _resolve_api_version("optimization_route_gepa_api_version", "v2")
MIPRO_API_VERSION: ApiVersion = _resolve_api_version("optimization_route_mipro_api_version", "v1")
EVAL_API_VERSION: ApiVersion = _resolve_api_version("optimization_route_eval_api_version", "v2")


def normalize_api_version(value: str) -> ApiVersion:
    return _as_api_version(value)


def _normalize_suffix(suffix: str) -> str:
    return suffix if suffix.startswith("/") else f"/{suffix}"


def offline_jobs_base(*, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_offline_jobs_base",
        f"{_api_prefix(version)}/offline/jobs",
        version,
    )


def offline_job_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_offline_job_path",
        f"{offline_jobs_base(api_version=version)}/{job_id}",
        job_id,
        version,
    )


def offline_job_subpath(job_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    normalized_suffix = _normalize_suffix(suffix)
    return _rs_or_fallback(
        "optimization_route_offline_job_subpath",
        f"{offline_job_path(job_id, api_version=version)}{normalized_suffix}",
        job_id,
        suffix,
        version,
    )


def offline_job_state_baseline_info_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_path(job_id, api_version=version)}/state/baseline-info"


def offline_job_state_envelope_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_path(job_id, api_version=version)}/state-envelope"


def offline_job_queue_trials_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_path(job_id, api_version=version)}/queue/trials"


def offline_job_queue_trial_path(job_id: str, trial_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_trials_path(job_id, api_version=version)}/{trial_id}"


def offline_job_queue_trials_reorder_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_trials_path(job_id, api_version=version)}/reorder"


def offline_job_queue_default_plan_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_path(job_id, api_version=version)}/queue/default-plan"


def offline_job_queue_rollouts_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_path(job_id, api_version=version)}/queue/rollouts"


def offline_job_queue_rollout_policy_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_rollouts_path(job_id, api_version=version)}/policy"


def offline_job_queue_rollout_metrics_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_rollouts_path(job_id, api_version=version)}/metrics"


def offline_job_queue_rollout_limiter_status_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_rollouts_path(job_id, api_version=version)}/limiter-status"


def offline_job_queue_rollout_retry_path(
    job_id: str,
    dispatch_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_rollouts_path(job_id, api_version=version)}/{dispatch_id}/retry"


def offline_job_queue_rollout_drain_path(job_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{offline_job_queue_rollouts_path(job_id, api_version=version)}/drain"


def online_sessions_base(*, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_online_sessions_base",
        f"{_api_prefix(version)}/online/sessions",
        version,
    )


def online_session_path(session_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_online_session_path",
        f"{online_sessions_base(api_version=version)}/{session_id}",
        session_id,
        version,
    )


def online_session_subpath(session_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    normalized_suffix = _normalize_suffix(suffix)
    return _rs_or_fallback(
        "optimization_route_online_session_subpath",
        f"{online_session_path(session_id, api_version=version)}{normalized_suffix}",
        session_id,
        suffix,
        version,
    )


def runtime_compatibility_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/runtime/compatibility"


def optimizer_events_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/optimizer/events"


def failures_query_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/failures/query"


def admin_optimizer_events_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/admin/optimizer/events"


def admin_failures_query_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/admin/failures/query"


def admin_victoria_logs_query_path(*, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/admin/victoria-logs/query"


def runtime_system_path(system_id: str, *, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/runtime/systems/{system_id}"


def runtime_system_subpath(system_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"{runtime_system_path(system_id, api_version=api_version)}{_normalize_suffix(suffix)}"


def runtime_session_path(session_id: str, *, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/runtime/sessions/{session_id}"


def runtime_session_subpath(session_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"{runtime_session_path(session_id, api_version=api_version)}{_normalize_suffix(suffix)}"


def runtime_container_rollout_checkpoint_dump_path(
    container_id: str,
    rollout_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    return (
        f"{_api_prefix(api_version)}/runtime/containers/{container_id}/rollouts/"
        f"{rollout_id}/checkpoint/dump"
    )


def runtime_container_rollout_checkpoint_restore_path(
    container_id: str,
    rollout_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    return (
        f"{_api_prefix(api_version)}/runtime/containers/{container_id}/rollouts/"
        f"{rollout_id}/checkpoint/restore"
    )


def runtime_queue_trials_path(system_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(system_id, "queue/trials", api_version=api_version)


def runtime_queue_contract_path(system_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(system_id, "queue/contract", api_version=api_version)


def runtime_queue_trial_path(system_id: str, trial_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(system_id, f"queue/trials/{trial_id}", api_version=api_version)


def runtime_queue_rollouts_path(system_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(system_id, "queue/rollouts", api_version=api_version)


def runtime_queue_rollout_path(system_id: str, rollout_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(
        system_id, f"queue/rollouts/{rollout_id}", api_version=api_version
    )


def runtime_queue_rollout_lease_path(system_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(system_id, "queue/rollouts/lease", api_version=api_version)


def runtime_queue_rollout_expire_leases_path(system_id: str, *, api_version: ApiVersion) -> str:
    return runtime_system_subpath(
        system_id,
        "queue/rollouts/expire-leases",
        api_version=api_version,
    )


def runtime_session_queue_trials_path(session_id: str, *, api_version: ApiVersion) -> str:
    return runtime_session_subpath(session_id, "queue/trials", api_version=api_version)


def runtime_session_queue_contract_path(session_id: str, *, api_version: ApiVersion) -> str:
    return runtime_session_subpath(session_id, "queue/contract", api_version=api_version)


def runtime_session_queue_trial_path(
    session_id: str,
    trial_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    return runtime_session_subpath(session_id, f"queue/trials/{trial_id}", api_version=api_version)


def runtime_session_queue_rollouts_path(session_id: str, *, api_version: ApiVersion) -> str:
    return runtime_session_subpath(session_id, "queue/rollouts", api_version=api_version)


def runtime_session_queue_rollout_path(
    session_id: str,
    rollout_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    return runtime_session_subpath(
        session_id,
        f"queue/rollouts/{rollout_id}",
        api_version=api_version,
    )


def runtime_session_queue_rollout_lease_path(session_id: str, *, api_version: ApiVersion) -> str:
    return runtime_session_subpath(session_id, "queue/rollouts/lease", api_version=api_version)


def runtime_session_queue_rollout_expire_leases_path(
    session_id: str,
    *,
    api_version: ApiVersion,
) -> str:
    return runtime_session_subpath(
        session_id,
        "queue/rollouts/expire-leases",
        api_version=api_version,
    )


def candidate_path(candidate_id: str, *, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/candidates/{candidate_id}"


def candidates_submit_path(*, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return f"{_api_prefix(version)}/candidates/submit"


def candidate_subpath(candidate_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"{candidate_path(candidate_id, api_version=api_version)}{_normalize_suffix(suffix)}"


def system_subpath(system_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"{_api_prefix(api_version)}/systems/{system_id}{_normalize_suffix(suffix)}"


def policy_systems_base(*, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_policy_systems_base",
        f"{_api_prefix(version)}/policy-optimization/systems",
        version,
    )


def policy_system_path(system_id: str, *, api_version: ApiVersion) -> str:
    version = normalize_api_version(api_version)
    return _rs_or_fallback(
        "optimization_route_policy_system_path",
        f"{policy_systems_base(api_version=version)}/{system_id}",
        system_id,
        version,
    )
