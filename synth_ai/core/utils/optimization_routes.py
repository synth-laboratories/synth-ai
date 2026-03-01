"""Canonical optimization route construction utilities.

Delegates to Rust core (synth_ai_py) only.
"""

from __future__ import annotations

from typing import Literal, cast

ApiVersion = Literal["v1", "v2"]

import synth_ai_py as _rs  # type: ignore


def _as_api_version(value: str) -> ApiVersion:
    normalized = value.strip().lower()
    if normalized in {"v1", "v2"}:
        return cast(ApiVersion, normalized)
    raise RuntimeError(f"Unsupported optimization API version from synth_ai_py: {value!r}")


GEPA_API_VERSION: ApiVersion = _as_api_version(_rs.optimization_route_gepa_api_version())
MIPRO_API_VERSION: ApiVersion = _as_api_version(_rs.optimization_route_mipro_api_version())
EVAL_API_VERSION: ApiVersion = _as_api_version(_rs.optimization_route_eval_api_version())


def normalize_api_version(value: str) -> ApiVersion:
    return _as_api_version(value)


def _normalize_suffix(suffix: str) -> str:
    return suffix if suffix.startswith("/") else f"/{suffix}"


def offline_jobs_base(*, api_version: ApiVersion) -> str:
    return _rs.optimization_route_offline_jobs_base(api_version)


def offline_job_path(job_id: str, *, api_version: ApiVersion) -> str:
    return _rs.optimization_route_offline_job_path(job_id, api_version)


def offline_job_subpath(job_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return _rs.optimization_route_offline_job_subpath(job_id, suffix, api_version)


def online_sessions_base(*, api_version: ApiVersion) -> str:
    return _rs.optimization_route_online_sessions_base(api_version)


def online_session_path(session_id: str, *, api_version: ApiVersion) -> str:
    return _rs.optimization_route_online_session_path(session_id, api_version)


def online_session_subpath(session_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return _rs.optimization_route_online_session_subpath(session_id, suffix, api_version)


def candidate_path(candidate_id: str, *, api_version: ApiVersion) -> str:
    return f"/{api_version}/candidates/{candidate_id}"


def candidate_subpath(candidate_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"{candidate_path(candidate_id, api_version=api_version)}{_normalize_suffix(suffix)}"


def system_subpath(system_id: str, suffix: str, *, api_version: ApiVersion) -> str:
    return f"/{api_version}/systems/{system_id}{_normalize_suffix(suffix)}"


def policy_systems_base(*, api_version: ApiVersion) -> str:
    return _rs.optimization_route_policy_systems_base(api_version)


def policy_system_path(system_id: str, *, api_version: ApiVersion) -> str:
    return _rs.optimization_route_policy_system_path(system_id, api_version)
