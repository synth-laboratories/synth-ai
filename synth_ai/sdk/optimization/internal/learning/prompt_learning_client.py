"""Client utilities for querying prompt learning job results."""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.core.utils.optimization_routes import (
    ApiVersion,
    candidate_path,
    candidate_subpath,
    candidates_submit_path,
    normalize_api_version,
    offline_job_path,
    offline_job_queue_default_plan_path,
    offline_job_queue_rollout_drain_path,
    offline_job_queue_rollout_limiter_status_path,
    offline_job_queue_rollout_metrics_path,
    offline_job_queue_rollout_policy_path,
    offline_job_queue_rollout_retry_path,
    offline_job_queue_rollouts_path,
    offline_job_queue_trial_path,
    offline_job_queue_trials_path,
    offline_job_queue_trials_reorder_path,
    offline_job_state_baseline_info_path,
    offline_job_state_envelope_path,
    offline_job_subpath,
    online_session_path,
    online_session_subpath,
    system_subpath,
)
from synth_ai.sdk.optimization.internal.utils import run_sync
from synth_ai.sdk.optimization.models import PolicyCandidate, PolicyCandidatePage

from .prompt_extraction import extract_candidate_content
from .prompt_learning_types import PromptResults


def _resolve_api_version(explicit: Optional[str]) -> ApiVersion:
    if explicit is not None:
        return normalize_api_version(explicit)
    import os

    for env_var in ("SYNTH_POLICY_API_VERSION", "SYNTH_PROMPT_OPT_API_VERSION"):
        raw = os.getenv(env_var)
        if raw:
            return normalize_api_version(raw)
    return "v1"


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _with_algorithm_kind(path: str, algorithm_kind: Optional[str]) -> str:
    normalized = str(algorithm_kind or "").strip().lower()
    if normalized not in {"gepa", "mipro"}:
        return path
    joiner = "&" if "?" in path else "?"
    return f"{path}{joiner}algorithm_kind={normalized}"


def _as_versioned_api_path(path: str) -> str:
    if path.startswith("/api/"):
        return path
    if path.startswith("/"):
        return f"/api{path}"
    return f"/api/{path}"


def _as_global_api_path(path: str) -> str:
    if path.startswith("/api/"):
        return path
    normalized = path if path.startswith("/") else f"/{path}"
    normalized = re.sub(r"^/v[0-9]+(?=/)", "", normalized)
    return f"/api{normalized}"


def _parse_lever_versions_raw(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    versions: Dict[str, int] = {}
    for key, value in raw.items():
        parsed = _coerce_int(value)
        if parsed is not None:
            versions[str(key)] = parsed
    return versions


def _parse_lever_versions(payload: Dict[str, Any]) -> Dict[str, int]:
    return _parse_lever_versions_raw(payload.get("lever_versions"))


def _first_present(candidates: Iterable[Any]) -> Any:
    for value in candidates:
        if value is not None:
            return value
    return None


def _merge_job_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        merged.update(metadata)
    job_metadata = payload.get("job_metadata")
    if isinstance(job_metadata, dict):
        merged.update(job_metadata)
    return merged


def _merge_metadata_from_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for payload in payloads:
        merged.update(_merge_job_metadata(payload))
    return merged


def _extract_candidate_content_text(candidate: Any) -> Optional[str]:
    if isinstance(candidate, str):
        candidate = candidate.strip()
        return candidate or None
    if not isinstance(candidate, dict):
        return None
    extracted = extract_candidate_content(candidate)
    if isinstance(extracted, str):
        extracted = extracted.strip()
        if extracted:
            return extracted

    messages = candidate.get("messages")
    if isinstance(messages, list):
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = message.get("content") or message.get("pattern") or message.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n\n".join(parts)

    sections = candidate.get("sections")
    if isinstance(sections, list):
        parts = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            text = section.get("content") or section.get("pattern")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n\n".join(parts)

    replacements = candidate.get("text_replacements")
    if isinstance(replacements, list):
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            text = replacement.get("new_text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _extract_best_candidate_content_from_sources(
    best_candidate: Optional[Dict[str, Any] | str],
    *sources: Dict[str, Any],
) -> Optional[str]:
    direct = _extract_candidate_content_text(best_candidate)
    if direct:
        return direct

    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in (
            "best_candidate_content",
            "candidate_content",
            "best_candidate_text",
            "candidate_code",
            "best_candidate_code",
        ):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("best_candidate", "best_prompt", "candidate", "program_candidate"):
            extracted = _extract_candidate_content_text(source.get(key))
            if extracted:
                return extracted
    return None


def _infer_lever_versions_from_summary(lever_summary: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(lever_summary, dict):
        return {}
    prompt_lever_id = lever_summary.get("prompt_lever_id")
    candidate_versions = lever_summary.get("candidate_lever_versions")
    if not isinstance(prompt_lever_id, str) or not isinstance(candidate_versions, dict):
        return {}
    candidate_id = (
        lever_summary.get("selected_candidate_id")
        or lever_summary.get("best_candidate_id")
        or (next(iter(candidate_versions.keys())) if len(candidate_versions) == 1 else None)
    )
    if candidate_id is None:
        return {}
    version = _coerce_int(candidate_versions.get(str(candidate_id)))
    if version is None:
        return {}
    return {prompt_lever_id: version}


def _infer_best_lever_version_from_summary(
    lever_summary: Optional[Dict[str, Any]],
    *,
    best_candidate_id: Optional[str] = None,
) -> Optional[int]:
    if not isinstance(lever_summary, dict):
        return None
    candidate_versions = lever_summary.get("candidate_lever_versions")
    if not isinstance(candidate_versions, dict):
        return None
    candidate_id = (
        best_candidate_id
        or lever_summary.get("selected_candidate_id")
        or lever_summary.get("best_candidate_id")
        or (next(iter(candidate_versions.keys())) if len(candidate_versions) == 1 else None)
    )
    if candidate_id is None:
        return None
    return _coerce_int(candidate_versions.get(str(candidate_id)))


def _extract_prompt_learning_fields_from_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _merge_job_metadata(payload)
    nested_result = payload.get("result") if isinstance(payload.get("result"), dict) else {}

    best_candidate = _first_present(
        [
            payload.get("best_candidate"),
            payload.get("best_prompt"),
            nested_result.get("best_candidate"),
            nested_result.get("best_prompt"),
            metadata.get("best_candidate"),
            metadata.get("best_prompt"),
        ]
    )
    if not isinstance(best_candidate, (dict, str)):
        best_candidate = None
    best_candidate_content = _extract_best_candidate_content_from_sources(
        best_candidate,
        payload,
        nested_result,
        metadata,
    )

    best_reward = _first_present(
        [
            _coerce_float(payload.get("best_reward")),
            _coerce_float(payload.get("best_score")),
            _coerce_float(nested_result.get("best_reward")),
            _coerce_float(nested_result.get("best_score")),
            _coerce_float(metadata.get("best_reward")),
            _coerce_float(metadata.get("best_score")),
            _coerce_float(metadata.get("prompt_best_average_reward")),
            _coerce_float(metadata.get("prompt_best_reward")),
        ]
    )

    lever_summary_raw = _first_present(
        [
            payload.get("lever_summary"),
            nested_result.get("lever_summary"),
            metadata.get("lever_summary"),
        ]
    )
    lever_summary = lever_summary_raw if isinstance(lever_summary_raw, dict) else None

    sensor_frames_raw = _first_present(
        [
            payload.get("sensor_frames"),
            nested_result.get("sensor_frames"),
            metadata.get("sensor_frames"),
        ]
    )
    sensor_frames = (
        [frame for frame in sensor_frames_raw if isinstance(frame, dict)]
        if isinstance(sensor_frames_raw, list)
        else []
    )

    lever_versions = _parse_lever_versions_raw(
        _first_present(
            [
                payload.get("lever_versions"),
                nested_result.get("lever_versions"),
                metadata.get("lever_versions"),
            ]
        )
    )
    inferred_from_summary = _infer_lever_versions_from_summary(lever_summary)
    summary_prompt_lever_id = (
        lever_summary.get("prompt_lever_id") if isinstance(lever_summary, dict) else None
    )
    if inferred_from_summary and (
        not lever_versions
        or (
            isinstance(summary_prompt_lever_id, str)
            and summary_prompt_lever_id not in lever_versions
        )
    ):
        lever_versions = inferred_from_summary

    best_lever_version = _first_present(
        [
            _coerce_int(payload.get("best_lever_version")),
            _coerce_int(nested_result.get("best_lever_version")),
            _coerce_int(metadata.get("best_lever_version")),
        ]
    )
    if best_lever_version is None:
        best_lever_version = _infer_best_lever_version_from_summary(lever_summary)
    if best_lever_version is None and lever_versions:
        best_lever_version = max(lever_versions.values())

    optimized_candidates = _first_present(
        [
            payload.get("optimized_candidates"),
            nested_result.get("optimized_candidates"),
            metadata.get("optimized_candidates"),
        ]
    )
    attempted_candidates = _first_present(
        [
            payload.get("attempted_candidates"),
            nested_result.get("attempted_candidates"),
            metadata.get("attempted_candidates"),
        ]
    )
    validation_results = _first_present(
        [
            payload.get("validation_results"),
            nested_result.get("validation_results"),
            payload.get("validation"),
            nested_result.get("validation"),
            metadata.get("validation"),
        ]
    )

    return {
        "best_candidate": best_candidate,
        "best_candidate_content": best_candidate_content,
        "best_reward": best_reward,
        "lever_summary": lever_summary,
        "sensor_frames": sensor_frames,
        "lever_versions": lever_versions,
        "best_lever_version": best_lever_version,
        "optimized_candidates": optimized_candidates
        if isinstance(optimized_candidates, list)
        else [],
        "attempted_candidates": attempted_candidates
        if isinstance(attempted_candidates, list)
        else [],
        "validation_results": validation_results if isinstance(validation_results, list) else [],
    }


def _merge_prompt_results_from_job_payload(result: PromptResults, payload: Dict[str, Any]) -> None:
    fields = _extract_prompt_learning_fields_from_job_payload(payload)
    if result.best_candidate is None and fields["best_candidate"] is not None:
        result.best_candidate = fields["best_candidate"]
    if result.best_candidate_content is None and fields["best_candidate_content"] is not None:
        result.best_candidate_content = fields["best_candidate_content"]
    if result.best_reward is None and fields["best_reward"] is not None:
        result.best_reward = fields["best_reward"]
    if result.lever_summary is None and fields["lever_summary"] is not None:
        result.lever_summary = fields["lever_summary"]
    if not result.sensor_frames and fields["sensor_frames"]:
        result.sensor_frames = fields["sensor_frames"]
    if not result.lever_versions and fields["lever_versions"]:
        result.lever_versions = fields["lever_versions"]
    if result.best_lever_version is None:
        result.best_lever_version = fields["best_lever_version"]
    if not result.optimized_candidates and fields["optimized_candidates"]:
        result.optimized_candidates = fields["optimized_candidates"]
    if not result.attempted_candidates and fields["attempted_candidates"]:
        result.attempted_candidates = fields["attempted_candidates"]
    if not result.validation_results and fields["validation_results"]:
        result.validation_results = fields["validation_results"]
    if result.best_lever_version is None and result.lever_versions:
        result.best_lever_version = max(result.lever_versions.values())


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg


def _coerce_events_list(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, list):
            return [event for event in events if isinstance(event, dict)]
        return None
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    return None


def _merge_job_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not payloads:
        return {}
    merged = dict(payloads[0])
    merged_metadata = _merge_metadata_from_payloads(payloads)
    if merged_metadata:
        merged["metadata"] = merged_metadata
        merged["job_metadata"] = merged_metadata

    best_fields: Dict[str, Any] = {
        "best_candidate": None,
        "best_candidate_content": None,
        "best_reward": None,
        "lever_summary": None,
        "sensor_frames": [],
        "lever_versions": {},
        "best_lever_version": None,
        "optimized_candidates": [],
        "attempted_candidates": [],
        "validation_results": [],
    }
    for payload in payloads:
        fields = _extract_prompt_learning_fields_from_job_payload(payload)
        if best_fields["best_candidate"] is None and fields["best_candidate"] is not None:
            best_fields["best_candidate"] = fields["best_candidate"]
        if (
            best_fields["best_candidate_content"] is None
            and fields["best_candidate_content"] is not None
        ):
            best_fields["best_candidate_content"] = fields["best_candidate_content"]
        if best_fields["best_reward"] is None and fields["best_reward"] is not None:
            best_fields["best_reward"] = fields["best_reward"]
        if best_fields["lever_summary"] is None and fields["lever_summary"] is not None:
            best_fields["lever_summary"] = fields["lever_summary"]
        if not best_fields["sensor_frames"] and fields["sensor_frames"]:
            best_fields["sensor_frames"] = fields["sensor_frames"]
        if not best_fields["lever_versions"] and fields["lever_versions"]:
            best_fields["lever_versions"] = fields["lever_versions"]
        if best_fields["best_lever_version"] is None and fields["best_lever_version"] is not None:
            best_fields["best_lever_version"] = fields["best_lever_version"]
        if not best_fields["optimized_candidates"] and fields["optimized_candidates"]:
            best_fields["optimized_candidates"] = fields["optimized_candidates"]
        if not best_fields["attempted_candidates"] and fields["attempted_candidates"]:
            best_fields["attempted_candidates"] = fields["attempted_candidates"]
        if not best_fields["validation_results"] and fields["validation_results"]:
            best_fields["validation_results"] = fields["validation_results"]

    if best_fields["best_reward"] is not None:
        merged["best_reward"] = best_fields["best_reward"]
    if best_fields["best_candidate"] is not None:
        merged["best_candidate"] = best_fields["best_candidate"]
        merged["best_prompt"] = best_fields["best_candidate"]
    if best_fields["best_candidate_content"] is not None:
        merged["best_candidate_content"] = best_fields["best_candidate_content"]
    merged["lever_summary"] = best_fields["lever_summary"]
    merged["sensor_frames"] = best_fields["sensor_frames"]
    merged["lever_versions"] = best_fields["lever_versions"]
    merged["best_lever_version"] = best_fields["best_lever_version"]
    if best_fields["optimized_candidates"]:
        merged["optimized_candidates"] = best_fields["optimized_candidates"]
    if best_fields["attempted_candidates"]:
        merged["attempted_candidates"] = best_fields["attempted_candidates"]
    if best_fields["validation_results"]:
        merged["validation_results"] = best_fields["validation_results"]
    return merged


def _extract_mipro_system_id_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for source in (payload, payload.get("metadata"), payload.get("job_metadata")):
        if not isinstance(source, dict):
            continue
        raw_system_id = source.get("mipro_system_id")
        if isinstance(raw_system_id, str) and raw_system_id.strip():
            return raw_system_id.strip()
    return None


def _extract_mipro_state_fields(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = state_payload.get("candidates")
    best_candidate_id = state_payload.get("best_candidate_id")
    if not isinstance(best_candidate_id, str):
        best_candidate_id = None

    if isinstance(candidates, dict) and best_candidate_id is None:
        best_id: Optional[str] = None
        best_reward_seen: Optional[float] = None
        for candidate_id, candidate_payload in candidates.items():
            if not isinstance(candidate_id, str) or not isinstance(candidate_payload, dict):
                continue
            candidate_reward = _coerce_float(candidate_payload.get("avg_reward"))
            if candidate_reward is None:
                continue
            if best_reward_seen is None or candidate_reward > best_reward_seen:
                best_id = candidate_id
                best_reward_seen = candidate_reward
        best_candidate_id = best_id

    best_candidate: Any = None
    if isinstance(candidates, dict) and isinstance(best_candidate_id, str):
        best_candidate = candidates.get(best_candidate_id)
    if not isinstance(best_candidate, dict):
        best_candidate = None

    best_reward = None
    if isinstance(best_candidate, dict):
        best_reward = _coerce_float(best_candidate.get("avg_reward"))
    if best_reward is None:
        best_reward = _coerce_float(state_payload.get("best_reward"))
    if best_reward is None:
        best_reward = _coerce_float(state_payload.get("best_score"))
    if best_reward is None and isinstance(candidates, dict):
        best_reward = max(
            (
                _coerce_float(candidate_payload.get("avg_reward"))
                for candidate_payload in candidates.values()
                if isinstance(candidate_payload, dict)
            ),
            default=None,
        )

    attempted_candidates = state_payload.get("attempted_candidates")
    if not isinstance(attempted_candidates, list):
        attempted_candidates = []
    optimized_candidates = state_payload.get("optimized_candidates")
    if not isinstance(optimized_candidates, list):
        optimized_candidates = []

    lever_summary = state_payload.get("lever_summary")
    if not isinstance(lever_summary, dict):
        lever_summary = None
    sensor_frames = state_payload.get("sensor_frames")
    if not isinstance(sensor_frames, list):
        sensor_frames = []
    sensor_frames = [frame for frame in sensor_frames if isinstance(frame, dict)]
    lever_versions = _parse_lever_versions_raw(state_payload.get("lever_versions"))
    inferred_from_summary = _infer_lever_versions_from_summary(lever_summary)
    summary_prompt_lever_id = (
        lever_summary.get("prompt_lever_id") if isinstance(lever_summary, dict) else None
    )
    if inferred_from_summary and (
        not lever_versions
        or (
            isinstance(summary_prompt_lever_id, str)
            and summary_prompt_lever_id not in lever_versions
        )
    ):
        lever_versions = inferred_from_summary

    if not lever_versions:
        candidate_versions = _parse_lever_versions_raw(
            state_payload.get("candidate_lever_versions")
        )
        prompt_lever_id = (
            state_payload.get("prompt_lever_id")
            if isinstance(state_payload.get("prompt_lever_id"), str)
            else summary_prompt_lever_id
        )
        selected_candidate_id = best_candidate_id
        if selected_candidate_id is None and isinstance(lever_summary, dict):
            summary_candidate = lever_summary.get("best_candidate_id") or lever_summary.get(
                "selected_candidate_id"
            )
            if isinstance(summary_candidate, str):
                selected_candidate_id = summary_candidate
        if (
            isinstance(prompt_lever_id, str)
            and isinstance(selected_candidate_id, str)
            and selected_candidate_id in candidate_versions
        ):
            lever_versions = {prompt_lever_id: candidate_versions[selected_candidate_id]}

    best_lever_version = _coerce_int(state_payload.get("best_lever_version"))
    if best_lever_version is None:
        best_lever_version = _infer_best_lever_version_from_summary(
            lever_summary,
            best_candidate_id=best_candidate_id,
        )
    if best_lever_version is None and isinstance(best_candidate_id, str):
        best_lever_version = _coerce_int(
            (
                state_payload.get("candidate_lever_versions", {})
                if isinstance(state_payload.get("candidate_lever_versions"), dict)
                else {}
            ).get(best_candidate_id)
        )
    if best_lever_version is None and lever_versions:
        best_lever_version = max(lever_versions.values())

    return {
        "best_candidate": best_candidate,
        "best_candidate_content": _extract_best_candidate_content_from_sources(
            best_candidate,
            state_payload,
        ),
        "best_reward": best_reward,
        "attempted_candidates": attempted_candidates,
        "optimized_candidates": optimized_candidates,
        "lever_summary": lever_summary,
        "sensor_frames": sensor_frames,
        "lever_versions": lever_versions,
        "best_lever_version": best_lever_version,
    }


def _coerce_state_events_list(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    normalized: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type") or event.get("type")
        if isinstance(event_type, str) and (
            event_type.startswith("mipro.") or event_type.startswith("gepa.")
        ):
            event_type = f"learning.policy.{event_type}"
        normalized.append(
            {
                "seq": event.get("seq"),
                "type": event_type,
                "message": event.get("message"),
                "data": event.get("data") if isinstance(event.get("data"), dict) else {},
                "created_at": event.get("created_at"),
                "ts": event.get("created_at"),
            }
        )
    return normalized


def _extract_outcome_reward(payload: Dict[str, Any]) -> Optional[float]:
    outcome_objectives = payload.get("outcome_objectives")
    if isinstance(outcome_objectives, dict):
        reward_val = _coerce_float(outcome_objectives.get("reward"))
        if reward_val is not None:
            return reward_val
    return _coerce_float(payload.get("outcome_reward"))


def _validate_job_id(job_id: str) -> None:
    """Validate that job_id has the expected prompt learning format.

    Args:
        job_id: Job ID to validate

    Raises:
        ValueError: If job_id doesn't start with 'pl_'
    """
    if not job_id.startswith("pl_"):
        raise ValueError(
            f"Invalid prompt learning job ID format: {job_id!r}. "
            f"Expected format: 'pl_<identifier>' (e.g., 'pl_9c58b711c2644083')"
        )


def _validate_system_id(system_id: str) -> None:
    system_id = str(system_id).strip()
    if not system_id:
        raise ValueError("system_id is required")


def _extract_reward_value(payload: Any, strict_keys: Optional[List[str]] = None) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    reward_val = _extract_outcome_reward(payload)
    if reward_val is not None:
        return float(reward_val)
    if strict_keys:
        for key in strict_keys:
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _extract_event_type(event: Dict[str, Any]) -> str:
    event_type = event.get("type") or event.get("event_type")
    if not event_type and isinstance(event.get("data"), dict):
        event_type = event["data"].get("type") or event["data"].get("event_type")
    return str(event_type or "")


def _extract_event_data(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("data")
    if data is None:
        data = event.get("payload") or event.get("data_json")
    if not isinstance(data, dict):
        return {}

    # Unwrap nested envelope if data still looks like an event wrapper.
    while (
        isinstance(data, dict)
        and "data" in data
        and any(
            key in data for key in ("type", "event_type", "message", "seq", "timestamp_ms", "ts")
        )
    ):
        inner = data.get("data")
        if not isinstance(inner, dict):
            break
        data = inner

    return data


def _append_event_bucket(
    bucket: Dict[str, Any],
    key: str,
    event_type: str,
    event_data: Dict[str, Any],
) -> None:
    if not event_data:
        return
    bucket.setdefault(key, []).append({"_event_type": event_type, **event_data})


def _merge_candidate_payload(event_data: Dict[str, Any]) -> Dict[str, Any]:
    program_candidate = event_data.get("program_candidate")
    if isinstance(program_candidate, dict):
        return {**event_data, **program_candidate}
    return event_data


class PromptLearningClient:
    """Client for interacting with prompt learning jobs and retrieving results."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        api_version: Optional[ApiVersion | str] = None,
    ) -> None:
        """Initialize the prompt learning client.

        Args:
            base_url: Base URL of the backend API (defaults to BACKEND_URL_BASE from urls.py)
            api_key: API key for authentication (defaults to SYNTH_API_KEY env var)
            timeout: Request timeout in seconds
        """
        import os

        from synth_ai.core.utils.urls import BACKEND_URL_BASE

        if not base_url:
            base_url = BACKEND_URL_BASE
        self._base_url = base_url

        # Resolve API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        self._api_key = api_key
        self._timeout = timeout
        self._api_version: ApiVersion = _resolve_api_version(api_version)

    async def _fetch_mipro_state(self, system_id: str) -> Optional[Dict[str, Any]]:
        system_id = system_id.strip()
        if not system_id:
            return None
        state_path = _as_versioned_api_path(
            online_session_path(system_id, api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.get(state_path)
            except Exception as e:
                logger.debug("Workflow state query failed: %s", e)
                return None
        return payload if isinstance(payload, dict) else None

    async def _fetch_mipro_events(
        self, system_id: str, *, since_seq: int = 0, limit: int = 5000
    ) -> Optional[List[Dict[str, Any]]]:
        system_id = system_id.strip()
        if not system_id:
            return None
        events_path = _as_versioned_api_path(
            online_session_subpath(system_id, "events", api_version=self._api_version)
        )
        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.get(events_path, params=params)
            except Exception as e:
                logger.debug("State events query failed: %s", e)
                return None
        return _coerce_state_events_list(payload)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job metadata and status.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            Job metadata including status, best_reward, created_at, etc.

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        job_paths = [_as_versioned_api_path(offline_job_path(job_id, api_version=self._api_version))]
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            last_not_found: Optional[Exception] = None
            first_error: Optional[Exception] = None
            payloads: List[Dict[str, Any]] = []
            for path in job_paths:
                try:
                    payload = await http.get(path)
                    if isinstance(payload, dict):
                        payloads.append(payload)
                        continue
                    if first_error is None:
                        first_error = RuntimeError(
                            f"Unexpected job payload type from {path}: {type(payload).__name__}"
                        )
                except Exception as exc:
                    if _is_not_found_error(exc):
                        last_not_found = exc
                        continue
                    if first_error is None:
                        first_error = exc
                    continue
            if payloads:
                merged = _merge_job_payloads(payloads)
                if merged.get("best_reward") is None and merged.get("best_score") is not None:
                    merged["best_reward"] = merged.get("best_score")
                if merged.get("best_score") is None and merged.get("best_reward") is not None:
                    merged["best_score"] = merged.get("best_reward")
                metadata = _merge_job_metadata(merged)
                algorithm = (
                    str(merged.get("algorithm") or metadata.get("algorithm") or "").strip().lower()
                )
                needs_state_strict = algorithm == "mipro" and (
                    merged.get("best_reward") is None or merged.get("best_candidate") is None
                )
                if needs_state_strict:
                    system_id = _extract_mipro_system_id_from_payload(merged)
                    if system_id:
                        state_payload = await self._fetch_mipro_state(system_id)
                        if isinstance(state_payload, dict):
                            state_fields = _extract_mipro_state_fields(state_payload)
                            if (
                                merged.get("best_reward") is None
                                and state_fields["best_reward"] is not None
                            ):
                                merged["best_reward"] = state_fields["best_reward"]
                            if (
                                merged.get("best_score") is None
                                and state_fields["best_reward"] is not None
                            ):
                                merged["best_score"] = state_fields["best_reward"]

                            if (
                                merged.get("best_candidate") is None
                                and state_fields["best_candidate"] is not None
                            ):
                                merged["best_candidate"] = state_fields["best_candidate"]
                                merged["best_prompt"] = state_fields["best_candidate"]
                            if (
                                merged.get("best_candidate_content") is None
                                and state_fields["best_candidate_content"] is not None
                            ):
                                merged["best_candidate_content"] = state_fields[
                                    "best_candidate_content"
                                ]
                            if (
                                not merged.get("attempted_candidates")
                                and state_fields["attempted_candidates"]
                            ):
                                merged["attempted_candidates"] = state_fields[
                                    "attempted_candidates"
                                ]
                            if (
                                not merged.get("optimized_candidates")
                                and state_fields["optimized_candidates"]
                            ):
                                merged["optimized_candidates"] = state_fields[
                                    "optimized_candidates"
                                ]
                            if (
                                merged.get("lever_summary") is None
                                and state_fields["lever_summary"] is not None
                            ):
                                merged["lever_summary"] = state_fields["lever_summary"]
                            if not merged.get("sensor_frames") and state_fields["sensor_frames"]:
                                merged["sensor_frames"] = state_fields["sensor_frames"]
                            if not merged.get("lever_versions") and state_fields["lever_versions"]:
                                merged["lever_versions"] = state_fields["lever_versions"]
                            if (
                                merged.get("best_lever_version") is None
                                and state_fields["best_lever_version"] is not None
                            ):
                                merged["best_lever_version"] = state_fields["best_lever_version"]
                            merged_metadata = _merge_job_metadata(merged)
                            if state_fields["best_reward"] is not None:
                                merged_metadata.setdefault(
                                    "prompt_best_reward", state_fields["best_reward"]
                                )
                                merged_metadata.setdefault(
                                    "best_reward", state_fields["best_reward"]
                                )
                            if state_fields["best_candidate"] is not None:
                                merged_metadata.setdefault(
                                    "best_candidate", state_fields["best_candidate"]
                                )
                            merged["metadata"] = merged_metadata
                            merged["job_metadata"] = merged_metadata
                return merged
            if last_not_found is not None:
                raise last_not_found
            if first_error is not None:
                raise first_error
            raise RuntimeError(f"Unable to load job payload for {job_id}")

    async def list_candidates(
        self,
        job_id: str,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List candidates for a job from canonical candidate CRUD endpoints."""
        _validate_job_id(job_id)
        params: Dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
        if algorithm:
            params["algorithm"] = algorithm
        if mode:
            params["mode"] = mode
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if sort:
            params["sort"] = sort
        if include:
            params["include"] = include
        path = _as_versioned_api_path(
            offline_job_subpath(job_id, "candidates", api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path, params=params)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from candidates endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def submit_candidates(
        self,
        *,
        job_id: str,
        algorithm_kind: str,
        candidates: List[Dict[str, Any]],
        proposal_session_id: Optional[str] = None,
        proposer_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit typed candidates via the v2 submit-candidates contract."""
        _validate_job_id(job_id)
        normalized_algorithm = str(algorithm_kind or "").strip().lower()
        if normalized_algorithm not in {"gepa", "mipro"}:
            raise ValueError("algorithm_kind must be one of: gepa, mipro")
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")
        payload = {
            "job_id": job_id,
            "algorithm_kind": normalized_algorithm,
            "candidates": candidates,
            "proposal_session_id": proposal_session_id,
            "proposer_metadata": dict(proposer_metadata or {}),
        }
        path = _as_versioned_api_path(candidates_submit_path(api_version=self._api_version))
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            response = await http.post_json(path, json=payload)
        if not isinstance(response, dict):
            raise ValueError(
                f"Unexpected response structure from submit candidates endpoint {path}: "
                f"{type(response).__name__}"
            )
        return response

    async def get_state_baseline_info(self, job_id: str) -> Dict[str, Any]:
        """Read persisted state-envelope baseline info projection for a job."""
        _validate_job_id(job_id)
        path = offline_job_state_baseline_info_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from state baseline endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_state_envelope(self, job_id: str) -> Dict[str, Any]:
        """Read full persisted state-envelope payload for a job."""
        _validate_job_id(job_id)
        path = offline_job_state_envelope_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from state envelope endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def list_trial_queue(self, job_id: str) -> Dict[str, Any]:
        """Read persisted trial queue for a job."""
        _validate_job_id(job_id)
        path = offline_job_queue_trials_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from trial queue endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def enqueue_trial(
        self,
        job_id: str,
        *,
        trial: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enqueue one trial spec in persisted trial queue."""
        _validate_job_id(job_id)
        path = _with_algorithm_kind(
            offline_job_queue_trials_path(job_id, api_version=self._api_version),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json=trial)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from enqueue trial endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def update_trial(
        self,
        job_id: str,
        trial_id: str,
        *,
        patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch one trial spec in persisted trial queue."""
        _validate_job_id(job_id)
        trial_id_norm = str(trial_id).strip()
        if not trial_id_norm:
            raise ValueError("trial_id is required")
        path = _with_algorithm_kind(
            offline_job_queue_trial_path(
                job_id,
                trial_id_norm,
                api_version=self._api_version,
            ),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json=patch)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from patch trial endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def cancel_trial(
        self,
        job_id: str,
        trial_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel one trial in persisted trial queue."""
        _validate_job_id(job_id)
        trial_id_norm = str(trial_id).strip()
        if not trial_id_norm:
            raise ValueError("trial_id is required")
        path = _with_algorithm_kind(
            offline_job_queue_trial_path(
                job_id,
                trial_id_norm,
                api_version=self._api_version,
            ),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.delete(path)
        if payload is None:
            return {"status": "ok"}
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from cancel trial endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def reorder_trials(
        self,
        job_id: str,
        *,
        trial_ids: List[str],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reorder trial queue by explicit trial id ordering."""
        _validate_job_id(job_id)
        path = _with_algorithm_kind(
            offline_job_queue_trials_reorder_path(job_id, api_version=self._api_version),
            algorithm_kind,
        )
        body = {"trial_ids": [str(trial_id) for trial_id in trial_ids]}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json=body)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from reorder trials endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def apply_default_trial_plan(
        self,
        job_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply default trial planner templates for the job."""
        _validate_job_id(job_id)
        path = _with_algorithm_kind(
            offline_job_queue_default_plan_path(job_id, api_version=self._api_version),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json={})
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from default plan endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_rollout_queue(self, job_id: str) -> Dict[str, Any]:
        """Read persisted rollout queue state for a job."""
        _validate_job_id(job_id)
        path = offline_job_queue_rollouts_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from rollout queue endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def set_rollout_queue_policy(
        self,
        job_id: str,
        *,
        policy_patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch rollout queue policies for a job."""
        _validate_job_id(job_id)
        path = _with_algorithm_kind(
            offline_job_queue_rollout_policy_path(job_id, api_version=self._api_version),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json=policy_patch)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from rollout policy endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_rollout_dispatch_metrics(self, job_id: str) -> Dict[str, Any]:
        """Read rollout dispatch metrics view for a job."""
        _validate_job_id(job_id)
        path = offline_job_queue_rollout_metrics_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from rollout metrics endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_rollout_limiter_status(self, job_id: str) -> Dict[str, Any]:
        """Read rollout scheduler limiter/runtime status for a job."""
        _validate_job_id(job_id)
        path = offline_job_queue_rollout_limiter_status_path(job_id, api_version=self._api_version)
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from rollout limiter endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def retry_rollout_dispatch(
        self,
        job_id: str,
        dispatch_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retry one dispatch item in persisted rollout queue state."""
        _validate_job_id(job_id)
        dispatch_id_norm = str(dispatch_id).strip()
        if not dispatch_id_norm:
            raise ValueError("dispatch_id is required")
        path = _with_algorithm_kind(
            offline_job_queue_rollout_retry_path(
                job_id,
                dispatch_id_norm,
                api_version=self._api_version,
            ),
            algorithm_kind,
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json={})
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from retry dispatch endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def drain_rollout_queue(
        self,
        job_id: str,
        *,
        cancel_queued: bool = False,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set rollout queue draining mode and optionally cancel queued dispatches."""
        _validate_job_id(job_id)
        path = _with_algorithm_kind(
            offline_job_queue_rollout_drain_path(job_id, api_version=self._api_version),
            algorithm_kind,
        )
        body = {"cancel_queued": bool(cancel_queued)}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.post_json(path, json=body)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from drain rollout endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def list_candidates_typed(
        self,
        job_id: str,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> PolicyCandidatePage:
        payload = await self.list_candidates(
            job_id,
            algorithm=algorithm,
            mode=mode,
            status=status,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )
        return PolicyCandidatePage.from_dict(payload)

    async def get_candidate(self, job_id: str, candidate_id: str) -> Dict[str, Any]:
        """Get a single candidate from canonical candidate CRUD endpoints."""
        _validate_job_id(job_id)
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        paths = [
            _as_versioned_api_path(
                offline_job_subpath(
                    job_id, f"candidates/{candidate_id}", api_version=self._api_version
                )
            ),
            _as_global_api_path(candidate_path(candidate_id, api_version=self._api_version)),
        ]
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            last_not_found: Optional[Exception] = None
            first_error: Optional[Exception] = None
            for path in paths:
                try:
                    payload = await http.get(path)
                    if isinstance(payload, dict):
                        return payload
                    if first_error is None:
                        first_error = ValueError(
                            f"Unexpected response structure from candidate endpoint {path}: "
                            f"{type(payload).__name__}"
                        )
                except Exception as exc:
                    if _is_not_found_error(exc):
                        last_not_found = exc
                        continue
                    if first_error is None:
                        first_error = exc
            if last_not_found is not None:
                raise last_not_found
            if first_error is not None:
                raise first_error
        raise RuntimeError(f"Unable to load candidate {candidate_id} for job {job_id}")

    async def get_candidate_typed(self, job_id: str, candidate_id: str) -> PolicyCandidate:
        payload = await self.get_candidate(job_id, candidate_id)
        return PolicyCandidate.from_dict(payload)

    async def list_system_candidates(
        self,
        system_id: str,
        *,
        job_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List candidates for a system from canonical candidate CRUD endpoints."""
        _validate_system_id(system_id)
        params: Dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
        if job_id:
            normalized_job_id = str(job_id).strip()
            if normalized_job_id:
                params["job_id"] = normalized_job_id
        if algorithm:
            params["algorithm"] = algorithm
        if mode:
            params["mode"] = mode
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        if sort:
            params["sort"] = sort
        if include:
            params["include"] = include
        path = _as_global_api_path(
            system_subpath(system_id, "candidates", api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path, params=params)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from system candidates endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_global_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Get a single candidate by global candidate id."""
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        path = _as_global_api_path(candidate_path(candidate_id, api_version=self._api_version))
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from candidate endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def list_seed_evals(
        self,
        job_id: str,
        *,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
        candidate_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List seed evaluations for a job from canonical seed-eval endpoints."""
        _validate_job_id(job_id)
        params: Dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
        if split:
            params["split"] = split
        if seed is not None:
            params["seed"] = int(seed)
        if success is not None:
            params["success"] = bool(success)
        if candidate_id:
            params["candidate_id"] = str(candidate_id).strip()
        if cursor:
            params["cursor"] = cursor
        if sort:
            params["sort"] = sort
        if include:
            params["include"] = include
        path = _as_versioned_api_path(
            offline_job_subpath(job_id, "seed-evals", api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path, params=params)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from seed-evals endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def list_system_seed_evals(
        self,
        system_id: str,
        *,
        job_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List seed evaluations for a system from canonical seed-eval endpoints."""
        _validate_system_id(system_id)
        params: Dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
        if job_id:
            normalized_job_id = str(job_id).strip()
            if normalized_job_id:
                params["job_id"] = normalized_job_id
        if candidate_id:
            params["candidate_id"] = str(candidate_id).strip()
        if split:
            params["split"] = split
        if seed is not None:
            params["seed"] = int(seed)
        if success is not None:
            params["success"] = bool(success)
        if cursor:
            params["cursor"] = cursor
        if sort:
            params["sort"] = sort
        if include:
            params["include"] = include
        path = _as_global_api_path(
            system_subpath(system_id, "seed-evals", api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path, params=params)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from system seed-evals endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def list_candidate_seed_evals(
        self,
        candidate_id: str,
        *,
        job_id: Optional[str] = None,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        success: Optional[bool] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List seed evaluations for a candidate from canonical seed-eval endpoints."""
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        params: Dict[str, Any] = {"limit": max(1, min(int(limit), 1000))}
        if job_id:
            normalized_job_id = str(job_id).strip()
            if normalized_job_id:
                params["job_id"] = normalized_job_id
        if split:
            params["split"] = split
        if seed is not None:
            params["seed"] = int(seed)
        if success is not None:
            params["success"] = bool(success)
        if cursor:
            params["cursor"] = cursor
        if sort:
            params["sort"] = sort
        if include:
            params["include"] = include
        path = _as_global_api_path(
            candidate_subpath(candidate_id, "seed-evals", api_version=self._api_version)
        )
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            payload = await http.get(path, params=params)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected response structure from candidate seed-evals endpoint {path}: "
                f"{type(payload).__name__}"
            )
        return payload

    async def get_events(
        self, job_id: str, *, since_seq: int = 0, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """Get events for a prompt learning job.

        Args:
            job_id: Job ID
            since_seq: Return events after this sequence number
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries with type, message, data, etc.

        Raises:
            ValueError: If job_id format is invalid or response structure is unexpected
        """
        _validate_job_id(job_id)
        import asyncio
        import os

        if os.getenv("SYNTH_USE_RUST_CORE", "").lower() in {"1", "true", "yes"}:
            try:
                import synth_ai_py

                resp = await asyncio.to_thread(
                    synth_ai_py.poll_events,
                    "prompt_learning",
                    job_id,
                    self._base_url,
                    self._api_key,
                    since_seq,
                    limit,
                    int(self._timeout * 1000),
                )
                raw_events = resp.get("events") if isinstance(resp, dict) else []
                events = []
                for ev in raw_events or []:
                    if not isinstance(ev, dict):
                        continue
                    events.append(
                        {
                            "seq": ev.get("seq", -1),
                            "type": ev.get("type") or ev.get("event_type") or "unknown",
                            "message": ev.get("message"),
                            "data": ev.get("data_json") or ev.get("data") or ev,
                            "ts": ev.get("ts"),
                        }
                    )
                return events
            except Exception as e:
                logger.debug("Rust tracker events query failed: %s", e)

        params = {"since_seq": since_seq, "limit": limit}
        event_paths = [
            _as_versioned_api_path(
                offline_job_subpath(job_id, "events", api_version=self._api_version)
            )
        ]
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            last_not_found: Optional[Exception] = None
            first_error: Optional[Exception] = None
            candidates: List[List[Dict[str, Any]]] = []
            for path in event_paths:
                try:
                    js = await http.get(path, params=params)
                    parsed = _coerce_events_list(js)
                    if parsed is not None:
                        candidates.append(parsed)
                    elif first_error is None:
                        first_error = ValueError(
                            f"Unexpected response structure from events endpoint {path}: "
                            f"{type(js).__name__}"
                        )
                except Exception as exc:
                    if _is_not_found_error(exc):
                        last_not_found = exc
                        continue
                    if first_error is None:
                        first_error = exc
                    continue
            best_events: List[Dict[str, Any]] = max(candidates, key=len) if candidates else []
            needs_state_strict = not best_events or len(best_events) <= 1
            if needs_state_strict:
                try:
                    job_payload = await self.get_job(job_id)
                except Exception as e:
                    logger.debug("Job payload fallback query failed: %s", e)
                    job_payload = None
                if isinstance(job_payload, dict):
                    metadata = _merge_job_metadata(job_payload)
                    algorithm = (
                        str(job_payload.get("algorithm") or metadata.get("algorithm") or "")
                        .strip()
                        .lower()
                    )
                    if algorithm == "mipro":
                        system_id = _extract_mipro_system_id_from_payload(job_payload)
                        if system_id:
                            state_events = await self._fetch_mipro_events(
                                system_id,
                                since_seq=since_seq,
                                limit=limit,
                            )
                            if state_events and len(state_events) > len(best_events):
                                return state_events
            if best_events:
                return best_events
            if last_not_found is not None:
                raise last_not_found
            if first_error is not None:
                raise first_error
            raise RuntimeError(f"Unable to load events for {job_id}")

    async def get_prompts(self, job_id: str) -> PromptResults:
        """Get the best prompts and scoring metadata from a completed job.

        Args:
            job_id: Job ID

        Returns:
            PromptResults dataclass containing:
                - best_candidate: The top-performing prompt with sections and metadata
                - best_reward: The best reward achieved
                - optimized_candidates: All frontier/Pareto-optimal candidates
                - attempted_candidates: All candidates tried during optimization

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        events: List[Dict[str, Any]] = []
        events_error: Optional[Exception] = None
        try:
            events = await self.get_events(job_id, limit=10000)
        except Exception as exc:
            events_error = exc

        result = PromptResults()

        # Extract results from events
        for event in events:
            event_type = _extract_event_type(event)
            event_data = _extract_event_data(event)
            if not event_type and isinstance(event_data, dict):
                event_type = str(event_data.get("type") or event_data.get("event_type") or "")

            if event_type:
                result.event_counts[event_type] = result.event_counts.get(event_type, 0) + 1
                result.event_history.append(
                    {
                        "type": event_type,
                        "seq": event.get("seq"),
                        "timestamp_ms": event.get("timestamp_ms"),
                        "ts": event.get("ts"),
                        "message": event.get("message"),
                        "data": event_data,
                    }
                )

            # Best prompt event (canonical)
            if event_type == "learning.policy.gepa.candidate.new_best":
                result.best_candidate = event_data.get("best_candidate") or event_data.get(
                    "best_prompt"
                )
                event_best_reward = _extract_reward_value(event_data, strict_keys=["best_reward"])
                if event_best_reward is not None:
                    result.best_reward = event_best_reward
                best_candidate = (
                    event_data.get("best_candidate")
                    or event_data.get("best_prompt")
                    or event_data.get("candidate")
                    or event_data.get("program_candidate")
                )
                if isinstance(best_candidate, dict):
                    result.best_candidate = best_candidate
                if result.best_candidate_content is None:
                    result.best_candidate_content = _extract_best_candidate_content_from_sources(
                        best_candidate,
                        event_data,
                    )
                _append_event_bucket(result.gepa, "best_candidates", event_type, event_data)

            # Candidate evaluated events
            elif event_type == "learning.policy.gepa.candidate.evaluated":
                candidate_view = _merge_candidate_payload(event_data)
                _append_event_bucket(result.gepa, "candidates", event_type, candidate_view)

            # Job completed event (contains all candidates) - canonical
            elif event_type == "learning.policy.gepa.job.completed":
                result.optimized_candidates = event_data.get("optimized_candidates", [])
                result.attempted_candidates = event_data.get("attempted_candidates", [])
                result.version_tree = event_data.get("version_tree")
                # Also extract best_candidate from final.results if not already set
                if result.best_candidate is None:
                    result.best_candidate = event_data.get("best_candidate") or event_data.get(
                        "best_prompt"
                    )
                if result.best_candidate_content is None:
                    result.best_candidate_content = _extract_best_candidate_content_from_sources(
                        result.best_candidate,
                        event_data,
                    )
                if result.best_reward is None:
                    event_best_reward = _extract_reward_value(
                        event_data, strict_keys=["best_reward"]
                    )
                    if event_best_reward is not None:
                        result.best_reward = event_best_reward

                # Extract rollout and proposal metrics
                # These may come from event_data directly or from nested state dict
                # Only update if we find non-zero values (preserve from earlier events)
                new_rollouts = event_data.get("total_rollouts") or event_data.get("state", {}).get(
                    "total_rollouts"
                )
                if new_rollouts:
                    result.total_rollouts = new_rollouts
                # trials_tried is the number of proposal/mutation calls
                new_proposals = event_data.get("trials_tried") or event_data.get("state", {}).get(
                    "total_trials"
                )
                if new_proposals:
                    result.total_proposal_calls = new_proposals

                # Extract validation results from validation field if present
                validation_data = event_data.get("validation")
                if isinstance(validation_data, list):
                    result.validation_results.extend(
                        val_item for val_item in validation_data if isinstance(val_item, dict)
                    )
                if isinstance(event_data.get("baseline"), dict):
                    result.gepa["baseline"] = event_data.get("baseline")
                _append_event_bucket(result.gepa, "job_completed", event_type, event_data)

            # Validation results
            elif event_type == "learning.policy.gepa.validation.completed":
                result.validation_results.append(event_data)
                _append_event_bucket(result.gepa, "validation", event_type, event_data)

            elif event_type.startswith("learning.policy.gepa."):
                if event_type.endswith(".baseline") or "baseline" in event_type:
                    result.gepa["baseline"] = event_data
                elif "frontier_updated" in event_type or "frontier.updated" in event_type:
                    _append_event_bucket(result.gepa, "frontier_updates", event_type, event_data)
                elif "generation.complete" in event_type or "generation.completed" in event_type:
                    _append_event_bucket(result.gepa, "generations", event_type, event_data)
                elif "progress" in event_type or "rollouts.progress" in event_type:
                    _append_event_bucket(result.gepa, "progress_updates", event_type, event_data)

            # MIPRO completion event - extract best_reward (canonical)
            elif event_type == "learning.policy.mipro.job.completed":
                if result.best_reward is None:
                    # Prefer unified best_reward field, strict to best_full_reward or best_minibatch_reward
                    result.best_reward = _extract_reward_value(
                        event_data,
                        strict_keys=["best_reward", "best_full_reward", "best_minibatch_reward"],
                    )
                if result.best_candidate is None:
                    candidate = event_data.get("best_candidate") or event_data.get("best_prompt")
                    if isinstance(candidate, (dict, str)):
                        result.best_candidate = candidate
                if result.best_candidate_content is None:
                    result.best_candidate_content = _extract_best_candidate_content_from_sources(
                        result.best_candidate,
                        event_data,
                    )
                if event_data.get("attempted_candidates") is not None:
                    result.attempted_candidates = event_data.get("attempted_candidates", [])
                if event_data.get("optimized_candidates") is not None:
                    result.optimized_candidates = event_data.get("optimized_candidates", [])
                if isinstance(event_data.get("lever_summary"), dict):
                    result.lever_summary = event_data.get("lever_summary")
                sensor_frames = event_data.get("sensor_frames")
                if isinstance(sensor_frames, list):
                    result.sensor_frames = [
                        frame for frame in sensor_frames if isinstance(frame, dict)
                    ]
                lever_versions = _parse_lever_versions(event_data)
                if lever_versions:
                    result.lever_versions = lever_versions
                best_lever_version = _coerce_int(event_data.get("best_lever_version"))
                if best_lever_version is not None:
                    result.best_lever_version = best_lever_version
                elif result.lever_versions:
                    result.best_lever_version = max(result.lever_versions.values())
                result.mipro["job"] = {"_event_type": event_type, **event_data}

            elif event_type.startswith("learning.policy.mipro."):
                # Capture MIPRO iteration/trial/incumbent/budget events for inspection
                if "iteration.started" in event_type or "iteration.completed" in event_type:
                    _append_event_bucket(result.mipro, "iterations", event_type, event_data)
                elif "trial.started" in event_type or "trial.completed" in event_type:
                    _append_event_bucket(result.mipro, "trials", event_type, event_data)
                elif "candidate.new_best" in event_type:
                    _append_event_bucket(result.mipro, "incumbents", event_type, event_data)
                    event_best_reward = _extract_reward_value(
                        event_data,
                        strict_keys=["best_reward", "reward", "full_reward", "minibatch_reward"],
                    )
                    if event_best_reward is not None and (
                        result.best_reward is None or event_best_reward > result.best_reward
                    ):
                        result.best_reward = event_best_reward
                    best_candidate = (
                        event_data.get("best_candidate")
                        or event_data.get("best_prompt")
                        or event_data.get("candidate")
                        or event_data.get("program_candidate")
                    )
                    if isinstance(best_candidate, dict):
                        result.best_candidate = best_candidate
                    if result.best_candidate_content is None:
                        result.best_candidate_content = (
                            _extract_best_candidate_content_from_sources(
                                best_candidate,
                                event_data,
                            )
                        )
                elif "candidate.evaluated" in event_type:
                    _append_event_bucket(result.mipro, "candidates", event_type, event_data)
                elif "budget" in event_type:
                    _append_event_bucket(result.mipro, "budget_updates", event_type, event_data)

        job_payload: Optional[Dict[str, Any]] = None
        job_error: Optional[Exception] = None
        try:
            job_payload = await self.get_job(job_id)
        except Exception as exc:
            job_error = exc
        if isinstance(job_payload, dict):
            _merge_prompt_results_from_job_payload(result, job_payload)
            metadata = _merge_job_metadata(job_payload)
            algorithm = (
                str(job_payload.get("algorithm") or metadata.get("algorithm") or "").strip().lower()
            )
            needs_state_strict = algorithm == "mipro" and (
                result.best_reward is None
                or result.best_candidate is None
                or not result.attempted_candidates
            )
            if needs_state_strict:
                system_id = _extract_mipro_system_id_from_payload(job_payload)
                if system_id:
                    state_payload = await self._fetch_mipro_state(system_id)
                    if isinstance(state_payload, dict):
                        state_fields = _extract_mipro_state_fields(state_payload)
                        state_job_payload: Dict[str, Any] = {
                            "best_candidate": state_fields["best_candidate"],
                            "best_candidate_content": state_fields["best_candidate_content"],
                            "best_reward": state_fields["best_reward"],
                            "best_reward": state_fields["best_reward"],
                            "attempted_candidates": state_fields["attempted_candidates"],
                            "optimized_candidates": state_fields["optimized_candidates"],
                            "lever_summary": state_fields["lever_summary"],
                            "sensor_frames": state_fields["sensor_frames"],
                            "lever_versions": state_fields["lever_versions"],
                            "best_lever_version": state_fields["best_lever_version"],
                        }
                        _merge_prompt_results_from_job_payload(result, state_job_payload)

        if result.best_lever_version is None and result.lever_versions:
            result.best_lever_version = max(result.lever_versions.values())
        if result.best_candidate_content is None:
            result.best_candidate_content = _extract_best_candidate_content_from_sources(
                result.best_candidate,
                job_payload if isinstance(job_payload, dict) else {},
            )

        if events_error is not None and not isinstance(job_payload, dict):
            if job_error is not None:
                raise RuntimeError(
                    f"Failed to load events and job payload for {job_id}: "
                    f"events_error={events_error}; job_error={job_error}"
                ) from events_error
            raise events_error

        return result

    async def get_scoring_summary(self, job_id: str) -> Dict[str, Any]:
        """Get a summary of scoring metrics for all candidates.

        Args:
            job_id: Job ID

        Returns:
            Dictionary with reward statistics:
                - best_train_accuracy: Best training accuracy
                - best_val_accuracy: Best validation accuracy (if available)
                - num_candidates_tried: Total candidates evaluated
                - num_frontier_candidates: Number in Pareto frontier
                - reward_distribution: Histogram of accuracy rewards

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        prompts_data = await self.get_prompts(job_id)

        attempted = prompts_data.attempted_candidates
        optimized = prompts_data.optimized_candidates
        validation = prompts_data.validation_results

        # Extract train accuracies (prefer objectives when available)
        train_accuracies: List[float] = []
        for candidate in attempted:
            if not isinstance(candidate, dict):
                continue
            reward_val = _extract_reward_value(candidate)
            if reward_val is None:
                reward = candidate.get("reward")
                reward_val = _extract_reward_value(reward)
            if reward_val is not None:
                train_accuracies.append(reward_val)

        # Extract val accuracies (prefer objectives, exclude baseline)
        # IMPORTANT: Exclude baseline from "best" calculation - baseline is for comparison only
        val_accuracies: List[float] = []
        for val_item in validation:
            if not isinstance(val_item, dict) or val_item.get("is_baseline", False):
                continue
            reward_val = _extract_reward_value(val_item)
            if reward_val is not None:
                val_accuracies.append(reward_val)

        # Reward distribution (bins)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        distribution = {f"{bins[i]:.1f}-{bins[i + 1]:.1f}": 0 for i in range(len(bins) - 1)}
        for acc in train_accuracies:
            for i in range(len(bins) - 1):
                if bins[i] <= acc < bins[i + 1] or (i == len(bins) - 2 and acc == bins[i + 1]):
                    distribution[f"{bins[i]:.1f}-{bins[i + 1]:.1f}"] += 1
                    break

        return {
            "best_train_accuracy": max(train_accuracies) if train_accuracies else None,
            "best_val_accuracy": max(val_accuracies) if val_accuracies else None,
            "num_candidates_tried": len(attempted),
            "num_frontier_candidates": len(optimized),
            "reward_distribution": distribution,
            "mean_train_accuracy": sum(train_accuracies) / len(train_accuracies)
            if train_accuracies
            else None,
        }


# Synchronous wrapper for convenience
def get_prompts(job_id: str, base_url: str, api_key: str) -> PromptResults:
    """Synchronous wrapper to get prompts from a job.

    Args:
        job_id: Job ID (e.g., "pl_9c58b711c2644083")
        base_url: Backend API base URL
        api_key: API key for authentication

    Returns:
        PromptResults dataclass with prompt results
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_prompts(job_id),
        label="get_prompts()",
    )


def get_scoring_summary(job_id: str, base_url: str, api_key: str) -> Dict[str, Any]:
    """Synchronous wrapper to get scoring summary.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        api_key: API key for authentication

    Returns:
        Dictionary with scoring statistics
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_scoring_summary(job_id),
        label="get_scoring_summary()",
    )
