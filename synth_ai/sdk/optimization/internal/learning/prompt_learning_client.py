"""Client utilities for querying prompt learning job results."""

from typing import Any, Dict, Iterable, List, Optional

from synth_ai.core.rust_core.http import RustCoreHttpClient
from synth_ai.sdk.optimization.internal.utils import run_sync

from .prompt_learning_types import PromptResults


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
            _coerce_float(metadata.get("prompt_best_score")),
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
        merged["best_score"] = best_fields["best_reward"]
    if best_fields["best_candidate"] is not None:
        merged["best_candidate"] = best_fields["best_candidate"]
        merged["best_prompt"] = best_fields["best_candidate"]
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
        best_score: Optional[float] = None
        for candidate_id, candidate_payload in candidates.items():
            if not isinstance(candidate_id, str) or not isinstance(candidate_payload, dict):
                continue
            score = _coerce_float(candidate_payload.get("avg_reward"))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_id = candidate_id
                best_score = score
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


def _extract_reward_value(
    payload: Any, fallback_keys: Optional[List[str]] = None
) -> Optional[float]:
    if not isinstance(payload, dict):
        return None
    reward_val = _extract_outcome_reward(payload)
    if reward_val is not None:
        return float(reward_val)
    if fallback_keys:
        for key in fallback_keys:
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except (TypeError, ValueError):
                continue
    return None


def _convert_template_to_pattern(template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sections = template.get("sections", [])
    if not sections:
        sections = template.get("prompt_sections", [])
    if not isinstance(sections, list) or not sections:
        return None
    messages: List[Dict[str, Any]] = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        content = sec.get("content")
        if not content:
            continue
        messages.append(
            {
                "role": sec.get("role", sec.get("name", "system")),
                "name": sec.get("name"),
                "content": content,
            }
        )
    if not messages:
        return None
    return {"messages": messages}


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
        self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 30.0
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

    async def _fetch_mipro_state(self, system_id: str) -> Optional[Dict[str, Any]]:
        system_id = system_id.strip()
        if not system_id:
            return None
        state_path = f"/api/prompt-learning/online/mipro/systems/{system_id}/state"
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.get(state_path)
            except Exception:
                return None
        return payload if isinstance(payload, dict) else None

    async def _fetch_mipro_events(
        self, system_id: str, *, since_seq: int = 0, limit: int = 5000
    ) -> Optional[List[Dict[str, Any]]]:
        system_id = system_id.strip()
        if not system_id:
            return None
        events_path = f"/api/prompt-learning/online/mipro/systems/{system_id}/events"
        params = {"since_seq": since_seq, "limit": limit}
        async with RustCoreHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            try:
                payload = await http.get(events_path, params=params)
            except Exception:
                return None
        return _coerce_state_events_list(payload)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job metadata and status.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            Job metadata including status, best_score, created_at, etc.

        Raises:
            ValueError: If job_id format is invalid
        """
        _validate_job_id(job_id)
        job_paths = [
            f"/api/jobs/{job_id}",
            f"/api/policy-optimization/online/jobs/{job_id}",
            f"/api/prompt-learning/online/jobs/{job_id}",
        ]
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
                metadata = _merge_job_metadata(merged)
                algorithm = (
                    str(merged.get("algorithm") or metadata.get("algorithm") or "").strip().lower()
                )
                needs_state_fallback = algorithm == "mipro" and (
                    merged.get("best_reward") is None or merged.get("best_candidate") is None
                )
                if needs_state_fallback:
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
                                merged["best_score"] = state_fields["best_reward"]
                            if (
                                merged.get("best_candidate") is None
                                and state_fields["best_candidate"] is not None
                            ):
                                merged["best_candidate"] = state_fields["best_candidate"]
                                merged["best_prompt"] = state_fields["best_candidate"]
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
                                    "prompt_best_score", state_fields["best_reward"]
                                )
                                merged_metadata.setdefault(
                                    "best_score", state_fields["best_reward"]
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
            except Exception:
                pass

        params = {"since_seq": since_seq, "limit": limit}
        event_paths = [
            f"/api/jobs/{job_id}/events",
            f"/api/policy-optimization/online/jobs/{job_id}/events",
            f"/api/prompt-learning/online/jobs/{job_id}/events",
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
            needs_state_fallback = not best_events or len(best_events) <= 1
            if needs_state_fallback:
                try:
                    job_payload = await self.get_job(job_id)
                except Exception:
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

    def _extract_full_text_from_template(self, template: Dict[str, Any]) -> str:
        """Extract full text from a serialized template dict (legacy)."""
        sections = template.get("sections", [])
        if not sections:
            sections = template.get("prompt_sections", [])
        full_text_parts = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_name = sec.get("name", "")
            sec_role = sec.get("role", "")
            sec_content = str(sec.get("content", ""))
            full_text_parts.append(f"[{sec_role} | {sec_name}]\n{sec_content}")
        return "\n\n".join(full_text_parts)

    def _extract_full_text_from_pattern(self, pattern: Dict[str, Any]) -> str:
        """Extract full text from a serialized pattern dict."""
        messages = pattern.get("messages", [])
        full_text_parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_name = msg.get("name", "")
            msg_role = msg.get("role", "")
            msg_content = str(msg.get("pattern") or msg.get("content") or "")
            full_text_parts.append(f"[{msg_role} | {msg_name}]\n{msg_content}")
        return "\n\n".join(full_text_parts)

    def _extract_full_text_from_object(self, obj: Dict[str, Any]) -> Optional[str]:
        """Extract full text from a candidate's object field.

        Args:
            obj: Candidate object dict (may have 'data' with 'sections' or 'text_replacements')

        Returns:
            Formatted full text string or None if extraction fails
        """
        # Try to get messages from object.data.messages (pattern format)
        data = obj.get("data", {})
        if isinstance(data, dict):
            messages = data.get("messages", [])
            if messages:
                return self._extract_full_text_from_pattern({"messages": messages})
            sections = data.get("sections", [])
            if sections:
                full_text_parts = []
                for sec in sections:
                    if not isinstance(sec, dict):
                        continue
                    sec_name = sec.get("name", "")
                    sec_role = sec.get("role", "")
                    sec_content = str(sec.get("content", ""))
                    full_text_parts.append(f"[{sec_role} | {sec_name}]\n{sec_content}")
                return "\n\n".join(full_text_parts)

            # Try text_replacements format (transformation format)
            text_replacements = data.get("text_replacements", [])
            if text_replacements and isinstance(text_replacements, list):
                full_text_parts = []
                for replacement in text_replacements:
                    if not isinstance(replacement, dict):
                        continue
                    new_text = replacement.get("new_text", "")
                    role = replacement.get("apply_to_role", "system")
                    if new_text:
                        full_text_parts.append(f"[{role}]\n{new_text}")
                if full_text_parts:
                    return "\n\n".join(full_text_parts)

        # Try direct messages on object
        messages = obj.get("messages", [])
        if messages:
            return self._extract_full_text_from_pattern({"messages": messages})

        # Try direct sections on object
        sections = obj.get("sections", [])
        if sections:
            return self._extract_full_text_from_template({"sections": sections})

        return None

    async def get_prompts(self, job_id: str) -> PromptResults:
        """Get the best prompts and scoring metadata from a completed job.

        Args:
            job_id: Job ID

        Returns:
            PromptResults dataclass containing:
                - best_candidate: The top-performing prompt with sections and metadata
                - best_score: The best accuracy score achieved
                - top_prompts: List of top-K prompts with train/val scores
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

        # Build validation score map by rank for later use
        validation_by_rank: Dict[int, float] = {}

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
                best_score = _extract_reward_value(event_data, fallback_keys=["best_score"])
                if best_score is not None:
                    result.best_reward = best_score
                best_candidate = (
                    event_data.get("best_candidate")
                    or event_data.get("best_prompt")
                    or event_data.get("candidate")
                    or event_data.get("program_candidate")
                )
                if isinstance(best_candidate, dict):
                    result.best_candidate = best_candidate
                _append_event_bucket(result.gepa, "best_candidates", event_type, event_data)

            # Candidate evaluated events may contain top-K prompt content
            elif event_type == "learning.policy.gepa.candidate.evaluated":
                candidate_view = _merge_candidate_payload(event_data)
                # Check if this is a top prompt content event (has rank)
                if event_data.get("rank") is not None:
                    pattern_payload = event_data.get("pattern")
                    if not pattern_payload:
                        template_payload = event_data.get("template")
                        if isinstance(template_payload, dict):
                            pattern_payload = _convert_template_to_pattern(template_payload)
                    prompt_entry: Dict[str, Any] = {
                        "rank": event_data.get("rank"),
                        "train_accuracy": candidate_view.get("train_accuracy"),
                        "val_accuracy": candidate_view.get("val_accuracy"),
                        "pattern": pattern_payload,
                        "full_text": event_data.get("full_text"),
                    }
                    candidate_id = candidate_view.get("candidate_id") or candidate_view.get(
                        "version_id"
                    )
                    if candidate_id is not None:
                        prompt_entry["candidate_id"] = candidate_id
                    for key in (
                        "parent_id",
                        "generation",
                        "accepted",
                        "is_pareto",
                        "mutation_type",
                        "mutation_params",
                        "prompt_summary",
                        "prompt_text",
                        "objectives",
                        "instance_scores",
                        "instance_objectives",
                        "seed_scores",
                        "seed_info",
                        "rollout_sample",
                        "token_usage",
                        "cost_usd",
                        "evaluation_duration_ms",
                        "skip_reason",
                        "stages",
                        "seeds_evaluated",
                        "full_score",
                    ):
                        value = candidate_view.get(key)
                        if value is not None:
                            prompt_entry[key] = value
                    mutation_type = candidate_view.get("mutation_type") or candidate_view.get(
                        "operator"
                    )
                    if mutation_type is not None:
                        prompt_entry["mutation_type"] = mutation_type
                    if candidate_view.get("minibatch_scores") is not None:
                        prompt_entry["minibatch_scores"] = candidate_view.get("minibatch_scores")
                    elif candidate_view.get("minibatch_score") is not None:
                        prompt_entry["minibatch_scores"] = [candidate_view.get("minibatch_score")]
                    result.top_prompts.append(prompt_entry)
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
                if result.best_reward is None:
                    best_score = _extract_reward_value(event_data, fallback_keys=["best_score"])
                    if best_score is not None:
                        result.best_reward = best_score

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
                    for val_item in validation_data:
                        if isinstance(val_item, dict):
                            rank = val_item.get("rank")
                            accuracy = _extract_reward_value(val_item)
                            if rank is not None and accuracy is not None:
                                validation_by_rank[rank] = accuracy
                if isinstance(event_data.get("baseline"), dict):
                    result.gepa["baseline"] = event_data.get("baseline")
                _append_event_bucket(result.gepa, "job_completed", event_type, event_data)

            # Validation results - build map by rank (canonical)
            elif event_type == "learning.policy.gepa.validation.completed":
                result.validation_results.append(event_data)
                # Try to extract rank and accuracy for mapping
                rank = event_data.get("rank")
                accuracy = _extract_reward_value(event_data)
                if rank is not None and accuracy is not None:
                    validation_by_rank[rank] = accuracy
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

            # MIPRO completion event - extract best_score (canonical)
            elif event_type == "learning.policy.mipro.job.completed":
                if result.best_reward is None:
                    # Prefer unified best_score field, fallback to best_full_score or best_minibatch_score
                    result.best_reward = _extract_reward_value(
                        event_data,
                        fallback_keys=["best_score", "best_full_score", "best_minibatch_score"],
                    )
                if result.best_candidate is None:
                    candidate = event_data.get("best_candidate") or event_data.get("best_prompt")
                    if isinstance(candidate, (dict, str)):
                        result.best_candidate = candidate
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
                    best_score = _extract_reward_value(
                        event_data,
                        fallback_keys=["best_score", "score", "full_score", "minibatch_score"],
                    )
                    if best_score is not None and (
                        result.best_reward is None or best_score > result.best_reward
                    ):
                        result.best_reward = best_score
                    best_candidate = (
                        event_data.get("best_candidate")
                        or event_data.get("best_prompt")
                        or event_data.get("candidate")
                        or event_data.get("program_candidate")
                    )
                    if isinstance(best_candidate, dict):
                        result.best_candidate = best_candidate
                elif "candidate.evaluated" in event_type:
                    _append_event_bucket(result.mipro, "candidates", event_type, event_data)
                elif "budget" in event_type:
                    _append_event_bucket(result.mipro, "budget_updates", event_type, event_data)

        # If top_prompts is empty but we have optimized_candidates, extract from them
        if not result.top_prompts and result.optimized_candidates:
            for idx, cand in enumerate(result.optimized_candidates):
                if not isinstance(cand, dict):
                    continue

                # Extract rank (use index+1 if rank not present)
                rank = cand.get("rank")
                if rank is None:
                    rank = idx + 1

                # Extract train accuracy from score
                score = cand.get("score", {})
                if not isinstance(score, dict):
                    score = {}
                train_accuracy = _extract_reward_value(score)
                if train_accuracy is None:
                    train_accuracy = _extract_reward_value(cand)

                # Extract val accuracy from validation map
                val_accuracy = validation_by_rank.get(rank)

                # Try to extract pattern/template and full_text
                pattern = None
                full_text = None

                # First try: pattern field (preferred)
                cand_pattern = cand.get("pattern")
                if cand_pattern and isinstance(cand_pattern, dict):
                    pattern = cand_pattern
                    full_text = self._extract_full_text_from_pattern(cand_pattern)

                # Second try: template field (legacy)
                cand_template = cand.get("template")
                if cand_template and isinstance(cand_template, dict):
                    pattern = _convert_template_to_pattern(cand_template)
                    full_text = self._extract_full_text_from_template(cand_template)
                # If it's not a dict, skip (might be a backend object that wasn't serialized)

                # Third try: object field
                if not full_text:
                    obj = cand.get("object", {})
                    if isinstance(obj, dict):
                        full_text = self._extract_full_text_from_object(obj)
                        # If we got full_text but no pattern/template, try to build structure
                        if full_text and not pattern:
                            # Try to extract pattern from object.data
                            obj_data = obj.get("data", {})
                            if isinstance(obj_data, dict):
                                if obj_data.get("messages"):
                                    pattern = {"messages": obj_data["messages"]}
                                elif obj_data.get("sections"):
                                    pattern = _convert_template_to_pattern(
                                        {"sections": obj_data["sections"]}
                                    )

                # Build prompt entry
                prompt_entry: Dict[str, Any] = {
                    "rank": rank,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                }
                for key in (
                    "candidate_id",
                    "version_id",
                    "parent_id",
                    "generation",
                    "accepted",
                    "is_pareto",
                    "mutation_type",
                    "mutation_params",
                    "prompt_summary",
                    "prompt_text",
                    "objectives",
                    "instance_scores",
                    "instance_objectives",
                    "seed_scores",
                    "seed_info",
                    "rollout_sample",
                    "token_usage",
                    "cost_usd",
                    "evaluation_duration_ms",
                    "skip_reason",
                    "stages",
                    "seeds_evaluated",
                    "full_score",
                ):
                    value = cand.get(key)
                    if value is not None:
                        prompt_entry[key] = value
                if pattern:
                    prompt_entry["pattern"] = pattern
                if full_text:
                    prompt_entry["full_text"] = full_text

                result.top_prompts.append(prompt_entry)

            # Sort by rank to ensure correct order
            result.top_prompts.sort(key=lambda p: p.get("rank", 999))

        # If we have validation results, prefer validation score for best_score
        # Rank 0 is the best prompt
        if validation_by_rank and 0 in validation_by_rank:
            # Use validation score for best_score when available
            result.best_reward = validation_by_rank[0]

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
            needs_state_fallback = algorithm == "mipro" and (
                result.best_reward is None
                or result.best_candidate is None
                or not result.attempted_candidates
            )
            if needs_state_fallback:
                system_id = _extract_mipro_system_id_from_payload(job_payload)
                if system_id:
                    state_payload = await self._fetch_mipro_state(system_id)
                    if isinstance(state_payload, dict):
                        state_fields = _extract_mipro_state_fields(state_payload)
                        state_job_payload: Dict[str, Any] = {
                            "best_candidate": state_fields["best_candidate"],
                            "best_reward": state_fields["best_reward"],
                            "best_score": state_fields["best_reward"],
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

        if events_error is not None and not isinstance(job_payload, dict):
            if job_error is not None:
                raise RuntimeError(
                    f"Failed to load events and job payload for {job_id}: "
                    f"events_error={events_error}; job_error={job_error}"
                ) from events_error
            raise events_error

        return result

    async def get_prompt_text(self, job_id: str, rank: int = 1) -> Optional[str]:
        """Get the full text of a specific prompt by rank.

        Args:
            job_id: Job ID
            rank: Prompt rank (1 = best, 2 = second best, etc.)

        Returns:
            Full prompt text or None if not found

        Raises:
            ValueError: If job_id format is invalid or rank < 1
        """
        _validate_job_id(job_id)
        if rank < 1:
            raise ValueError(f"Rank must be >= 1, got: {rank}")
        prompts_data = await self.get_prompts(job_id)
        top_prompts = prompts_data.top_prompts

        for prompt_info in top_prompts:
            if prompt_info.get("rank") == rank:
                return prompt_info.get("full_text")

        return None

    async def get_scoring_summary(self, job_id: str) -> Dict[str, Any]:
        """Get a summary of scoring metrics for all candidates.

        Args:
            job_id: Job ID

        Returns:
            Dictionary with scoring statistics:
                - best_train_accuracy: Best training accuracy
                - best_val_accuracy: Best validation accuracy (if available)
                - num_candidates_tried: Total candidates evaluated
                - num_frontier_candidates: Number in Pareto frontier
                - score_distribution: Histogram of accuracy scores

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
                score = candidate.get("score")
                reward_val = _extract_reward_value(score)
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

        # Score distribution (bins)
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
            "score_distribution": distribution,
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


def get_prompt_text(job_id: str, base_url: str, api_key: str, rank: int = 1) -> Optional[str]:
    """Synchronous wrapper to get prompt text by rank.

    Args:
        job_id: Job ID
        base_url: Backend API base URL
        api_key: API key for authentication
        rank: Prompt rank (1 = best, 2 = second best, etc.)

    Returns:
        Full prompt text or None if not found
    """
    client = PromptLearningClient(base_url, api_key)
    return run_sync(
        client.get_prompt_text(job_id, rank),
        label="get_prompt_text()",
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
