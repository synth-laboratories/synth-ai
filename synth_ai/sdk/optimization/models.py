from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from synth_ai.core.levers import MiproLeverSummary
from synth_ai.core.sensors import SensorFrameSummary


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _from_data_or_metadata(data: Dict[str, Any], key: str) -> Optional[Any]:
    value = data.get(key)
    if value is not None:
        return value
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get(key)
        if nested is not None:
            return nested
    job_metadata = data.get("job_metadata")
    if isinstance(job_metadata, dict):
        nested = job_metadata.get(key)
        if nested is not None:
            return nested
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_mapping_list(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _extract_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return out


def _extract_string_float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key, raw in value.items():
        key_str = _coerce_str(key)
        if key_str is None:
            continue
        parsed = _coerce_float(raw)
        if parsed is None:
            continue
        out[key_str] = parsed
    return out


def _extract_candidate_reward(candidate: Dict[str, Any]) -> Optional[float]:
    for key in ("mean_reward", "reward", "avg_reward", "train_reward", "validation_reward"):
        parsed = _coerce_float(candidate.get(key))
        if parsed is not None:
            return parsed
    objectives = candidate.get("instance_objectives")
    if isinstance(objectives, list):
        values = [
            parsed
            for parsed in (
                _coerce_float(obj.get("reward")) for obj in objectives if isinstance(obj, dict)
            )
            if parsed is not None
        ]
        if values:
            return sum(values) / len(values)
    for value in candidate.values():
        if not isinstance(value, dict):
            continue
        parsed = _coerce_float(value.get("reward"))
        if parsed is not None:
            return parsed
    return None


def _extract_candidate_objective(candidate: Dict[str, Any]) -> Optional[float]:
    objective = candidate.get("objective")
    if isinstance(objective, dict):
        reward = _coerce_float(objective.get("reward"))
        if reward is not None:
            return reward
    return _coerce_float(candidate.get("objective"))


def _extract_best_reward_value(data: Dict[str, Any], include_train: bool = True) -> Optional[float]:
    if include_train:
        reward_keys = ("best_reward", "best_train_reward")
    else:
        reward_keys = ("best_reward",)
    parsed = _coerce_float(_first_present(data, reward_keys))
    if parsed is not None:
        return parsed

    for key in ("metadata", "job_metadata"):
        metadata = data.get(key)
        if not isinstance(metadata, dict):
            continue
        parsed = _coerce_float(_first_present(metadata, reward_keys))
        if parsed is not None:
            return parsed

    candidate_collections: list[Any] = [
        data.get("candidates"),
        data.get("frontier"),
        data.get("archive"),
    ]
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        candidate_collections.append(metadata.get("candidates"))
    job_metadata = data.get("job_metadata")
    if isinstance(job_metadata, dict):
        candidate_collections.append(job_metadata.get("candidates"))

    reward_values: list[float] = []
    for collection in candidate_collections:
        if not isinstance(collection, list):
            continue
        for candidate in collection:
            if not isinstance(candidate, dict):
                continue
            candidate_reward = _extract_candidate_reward(candidate)
            if candidate_reward is not None:
                reward_values.append(candidate_reward)
    if reward_values:
        return max(reward_values)
    return None


def _parse_lever_versions(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    versions: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            versions[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return versions


def _extract_candidate_list(data: Dict[str, Any], key: str) -> list[PolicyCandidate]:
    raw = _from_data_or_metadata(data, key)
    if not isinstance(raw, list):
        return []
    candidates: list[PolicyCandidate] = []
    for item in raw:
        if isinstance(item, dict):
            candidates.append(PolicyCandidate.from_dict(item))
    return candidates


def _extract_long_horizon_payload(
    data: Dict[str, Any],
    *,
    canonical_key: str,
    v2_key: str,
) -> Optional[Dict[str, Any]]:
    for key in (canonical_key, v2_key):
        raw = _from_data_or_metadata(data, key)
        if isinstance(raw, dict):
            return dict(raw)
    return None


def _normalize_status(status: str) -> str:
    return status.strip().lower().replace(" ", "_")


def _extract_system_prompt_from_dict(prompt: Dict[str, Any]) -> Optional[str]:
    """Extract system prompt from a structured prompt dict."""
    # Try messages format (most common for GEPA)
    messages = prompt.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return msg.get("pattern") or msg.get("content")

    # Try sections format (canonical)
    sections = prompt.get("sections", [])
    for sec in sections:
        if isinstance(sec, dict) and sec.get("role") == "system":
            return sec.get("content") or sec.get("pattern")

    # Try text_replacements format (transformation)
    text_replacements = prompt.get("text_replacements", [])
    for tr in text_replacements:
        if isinstance(tr, dict) and tr.get("apply_to_role") == "system":
            return tr.get("new_text")

    return None


def _extract_candidate_content_from_dict(candidate: Dict[str, Any]) -> Optional[str]:
    """Extract generic candidate artifact text from structured candidate payloads."""
    for key in (
        "candidate_content",
        "candidate_code",
        "solver_code",
        "program_text",
        "code",
        "instruction_text",
        "instruction",
        "prompt_text",
        "text",
        "system_prompt",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    program_candidate = candidate.get("program_candidate")
    if isinstance(program_candidate, dict):
        nested = _extract_candidate_content_from_dict(program_candidate)
        if nested:
            return nested

    stage_payloads = candidate.get("stage_payloads")
    if isinstance(stage_payloads, dict):
        for payload in stage_payloads.values():
            if not isinstance(payload, dict):
                continue
            nested = _extract_candidate_content_from_dict(payload)
            if nested:
                return nested

    messages = candidate.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content") or message.get("pattern") or message.get("text")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
        if parts:
            return "\n\n".join(parts)

    nested_candidate = candidate.get("candidate")
    if isinstance(nested_candidate, dict):
        nested = _extract_candidate_content_from_dict(nested_candidate)
        if nested:
            return nested

    return None


def _extract_candidate_content(
    best_candidate: Optional[str | Dict[str, Any]],
    raw: Dict[str, Any],
) -> Optional[str]:
    """Extract generic best-candidate content for prompt and non-prompt artifacts."""
    if isinstance(best_candidate, str) and best_candidate.strip():
        return best_candidate.strip()
    if isinstance(best_candidate, dict):
        result = _extract_candidate_content_from_dict(best_candidate)
        if result:
            return result

    for key in ("best_candidate", "best_prompt"):
        raw_best = raw.get(key)
        if isinstance(raw_best, str) and raw_best.strip():
            return raw_best.strip()
        if isinstance(raw_best, dict):
            result = _extract_candidate_content_from_dict(raw_best)
            if result:
                return result

    for key in ("optimized_candidates", "frontier", "candidates", "archive"):
        candidates = raw.get(key)
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            result = _extract_candidate_content_from_dict(candidate)
            if result:
                return result
    return None


def _extract_candidate_mapping(data: Dict[str, Any], *keys: str) -> Optional[Dict[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return dict(value)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, dict):
                return dict(value)
    candidate_object = data.get("object")
    if isinstance(candidate_object, dict):
        for key in keys:
            value = candidate_object.get(key)
            if isinstance(value, dict):
                return dict(value)
        object_metadata = candidate_object.get("metadata")
        if isinstance(object_metadata, dict):
            for key in keys:
                value = object_metadata.get(key)
                if isinstance(value, dict):
                    return dict(value)
    return None


def _extract_system_prompt(
    best_candidate: Optional[str | Dict[str, Any]],
    raw: Dict[str, Any],
) -> Optional[str]:
    """Extract system prompt from result data, trying multiple sources."""
    # Direct string
    if isinstance(best_candidate, str) and best_candidate:
        return best_candidate

    # Structured dict
    if isinstance(best_candidate, dict):
        result = _extract_system_prompt_from_dict(best_candidate)
        if result:
            return result

    # Try raw response fields
    raw_best = raw.get("best_candidate")
    if raw_best is None:
        raw_best = raw.get("best_prompt")
    if isinstance(raw_best, str) and raw_best:
        return raw_best
    if isinstance(raw_best, dict):
        result = _extract_system_prompt_from_dict(raw_best)
        if result:
            return result

    # Try candidates in raw data
    for key in ("optimized_candidates", "frontier", "candidates", "archive"):
        candidates = raw.get(key, [])
        if isinstance(candidates, list):
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                pattern = cand.get("pattern") or cand.get("object", {}).get("pattern")
                if isinstance(pattern, dict):
                    result = _extract_system_prompt_from_dict(pattern)
                    if result:
                        return result

    # Strict for non-prompt artifacts that still expose candidate text.
    return _extract_candidate_content(best_candidate, raw)


def _normalize_error_message(error_text: Any) -> Optional[str]:
    text = str(error_text or "").strip()
    if not text:
        return None
    lower = text.lower()
    is_container_health_failure = (
        "container health check failed for" in lower
        or "container health check failed:" in lower
        or "health check failed for" in lower
    ) and "/health" in lower
    if not is_container_health_failure:
        return text
    if "skip_health_check=true only skips sdk pre-submit checks" in lower:
        return text
    return (
        f"{text} "
        "Hint: This health check runs in backend workers. "
        "skip_health_check=True only skips SDK pre-submit checks. "
        "Ensure container_url is reachable from backend workers and that the eval server is running."
    )


class PolicyJobStatus(str, Enum):
    """Status of a policy optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> PolicyJobStatus:
        normalized = _normalize_status(status)
        if normalized in ("success", "succeeded", "completed", "complete"):
            return cls.SUCCEEDED
        if normalized in ("cancelled", "canceled", "cancel"):
            return cls.CANCELLED
        if normalized in ("failed", "failure", "error"):
            return cls.FAILED
        if normalized in ("running", "in_progress"):
            return cls.RUNNING
        if normalized == "paused":
            return cls.PAUSED
        if normalized == "queued":
            return cls.QUEUED
        if normalized == "pending":
            return cls.PENDING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (
            PolicyJobStatus.SUCCEEDED,
            PolicyJobStatus.FAILED,
            PolicyJobStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        return self == PolicyJobStatus.SUCCEEDED


class GraphJobStatus(str, Enum):
    """Status of a graph optimization job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> GraphJobStatus:
        normalized = _normalize_status(status)
        if normalized in ("success", "succeeded"):
            return cls.SUCCEEDED
        if normalized in ("completed", "complete"):
            return cls.COMPLETED
        if normalized in ("cancelled", "canceled", "cancel"):
            return cls.CANCELLED
        if normalized in ("failed", "failure", "error"):
            return cls.FAILED
        if normalized in ("running", "in_progress"):
            return cls.RUNNING
        if normalized == "queued":
            return cls.QUEUED
        if normalized == "pending":
            return cls.PENDING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (
            GraphJobStatus.COMPLETED,
            GraphJobStatus.SUCCEEDED,
            GraphJobStatus.FAILED,
            GraphJobStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        return self in (GraphJobStatus.COMPLETED, GraphJobStatus.SUCCEEDED)


@dataclass
class PolicyCandidate:
    """Canonical typed candidate artifact model."""

    candidate_id: str
    candidate_type: Optional[str] = None
    artifact_kind: str = "unknown"
    artifact_payload: Optional[Any] = None
    artifact_preview: Optional[str] = None
    candidate_content: Optional[str] = None
    status: Optional[str] = None
    optimization_mode: Optional[str] = None
    score: Optional[float] = None
    reward: Optional[float] = None
    objective: Optional[float] = None
    candidate_priority: Optional[Dict[str, Any]] = None
    candidate_priority_score: Optional[float] = None
    long_horizon_metrics: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PolicyCandidate:
        candidate_id = str(data.get("candidate_id") or "").strip()
        artifact_kind = str(data.get("artifact_kind") or "").strip() or "unknown"
        artifact_payload = data.get("artifact_payload")
        if artifact_payload is None:
            artifact_payload = data.get("candidate_artifact")
        artifact_preview = data.get("artifact_preview")
        if isinstance(artifact_preview, str):
            artifact_preview = artifact_preview.strip() or None
        else:
            artifact_preview = None

        candidate_content = data.get("candidate_content")
        if not isinstance(candidate_content, str) or not candidate_content.strip():
            candidate_content = None
            for key in ("candidate_code", "prompt_text", "instruction", "text"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_content = value.strip()
                    break
        if candidate_content is None and isinstance(artifact_payload, dict):
            for key in (
                "candidate_code",
                "candidate_content",
                "prompt_text",
                "instruction",
                "text",
            ):
                value = artifact_payload.get(key)
                if isinstance(value, str) and value.strip():
                    candidate_content = value.strip()
                    break

        objective = _extract_candidate_objective(data)
        reward = _extract_candidate_reward(data)
        score = _coerce_float(data.get("score")) or objective or reward
        candidate_priority = _extract_candidate_mapping(data, "candidate_priority")
        candidate_priority_score = _first_non_none(
            _coerce_float(data.get("candidate_priority_score")),
            _coerce_float(
                candidate_priority.get("priority_score")
                if isinstance(candidate_priority, dict)
                else None
            ),
        )
        long_horizon_metrics = _extract_candidate_mapping(
            data,
            "long_horizon_metrics",
            "v2_long_horizon_metrics",
        )
        return cls(
            candidate_id=candidate_id,
            candidate_type=(
                str(data.get("candidate_type")).strip()
                if data.get("candidate_type") is not None
                else None
            ),
            artifact_kind=artifact_kind,
            artifact_payload=artifact_payload,
            artifact_preview=artifact_preview,
            candidate_content=candidate_content,
            status=str(data.get("status")).strip() if data.get("status") is not None else None,
            optimization_mode=(
                str(data.get("optimization_mode")).strip()
                if data.get("optimization_mode") is not None
                else (str(data.get("mode")).strip() if data.get("mode") is not None else None)
            ),
            score=score,
            reward=reward,
            objective=objective,
            candidate_priority=candidate_priority,
            candidate_priority_score=candidate_priority_score,
            long_horizon_metrics=long_horizon_metrics,
            raw=dict(data),
        )

    @property
    def long_horizon_metrics_typed(self) -> Optional[LongHorizonMetricsView]:
        """Best-effort typed parsing of candidate-level long-horizon metrics."""
        if not isinstance(self.long_horizon_metrics, dict):
            return None
        return LongHorizonMetricsView.from_dict(self.long_horizon_metrics)


@dataclass
class PolicyCandidatePage:
    """Typed page of canonical candidates."""

    items: list[PolicyCandidate] = field(default_factory=list)
    next_cursor: Optional[str] = None
    job_id: Optional[str] = None
    system_id: Optional[str] = None
    algorithm: Optional[str] = None
    mode: Optional[str] = None
    sort: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> PolicyCandidatePage:
        raw_items = payload.get("items")
        items = []
        if isinstance(raw_items, list):
            items = [
                PolicyCandidate.from_dict(item) for item in raw_items if isinstance(item, dict)
            ]
        next_cursor = payload.get("next_cursor")
        return cls(
            items=items,
            next_cursor=next_cursor
            if isinstance(next_cursor, str) and next_cursor.strip()
            else None,
            job_id=payload.get("job_id") if isinstance(payload.get("job_id"), str) else None,
            system_id=payload.get("system_id")
            if isinstance(payload.get("system_id"), str)
            else None,
            algorithm=payload.get("algorithm")
            if isinstance(payload.get("algorithm"), str)
            else None,
            mode=payload.get("mode") if isinstance(payload.get("mode"), str) else None,
            sort=payload.get("sort") if isinstance(payload.get("sort"), str) else None,
            raw=dict(payload),
        )


@dataclass
class AchievementEntityView:
    """Typed achievement entity surfaced in long-horizon metrics."""

    achievement_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    global_success_rate: Optional[float] = None
    unlock_rate_by_seed_slice: Dict[str, float] = field(default_factory=dict)
    prereq_achievement_ids: list[str] = field(default_factory=list)
    learned: Optional[bool] = None
    user_defined: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AchievementEntityView:
        payload = dict(data)
        return cls(
            achievement_id=_first_non_none(
                _coerce_str(payload.get("achievement_id")),
                _coerce_str(payload.get("id")),
                _coerce_str(payload.get("key")),
            ),
            name=_coerce_str(payload.get("name")),
            description=_coerce_str(payload.get("description")),
            status=_coerce_str(payload.get("status")),
            global_success_rate=_first_non_none(
                _coerce_float(payload.get("global_success_rate")),
                _coerce_float(payload.get("success_rate")),
            ),
            unlock_rate_by_seed_slice=_extract_string_float_map(
                payload.get("unlock_rate_by_seed_slice")
            ),
            prereq_achievement_ids=_extract_string_list(payload.get("prereq_achievement_ids")),
            learned=_coerce_bool(payload.get("learned")),
            user_defined=_coerce_bool(payload.get("user_defined")),
            raw=payload,
        )


@dataclass
class SubgoalEntityView:
    """Typed subgoal entity surfaced in long-horizon metrics."""

    subgoal_id: Optional[str] = None
    text: Optional[str] = None
    status: Optional[str] = None
    completion_confidence: Optional[float] = None
    completion_rate: Optional[float] = None
    seed: Optional[int] = None
    stage_idx: Optional[int] = None
    achievement_refs: list[str] = field(default_factory=list)
    learned: Optional[bool] = None
    user_defined: Optional[bool] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubgoalEntityView:
        payload = dict(data)
        seed = payload.get("seed")
        stage_idx = payload.get("stage_idx")
        try:
            seed_int = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed_int = None
        try:
            stage_idx_int = int(stage_idx) if stage_idx is not None else None
        except (TypeError, ValueError):
            stage_idx_int = None
        return cls(
            subgoal_id=_first_non_none(
                _coerce_str(payload.get("subgoal_id")),
                _coerce_str(payload.get("id")),
                _coerce_str(payload.get("key")),
            ),
            text=_first_non_none(
                _coerce_str(payload.get("text")), _coerce_str(payload.get("name"))
            ),
            status=_coerce_str(payload.get("status")),
            completion_confidence=_coerce_float(payload.get("completion_confidence")),
            completion_rate=_coerce_float(payload.get("completion_rate")),
            seed=seed_int,
            stage_idx=stage_idx_int,
            achievement_refs=_extract_string_list(payload.get("achievement_refs")),
            learned=_coerce_bool(payload.get("learned")),
            user_defined=_coerce_bool(payload.get("user_defined")),
            raw=payload,
        )


@dataclass
class AgentEntityView:
    """Typed per-agent summary surfaced in long-horizon metrics/progress."""

    agent_id: Optional[str] = None
    role: Optional[str] = None
    eta: Optional[float] = None
    reach: Optional[float] = None
    pi_gap: Optional[float] = None
    failure_codes: list[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentEntityView:
        payload = dict(data)
        return cls(
            agent_id=_first_non_none(
                _coerce_str(payload.get("agent_id")),
                _coerce_str(payload.get("id")),
                _coerce_str(payload.get("name")),
            ),
            role=_coerce_str(payload.get("role")),
            eta=_first_non_none(
                _coerce_float(payload.get("eta")), _coerce_float(payload.get("eta_mean"))
            ),
            reach=_first_non_none(
                _coerce_float(payload.get("reach")),
                _coerce_float(payload.get("reach_score")),
            ),
            pi_gap=_first_non_none(
                _coerce_float(payload.get("pi_gap")),
                _coerce_float(payload.get("pi_gap_candidate")),
            ),
            failure_codes=_extract_string_list(payload.get("failure_codes")),
            raw=payload,
        )


@dataclass
class CounterfactualEdgeView:
    """Typed counterfactual credit edge."""

    source: Optional[str] = None
    target: Optional[str] = None
    lag: Optional[int] = None
    weight: Optional[float] = None
    evidence: Optional[float] = None
    credit: Optional[float] = None
    delta_reward: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CounterfactualEdgeView:
        payload = dict(data)
        lag = payload.get("lag")
        try:
            lag_int = int(lag) if lag is not None else None
        except (TypeError, ValueError):
            lag_int = None
        return cls(
            source=_first_non_none(
                _coerce_str(payload.get("source")),
                _coerce_str(payload.get("from")),
                _coerce_str(payload.get("source_node")),
            ),
            target=_first_non_none(
                _coerce_str(payload.get("target")),
                _coerce_str(payload.get("to")),
                _coerce_str(payload.get("target_node")),
            ),
            lag=lag_int,
            weight=_first_non_none(
                _coerce_float(payload.get("weight")),
                _coerce_float(payload.get("edge_weight")),
            ),
            evidence=_first_non_none(
                _coerce_float(payload.get("evidence")),
                _coerce_float(payload.get("evidence_strength")),
            ),
            credit=_coerce_float(payload.get("credit")),
            delta_reward=_coerce_float(payload.get("delta_reward")),
            raw=payload,
        )


@dataclass
class LongHorizonMetricsView:
    """Best-effort typed view over long-horizon metrics payloads."""

    enabled: bool = False
    available: bool = False
    source: Optional[str] = None
    reason: Optional[str] = None
    candidate_id: Optional[str] = None
    reward_total: Optional[float] = None
    reward_total_eligible: Optional[bool] = None
    eta: Optional[float] = None
    reach: Optional[float] = None
    pi_gap_candidate: Optional[float] = None
    base_coverage_ok: Optional[bool] = None
    planner_used_rate: Optional[float] = None
    planner_fallback_rate: Optional[float] = None
    achievement_entities: list[Dict[str, Any]] = field(default_factory=list)
    subgoal_entities: list[Dict[str, Any]] = field(default_factory=list)
    agent_entities: list[Dict[str, Any]] = field(default_factory=list)
    coordination_failure_codes: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LongHorizonMetricsView:
        payload = dict(data)
        leader = payload.get("leader")
        global_metrics = payload.get("global")
        selected = leader if isinstance(leader, dict) else global_metrics
        if not isinstance(selected, dict):
            selected = payload

        enabled_raw = _first_non_none(
            _coerce_bool(payload.get("enabled")),
            _coerce_bool(selected.get("enabled")),
        )
        available_raw = _first_non_none(
            _coerce_bool(payload.get("available")),
            _coerce_bool(selected.get("available")),
        )

        coordination_failure_codes: Dict[str, int] = {}
        raw_failure_codes = selected.get("coordination_failure_codes")
        if isinstance(raw_failure_codes, dict):
            for code, count in raw_failure_codes.items():
                code_str = _coerce_str(code)
                if code_str is None:
                    continue
                try:
                    coordination_failure_codes[code_str] = int(count)
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw_failure_codes, list):
            for code in _extract_string_list(raw_failure_codes):
                coordination_failure_codes[code] = coordination_failure_codes.get(code, 0) + 1

        return cls(
            enabled=bool(enabled_raw) if enabled_raw is not None else False,
            available=bool(available_raw) if available_raw is not None else False,
            source=_first_non_none(
                _coerce_str(selected.get("source")), _coerce_str(payload.get("source"))
            ),
            reason=_first_non_none(
                _coerce_str(payload.get("reason")), _coerce_str(selected.get("reason"))
            ),
            candidate_id=_first_non_none(
                _coerce_str(selected.get("candidate_id")),
                _coerce_str(payload.get("candidate_id")),
            ),
            reward_total=_coerce_float(selected.get("reward_total")),
            reward_total_eligible=_coerce_bool(selected.get("reward_total_eligible")),
            eta=_first_non_none(
                _coerce_float(selected.get("eta_mean")),
                _coerce_float(selected.get("eta")),
            ),
            reach=_first_non_none(
                _coerce_float(selected.get("reach")),
                _coerce_float(selected.get("reach_score")),
            ),
            pi_gap_candidate=_first_non_none(
                _coerce_float(selected.get("pi_gap_candidate")),
                _coerce_float(selected.get("pi_gap_mean")),
                _coerce_float(selected.get("pi_gap")),
            ),
            base_coverage_ok=_coerce_bool(selected.get("base_coverage_ok")),
            planner_used_rate=_coerce_float(selected.get("planner_used_rate")),
            planner_fallback_rate=_coerce_float(selected.get("planner_fallback_rate")),
            achievement_entities=_extract_mapping_list(selected.get("achievement_entities")),
            subgoal_entities=_extract_mapping_list(selected.get("subgoal_entities")),
            agent_entities=_extract_mapping_list(selected.get("agent_entities")),
            coordination_failure_codes=coordination_failure_codes,
            raw=payload,
        )

    @property
    def achievement_entities_typed(self) -> list[AchievementEntityView]:
        out: list[AchievementEntityView] = []
        for item in self.achievement_entities:
            if isinstance(item, dict):
                out.append(AchievementEntityView.from_dict(item))
        return out

    @property
    def subgoal_entities_typed(self) -> list[SubgoalEntityView]:
        out: list[SubgoalEntityView] = []
        for item in self.subgoal_entities:
            if isinstance(item, dict):
                out.append(SubgoalEntityView.from_dict(item))
        return out

    @property
    def agent_entities_typed(self) -> list[AgentEntityView]:
        out: list[AgentEntityView] = []
        for item in self.agent_entities:
            if isinstance(item, dict):
                out.append(AgentEntityView.from_dict(item))
        return out

    @property
    def counterfactual_edges_typed(self) -> list[CounterfactualEdgeView]:
        counterfactual = self.raw.get("counterfactual_credit")
        if not isinstance(counterfactual, dict):
            return []
        raw_entries = counterfactual.get("edges")
        if not isinstance(raw_entries, list):
            raw_entries = counterfactual.get("top_contributions")
        if not isinstance(raw_entries, list):
            return []
        out: list[CounterfactualEdgeView] = []
        for item in raw_entries:
            if isinstance(item, dict):
                out.append(CounterfactualEdgeView.from_dict(item))
        return out


@dataclass
class LongHorizonProgressView:
    """Best-effort typed view over long-horizon progress payloads."""

    enabled: bool = False
    available: bool = False
    source: Optional[str] = None
    reason: Optional[str] = None
    frontier: list[Dict[str, Any]] = field(default_factory=list)
    agent_frontier: list[Dict[str, Any]] = field(default_factory=list)
    multi_agent: Optional[Dict[str, Any]] = None
    counterfactual_credit: Optional[Dict[str, Any]] = None
    eta: Optional[float] = None
    coordination_failure_codes: list[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LongHorizonProgressView:
        payload = dict(data)
        enabled_raw = _coerce_bool(payload.get("enabled"))
        available_raw = _coerce_bool(payload.get("available"))

        coordination_failure_codes = _extract_string_list(payload.get("coordination_failure_codes"))
        if not coordination_failure_codes:
            raw_multi_agent = payload.get("multi_agent")
            if isinstance(raw_multi_agent, dict):
                coordination_failure_codes = _extract_string_list(
                    raw_multi_agent.get("coordination_failure_codes")
                )

        raw_multi_agent = payload.get("multi_agent")
        raw_counterfactual = payload.get("counterfactual_credit")

        return cls(
            enabled=bool(enabled_raw) if enabled_raw is not None else False,
            available=bool(available_raw) if available_raw is not None else False,
            source=_coerce_str(payload.get("source")),
            reason=_coerce_str(payload.get("reason")),
            frontier=_extract_mapping_list(payload.get("frontier")),
            agent_frontier=_extract_mapping_list(payload.get("agent_frontier")),
            multi_agent=dict(raw_multi_agent) if isinstance(raw_multi_agent, dict) else None,
            counterfactual_credit=(
                dict(raw_counterfactual) if isinstance(raw_counterfactual, dict) else None
            ),
            eta=_first_non_none(
                _coerce_float(payload.get("eta")), _coerce_float(payload.get("eta_mean"))
            ),
            coordination_failure_codes=coordination_failure_codes,
            raw=payload,
        )

    @property
    def frontier_agents_typed(self) -> list[AgentEntityView]:
        out: list[AgentEntityView] = []
        for item in self.agent_frontier:
            if isinstance(item, dict):
                out.append(AgentEntityView.from_dict(item))
        return out

    @property
    def counterfactual_edges_typed(self) -> list[CounterfactualEdgeView]:
        if not isinstance(self.counterfactual_credit, dict):
            return []
        edges = self.counterfactual_credit.get("edges")
        if not isinstance(edges, list):
            edges = self.counterfactual_credit.get("top_contributions")
        if not isinstance(edges, list):
            edges = self.counterfactual_credit.get("counterfactual_recent_contributions")
        if not isinstance(edges, list):
            return []
        out: list[CounterfactualEdgeView] = []
        for item in edges:
            if isinstance(item, dict):
                out.append(CounterfactualEdgeView.from_dict(item))
        return out


@dataclass
class PolicyOptimizationResult:
    """Typed result from a policy optimization job."""

    job_id: str
    status: PolicyJobStatus
    algorithm: Optional[str] = None
    best_reward: Optional[float] = None
    best_candidate: Optional[str | Dict[str, Any]] = None
    attempted_candidates: list[PolicyCandidate] = field(default_factory=list)
    optimized_candidates: list[PolicyCandidate] = field(default_factory=list)
    long_horizon_metrics: Optional[Dict[str, Any]] = None
    long_horizon_progress: Optional[Dict[str, Any]] = None
    lever_summary: Optional[Dict[str, Any]] = None
    sensor_frames: list[Dict[str, Any]] = field(default_factory=list)
    lever_versions: Dict[str, int] = field(default_factory=dict)
    best_lever_version: Optional[int] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, job_id: str, data: Dict[str, Any], *, algorithm: Optional[str] = None
    ) -> PolicyOptimizationResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=True)
        attempted_candidates = _extract_candidate_list(data, "attempted_candidates")
        optimized_candidates = _extract_candidate_list(data, "optimized_candidates")
        long_horizon_metrics = _extract_long_horizon_payload(
            data,
            canonical_key="long_horizon_metrics",
            v2_key="v2_long_horizon_metrics",
        )
        long_horizon_progress = _extract_long_horizon_payload(
            data,
            canonical_key="long_horizon_progress",
            v2_key="v2_long_horizon_progress",
        )
        lever_summary_raw = _from_data_or_metadata(data, "lever_summary")
        lever_summary = lever_summary_raw if isinstance(lever_summary_raw, dict) else None
        sensor_frames_raw = _from_data_or_metadata(data, "sensor_frames")
        sensor_frames = sensor_frames_raw if isinstance(sensor_frames_raw, list) else []
        lever_versions = _parse_lever_versions(_from_data_or_metadata(data, "lever_versions"))
        best_lever_version_raw = _from_data_or_metadata(data, "best_lever_version")
        best_lever_version = None
        if best_lever_version_raw is not None:
            try:
                best_lever_version = int(best_lever_version_raw)
            except (TypeError, ValueError):
                best_lever_version = None
        if lever_versions:
            best_lever_version = best_lever_version or max(int(v) for v in lever_versions.values())
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm or data.get("algorithm"),
            best_reward=best_reward,
            best_candidate=_from_data_or_metadata(data, "best_candidate")
            or _from_data_or_metadata(data, "best_prompt"),
            attempted_candidates=attempted_candidates,
            optimized_candidates=optimized_candidates,
            long_horizon_metrics=long_horizon_metrics,
            long_horizon_progress=long_horizon_progress,
            lever_summary=lever_summary,
            sensor_frames=[frame for frame in sensor_frames if isinstance(frame, dict)],
            lever_versions=lever_versions,
            best_lever_version=best_lever_version,
            error=_normalize_error_message(
                _first_present(data, ("error", "error_message", "failure_reason", "message"))
            ),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == PolicyJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def best_prompt(self) -> Optional[str | Dict[str, Any]]:
        """Backward-compatible alias for `best_candidate`."""
        return self.best_candidate

    @property
    def best_candidate_content(self) -> Optional[str]:
        """Generic best-candidate content (prompt text or non-prompt artifact text)."""
        return _extract_candidate_content(self.best_candidate, self.raw)

    @property
    def v2_long_horizon_metrics(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for long-horizon metrics payload."""
        return self.long_horizon_metrics

    @property
    def v2_long_horizon_progress(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for long-horizon progress payload."""
        return self.long_horizon_progress

    @property
    def long_horizon_metrics_typed(self) -> Optional[LongHorizonMetricsView]:
        """Best-effort typed parsing of long-horizon metrics payload."""
        if not isinstance(self.long_horizon_metrics, dict):
            return None
        return LongHorizonMetricsView.from_dict(self.long_horizon_metrics)

    @property
    def long_horizon_progress_typed(self) -> Optional[LongHorizonProgressView]:
        """Best-effort typed parsing of long-horizon progress payload."""
        if not isinstance(self.long_horizon_progress, dict):
            return None
        return LongHorizonProgressView.from_dict(self.long_horizon_progress)

    @property
    def lever_summary_typed(self) -> Optional[MiproLeverSummary]:
        """Best-effort typed parsing of `lever_summary` for MIPRO runs."""
        return MiproLeverSummary.from_dict(self.lever_summary) if self.lever_summary else None

    @property
    def sensor_frame_summaries_typed(self) -> list[SensorFrameSummary]:
        """Best-effort typed parsing of `sensor_frames` summaries."""
        out: list[SensorFrameSummary] = []
        for frame in self.sensor_frames:
            parsed = SensorFrameSummary.from_dict(frame)
            if parsed is not None:
                out.append(parsed)
        return out


@dataclass
class PromptLearningResult:
    """Typed result from a prompt learning job."""

    job_id: str
    status: PolicyJobStatus
    best_reward: Optional[float] = None
    best_candidate: Optional[str | Dict[str, Any]] = None
    attempted_candidates: list[PolicyCandidate] = field(default_factory=list)
    optimized_candidates: list[PolicyCandidate] = field(default_factory=list)
    long_horizon_metrics: Optional[Dict[str, Any]] = None
    long_horizon_progress: Optional[Dict[str, Any]] = None
    lever_summary: Optional[Dict[str, Any]] = None
    sensor_frames: list[Dict[str, Any]] = field(default_factory=list)
    lever_versions: Dict[str, int] = field(default_factory=dict)
    best_lever_version: Optional[int] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, job_id: str, data: Dict[str, Any]) -> PromptLearningResult:
        status_str = data.get("status", "pending")
        status = PolicyJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=True)
        attempted_candidates = _extract_candidate_list(data, "attempted_candidates")
        optimized_candidates = _extract_candidate_list(data, "optimized_candidates")
        long_horizon_metrics = _extract_long_horizon_payload(
            data,
            canonical_key="long_horizon_metrics",
            v2_key="v2_long_horizon_metrics",
        )
        long_horizon_progress = _extract_long_horizon_payload(
            data,
            canonical_key="long_horizon_progress",
            v2_key="v2_long_horizon_progress",
        )
        lever_summary_raw = _from_data_or_metadata(data, "lever_summary")
        lever_summary = lever_summary_raw if isinstance(lever_summary_raw, dict) else None
        sensor_frames_raw = _from_data_or_metadata(data, "sensor_frames")
        sensor_frames = sensor_frames_raw if isinstance(sensor_frames_raw, list) else []
        lever_versions = _parse_lever_versions(_from_data_or_metadata(data, "lever_versions"))
        best_lever_version_raw = _from_data_or_metadata(data, "best_lever_version")
        best_lever_version = None
        if best_lever_version_raw is not None:
            try:
                best_lever_version = int(best_lever_version_raw)
            except (TypeError, ValueError):
                best_lever_version = None
        if lever_versions:
            best_lever_version = best_lever_version or max(int(v) for v in lever_versions.values())
        return cls(
            job_id=job_id,
            status=status,
            best_reward=best_reward,
            best_candidate=_from_data_or_metadata(data, "best_candidate")
            or _from_data_or_metadata(data, "best_prompt"),
            attempted_candidates=attempted_candidates,
            optimized_candidates=optimized_candidates,
            long_horizon_metrics=long_horizon_metrics,
            long_horizon_progress=long_horizon_progress,
            lever_summary=lever_summary,
            sensor_frames=[frame for frame in sensor_frames if isinstance(frame, dict)],
            lever_versions=lever_versions,
            best_lever_version=best_lever_version,
            error=_normalize_error_message(
                _first_present(data, ("error", "error_message", "failure_reason", "message"))
            ),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == PolicyJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def get_system_prompt(self) -> Optional[str]:
        """Extract the optimized system prompt text.

        Handles various prompt formats (pattern, template, transformation)
        and returns the system prompt as a clean string.

        Returns:
            System prompt text, or None if extraction fails
        """
        return _extract_system_prompt(self.best_candidate, self.raw)

    @property
    def best_prompt(self) -> Optional[str | Dict[str, Any]]:
        """Backward-compatible alias for `best_candidate`."""
        return self.best_candidate

    @property
    def best_candidate_content(self) -> Optional[str]:
        """Generic best-candidate content (prompt text or non-prompt artifact text)."""
        return _extract_candidate_content(self.best_candidate, self.raw)

    @property
    def v2_long_horizon_metrics(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for long-horizon metrics payload."""
        return self.long_horizon_metrics

    @property
    def v2_long_horizon_progress(self) -> Optional[Dict[str, Any]]:
        """Compatibility alias for long-horizon progress payload."""
        return self.long_horizon_progress

    @property
    def long_horizon_metrics_typed(self) -> Optional[LongHorizonMetricsView]:
        """Best-effort typed parsing of long-horizon metrics payload."""
        if not isinstance(self.long_horizon_metrics, dict):
            return None
        return LongHorizonMetricsView.from_dict(self.long_horizon_metrics)

    @property
    def long_horizon_progress_typed(self) -> Optional[LongHorizonProgressView]:
        """Best-effort typed parsing of long-horizon progress payload."""
        if not isinstance(self.long_horizon_progress, dict):
            return None
        return LongHorizonProgressView.from_dict(self.long_horizon_progress)

    @property
    def lever_summary_typed(self) -> Optional[MiproLeverSummary]:
        """Best-effort typed parsing of `lever_summary` for MIPRO runs."""
        return MiproLeverSummary.from_dict(self.lever_summary) if self.lever_summary else None

    @property
    def sensor_frame_summaries_typed(self) -> list[SensorFrameSummary]:
        """Best-effort typed parsing of `sensor_frames` summaries."""
        out: list[SensorFrameSummary] = []
        for frame in self.sensor_frames:
            parsed = SensorFrameSummary.from_dict(frame)
            if parsed is not None:
                out.append(parsed)
        return out


@dataclass
class GraphOptimizationResult:
    """Typed result from a graph optimization job."""

    job_id: str
    status: GraphJobStatus
    algorithm: Optional[str] = None
    best_reward: Optional[float] = None
    best_yaml: Optional[str] = None
    best_snapshot_id: Optional[str] = None
    generations_completed: Optional[int] = None
    total_candidates_evaluated: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(
        cls, job_id: str, data: Dict[str, Any], *, algorithm: Optional[str] = None
    ) -> GraphOptimizationResult:
        status_str = data.get("status", "pending")
        status = GraphJobStatus.from_string(status_str)
        best_reward = _extract_best_reward_value(data, include_train=False)
        return cls(
            job_id=job_id,
            status=status,
            algorithm=algorithm,
            best_reward=best_reward,
            best_yaml=data.get("best_yaml"),
            best_snapshot_id=data.get("best_snapshot_id"),
            generations_completed=data.get("generations_completed"),
            total_candidates_evaluated=data.get("total_candidates_evaluated"),
            duration_seconds=data.get("duration_seconds"),
            error=data.get("error"),
            raw=data,
        )

    @property
    def succeeded(self) -> bool:
        return self.status.is_success

    @property
    def failed(self) -> bool:
        return self.status == GraphJobStatus.FAILED

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


__all__ = [
    "PolicyJobStatus",
    "GraphJobStatus",
    "PolicyCandidate",
    "PolicyCandidatePage",
    "AchievementEntityView",
    "SubgoalEntityView",
    "AgentEntityView",
    "CounterfactualEdgeView",
    "LongHorizonMetricsView",
    "LongHorizonProgressView",
    "PolicyOptimizationResult",
    "PromptLearningResult",
    "GraphOptimizationResult",
]
