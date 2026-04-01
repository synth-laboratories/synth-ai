"""Compatibility wrapper for legacy policy job imports.

This module preserves the pre-frontdoor `policy.job` import path while routing
to canonical prompt-learning internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from synth_ai.core.utils.urls import is_cloudflare_tunnel_url, is_local_hostname
from synth_ai.sdk.container.auth import has_container_token_signing_key
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob


def _extract_container_url(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    section = payload.get("policy_optimization")
    if not isinstance(section, dict):
        return None
    value = section.get("container_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(overrides or {})
    for key in (
        "prompt_learning.container_url",
        "policy_optimization.container_url",
        "container_url",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value.strip():
            normalized["container_url"] = value.strip()
            break
    return normalized


def _is_non_local_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    host = (parsed.hostname or "").lower()
    return not (
        is_local_hostname(host) and (parsed.scheme or "http").lower() == "http"
    )


@dataclass
class PolicyOptimizationJobConfig:
    config_dict: dict[str, Any]
    backend_url: str
    api_key: str
    container_api_key: str | None = None
    container_worker_token: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config_dict, dict):
            raise ValueError("config_dict is required")
        if not str(self.backend_url or "").strip():
            raise ValueError("backend_url is required")
        if not str(self.api_key or "").strip():
            raise ValueError("api_key is required")

        url = _infer_container_url(self)
        if url and is_cloudflare_tunnel_url(url):
            raise ValueError(
                "Cloudflare tunnel URLs are forbidden. Use SynthTunnel or a Synth-managed ngrok-compatible URL."
            )
        if (
            self.algorithm == "gepa"
            and url
            and _is_non_local_url(url)
            and not (self.container_api_key or "").strip()
            and not has_container_token_signing_key()
        ):
            raise ValueError(
                "GEPA non-local container_url requires SYNTH_CONTAINER_AUTH_PRIVATE_KEY or SYNTH_CONTAINER_AUTH_PRIVATE_KEYS."
            )

    @property
    def algorithm(self) -> str:
        section = self.config_dict.get("policy_optimization")
        if isinstance(section, dict):
            raw = section.get("algorithm")
            if isinstance(raw, str) and raw.strip():
                return raw.strip().lower()
        return "gepa"

    def to_prompt_learning_config(self) -> dict[str, Any]:
        converted = deepcopy(self.config_dict)
        if not isinstance(converted, dict):
            return {"prompt_learning": {}}
        policy_section = converted.pop("policy_optimization", None)
        if isinstance(policy_section, dict):
            converted["prompt_learning"] = deepcopy(policy_section)
        elif not isinstance(converted.get("prompt_learning"), dict):
            converted["prompt_learning"] = {}

        inferred = _infer_container_url(self)
        if inferred and isinstance(converted.get("prompt_learning"), dict):
            converted["prompt_learning"]["container_url"] = inferred
        return converted


def _infer_container_url(config: PolicyOptimizationJobConfig) -> str | None:
    normalized_overrides = _normalize_overrides(config.overrides)
    value = normalized_overrides.get("container_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _extract_container_url(config.config_dict)


class PolicyOptimizationJob:
    """Legacy policy job wrapper backed by PromptLearningJob."""

    def __init__(self, config: PolicyOptimizationJobConfig, delegate: PromptLearningJob) -> None:
        self.config = config
        self._delegate = delegate

    @classmethod
    def from_dict(
        cls,
        *,
        config_dict: dict[str, Any],
        backend_url: str,
        api_key: str,
        container_api_key: str | None = None,
        container_worker_token: str | None = None,
        overrides: dict[str, Any] | None = None,
        skip_health_check: bool = False,
    ) -> PolicyOptimizationJob:
        cfg = PolicyOptimizationJobConfig(
            config_dict=config_dict,
            backend_url=backend_url,
            api_key=api_key,
            container_api_key=container_api_key,
            container_worker_token=container_worker_token,
            overrides=dict(overrides or {}),
        )
        delegate = PromptLearningJob.from_dict(
            config_dict=cfg.to_prompt_learning_config(),
            backend_url=backend_url,
            api_key=api_key,
            container_worker_token=container_worker_token,
            overrides=_normalize_overrides(overrides),
            skip_health_check=skip_health_check,
        )
        return cls(config=cfg, delegate=delegate)

    def _get_delegate(self) -> Any:
        return self._delegate

    @property
    def _job_id(self) -> str | None:
        return getattr(self._delegate, "_job_id", None)

    @_job_id.setter
    def _job_id(self, value: str | None) -> None:
        self._delegate._job_id = value

    def list_candidates_typed(self, **kwargs: Any) -> Any:
        return self._get_delegate().list_candidates_typed(**kwargs)

    def get_candidate_typed(self, candidate_id: str) -> Any:
        return self._get_delegate().get_candidate_typed(candidate_id)

    def submit_candidates(
        self,
        *,
        algorithm_kind: str,
        candidates: list[dict[str, Any]],
        proposal_session_id: str | None = None,
        proposer_metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self._get_delegate().submit_candidates(
            algorithm_kind=algorithm_kind,
            candidates=candidates,
            proposal_session_id=proposal_session_id,
            proposer_metadata=proposer_metadata,
        )

    async def submit_candidates_async(
        self,
        *,
        algorithm_kind: str,
        candidates: list[dict[str, Any]],
        proposal_session_id: str | None = None,
        proposer_metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await self._get_delegate().submit_candidates_async(
            algorithm_kind=algorithm_kind,
            candidates=candidates,
            proposal_session_id=proposal_session_id,
            proposer_metadata=proposer_metadata,
        )

    def get_state_baseline_info(self) -> Any:
        return self._get_delegate().get_state_baseline_info()

    async def get_state_baseline_info_async(self) -> Any:
        return await self._get_delegate().get_state_baseline_info_async()

    def get_state_envelope(self) -> Any:
        return self._get_delegate().get_state_envelope()

    async def get_state_envelope_async(self) -> Any:
        return await self._get_delegate().get_state_envelope_async()

    def list_trial_queue(self) -> Any:
        return self._get_delegate().list_trial_queue()

    async def list_trial_queue_async(self) -> Any:
        return await self._get_delegate().list_trial_queue_async()

    def enqueue_trial(
        self,
        *,
        trial: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().enqueue_trial(
            trial=trial,
            algorithm_kind=algorithm_kind,
        )

    async def enqueue_trial_async(
        self,
        *,
        trial: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().enqueue_trial_async(
            trial=trial,
            algorithm_kind=algorithm_kind,
        )

    def update_trial(
        self,
        trial_id: str,
        *,
        patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().update_trial(
            trial_id,
            patch=patch,
            algorithm_kind=algorithm_kind,
        )

    async def update_trial_async(
        self,
        trial_id: str,
        *,
        patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().update_trial_async(
            trial_id,
            patch=patch,
            algorithm_kind=algorithm_kind,
        )

    def cancel_trial(
        self,
        trial_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().cancel_trial(
            trial_id,
            algorithm_kind=algorithm_kind,
        )

    async def cancel_trial_async(
        self,
        trial_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().cancel_trial_async(
            trial_id,
            algorithm_kind=algorithm_kind,
        )

    def reorder_trials(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().reorder_trials(
            trial_ids=trial_ids,
            algorithm_kind=algorithm_kind,
        )

    async def reorder_trials_async(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().reorder_trials_async(
            trial_ids=trial_ids,
            algorithm_kind=algorithm_kind,
        )

    def apply_default_trial_plan(
        self,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().apply_default_trial_plan(
            algorithm_kind=algorithm_kind,
        )

    async def apply_default_trial_plan_async(
        self,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().apply_default_trial_plan_async(
            algorithm_kind=algorithm_kind,
        )

    def get_rollout_queue(self) -> Any:
        return self._get_delegate().get_rollout_queue()

    async def get_rollout_queue_async(self) -> Any:
        return await self._get_delegate().get_rollout_queue_async()

    def set_rollout_queue_policy(
        self,
        *,
        policy_patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().set_rollout_queue_policy(
            policy_patch=policy_patch,
            algorithm_kind=algorithm_kind,
        )

    async def set_rollout_queue_policy_async(
        self,
        *,
        policy_patch: dict[str, Any],
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().set_rollout_queue_policy_async(
            policy_patch=policy_patch,
            algorithm_kind=algorithm_kind,
        )

    def get_rollout_dispatch_metrics(self) -> Any:
        return self._get_delegate().get_rollout_dispatch_metrics()

    async def get_rollout_dispatch_metrics_async(self) -> Any:
        return await self._get_delegate().get_rollout_dispatch_metrics_async()

    def get_rollout_limiter_status(self) -> Any:
        return self._get_delegate().get_rollout_limiter_status()

    async def get_rollout_limiter_status_async(self) -> Any:
        return await self._get_delegate().get_rollout_limiter_status_async()

    def retry_rollout_dispatch(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().retry_rollout_dispatch(
            dispatch_id,
            algorithm_kind=algorithm_kind,
        )

    async def retry_rollout_dispatch_async(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().retry_rollout_dispatch_async(
            dispatch_id,
            algorithm_kind=algorithm_kind,
        )

    def drain_rollout_queue(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: str | None = None,
    ) -> Any:
        return self._get_delegate().drain_rollout_queue(
            cancel_queued=cancel_queued,
            algorithm_kind=algorithm_kind,
        )

    async def drain_rollout_queue_async(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: str | None = None,
    ) -> Any:
        return await self._get_delegate().drain_rollout_queue_async(
            cancel_queued=cancel_queued,
            algorithm_kind=algorithm_kind,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


__all__ = [
    "PolicyOptimizationJob",
    "PolicyOptimizationJobConfig",
    "_extract_container_url",
    "_infer_container_url",
    "has_container_token_signing_key",
]
