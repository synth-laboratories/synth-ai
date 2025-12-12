"""First-class SDK API for Context Learning (Context Engineering).

Context Learning is an infra training job type that optimizes environment
setup scripts (pre-flight / post-flight bash) for terminal/coding agents.

CLI usage:
    uvx synth-ai train --type context_learning --config my_context.toml

SDK usage:
    from synth_ai.sdk.api.train.context_learning import ContextLearningJob

    job = ContextLearningJob.from_config("context.toml")
    job.submit()
    final = job.stream_until_complete()
    best = job.download_best_script()
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from synth_ai.core.telemetry import log_info

from synth_ai.sdk.streaming import (
    ContextLearningHandler,
    JobStreamer,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)

from .utils import ensure_api_base, http_get, http_post, load_toml, TrainError


@dataclass
class ContextLearningSubmitResult:
    job_id: str
    status: str
    created_at: Optional[str] = None


@dataclass
class BestScriptResult:
    job_id: str
    best_score: float
    preflight_script: str
    generation: int
    variation_id: str
    metadata: Dict[str, Any]


@dataclass
class ContextLearningJobConfig:
    config_path: Path
    backend_url: str
    api_key: str
    task_app_api_key: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not self.backend_url:
            raise ValueError("backend_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.task_app_api_key:
            self.task_app_api_key = os.environ.get("ENVIRONMENT_API_KEY")


class ContextLearningJob:
    """High-level SDK class for running Context Learning jobs."""

    def __init__(self, config: ContextLearningJobConfig, job_id: Optional[str] = None) -> None:
        self.config = config
        self._job_id = job_id
        self._payload: Optional[Dict[str, Any]] = None

    @property
    def job_id(self) -> Optional[str]:
        return self._job_id

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        task_app_api_key: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ContextLearningJob:
        from synth_ai.core.env import get_backend_from_env

        config_path_obj = Path(config_path)

        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        cfg = ContextLearningJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            overrides=overrides or {},
        )
        return cls(cfg)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        *,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> ContextLearningJob:
        from synth_ai.core.env import get_backend_from_env

        if not backend_url:
            backend_url = os.environ.get("BACKEND_BASE_URL", "").strip()
            if not backend_url:
                base, _ = get_backend_from_env()
                backend_url = f"{base}/api" if not base.endswith("/api") else base

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        cfg = ContextLearningJobConfig(
            config_path=Path("/dev/null"),
            backend_url=backend_url,
            api_key=api_key,
        )
        return cls(cfg, job_id=job_id)

    def _build_payload(self) -> Dict[str, Any]:
        if self._payload is not None:
            return self._payload

        cfg = load_toml(self.config.config_path)
        section = cfg.get("context_learning")
        if not isinstance(section, dict):
            raise TrainError("Missing required [context_learning] section.")

        task_app_url = section.get("task_app_url") or section.get("task_url")
        if not task_app_url:
            raise TrainError("[context_learning].task_app_url is required")

        evaluation_seeds = section.get("evaluation_seeds") or []
        if not isinstance(evaluation_seeds, list):
            raise TrainError("[context_learning].evaluation_seeds must be a list")

        env_section = section.get("environment") if isinstance(section.get("environment"), dict) else {}
        preflight_script = env_section.get("preflight_script")
        postflight_script = env_section.get("postflight_script")

        # Support script paths
        preflight_path = env_section.get("baseline_preflight_script_path")
        if preflight_path and not preflight_script:
            p = Path(str(preflight_path))
            preflight_script = p.read_text(encoding="utf-8")

        postflight_path = env_section.get("baseline_postflight_script_path")
        if postflight_path and not postflight_script:
            p = Path(str(postflight_path))
            postflight_script = p.read_text(encoding="utf-8")

        algo_section = section.get("algorithm") if isinstance(section.get("algorithm"), dict) else {}

        algorithm_config: Dict[str, Any] = {}
        for key in (
            "initial_population_size",
            "num_generations",
            "mutation_llm_model",
            "policy_config",
        ):
            if key in algo_section and algo_section[key] is not None:
                algorithm_config[key] = algo_section[key]

        metadata = section.get("metadata") if isinstance(section.get("metadata"), dict) else {}

        payload: Dict[str, Any] = {
            "task_app_url": task_app_url,
            "task_app_api_key": self.config.task_app_api_key,
            "evaluation_seeds": evaluation_seeds,
            "environment": {
                "preflight_script": preflight_script,
                "postflight_script": postflight_script,
            },
            "algorithm_config": algorithm_config,
            "metadata": metadata,
            "auto_start": True,
        }

        # Merge overrides (CLI and caller)
        for k, v in (self.config.overrides or {}).items():
            if v is not None:
                payload[k] = v

        self._payload = payload
        return payload

    def submit(self) -> ContextLearningSubmitResult:
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        payload = self._build_payload()
        url = f"{self.config.backend_url.rstrip('/')}/context-learning/jobs"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        log_info("ContextLearningJob.submit invoked", ctx={"url": url})
        resp = http_post(url, headers=headers, json_body=payload)
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Context learning submission failed: {resp.status_code} {resp.text[:500]}"
            )
        js = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        job_id = js.get("job_id")
        if not job_id:
            raise RuntimeError("Response missing job_id")
        self._job_id = job_id
        return ContextLearningSubmitResult(
            job_id=job_id,
            status=js.get("status", "pending"),
            created_at=js.get("created_at"),
        )

    def stream_until_complete(
        self,
        *,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            max_events_per_poll=500,
            deduplicate=True,
        )

        if handlers is None:
            handlers = [ContextLearningHandler()]

        streamer = JobStreamer(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            endpoints=StreamEndpoints.context_learning(self._job_id),
            config=config,
            handlers=list(handlers),
        )

        final_status = asyncio.run(streamer.stream_until_terminal())

        if on_event and isinstance(final_status, dict):
            try:
                on_event(final_status)
            except Exception:
                pass

        return final_status

    def download_best_script(self) -> BestScriptResult:
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        url = f"{self.config.backend_url.rstrip('/')}/context-learning/jobs/{self._job_id}/best-script"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        resp = http_get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch best script: {resp.status_code} {resp.text[:500]}"
            )
        js = resp.json()
        return BestScriptResult(
            job_id=js.get("job_id", self._job_id),
            best_score=float(js.get("best_score") or 0.0),
            preflight_script=str(js.get("preflight_script") or ""),
            generation=int(js.get("generation") or 0),
            variation_id=str(js.get("variation_id") or ""),
            metadata=js.get("metadata") or {},
        )


__all__ = [
    "ContextLearningJob",
    "ContextLearningJobConfig",
    "ContextLearningSubmitResult",
    "BestScriptResult",
]

