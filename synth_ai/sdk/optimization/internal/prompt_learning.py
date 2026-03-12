"""Internal prompt learning implementation.

Public API: Use canonical policy v1 clients under `synth_ai.sdk.optimization`.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
from urllib.parse import urlparse, urlunparse

from synth_ai.core.utils.optimization_routes import GEPA_API_VERSION, offline_job_path
from synth_ai.core.utils.urls import (
    BACKEND_URL_BASE,
    RUST_BACKEND_URL_BASE,
    infer_prompt_learning_container_url,
    is_cloudflare_tunnel_url,
    is_free_ngrok_url,
    is_local_http_container_url,
    is_synth_managed_ngrok_url,
    is_synthtunnel_url,
)
from synth_ai.sdk.container.auth import ensure_container_auth, has_container_token_signing_key
from synth_ai.sdk.optimization.models import (
    PolicyCandidate,
    PolicyCandidatePage,
    PromptLearningResult,
)

from .builders import (
    PromptLearningBuildResult,
    build_prompt_learning_payload,
    build_prompt_learning_payload_from_mapping,
)
from .container_api import check_container_health
from .pollers import JobPoller, PollOutcome
from .prompt_learning_service import (
    cancel_prompt_learning_job,
    pause_prompt_learning_job,
    query_prompt_learning_workflow_state,
    resume_prompt_learning_job,
)
from .utils import ensure_api_base, run_sync
from .validators import reject_legacy_policy_optimization

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.prompt_learning.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "PromptLearningJob"):
        raise RuntimeError("Rust core PromptLearningJob required; synth_ai_py is unavailable.")
    return synth_ai_py


_LOCAL_BACKEND_HOSTS = {"localhost", "127.0.0.1", "host.docker.internal"}


def _strip_api_suffix(base: str) -> str:
    """Normalize a backend base URL for Rust clients (no trailing /api[/v1])."""
    b = (base or "").strip().rstrip("/")
    if not b:
        return b
    for suffix in ("/api/v1", "/api"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
            break
    return b


def _resolve_python_backend_base_runtime() -> str:
    """Resolve backend base URL at call-time (not import-time).

    This avoids stale BACKEND_URL_BASE surprises when users `import synth_ai` and
    only later export `SYNTH_BACKEND_URL` (common .env workflow).
    """
    for k in ("SYNTH_BACKEND_URL", "SYNTH_API_URL", "BACKEND_URL"):
        v = (os.getenv(k) or "").strip()
        if v:
            return v.rstrip("/")
    return (BACKEND_URL_BASE or "").rstrip("/")


def _resolve_rust_backend_api_base(python_backend_api_base: str) -> str:
    """Resolve Rust backend base URL for prompt-learning operations.

    Local dev runs split services (Python :800x, Rust :808x). Job create/poll
    must hit Rust even when SYNTH_BACKEND_URL points at the Python API.
    """
    # 1) Explicit override wins (user intentionally points at a specific Rust backend).
    override = (os.getenv("SYNTH_RUST_BACKEND_URL_OVERRIDE") or "").strip()
    if override:
        return _strip_api_suffix(override)

    # 2) When Python backend is a known remote Synth URL, derive Rust backend from it.
    # This avoids RUST_BACKEND_URL (often localhost from local dev) overriding when
    # SYNTH_BACKEND_URL points at Railway dev/prod.
    try:
        parsed = urlparse((python_backend_api_base or "").strip())
        host = (parsed.hostname or "").strip().lower()
        scheme = parsed.scheme or "https"
        rust_host = None
        if host == "api-dev.usesynth.ai":
            rust_host = "infra-api-dev.usesynth.ai"
        elif host == "api.usesynth.ai":
            rust_host = "infra-api.usesynth.ai"
        if rust_host is not None:
            netloc = f"{rust_host}:{parsed.port}" if parsed.port else rust_host
            new = parsed._replace(scheme=scheme, netloc=netloc)
            return _strip_api_suffix(urlunparse(new))
    except Exception:
        pass

    # 3) Other env vars (RUST_BACKEND_URL, DEV_RUST_BACKEND_URL, etc.).
    for k in (
        "SYNTH_RUST_BACKEND_URL",
        "LOCAL_RUST_BACKEND_URL",
        "DEV_RUST_BACKEND_URL",
        "PROD_RUST_BACKEND_URL",
        "RUST_BACKEND_URL",
    ):
        v = (os.getenv(k) or "").strip()
        if v:
            return _strip_api_suffix(v)

    # 4) If this looks like a local python slot URL (800x), translate to the rust slot (808x).
    try:
        parsed = urlparse((python_backend_api_base or "").strip())
        host = (parsed.hostname or "").strip().lower()
        port = parsed.port
        if host in _LOCAL_BACKEND_HOSTS and isinstance(port, int) and 8000 <= port <= 8009:
            rust_port = port + 80  # 8000->8080, 8001->8081, ...
            new = parsed._replace(netloc=f"{host}:{rust_port}")
            return _strip_api_suffix(urlunparse(new))
    except Exception:
        pass

    # 5) Fall back to the env-aware rust base.
    return _strip_api_suffix(RUST_BACKEND_URL_BASE)


def _extract_algorithm(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("algorithm",):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    section = payload
    if isinstance(payload.get("prompt_learning"), dict):
        section = payload.get("prompt_learning", {})
    if isinstance(section, dict):
        value = section.get("algorithm")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _infer_algorithm(config: PromptLearningJobConfig) -> str:
    overrides = config.overrides or {}
    for key in ("algorithm", "prompt_learning.algorithm"):
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if config.config_dict:
        algo = _extract_algorithm(config.config_dict)
        if algo:
            return algo
    if config.config_path:
        try:
            payload = synth_ai_py.load_toml(str(config.config_path))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            algo = _extract_algorithm(payload)
            if algo:
                return algo
    return "gepa"


def _algorithm_to_kind(algorithm: str) -> str | None:
    algo = algorithm.strip().lower()
    if algo == "gepa":
        return "gepa_offline"
    if algo in {"mipro", "voyager"}:
        return "mipro_offline"
    return None


def _normalize_offline_config_value(config_value: Any, kind: str) -> dict[str, Any]:
    if isinstance(config_value, dict):
        if kind == "eval":
            return dict(config_value)
        reject_legacy_policy_optimization(config_value)
        if "prompt_learning" in config_value:
            return dict(config_value)
        return {"prompt_learning": dict(config_value)}
    if kind == "eval":
        raise ValueError("eval offline jobs require config to be a dictionary")
    return {"prompt_learning": config_value}


def _default_system_name(kind: str) -> str:
    if kind == "gepa_offline":
        return "sdk-gepa-offline"
    if kind == "mipro_offline":
        return "sdk-mipro-offline"
    return "sdk-offline-job"


def _canonicalize_offline_create_payload(config_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize SDK payloads to canonical offline create contract.

    Backend requires top-level `kind` and canonical `config`.
    Legacy shapes (`algorithm`, `config_body`, `policy_optimization`) are rejected.
    """
    payload = dict(config_payload)

    if "algorithm" in payload:
        raise ValueError("Top-level 'algorithm' is no longer supported; use top-level 'kind'.")
    # Accept config_body from SDK builder and rename to config
    if "config_body" in payload and "config" not in payload:
        payload["config"] = payload.pop("config_body")
    reject_legacy_policy_optimization(payload)

    kind_raw = payload.get("kind")
    kind = kind_raw.strip() if isinstance(kind_raw, str) and kind_raw.strip() else None
    if kind is None:
        # Derive kind from algorithm_name + execution_mode (builder output shape)
        algo = payload.get("algorithm_name", "")
        mode = payload.get("execution_mode", "offline")
        if algo:
            kind = f"{algo}_{mode}"
    if kind is None:
        raise ValueError(
            "request kind is required; provide top-level kind in {gepa_offline,mipro_offline,eval}."
        )

    config_value = payload.get("config")
    if config_value is None:
        config_value = payload
    canonical_config = _normalize_offline_config_value(config_value, kind)
    if kind == "eval":
        inference_mode = payload.get("inference_mode")
        if inference_mode is not None and isinstance(canonical_config, dict):
            canonical_config.setdefault("inference_mode", inference_mode)

    technique = payload.get("technique")
    if not isinstance(technique, str) or not technique.strip():
        technique = "discrete_optimization"

    config_mode = payload.get("config_mode")
    if not isinstance(config_mode, str) or not config_mode.strip():
        config_mode = "DEFAULT"

    system = payload.get("system")
    if not isinstance(system, dict):
        system = {}
    if not isinstance(system.get("name"), str) or not str(system.get("name")).strip():
        system["name"] = _default_system_name(kind)
    if "reuse" not in system:
        system["reuse"] = True

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    auto_start = payload.get("auto_start")
    if not isinstance(auto_start, bool):
        auto_start = True

    return {
        "kind": kind,
        "technique": technique,
        "config_mode": config_mode,
        "system": system,
        "config": canonical_config,
        "metadata": metadata,
        "auto_start": auto_start,
    }


_ROLLOUT_HEALTH_CHECK_MODE_ALIASES: Dict[str, str] = {
    "required": "strict",
    "on": "strict",
    "enabled": "strict",
    "true": "strict",
    "best_effort": "warn",
    "continue_on_failure": "warn",
    "relaxed": "warn",
    "skip": "off",
    "disabled": "off",
    "false": "off",
}
_ROLLOUT_HEALTH_CHECK_MODE_VALUES = {"strict", "warn", "off"}


def _normalize_rollout_health_check_mode_for_sdk(value: str) -> str:
    normalized = _ROLLOUT_HEALTH_CHECK_MODE_ALIASES.get(
        value.strip().lower(), value.strip().lower()
    )
    if normalized not in _ROLLOUT_HEALTH_CHECK_MODE_VALUES:
        valid = ", ".join(sorted(_ROLLOUT_HEALTH_CHECK_MODE_VALUES))
        raise ValueError(
            f"Invalid rollout_health_check_mode. Expected one of: {valid}. Got {value!r}."
        )
    return normalized


def _extract_rollout_health_check_mode(payload: dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    section = payload
    if isinstance(payload.get("prompt_learning"), dict):
        section = payload["prompt_learning"]
    if not isinstance(section, dict):
        return None

    for key in ("rollout_health_check_mode", "backend_rollout_health_check_mode"):
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for nested_key in ("gepa", "mipro"):
        nested = section.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in ("rollout_health_check_mode", "backend_rollout_health_check_mode"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


@dataclass
class PromptLearningJobConfig:
    """Configuration for a prompt learning job.

    This dataclass holds all the configuration needed to submit and run
    a prompt learning job (GEPA optimization).

    Supports two modes:
    1. **File-based**: Provide `config_path` pointing to a TOML file
    2. **Programmatic**: Provide `config_dict` with the configuration directly

    Both modes go through the same `PromptLearningConfig` Pydantic validation.

    Attributes:
        config_path: Path to the TOML configuration file. Mutually exclusive with config_dict.
        config_dict: Dictionary with prompt learning configuration. Mutually exclusive with config_path.
            Should have the same structure as the TOML file (with 'prompt_learning' section).
        backend_url: Base URL of the Synth API backend (e.g., "https://api.usesynth.ai").
        api_key: Synth API key for authentication.
        container_worker_token: SynthTunnel worker token for relay auth when using st.usesynth.ai URLs.
        allow_experimental: If True, allows use of experimental models.
        overrides: Dictionary of config overrides.

    Auth lifecycle:
        - SDK/backend auth uses `SYNTH_API_KEY` bearer auth.
        - SynthTunnel relay auth uses `container_worker_token`.
        - GEPA rollout auth for SynthTunnel/non-local container URLs requires
          signer keys (`SYNTH_CONTAINER_AUTH_PRIVATE_KEY(S)`) for signed
          container auth tokens.
        - `container_api_key` is not part of this API.

    Example (file-based):
        >>> config = PromptLearningJobConfig(
        ...     config_path=Path("my_config.toml"),
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )

    Example (programmatic):
        >>> config = PromptLearningJobConfig(
        ...     config_dict={
        ...         "prompt_learning": {
        ...             "algorithm": "gepa",
        ...             "container_url": "https://tunnel.example.com",
        ...             "task_data": {
        ...                 "split": "train",
        ...                 "train_pools": {"reflection_seeds": [0], "pareto_seeds": []},
        ...                 "validation_seeds": [1],
        ...             },
        ...             "gepa": {
        ...                 "initial_candidate": {"stages": [...]},
        ...                 "termination_conditions": {"total_rollouts": 20},
        ...             },
        ...         }
        ...     },
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key="sk_live_...",
        ... )
    """

    backend_url: str
    api_key: str
    config_path: Optional[Path] = None
    config_dict: Optional[Dict[str, Any]] = None
    container_api_key: Optional[str] = field(default=None, repr=False)
    container_key: Optional[str] = field(default=None, repr=False)
    container_worker_token: Optional[str] = field(default=None, repr=False)
    allow_experimental: Optional[bool] = None
    overrides: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Must provide exactly one of config_path or config_dict
        has_path = self.config_path is not None
        has_dict = self.config_dict is not None

        if has_path and has_dict:
            raise ValueError("Provide either config_path OR config_dict, not both")
        if not has_path and not has_dict:
            raise ValueError("Either config_path or config_dict is required")

        if has_path and not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        if not self.backend_url:
            raise ValueError("backend_url is required (pass backend_url or set SYNTH_BACKEND_URL).")
        if not self.api_key:
            raise ValueError("api_key is required (pass api_key or set SYNTH_API_KEY).")

        if (not self.container_api_key) and isinstance(self.container_key, str):
            container_key = self.container_key.strip()
            self.container_api_key = container_key or None

        task_url = infer_prompt_learning_container_url(
            overrides=self.overrides or {},
            config_dict=self.config_dict,
            config_path=str(self.config_path) if self.config_path else None,
        )
        algorithm = _infer_algorithm(self)
        is_gepa = algorithm == "gepa"
        has_token_signer = has_container_token_signing_key()
        if task_url and is_cloudflare_tunnel_url(task_url):
            raise ValueError(
                "Cloudflare tunnel URLs are forbidden. Use SynthTunnel or a Synth-managed ngrok-compatible URL."
            )
        if task_url:
            host = (urlparse(task_url).hostname or "").lower()
            if "ngrok" in host and (
                is_free_ngrok_url(task_url) or not is_synth_managed_ngrok_url(task_url)
            ):
                raise ValueError(
                    "ngrok URL is not allowed. Use a Synth-managed ngrok-compatible URL allow-listed in SYNTH_MANAGED_TUNNEL_HOSTS."
                )
        if task_url and is_synthtunnel_url(task_url):
            if is_gepa and not has_token_signer:
                raise ValueError(
                    "GEPA SynthTunnel rollout auth requires "
                    "SYNTH_CONTAINER_AUTH_PRIVATE_KEY or SYNTH_CONTAINER_AUTH_PRIVATE_KEYS."
                )
            ensure_container_auth(
                backend_base=self.backend_url,
                synth_api_key=self.api_key,
            )
            if not (self.container_worker_token or "").strip():
                raise ValueError(
                    "container_worker_token is required for SynthTunnel container_url. "
                    "Pass tunnel.worker_token or set container_worker_token."
                )
        else:
            if (
                is_gepa
                and task_url
                and not has_token_signer
                and not (self.container_api_key or "").strip()
                and not is_local_http_container_url(task_url)
            ):
                raise ValueError(
                    "GEPA rollout auth for non-local container_url requires "
                    "SYNTH_CONTAINER_AUTH_PRIVATE_KEY or SYNTH_CONTAINER_AUTH_PRIVATE_KEYS."
                )


class PromptLearningJobPoller(JobPoller):
    """Poller for prompt learning jobs."""

    def poll_job(self, job_id: str) -> PollOutcome:
        """Poll a prompt learning job by ID.

        Args:
            job_id: Job ID (e.g., "pl_9c58b711c2644083")

        Returns:
            PollOutcome with status and payload
        """
        return super().poll(offline_job_path(job_id, api_version=GEPA_API_VERSION))


class PromptLearningJob:
    """High-level SDK class for running prompt learning jobs (GEPA).

    This class provides a clean API for:
    1. Submitting prompt learning jobs
    2. Polling job status
    3. Retrieving results

    Example:
        >>> from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
        >>>
        >>> # Create job from config
        >>> job = PromptLearningJob.from_config(
        ...     config_path="my_config.toml",
        ...     backend_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"]
        ... )
        >>>
        >>> # Submit job
        >>> job_id = job.submit()
        >>> print(f"Job submitted: {job_id}")
        >>>
        >>> # Stream until complete
        >>> result = job.stream_until_complete(timeout=3600.0)
        >>> print(f"Best reward: {result['best_reward']}")
        >>>
        >>> # Or poll manually
        >>> status = job.get_status()
        >>> print(f"Status: {status['status']}")
    """

    def __init__(
        self,
        config: PromptLearningJobConfig,
        job_id: Optional[str] = None,
        skip_health_check: bool = False,
    ) -> None:
        """Initialize a prompt learning job.

        Args:
            config: Job configuration
            job_id: Existing job ID (if resuming a previous job)
            skip_health_check: If True, skip container health check before submission.
                              Useful when using tunnels where DNS may not have propagated yet.
                              When enabled and no explicit rollout_health_check_mode is set
                              in config/overrides, backend rollout preflight defaults to
                              prompt_learning.rollout_health_check_mode='warn'.
        """
        self.config = config
        self._job_id = job_id
        self._build_result: Optional[PromptLearningBuildResult] = None
        self._skip_health_check = skip_health_check
        self._rust_job: Any | None = None

        rust = _require_rust()
        if job_id:
            rust_base = _resolve_rust_backend_api_base(self.config.backend_url)
            self._rust_job = rust.PromptLearningJob.from_job_id(
                job_id, self.config.api_key, rust_base
            )

    def _ensure_rust_job(self, config_payload: Optional[Dict[str, Any]] = None) -> Any | None:
        rust = _require_rust()
        if self._rust_job is not None:
            return self._rust_job
        rust_base = _resolve_rust_backend_api_base(self.config.backend_url)
        if self._job_id:
            self._rust_job = rust.PromptLearningJob.from_job_id(
                self._job_id, self.config.api_key, rust_base
            )
        elif config_payload is not None:
            self._rust_job = rust.PromptLearningJob.from_dict(
                config_payload,
                self.config.api_key,
                rust_base,
                self.config.container_worker_token,
            )
        return self._rust_job

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        container_worker_token: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PromptLearningJob:
        """Create a job from a TOML config file.

        Args:
            config_path: Path to TOML config file
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            container_worker_token: SynthTunnel worker token for relay auth
            allow_experimental: Allow experimental models
            overrides: Config overrides

        Returns:
            PromptLearningJob instance

        Raises:
            ValueError: If required config is missing
            FileNotFoundError: If config file doesn't exist
        """

        config_path_obj = Path(config_path)

        if not backend_url:
            backend_url = _resolve_python_backend_base_runtime()

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = PromptLearningJobConfig(
            config_path=config_path_obj,
            backend_url=backend_url,
            api_key=api_key,
            container_worker_token=container_worker_token,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )

        return cls(config)

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        container_worker_token: Optional[str] = None,
        allow_experimental: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
        skip_health_check: bool = False,
    ) -> PromptLearningJob:
        """Create a job from a configuration dictionary (programmatic use).

        This allows creating prompt learning jobs without a TOML file, enabling
        programmatic use in notebooks, scripts, and applications.

        The config_dict should have the same structure as a TOML file:
        ```python
        {
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": "https://...",
                "task_data": {
                    "split": "train",
                    "train_pools": {"reflection_seeds": [0], "pareto_seeds": []},
                    "validation_seeds": [1],
                },
                "gepa": {
                    "initial_candidate": {"stages": [...]},
                    "termination_conditions": {"total_rollouts": 20},
                },
            }
        }
        ```

        Args:
            config_dict: Configuration dictionary with 'prompt_learning' section
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)
            container_worker_token: SynthTunnel worker token for relay auth
            allow_experimental: Allow experimental models
            overrides: Config overrides
            skip_health_check: If True, skip SDK pre-submit container health check.
                Also defaults backend rollout preflight mode to 'warn' unless
                rollout_health_check_mode is explicitly set in config/overrides.

        Returns:
            PromptLearningJob instance

        Raises:
            ValueError: If required config is missing or invalid

        Example:
            >>> job = PromptLearningJob.from_dict(
            ...     config_dict={
            ...         "prompt_learning": {
            ...             "algorithm": "gepa",
            ...             "container_url": "https://tunnel.example.com",
            ...             "gepa": {
            ...                 "initial_candidate": {"stages": [...]},
            ...                 "termination_conditions": {"total_rollouts": 50},
            ...             },
            ...         }
            ...     },
            ...     api_key="sk_live_...",
            ... )
            >>> job_id = job.submit()
        """

        if not backend_url:
            backend_url = _resolve_python_backend_base_runtime()

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        config = PromptLearningJobConfig(
            config_dict=config_dict,
            backend_url=backend_url,
            api_key=api_key,
            container_worker_token=container_worker_token,
            allow_experimental=allow_experimental,
            overrides=overrides or {},
        )

        # Auto-detect tunnel URLs and skip health check if not explicitly set
        if skip_health_check is False:  # Only auto-detect if not explicitly True
            pl = config_dict.get("prompt_learning", {}) if isinstance(config_dict, dict) else {}
            task_url = pl.get("container_url")
            if task_url and is_synthtunnel_url(task_url):
                skip_health_check = True

        return cls(config, skip_health_check=skip_health_check)

    @classmethod
    def from_job_id(
        cls,
        job_id: str,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> PromptLearningJob:
        """Resume an existing job by ID.

        Args:
            job_id: Existing job ID
            backend_url: Backend API URL (defaults to env or production)
            api_key: API key (defaults to SYNTH_API_KEY env var)

        Returns:
            PromptLearningJob instance for the existing job
        """

        if not backend_url:
            backend_url = _resolve_python_backend_base_runtime()

        if not api_key:
            api_key = os.environ.get("SYNTH_API_KEY")
            if not api_key:
                raise ValueError(
                    "api_key is required (provide explicitly or set SYNTH_API_KEY env var)"
                )

        # Create minimal config (we don't need the config for resuming - use empty dict)
        # The config_dict is never used when resuming since we have the job_id
        config = PromptLearningJobConfig(
            config_dict={"prompt_learning": {"_resumed": True}},  # Placeholder for resume mode
            backend_url=backend_url,
            api_key=api_key,
        )

        return cls(config, job_id=job_id)

    def _build_payload(self) -> PromptLearningBuildResult:
        """Build the job payload from config.

        Supports both file-based (config_path) and programmatic (config_dict) modes.
        Both modes route through the same PromptLearningConfig Pydantic validation.
        """
        if self._build_result is None:
            overrides = dict(self.config.overrides or {})
            overrides["backend"] = self.config.backend_url
            override_mode_value = next(
                (
                    value
                    for key, value in overrides.items()
                    if key
                    in (
                        "prompt_learning.rollout_health_check_mode",
                        "prompt_learning.backend_rollout_health_check_mode",
                        "prompt_learning.gepa.rollout_health_check_mode",
                        "prompt_learning.gepa.backend_rollout_health_check_mode",
                        "prompt_learning.mipro.rollout_health_check_mode",
                        "prompt_learning.mipro.backend_rollout_health_check_mode",
                    )
                    and isinstance(value, str)
                    and value.strip()
                ),
                None,
            )

            declared_mode_value: Optional[str] = None
            if self.config.config_dict is not None:
                declared_mode_value = _extract_rollout_health_check_mode(self.config.config_dict)
            elif self.config.config_path is not None:
                try:
                    payload = synth_ai_py.load_toml(str(self.config.config_path))
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    declared_mode_value = _extract_rollout_health_check_mode(payload)

            if isinstance(declared_mode_value, str) and declared_mode_value.strip():
                _normalize_rollout_health_check_mode_for_sdk(declared_mode_value)
            if isinstance(override_mode_value, str) and override_mode_value.strip():
                overrides["prompt_learning.rollout_health_check_mode"] = (
                    _normalize_rollout_health_check_mode_for_sdk(override_mode_value)
                )
            elif self._skip_health_check and not declared_mode_value:
                # Keep default strict behavior unless caller explicitly asked to skip SDK preflight.
                # In that case, relax backend preflight to warn so backend rollout workers do not
                # hard-fail before the run starts.
                overrides["prompt_learning.rollout_health_check_mode"] = "warn"

            # Route to appropriate builder based on config mode
            if self.config.config_dict is not None:
                # Programmatic mode: use dict-based builder
                self._build_result = build_prompt_learning_payload_from_mapping(
                    raw_config=self.config.config_dict,
                    task_url=None,
                    overrides=overrides,
                    allow_experimental=self.config.allow_experimental,
                    source_label="PromptLearningJob.from_dict",
                )
            elif self.config.config_path is not None:
                # File-based mode: use path-based builder
                if not self.config.config_path.exists():
                    raise RuntimeError(
                        f"Config file not found: {self.config.config_path}. "
                        "Use from_dict() for programmatic config or from_job_id() to resume."
                    )

                self._build_result = build_prompt_learning_payload(
                    config_path=self.config.config_path,
                    task_url=None,
                    overrides=overrides,
                    allow_experimental=self.config.allow_experimental,
                )
            else:
                raise RuntimeError(
                    "Cannot build payload: either config_path or config_dict is required. "
                    "Use from_config() for file-based config, from_dict() for programmatic config, "
                    "or from_job_id() to resume an existing job."
                )
        return self._build_result

    def submit(self) -> str:
        """Submit the job to the backend.

        Returns:
            Job ID

        Raises:
            RuntimeError: If job submission fails
            ValueError: If container health check fails
        """
        if self._job_id:
            raise RuntimeError(f"Job already submitted: {self._job_id}")

        build = self._build_payload()
        task_url = (build.task_url or "").strip()
        is_synth = is_synthtunnel_url(task_url)
        if is_synth and not (self.config.container_worker_token or "").strip():
            raise ValueError(
                "container_worker_token is required for SynthTunnel container_url. "
                "Pass tunnel.worker_token or set container_worker_token."
            )

        # Header-only worker-token transport policy:
        # worker token must be sent via dedicated transport header in backend flows.
        config_payload = _canonicalize_offline_create_payload(build.payload)

        # Health check (skip if _skip_health_check is set - useful for tunnels with DNS delay)
        if not self._skip_health_check and task_url:
            if is_synth:
                health = check_container_health(
                    task_url,
                    "",
                    worker_token=self.config.container_worker_token,
                )
                if not health.ok:
                    raise ValueError(
                        f"Container health check failed for container_url={task_url!r}: {health.detail}. "
                        "If this URL is a fresh tunnel, retry after DNS propagation or use skip_health_check=True."
                    )
            else:
                health = check_container_health(task_url, "")
                if not health.ok:
                    raise ValueError(
                        f"Container health check failed for container_url={task_url!r}: {health.detail}. "
                        "If this URL is a fresh tunnel, retry after DNS propagation or use skip_health_check=True."
                    )

        # Submit job
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Submitting job to: %s", self.config.backend_url)

        rust_job = self._ensure_rust_job(config_payload=config_payload)
        job_id = rust_job.submit()
        if not job_id:
            raise RuntimeError("Response missing job ID")
        self._job_id = job_id
        return job_id

    @property
    def job_id(self) -> Optional[str]:
        """Get the job ID (None if not yet submitted)."""
        return self._job_id

    async def get_status_async(self) -> Dict[str, Any]:
        """Get current job status (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        result = await asyncio.to_thread(rust_job.get_status)
        return dict(result) if isinstance(result, dict) else {}

    def get_status(self) -> Dict[str, Any]:
        """Get current job status.

        Returns:
            Job status dictionary

        Raises:
            RuntimeError: If job hasn't been submitted yet
            ValueError: If job ID format is invalid
        """
        rust_job = self._ensure_rust_job()
        result = rust_job.get_status()
        return dict(result) if isinstance(result, dict) else {}

    async def stream_until_complete_async(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PromptLearningResult:
        """Stream job events until completion using SSE (async)."""
        import contextlib

        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        if rust_job is not None and handlers is None:
            result = await asyncio.to_thread(rust_job.stream_until_complete, timeout)
            payload = dict(result) if isinstance(result, dict) else {}
            if on_event and isinstance(payload, dict):
                with contextlib.suppress(Exception):
                    on_event(payload)
            return PromptLearningResult.from_response(self._job_id, payload)

        from .prompt_learning_streaming import build_prompt_learning_streamer

        streamer = build_prompt_learning_streamer(
            backend_url=ensure_api_base(self.config.backend_url),
            api_key=self.config.api_key,
            job_id=self._job_id,
            handlers=handlers,
            interval=interval,
            timeout=timeout,
        )

        final_status = await streamer.stream_until_terminal()

        if on_event and isinstance(final_status, dict):
            with contextlib.suppress(Exception):
                on_event(final_status)

        # SSE final_status may not have full results - fetch them if job succeeded
        status_str = str(final_status.get("status", "")).lower()
        if status_str in ("succeeded", "completed", "success"):
            with contextlib.suppress(Exception):
                full_results = await self.get_results_async()
                # Merge full results into final_status
                final_status.update(
                    {
                        "best_candidate": full_results.get("best_candidate"),
                        "best_reward": full_results.get("best_reward"),
                        "lever_summary": full_results.get("lever_summary"),
                        "sensor_frames": full_results.get("sensor_frames"),
                        "lever_versions": full_results.get("lever_versions"),
                        "best_lever_version": full_results.get("best_lever_version"),
                    }
                )
        if final_status.get("best_reward") is None and final_status.get("best_score") is not None:
            final_status["best_reward"] = final_status.get("best_score")

        return PromptLearningResult.from_response(self._job_id, final_status)

    def stream_until_complete(
        self,
        *,
        timeout: float = 3600.0,
        interval: float = 15.0,
        handlers: Optional[Sequence[Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PromptLearningResult:
        """Stream job events until completion using SSE.

        This provides real-time event streaming instead of polling.
        Note: on_event is invoked once with the final status payload; use handlers
        to receive per-event callbacks.
        """
        return run_sync(
            self.stream_until_complete_async(
                timeout=timeout,
                interval=interval,
                handlers=handlers,
                on_event=on_event,
            ),
            label="stream_until_complete() (use stream_until_complete_async in async contexts)",
        )

    async def get_results_async(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.) (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        rust_job = self._ensure_rust_job()
        result = await asyncio.to_thread(rust_job.get_results)
        return dict(result) if isinstance(result, dict) else {}

    def get_results(self) -> Dict[str, Any]:
        """Get job results (prompts, scores, etc.)."""
        rust_job = self._ensure_rust_job()
        result = rust_job.get_results()
        return dict(result) if isinstance(result, dict) else {}

    async def list_candidates_async(
        self,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical candidates for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.list_candidates(
            self._job_id,
            algorithm=algorithm,
            mode=mode,
            status=status,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )

    async def list_candidates_typed_async(
        self,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> PolicyCandidatePage:
        """List canonical candidates for this job as typed artifacts (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.list_candidates_typed(
            self._job_id,
            algorithm=algorithm,
            mode=mode,
            status=status,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )

    def list_candidates(
        self,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List canonical candidates for this job."""
        return run_sync(
            self.list_candidates_async(
                algorithm=algorithm,
                mode=mode,
                status=status,
                limit=limit,
                cursor=cursor,
                sort=sort,
                include=include,
            ),
            label="list_candidates() (use list_candidates_async in async contexts)",
        )

    def list_candidates_typed(
        self,
        *,
        algorithm: Optional[str] = None,
        mode: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        sort: Optional[str] = None,
        include: Optional[str] = None,
    ) -> PolicyCandidatePage:
        """List canonical candidates for this job as typed artifacts."""
        return run_sync(
            self.list_candidates_typed_async(
                algorithm=algorithm,
                mode=mode,
                status=status,
                limit=limit,
                cursor=cursor,
                sort=sort,
                include=include,
            ),
            label="list_candidates_typed() (use list_candidates_typed_async in async contexts)",
        )

    async def get_candidate_async(self, candidate_id: str) -> Dict[str, Any]:
        """Get a canonical candidate by id for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_candidate(self._job_id, candidate_id)

    async def get_candidate_typed_async(self, candidate_id: str) -> PolicyCandidate:
        """Get a canonical candidate as typed artifact for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_candidate_typed(self._job_id, candidate_id)

    def get_candidate(self, candidate_id: str) -> Dict[str, Any]:
        """Get a canonical candidate by id for this job."""
        return run_sync(
            self.get_candidate_async(candidate_id),
            label="get_candidate() (use get_candidate_async in async contexts)",
        )

    def get_candidate_typed(self, candidate_id: str) -> PolicyCandidate:
        """Get a canonical candidate as typed artifact for this job."""
        return run_sync(
            self.get_candidate_typed_async(candidate_id),
            label="get_candidate_typed() (use get_candidate_typed_async in async contexts)",
        )

    async def submit_candidates_async(
        self,
        *,
        algorithm_kind: str,
        candidates: list[Dict[str, Any]],
        proposal_session_id: Optional[str] = None,
        proposer_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit typed GEPA/MIPRO candidates through the explicit submit contract."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.submit_candidates(
            job_id=self._job_id,
            algorithm_kind=algorithm_kind,
            candidates=candidates,
            proposal_session_id=proposal_session_id,
            proposer_metadata=proposer_metadata,
        )

    def submit_candidates(
        self,
        *,
        algorithm_kind: str,
        candidates: list[Dict[str, Any]],
        proposal_session_id: Optional[str] = None,
        proposer_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit typed GEPA/MIPRO candidates through the explicit submit contract."""
        return run_sync(
            self.submit_candidates_async(
                algorithm_kind=algorithm_kind,
                candidates=candidates,
                proposal_session_id=proposal_session_id,
                proposer_metadata=proposer_metadata,
            ),
            label="submit_candidates() (use submit_candidates_async in async contexts)",
        )

    async def get_state_baseline_info_async(self) -> Dict[str, Any]:
        """Get persisted state-envelope baseline_info for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_state_baseline_info(self._job_id)

    def get_state_baseline_info(self) -> Dict[str, Any]:
        """Get persisted state-envelope baseline_info for this job."""
        return run_sync(
            self.get_state_baseline_info_async(),
            label="get_state_baseline_info() (use get_state_baseline_info_async in async contexts)",
        )

    async def get_state_envelope_async(self) -> Dict[str, Any]:
        """Get full persisted state-envelope payload for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_state_envelope(self._job_id)

    def get_state_envelope(self) -> Dict[str, Any]:
        """Get full persisted state-envelope payload for this job."""
        return run_sync(
            self.get_state_envelope_async(),
            label="get_state_envelope() (use get_state_envelope_async in async contexts)",
        )

    async def list_trial_queue_async(self) -> Dict[str, Any]:
        """List persisted trial queue for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.list_trial_queue(self._job_id)

    def list_trial_queue(self) -> Dict[str, Any]:
        """List persisted trial queue for this job."""
        return run_sync(
            self.list_trial_queue_async(),
            label="list_trial_queue() (use list_trial_queue_async in async contexts)",
        )

    async def enqueue_trial_async(
        self,
        *,
        trial: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enqueue one persisted trial spec for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.enqueue_trial(
            self._job_id,
            trial=trial,
            algorithm_kind=algorithm_kind,
        )

    def enqueue_trial(
        self,
        *,
        trial: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enqueue one persisted trial spec for this job."""
        return run_sync(
            self.enqueue_trial_async(trial=trial, algorithm_kind=algorithm_kind),
            label="enqueue_trial() (use enqueue_trial_async in async contexts)",
        )

    async def update_trial_async(
        self,
        trial_id: str,
        *,
        patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch one persisted trial spec for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.update_trial(
            self._job_id,
            trial_id,
            patch=patch,
            algorithm_kind=algorithm_kind,
        )

    def update_trial(
        self,
        trial_id: str,
        *,
        patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch one persisted trial spec for this job."""
        return run_sync(
            self.update_trial_async(
                trial_id,
                patch=patch,
                algorithm_kind=algorithm_kind,
            ),
            label="update_trial() (use update_trial_async in async contexts)",
        )

    async def cancel_trial_async(
        self,
        trial_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel one persisted trial for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.cancel_trial(
            self._job_id,
            trial_id,
            algorithm_kind=algorithm_kind,
        )

    def cancel_trial(
        self,
        trial_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel one persisted trial for this job."""
        return run_sync(
            self.cancel_trial_async(trial_id, algorithm_kind=algorithm_kind),
            label="cancel_trial() (use cancel_trial_async in async contexts)",
        )

    async def reorder_trials_async(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reorder persisted trial queue for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.reorder_trials(
            self._job_id,
            trial_ids=trial_ids,
            algorithm_kind=algorithm_kind,
        )

    def reorder_trials(
        self,
        *,
        trial_ids: list[str],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reorder persisted trial queue for this job."""
        return run_sync(
            self.reorder_trials_async(trial_ids=trial_ids, algorithm_kind=algorithm_kind),
            label="reorder_trials() (use reorder_trials_async in async contexts)",
        )

    async def apply_default_trial_plan_async(
        self,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply default persisted trial plan templates for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.apply_default_trial_plan(
            self._job_id,
            algorithm_kind=algorithm_kind,
        )

    def apply_default_trial_plan(
        self,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply default persisted trial plan templates for this job."""
        return run_sync(
            self.apply_default_trial_plan_async(algorithm_kind=algorithm_kind),
            label="apply_default_trial_plan() (use apply_default_trial_plan_async in async contexts)",
        )

    async def get_rollout_queue_async(self) -> Dict[str, Any]:
        """Get persisted rollout queue state for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_rollout_queue(self._job_id)

    def get_rollout_queue(self) -> Dict[str, Any]:
        """Get persisted rollout queue state for this job."""
        return run_sync(
            self.get_rollout_queue_async(),
            label="get_rollout_queue() (use get_rollout_queue_async in async contexts)",
        )

    async def set_rollout_queue_policy_async(
        self,
        *,
        policy_patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch persisted rollout queue policy for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.set_rollout_queue_policy(
            self._job_id,
            policy_patch=policy_patch,
            algorithm_kind=algorithm_kind,
        )

    def set_rollout_queue_policy(
        self,
        *,
        policy_patch: Dict[str, Any],
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch persisted rollout queue policy for this job."""
        return run_sync(
            self.set_rollout_queue_policy_async(
                policy_patch=policy_patch,
                algorithm_kind=algorithm_kind,
            ),
            label="set_rollout_queue_policy() (use set_rollout_queue_policy_async in async contexts)",
        )

    async def get_rollout_dispatch_metrics_async(self) -> Dict[str, Any]:
        """Get persisted rollout dispatch metrics for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_rollout_dispatch_metrics(self._job_id)

    def get_rollout_dispatch_metrics(self) -> Dict[str, Any]:
        """Get persisted rollout dispatch metrics for this job."""
        return run_sync(
            self.get_rollout_dispatch_metrics_async(),
            label=(
                "get_rollout_dispatch_metrics() "
                "(use get_rollout_dispatch_metrics_async in async contexts)"
            ),
        )

    async def get_rollout_limiter_status_async(self) -> Dict[str, Any]:
        """Get rollout scheduler limiter status for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.get_rollout_limiter_status(self._job_id)

    def get_rollout_limiter_status(self) -> Dict[str, Any]:
        """Get rollout scheduler limiter status for this job."""
        return run_sync(
            self.get_rollout_limiter_status_async(),
            label=(
                "get_rollout_limiter_status() "
                "(use get_rollout_limiter_status_async in async contexts)"
            ),
        )

    async def retry_rollout_dispatch_async(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retry a rollout dispatch for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.retry_rollout_dispatch(
            self._job_id,
            dispatch_id,
            algorithm_kind=algorithm_kind,
        )

    def retry_rollout_dispatch(
        self,
        dispatch_id: str,
        *,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retry a rollout dispatch for this job."""
        return run_sync(
            self.retry_rollout_dispatch_async(
                dispatch_id,
                algorithm_kind=algorithm_kind,
            ),
            label=("retry_rollout_dispatch() (use retry_rollout_dispatch_async in async contexts)"),
        )

    async def drain_rollout_queue_async(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set rollout queue draining mode for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.drain_rollout_queue(
            self._job_id,
            cancel_queued=cancel_queued,
            algorithm_kind=algorithm_kind,
        )

    def drain_rollout_queue(
        self,
        *,
        cancel_queued: bool = False,
        algorithm_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set rollout queue draining mode for this job."""
        return run_sync(
            self.drain_rollout_queue_async(
                cancel_queued=cancel_queued,
                algorithm_kind=algorithm_kind,
            ),
            label="drain_rollout_queue() (use drain_rollout_queue_async in async contexts)",
        )

    async def list_seed_evals_async(
        self,
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
        """List canonical seed evaluations for this job (async)."""
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")
        from .learning.prompt_learning_client import PromptLearningClient

        client = PromptLearningClient(
            base_url=self.config.backend_url,
            api_key=self.config.api_key,
            api_version=GEPA_API_VERSION,
        )
        return await client.list_seed_evals(
            self._job_id,
            split=split,
            seed=seed,
            success=success,
            candidate_id=candidate_id,
            limit=limit,
            cursor=cursor,
            sort=sort,
            include=include,
        )

    def list_seed_evals(
        self,
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
        """List canonical seed evaluations for this job."""
        return run_sync(
            self.list_seed_evals_async(
                split=split,
                seed=seed,
                success=success,
                candidate_id=candidate_id,
                limit=limit,
                cursor=cursor,
                sort=sort,
                include=include,
            ),
            label="list_seed_evals() (use list_seed_evals_async in async contexts)",
        )

    def pause(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Pause a running prompt learning job.

        Args:
            reason: Optional reason for pausing

        Returns:
            Dict with pause status

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return pause_prompt_learning_job(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            reason=reason,
        )

    def resume(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Resume a paused prompt learning job.

        Args:
            reason: Optional reason for resuming

        Returns:
            Dict with resume status

        Raises:
            RuntimeError: If job hasn't been submitted yet
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return resume_prompt_learning_job(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            reason=reason,
        )

    def cancel(self, *, reason: Optional[str] = None) -> Dict[str, Any]:
        """Cancel a running prompt learning job.

        Sends a cancellation request to the backend. For GEPA/MiPRO jobs,
        this sets the cancel_requested flag and the optimizer will stop
        at the next checkpoint.

        Args:
            reason: Optional reason for cancellation (recorded in job metadata)

        Returns:
            Dict with cancellation status:
            - job_id: The job ID
            - status: "succeeded", "partial", or "failed"
            - message: Human-readable status message
            - attempt_id: ID of the cancel attempt (for debugging)

        Raises:
            RuntimeError: If job hasn't been submitted yet
            httpx.HTTPStatusError: If the cancellation request fails

        Example:
            >>> job.submit()
            >>> # Later...
            >>> result = job.cancel(reason="No longer needed")
            >>> print(result["message"])
            "Cancellation requested. Job will stop at next checkpoint."
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return cancel_prompt_learning_job(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
            reason=reason,
        )

    def query_workflow_state(self) -> Dict[str, Any]:
        """Query the Temporal workflow state for instant polling.

        This queries the workflow directly using its @workflow.query handler,
        providing instant state without database lookups. Useful for real-time
        progress monitoring.

        Returns:
            Dict with workflow state:
            - job_id: The job ID
            - workflow_state: State from the query handler (or None if unavailable)
                - job_id: Job identifier
                - run_id: Current run ID
                - status: Current status (pending, running, succeeded, failed, cancelled)
                - progress: Human-readable progress string
                - algorithm: Algorithm being used (gepa, mipro)
                - error: Error message if failed
            - query_name: Name of the query that was executed
            - error: Error message if query failed (workflow may have completed)

        Raises:
            RuntimeError: If job hasn't been submitted yet

        Example:
            >>> state = job.query_workflow_state()
            >>> if state["workflow_state"]:
            ...     print(f"Status: {state['workflow_state']['status']}")
            ...     print(f"Progress: {state['workflow_state']['progress']}")
            >>> else:
            ...     print(f"Query failed: {state.get('error')}")
        """
        if not self._job_id:
            raise RuntimeError("Job not yet submitted. Call submit() first.")

        return query_prompt_learning_workflow_state(
            backend_url=self.config.backend_url,
            api_key=self.config.api_key,
            job_id=self._job_id,
        )


__all__ = [
    "PromptLearningJob",
    "PromptLearningJobConfig",
    "PromptLearningJobPoller",
    "PromptLearningResult",
]
