import importlib
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
from pydantic import ValidationError

from synth_ai.sdk.shared.models import normalize_model_identifier


def ensure_allowed_model(model: str) -> None:
    """Validate that a model identifier is allowed.

    Args:
        model: Model identifier to validate

    Raises:
        UnsupportedModelError: If the model is not valid
    """
    normalize_model_identifier(model)  # Will raise if invalid


# SFT module import is optional (moved to research repo)
try:
    _sft_module = cast(Any, importlib.import_module("synth_ai.sdk.learning.sft.config"))
    prepare_sft_job_payload = cast(
        Callable[..., dict[str, Any]], _sft_module.prepare_sft_job_payload
    )
    _SFT_AVAILABLE = True
except Exception:  # pragma: no cover - SFT moved to research repo
    prepare_sft_job_payload = None  # type: ignore[assignment]
    _SFT_AVAILABLE = False

try:
    pass
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.builders.") from exc

from synth_ai.core.config.resolver import ConfigResolver  # noqa: E402
from synth_ai.core.utils.urls import (  # noqa: E402
    is_local_http_container_url,
    is_synthtunnel_url,
)
from synth_ai.sdk.container.auth import (  # noqa: E402
    has_container_token_signing_key,
)

from .configs import PromptLearningConfig  # noqa: E402
from .utils import ensure_api_base  # noqa: E402


@dataclass(slots=True)
class PromptLearningBuildResult:
    payload: dict[str, Any]
    task_url: str


def _default_verifier_backend_base(config_dict: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Backfill verifier.backend_base to the job backend when omitted.

    Many local configs want verifier scoring to run against the same backend the job is
    submitted to. If backend_base is empty, Rust defaults may point at production, which
    breaks local runs (dev keys are invalid on prod).
    """
    backend = overrides.get("backend")
    if not backend:
        return

    pl = config_dict.get("prompt_learning")
    if not isinstance(pl, dict):
        return

    verifier = pl.get("verifier")
    if not isinstance(verifier, dict):
        return

    if not verifier.get("enabled", False):
        return

    backend_base = str(verifier.get("backend_base") or "").strip()
    if backend_base:
        return

    # Prefer a base URL (no "/api" suffix) for verifier.backend_base.
    base = str(backend).strip().rstrip("/")
    if base.endswith("/api"):
        base = base[: -len("/api")]
    verifier["backend_base"] = base


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "invalid value")
        lines.append(f"{loc or '<root>'}: {msg}")
    details = "\n".join(f"  - {line}" for line in lines) or "  - Invalid configuration"
    return f"Config validation failed ({path}):\n{details}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_FORBIDDEN_CONTAINER_AUTH_FIELDS = {"container_api_key", "container_api_keys"}


def _collect_forbidden_container_auth_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            next_path = f"{path}.{key_str}" if path else key_str
            lower = key_str.strip().lower()
            if (
                lower in _FORBIDDEN_CONTAINER_AUTH_FIELDS
                or lower.endswith(".container_api_key")
                or lower.endswith(".container_api_keys")
            ):
                paths.append(next_path)
            paths.extend(_collect_forbidden_container_auth_paths(child, next_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            next_path = f"{path}[{index}]" if path else f"[{index}]"
            paths.extend(_collect_forbidden_container_auth_paths(child, next_path))
    return paths


def _assert_no_forbidden_container_auth_overrides(overrides_value: Any) -> None:
    forbidden_paths = _collect_forbidden_container_auth_paths(overrides_value)
    if not forbidden_paths:
        return
    raise ValueError(
        "container_api_key/container_api_keys must never be embedded in policy-optimization "
        "job payload overrides. This auth is server-resolved only. Forbidden override paths: "
        + ", ".join(sorted(forbidden_paths))
    )


def _strip_forbidden_container_auth_fields(config_dict: dict[str, Any]) -> None:
    config_dict.pop("container_api_key", None)
    config_dict.pop("container_api_keys", None)
    for key in list(config_dict.keys()):
        lower = str(key).strip().lower()
        if lower.endswith(".container_api_key") or lower.endswith(".container_api_keys"):
            config_dict.pop(key, None)
    prompt_learning = config_dict.get("prompt_learning")
    if isinstance(prompt_learning, dict):
        prompt_learning.pop("container_api_key", None)
        prompt_learning.pop("container_api_keys", None)
        for key in list(prompt_learning.keys()):
            lower = str(key).strip().lower()
            if lower.endswith(".container_api_key") or lower.endswith(".container_api_keys"):
                prompt_learning.pop(key, None)


def _normalize_mipro_section(
    pl_cfg: PromptLearningConfig,
    config_dict: dict[str, Any],
    *,
    source: str,
    prefer_model: bool,
) -> None:
    _require(
        pl_cfg.mipro is not None,
        f"{source}: MIPRO config missing [prompt_learning.mipro] section.",
    )

    pl_section = config_dict.get("prompt_learning")
    if not isinstance(pl_section, dict):
        pl_section = {}
        config_dict["prompt_learning"] = pl_section

    mipro_section = pl_section.get("mipro", {})
    if hasattr(mipro_section, "model_dump"):
        mipro_section = mipro_section.model_dump(mode="python")
    elif hasattr(mipro_section, "dict"):
        mipro_section = mipro_section.dict()
    if not isinstance(mipro_section, dict):
        mipro_section = {}

    bootstrap_from_model = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
    online_from_model = getattr(pl_cfg.mipro, "online_pool", None)
    test_from_model = getattr(pl_cfg.mipro, "test_pool", None)
    reference_from_model = getattr(pl_cfg.mipro, "reference_pool", None)

    def _maybe_set(key: str, value: Any) -> None:
        if value is None:
            return
        if prefer_model or mipro_section.get(key) is None:
            mipro_section[key] = value

    _maybe_set("bootstrap_train_seeds", bootstrap_from_model)
    _maybe_set("online_pool", online_from_model)
    _maybe_set("test_pool", test_from_model)
    _maybe_set("reference_pool", reference_from_model)

    for key in ("bootstrap_train_seeds", "online_pool", "test_pool", "reference_pool"):
        if mipro_section.get(key) is None and pl_section.get(key) is not None:
            mipro_section[key] = pl_section[key]

    # compatibility removed: older benchmark configs used online_train_seeds.
    if mipro_section.get("online_pool") is None:
        canonical_online_pool = mipro_section.get("online_train_seeds")
        if canonical_online_pool is None:
            canonical_online_pool = pl_section.get("online_train_seeds")
        if canonical_online_pool is not None:
            mipro_section["online_pool"] = canonical_online_pool
            warnings.warn(
                (
                    f"{source}: prompt_learning.mipro.online_train_seeds is deprecated; "
                    "use prompt_learning.mipro.online_pool."
                ),
                UserWarning,
                stacklevel=3,
            )

    mipro_env_name = mipro_section.get("env_name")
    if mipro_env_name and not pl_section.get("env_name") and not pl_section.get("container_id"):
        pl_section["env_name"] = mipro_env_name

    pl_section["mipro"] = mipro_section
    config_dict["prompt_learning"] = pl_section

    _require(
        mipro_section.get("bootstrap_train_seeds") is not None,
        f"{source}: bootstrap_train_seeds missing for MIPRO config.",
    )
    _require(
        mipro_section.get("online_pool") is not None,
        f"{source}: online_pool missing for MIPRO config.",
    )


def _resolve_execution_mode(config_dict: dict[str, Any], pl_cfg: PromptLearningConfig) -> str:
    prompt_learning = config_dict.get("prompt_learning")
    if isinstance(prompt_learning, dict):
        explicit = prompt_learning.get("execution_mode")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()
        if pl_cfg.algorithm == "gepa":
            gepa = prompt_learning.get("gepa")
            if isinstance(gepa, dict):
                gepa_mode = gepa.get("execution_mode") or gepa.get("mode")
                if isinstance(gepa_mode, str) and gepa_mode.strip():
                    return gepa_mode.strip().lower()
        if pl_cfg.algorithm == "mipro":
            mipro = prompt_learning.get("mipro")
            if isinstance(mipro, dict):
                mipro_mode = mipro.get("execution_mode") or mipro.get("mode")
                if isinstance(mipro_mode, str) and mipro_mode.strip():
                    return mipro_mode.strip().lower()
    return "offline"


def _extract_train_seeds_from_task_data(task_data: dict[str, Any]) -> list | None:
    """Extract train seeds from task_data.train_pools (reflection + pareto merge)."""
    pools = task_data.get("train_pools") if isinstance(task_data.get("train_pools"), dict) else {}
    reflection = (
        pools.get("reflection_seeds") if isinstance(pools.get("reflection_seeds"), list) else []
    )
    pareto = pools.get("pareto_seeds") if isinstance(pools.get("pareto_seeds"), list) else []
    merged = list(reflection)
    for seed in pareto:
        if seed not in merged:
            merged.append(seed)
    return merged or None


def _extract_val_seeds_from_task_data(task_data: dict[str, Any]) -> list | None:
    """Extract validation seeds from task_data."""
    validation_pools = task_data.get("validation_pools")
    if isinstance(validation_pools, dict) and isinstance(validation_pools.get("main_seeds"), list):
        return validation_pools.get("main_seeds")
    if isinstance(task_data.get("validation_seeds"), list):
        return task_data.get("validation_seeds")
    return None


def _validate_gepa_container_auth(candidate_task_url: str, is_gepa: bool) -> bool:
    """Validate GEPA container auth and return signer-key availability."""
    signer_configured = has_container_token_signing_key() if is_gepa else False
    if candidate_task_url and is_gepa and not signer_configured:
        if is_synthtunnel_url(candidate_task_url):
            raise ValueError(
                "GEPA SynthTunnel rollout auth requires "
                "SYNTH_CONTAINER_AUTH_PRIVATE_KEY or SYNTH_CONTAINER_AUTH_PRIVATE_KEYS."
            )
        if not is_local_http_container_url(candidate_task_url):
            raise ValueError(
                "GEPA rollout auth for non-local container_url requires "
                "SYNTH_CONTAINER_AUTH_PRIVATE_KEY or SYNTH_CONTAINER_AUTH_PRIVATE_KEYS."
            )
    return signer_configured


def build_prompt_learning_payload(
    *,
    config_path: Path,
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job (MIPRO or GEPA)."""
    from pydantic import ValidationError

    from .configs.prompt_learning import load_toml

    # SDK-SIDE VALIDATION: Catch errors BEFORE sending to backend
    from .validators import validate_prompt_learning_config

    raw_config = load_toml(config_path)
    validate_prompt_learning_config(raw_config, config_path)

    try:
        pl_cfg = PromptLearningConfig.from_path(config_path)
    except ValidationError as exc:
        raise click.ClickException(_format_validation_error(config_path, exc)) from exc

    # Early validation: Check required fields for GEPA
    if pl_cfg.algorithm == "gepa":
        if not pl_cfg.gepa:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa] section is required"
            )
        train_seeds = None
        val_seeds = None
        if pl_cfg.gepa.evaluation:
            train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                pl_cfg.gepa.evaluation, "seeds", None
            )
            val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
                pl_cfg.gepa.evaluation, "validation_seeds", None
            )
        task_data = (raw_config.get("prompt_learning") or {}).get("task_data", {})
        if isinstance(task_data, dict):
            if not train_seeds:
                train_seeds = _extract_train_seeds_from_task_data(task_data)
            if not val_seeds:
                val_seeds = _extract_val_seeds_from_task_data(task_data)
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train seeds: provide prompt_learning.task_data.train_pools.{reflection_seeds,pareto_seeds} or prompt_learning.gepa.evaluation.seeds"
            )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing validation seeds: provide prompt_learning.task_data.validation_seeds or prompt_learning.gepa.evaluation.validation_seeds"
            )

    candidate_task_url = (
        (overrides.get("task_url") or task_url or "").strip()
        or (pl_cfg.container_url or "").strip()
        or (os.environ.get("CONTAINER_URL") or "").strip()
    )
    is_gepa = pl_cfg.algorithm == "gepa"
    _validate_gepa_container_auth(candidate_task_url, is_gepa)

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()
    raw_prompt_learning = raw_config.get("prompt_learning")
    if isinstance(raw_prompt_learning, dict) and isinstance(
        raw_prompt_learning.get("policy"), dict
    ):
        config_dict.setdefault("prompt_learning", {})["policy"] = dict(
            raw_prompt_learning["policy"]
        )
    _default_verifier_backend_base(config_dict, overrides)

    # Canonical path: build payload in Python to avoid requiring legacy policy fields.
    # We intentionally do not route through synth_ai_py.build_prompt_learning_payload().

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("CONTAINER_URL")
    config_task_url = (pl_cfg.container_url or "").strip() or None
    config_container_id = (pl_cfg.container_id or "").strip() or None

    # For prompt learning, prefer config value over env if config is explicitly set
    # This allows TOML files to specify container_url without env var interference
    # But CLI override always wins
    if cli_task_url:
        # CLI override takes precedence
        final_task_url = ConfigResolver.resolve(
            "container_url",
            cli_value=cli_task_url,
            env_value=None,  # Don't check env when CLI is set
            config_value=config_task_url,
            required=True,
        )
    elif config_task_url:
        # Config explicitly set - use it (ignore env var to avoid conflicts)
        final_task_url = config_task_url
    else:
        # No config, fall back to env or error
        final_task_url = ConfigResolver.resolve(
            "container_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=False,
        )
    _require(
        final_task_url is not None or config_container_id is not None,
        "container_url or container_id is required",
    )

    # Ensure container routing is set. For hosted containers, container_id can be used
    # without a direct container_url.
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        if final_task_url:
            pl_section["container_url"] = final_task_url
        if config_container_id and not pl_section.get("container_id"):
            pl_section["container_id"] = config_container_id
        # Spec-compliant behavior: container auth is server-resolved and must not
        # be embedded in job payloads.

        # GEPA canonical seed surface is task_data / gepa.evaluation only.
        if pl_cfg.algorithm == "gepa" and pl_cfg.gepa:
            # Try to get train_seeds directly from the gepa config object first
            train_seeds = None
            if pl_cfg.gepa.evaluation:
                train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                    pl_cfg.gepa.evaluation, "seeds", None
                )

            # If not found, try from serialized dict
            if not train_seeds:
                gepa_section = pl_section.get("gepa", {})
                # Handle case where gepa_section might still be a Pydantic model
                if hasattr(gepa_section, "model_dump"):
                    gepa_section = gepa_section.model_dump(mode="python")
                elif hasattr(gepa_section, "dict"):
                    gepa_section = gepa_section.dict()

                if isinstance(gepa_section, dict):
                    eval_section = gepa_section.get("evaluation", {})
                    # Handle case where eval_section might still be a Pydantic model
                    if hasattr(eval_section, "model_dump"):
                        eval_section = eval_section.model_dump(mode="python")
                    elif hasattr(eval_section, "dict"):
                        eval_section = eval_section.dict()

                    if isinstance(eval_section, dict):
                        train_seeds = eval_section.get("train_seeds") or eval_section.get("seeds")

                    # Update gepa_section back to pl_section in case we converted it
                    pl_section["gepa"] = gepa_section

            # Preferred shape: task_data.train_pools
            if not train_seeds and isinstance(pl_section, dict):
                task_data = pl_section.get("task_data")
                if isinstance(task_data, dict):
                    train_seeds = _extract_train_seeds_from_task_data(task_data)

            # Canonical-only: do not shadow seeds into deprecated top-level aliases.

        if pl_cfg.algorithm == "mipro":
            _normalize_mipro_section(pl_cfg, config_dict, source="pre-merge", prefer_model=True)
    else:
        replacement: dict[str, Any] = {}
        if final_task_url:
            replacement["container_url"] = final_task_url
        if config_container_id:
            replacement["container_id"] = config_container_id
        config_dict["prompt_learning"] = replacement

    # Build payload matching backend API format
    # Extract nested overrides if present, otherwise use flat overrides directly.
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    _assert_no_forbidden_container_auth_overrides(config_overrides)
    # Remove non-override keys (backend, task_url, metadata, auto_start)
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start", "container_api_key")
    }

    forbidden_legacy_override_prefixes = ("policy_optimization.",)
    forbidden_legacy_override_exact = {"policy_optimization"}
    for override_key in list(config_overrides.keys()):
        if override_key in forbidden_legacy_override_exact or override_key.startswith(
            forbidden_legacy_override_prefixes
        ):
            raise click.ClickException(
                f"Legacy override '{override_key}' is no longer supported; use canonical prompt_learning fields only."
            )

    # CRITICAL: Merge overrides into config_dict BEFORE sending to backend
    # This ensures early validation in backend sees merged values
    # Use the same deep_update logic used throughout core utilities.
    if config_overrides:
        from synth_ai.core.utils.dict import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)

    _strip_forbidden_container_auth_fields(config_dict)

    if pl_cfg.algorithm == "mipro":
        _normalize_mipro_section(pl_cfg, config_dict, source="post-merge", prefer_model=False)

    # ASSERT: Verify critical overrides are reflected in config payload
    pl_section_in_dict = config_dict.get("prompt_learning", {})
    if config_overrides:
        # Check rollout budget override
        rollout_budget_key = "prompt_learning.gepa.rollout.budget"
        if rollout_budget_key in config_overrides:
            expected_budget = config_overrides[rollout_budget_key]
            gepa_section = pl_section_in_dict.get("gepa", {})
            actual_budget = (
                gepa_section.get("rollout", {}).get("budget")
                if isinstance(gepa_section, dict)
                else None
            )
            if actual_budget is not None:
                _require(
                    actual_budget == expected_budget,
                    f"Rollout budget mismatch: config_body has {actual_budget} but override specifies {expected_budget}. "
                    "This indicates the override wasn't applied correctly.",
                )

    # FINAL CHECK: Ensure config_body has correct structure for backend
    # Backend expects: {"prompt_learning": {...}} (full TOML structure)
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    payload: dict[str, Any] = {
        "job_kind": "optimization",
        "algorithm_name": pl_cfg.algorithm,
        "execution_mode": _resolve_execution_mode(config_dict, pl_cfg),
        "config_schema_version": (
            (
                config_dict.get("prompt_learning", {}).get("config_schema_version")
                if isinstance(config_dict.get("prompt_learning"), dict)
                else None
            )
            or "v2"
        ),
        "config_body": config_dict,
        "overrides": config_overrides,
        "metadata": overrides.get("metadata", {}),
        "auto_start": overrides.get("auto_start", True),
    }

    backend = overrides.get("backend")
    if backend:
        metadata_default: dict[str, Any] = {}
        metadata = cast(dict[str, Any], payload.setdefault("metadata", metadata_default))
        metadata["backend_base_url"] = ensure_api_base(str(backend))

    return PromptLearningBuildResult(payload=payload, task_url=final_task_url)


def build_prompt_learning_payload_from_mapping(
    *,
    raw_config: dict[str, Any],
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
    source_label: str = "programmatic",
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job from a dictionary (programmatic use).

    This is the same as build_prompt_learning_payload but accepts a dict instead of a file path.
    Both functions route through the same PromptLearningConfig Pydantic validation.

    Args:
        raw_config: Configuration dictionary with the same structure as the TOML file.
                   Should have a 'prompt_learning' section.
        task_url: Override for container_url
        overrides: Config overrides (merged into config)
        allow_experimental: Allow experimental models
        source_label: Label for logging/error messages (default: "programmatic")

    Returns:
        PromptLearningBuildResult with payload and task_url

    Example:
        >>> result = build_prompt_learning_payload_from_mapping(
        ...     raw_config={
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
        ...     task_url=None,
        ...     overrides={},
        ... )
    """
    from pydantic import ValidationError

    # SDK-SIDE VALIDATION: Catch errors BEFORE sending to backend
    from .validators import validate_prompt_learning_config

    # Use a pseudo-path for error messages (validator expects Path object)
    pseudo_path = Path(f"<{source_label}>")
    validate_prompt_learning_config(raw_config, pseudo_path)

    try:
        pl_cfg = PromptLearningConfig.from_mapping(raw_config)
    except ValidationError as exc:
        # Format validation errors for dict-based config
        lines: list[str] = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ()))
            msg = error.get("msg", "invalid value")
            lines.append(f"{loc or '<root>'}: {msg}")
        details = "\n".join(f"  - {line}" for line in lines) or "  - Invalid configuration"
        raise click.ClickException(
            f"Config validation failed ({source_label}):\n{details}"
        ) from exc

    # Early validation: Check required fields for GEPA
    if pl_cfg.algorithm == "gepa":
        if not pl_cfg.gepa:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa] section is required"
            )
        train_seeds = None
        val_seeds = None
        if pl_cfg.gepa.evaluation:
            train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                pl_cfg.gepa.evaluation, "seeds", None
            )
            val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
                pl_cfg.gepa.evaluation, "validation_seeds", None
            )
        task_data = (raw_config.get("prompt_learning") or {}).get("task_data", {})
        if isinstance(task_data, dict):
            if not train_seeds:
                train_seeds = _extract_train_seeds_from_task_data(task_data)
            if not val_seeds:
                val_seeds = _extract_val_seeds_from_task_data(task_data)
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train seeds: provide prompt_learning.task_data.train_pools.{reflection_seeds,pareto_seeds} or prompt_learning.gepa.evaluation.seeds"
            )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing validation seeds: provide prompt_learning.task_data.validation_pools.main_seeds (or validation_seeds) or prompt_learning.gepa.evaluation.validation_seeds"
            )

    candidate_task_url = (
        (overrides.get("task_url") or task_url or "").strip()
        or (pl_cfg.container_url or "").strip()
        or (os.environ.get("CONTAINER_URL") or "").strip()
    )
    is_gepa = pl_cfg.algorithm == "gepa"
    _validate_gepa_container_auth(candidate_task_url, is_gepa)

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()
    raw_prompt_learning = raw_config.get("prompt_learning")
    if isinstance(raw_prompt_learning, dict) and isinstance(
        raw_prompt_learning.get("policy"), dict
    ):
        config_dict.setdefault("prompt_learning", {})["policy"] = dict(
            raw_prompt_learning["policy"]
        )
    _default_verifier_backend_base(config_dict, overrides)

    # Canonical path: build payload in Python to avoid requiring legacy policy fields.
    # We intentionally do not route through synth_ai_py.build_prompt_learning_payload().

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("CONTAINER_URL")
    config_task_url = (pl_cfg.container_url or "").strip() or None
    config_container_id = (pl_cfg.container_id or "").strip() or None

    # Resolve container_url with same precedence as file-based builder
    if cli_task_url:
        final_task_url = ConfigResolver.resolve(
            "container_url",
            cli_value=cli_task_url,
            env_value=None,
            config_value=config_task_url,
            required=True,
        )
    elif config_task_url:
        final_task_url = config_task_url
    else:
        final_task_url = ConfigResolver.resolve(
            "container_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=False,
        )
    _require(
        final_task_url is not None or config_container_id is not None,
        "container_url or container_id is required",
    )

    # Ensure container routing is set. For hosted containers, container_id can be used
    # without a direct container_url.
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        if final_task_url:
            pl_section["container_url"] = final_task_url
        if config_container_id and not pl_section.get("container_id"):
            pl_section["container_id"] = config_container_id
        # Spec-compliant behavior: container auth is server-resolved and must not
        # be embedded in job payloads.

        # GEPA: Extract train_seeds from nested structure
        if pl_cfg.algorithm == "gepa" and pl_cfg.gepa:
            train_seeds = None
            if pl_cfg.gepa.evaluation:
                train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                    pl_cfg.gepa.evaluation, "seeds", None
                )
            if not train_seeds:
                task_data = pl_section.get("task_data")
                if isinstance(task_data, dict):
                    train_seeds = _extract_train_seeds_from_task_data(task_data)

            if train_seeds and not pl_section.get("train_seeds"):
                pl_section["train_seeds"] = train_seeds
            if train_seeds and not pl_section.get("evaluation_seeds"):
                pl_section["evaluation_seeds"] = train_seeds

        if pl_cfg.algorithm == "mipro":
            _normalize_mipro_section(pl_cfg, config_dict, source="pre-merge", prefer_model=True)
    else:
        replacement: dict[str, Any] = {}
        if final_task_url:
            replacement["container_url"] = final_task_url
        if config_container_id:
            replacement["container_id"] = config_container_id
        config_dict["prompt_learning"] = replacement

    # Build payload matching backend API format
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    _assert_no_forbidden_container_auth_overrides(config_overrides)
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start", "container_api_key")
    }

    forbidden_legacy_override_prefixes = ("policy_optimization.",)
    forbidden_legacy_override_exact = {"policy_optimization"}
    for override_key in list(config_overrides.keys()):
        if override_key in forbidden_legacy_override_exact or override_key.startswith(
            forbidden_legacy_override_prefixes
        ):
            raise click.ClickException(
                f"Legacy override '{override_key}' is no longer supported; use canonical prompt_learning fields only."
            )

    # Merge overrides into config_dict
    if config_overrides:
        from synth_ai.core.utils.dict import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)

    _strip_forbidden_container_auth_fields(config_dict)

    if pl_cfg.algorithm == "mipro":
        _normalize_mipro_section(pl_cfg, config_dict, source="post-merge", prefer_model=False)

    # Final validation
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    payload: dict[str, Any] = {
        "job_kind": "optimization",
        "algorithm_name": pl_cfg.algorithm,
        "execution_mode": _resolve_execution_mode(config_dict, pl_cfg),
        "config_schema_version": (
            (
                config_dict.get("prompt_learning", {}).get("config_schema_version")
                if isinstance(config_dict.get("prompt_learning"), dict)
                else None
            )
            or "v2"
        ),
        "config_body": config_dict,
        "overrides": config_overrides,
        "metadata": overrides.get("metadata", {}),
        "auto_start": overrides.get("auto_start", True),
    }

    backend = overrides.get("backend")
    if backend:
        metadata_default: dict[str, Any] = {}
        metadata = cast(dict[str, Any], payload.setdefault("metadata", metadata_default))
        metadata["backend_base_url"] = ensure_api_base(str(backend))

    return PromptLearningBuildResult(payload=payload, task_url=final_task_url)


__all__ = [
    "PromptLearningBuildResult",
    "_assert_no_forbidden_container_auth_overrides",
    "_strip_forbidden_container_auth_fields",
    "build_prompt_learning_payload",
    "build_prompt_learning_payload_from_mapping",
]
