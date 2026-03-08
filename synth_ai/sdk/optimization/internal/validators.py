"""Rust-backed validation for optimization configs."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

try:
    import tomllib as _toml  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]

from synth_ai.sdk.optimization.internal.validation.prompt_learning_validation import (
    validate_prompt_learning_config as _validate_unknown_fields,
)
from synth_ai.sdk.shared.models import normalize_model_identifier

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.validators.") from exc


class ConfigValidationError(Exception):
    """Raised when a training config is invalid."""

    pass


_GEPA_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'proposer_backend' in [prompt_learning.gepa].",
    "Unknown field 'proposer' in [prompt_learning.gepa].",
    "Unknown field 'context_override' in [prompt_learning.gepa].",
    "Unknown field 'proposer_mode' in [prompt_learning.gepa].",
    "Unknown field 'proposal_pipeline' in [prompt_learning.gepa].",
    "Unknown field 'actionable_upfront_context' in [prompt_learning.gepa].",
    "Unknown field 'task_context' in [prompt_learning.gepa].",
    "Unknown field 'termination_conditions' in [prompt_learning.gepa].",
    "Unknown field 'initial_candidate' in [prompt_learning.gepa].",
    "Unknown field 'policy_config' in [prompt_learning.gepa].",
)

_MIPRO_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'mode' in [prompt_learning.mipro].",
    "Unknown field 'val_seeds' in [prompt_learning.mipro].",
    "Unknown field 'proposer' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_mode' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_min_rollouts' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_min_rewards' in [prompt_learning.mipro].",
    "Unknown field 'online_proposer_max_candidates' in [prompt_learning.mipro].",
)

_PROMPT_LEARNING_ALIAS_WARNING_PREFIXES: tuple[str, ...] = (
    "Unknown field 'container_url' in [prompt_learning].",
    "Unknown field 'container_id' in [prompt_learning].",
    "Unknown field 'policy' in [prompt_learning].",
    # Canonical optimize-anything fields.
    "Unknown field 'optimization_mode' in [prompt_learning].",
    "Unknown field 'target_mode' in [prompt_learning].",
    "Unknown field 'target_kind' in [prompt_learning].",
    "Unknown field 'artifact' in [prompt_learning].",
    "Unknown field 'artifact_kind' in [prompt_learning].",
    "Unknown field 'default_artifact_kind' in [prompt_learning].",
    "Unknown field 'artifact_schema' in [prompt_learning].",
    "Unknown field 'artifact_bounds' in [prompt_learning].",
    # New GEPA task-data ownership surface.
    "Unknown field 'task_data' in [prompt_learning].",
    # Kind/mode disambiguation canonical surface.
    "Unknown field 'job_kind' in [prompt_learning].",
    "Unknown field 'algorithm_name' in [prompt_learning].",
    "Unknown field 'execution_mode' in [prompt_learning].",
    "Unknown field 'config_schema_version' in [prompt_learning].",
)

_GEPA_ROLLOUT_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'max_concurrent_rollouts' in [prompt_learning.gepa.rollout].",
)

_GEPA_THROUGHPUT_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'throughput' in [prompt_learning.gepa].",
    "Unknown field 'max_concurrent_rollouts' in [prompt_learning.gepa.throughput].",
)

_VERIFIER_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'model' in [prompt_learning.verifier].",
    "Unknown field 'reward_on_trace' in [prompt_learning.verifier].",
    "Unknown field 'source' in [prompt_learning.verifier].",
)

_GEPA_SPEC_ALIAS_WARNING_PREFIXES = (
    "Unknown field 'spec_path' in [prompt_learning.gepa].",
    "Unknown field 'spec_max_tokens' in [prompt_learning.gepa].",
    "Unknown field 'spec_include_examples' in [prompt_learning.gepa].",
    "Unknown field 'spec_priority_threshold' in [prompt_learning.gepa].",
)


def _collect_forbidden_policy_paths(payload: Any, path: str = "") -> list[str]:
    """Return non-canonical policy locations while allowing prompt_learning.policy."""
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = f"{path}.{key}" if path else key
            if key == "policy" and not next_path.startswith("prompt_learning."):
                paths.append(next_path)
            paths.extend(_collect_forbidden_policy_paths(value, next_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            next_path = f"{path}[{index}]"
            paths.extend(_collect_forbidden_policy_paths(value, next_path))
    return paths


def _raise_if_forbidden_policy_fields(config_data: dict[str, Any], config_path: Path) -> None:
    """Reject non-canonical policy sections while keeping prompt_learning.policy compatible."""
    policy_paths = _collect_forbidden_policy_paths(config_data)
    if policy_paths:
        joined_paths = ", ".join(policy_paths)
        raise click.ClickException(
            f"{config_path}: user-submitted policy config is forbidden. "
            f"Remove these fields: {joined_paths}"
        )


def reject_legacy_policy_optimization(
    config_data: dict | Mapping, config_path: Path | str | None = None
) -> None:
    """Reject legacy policy_optimization top-level section."""
    if not isinstance(config_data, Mapping):
        return
    if "policy_optimization" not in config_data:
        return
    prefix = f"{config_path}: " if config_path else ""
    raise ValueError(
        f"{prefix}top-level policy_optimization is no longer supported; use prompt_learning."
    )


def _raise_if_legacy_sections(config_data: dict[str, Any], config_path: Path) -> None:
    """Reject legacy top-level config sections that are no longer supported."""
    reject_legacy_policy_optimization(config_data, config_path)


def _iter_model_values(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "model" and isinstance(value, str):
                yield value.strip()
            yield from _iter_model_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_model_values(value)


def _normalize_supported_model_errors(config_data: dict[str, Any], errors: list[str]) -> list[str]:
    """Drop stale strict-validation errors for models supported by core/backend.

    Some local environments may load an older prebuilt ``synth_ai_py`` extension while
    using a newer Python SDK checkout. Allowing these known-safe aliases keeps local
    benchmark/eval scripts unblocked without weakening other validation paths.
    """
    requested_models = {model.lower() for model in _iter_model_values(config_data)}
    if "gpt-5.3-codex" not in requested_models:
        return errors

    normalized: list[str] = []
    for err in errors:
        lowered = err.lower()
        if "unsupported openai model" in lowered and "gpt-5.3-codex" in lowered:
            continue
        normalized.append(err)
    return normalized


def _normalize_container_fields(config_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize canonical container fields before strict validation (mutates in-place)."""
    pl = config_data.get("prompt_learning")
    if not isinstance(pl, dict):
        return config_data

    def _first_str(*vals: Any) -> str | None:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    url = _first_str(pl.get("container_url"))
    if url:
        pl.setdefault("container_url", url)

    cid = _first_str(pl.get("container_id"))
    if cid:
        pl.setdefault("container_id", cid)
    return config_data


def _normalize_kind_mode_aliases(config_data: dict[str, Any]) -> None:
    """Normalize canonical v2 algorithm/mode aliases (mutates in-place)."""
    pl = config_data.get("prompt_learning")
    if not isinstance(pl, dict):
        return

    algorithm_name = pl.get("algorithm_name")
    if pl.get("algorithm") is None and isinstance(algorithm_name, str) and algorithm_name.strip():
        pl["algorithm"] = algorithm_name.strip().lower()

    def _norm_target_mode(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_")
        if normalized == "prompt":
            return "prompt"
        if normalized in {"anything", "optimize_anything", "context"}:
            return "anything"
        return None

    def _norm_target_kind(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"generic", "context"}:
            return normalized
        if normalized in {"skills", "skill", "skill_bundle", "trajectory_index"}:
            return "skills"
        return None

    def _norm_optimization_mode(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"prompt", "context", "optimize_anything"}:
            return normalized
        return None

    target_mode = _norm_target_mode(pl.get("target_mode"))
    target_kind = _norm_target_kind(pl.get("target_kind"))
    optimization_mode = _norm_optimization_mode(pl.get("optimization_mode"))

    if target_mode is None and optimization_mode is not None:
        if optimization_mode == "prompt":
            target_mode = "prompt"
        else:
            target_mode = "anything"
            if target_kind is None:
                target_kind = "context" if optimization_mode == "context" else "generic"

    if target_mode == "prompt":
        target_kind = None
        optimization_mode = "prompt"
    elif target_mode == "anything":
        if target_kind is None:
            target_kind = "generic"
        optimization_mode = "context" if target_kind == "context" else "optimize_anything"

    if target_mode is not None:
        pl["target_mode"] = target_mode
    if target_kind is None:
        pl["target_kind"] = None
    else:
        pl["target_kind"] = target_kind
    if optimization_mode is not None:
        pl["optimization_mode"] = optimization_mode


def _filter_known_gepa_alias_warnings(warnings_list: list[str]) -> list[str]:
    """Hide unknown-field warnings for accepted canonical aliases."""
    filtered: list[str] = []
    for warning in warnings_list:
        if any(warning.startswith(prefix) for prefix in _PROMPT_LEARNING_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _GEPA_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _MIPRO_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _GEPA_ROLLOUT_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _GEPA_THROUGHPUT_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _VERIFIER_ALIAS_WARNING_PREFIXES):
            continue
        if any(warning.startswith(prefix) for prefix in _GEPA_SPEC_ALIAS_WARNING_PREFIXES):
            continue
        filtered.append(warning)
    return filtered


def _raise_validation_errors(errors: list[str], config_path: Path) -> None:
    if not errors:
        return
    msg = "\n".join(errors)
    raise click.ClickException(f"{config_path}: {msg}")


def _has_nonempty_string_field(config_data: dict[str, Any], field: str) -> bool:
    pl = config_data.get("prompt_learning")
    if not isinstance(pl, dict):
        pl = config_data
    value = pl.get(field) if isinstance(pl, dict) else None
    return isinstance(value, str) and bool(value.strip())


def _has_container_id(config_data: dict[str, Any]) -> bool:
    return _has_nonempty_string_field(config_data, "container_id")


def _has_container_url(config_data: dict[str, Any]) -> bool:
    return _has_nonempty_string_field(config_data, "container_url")


def _normalize_strict_errors(config_data: dict[str, Any], errors: list[str]) -> list[str]:
    """Filter strict validation errors that conflict with canonical container fields."""
    has_container_id = _has_container_id(config_data)
    has_container_url = _has_container_url(config_data)
    normalized: list[str] = []
    for err in errors:
        if (
            "prompt_learning.policy is forbidden in canonical config; model/provider are server-owned"
            in err
        ):
            continue
        if (
            "Missing required field: prompt_learning.gepa.initial_candidate" in err
            and isinstance(config_data.get("prompt_learning"), dict)
            and config_data["prompt_learning"].get("algorithm") == "gepa"
        ):
            continue
        if (has_container_id or has_container_url) and (
            "Missing required field: prompt_learning.container_url" in err
            # Older prebuilt synth_ai_py may still emit legacy field names.
            # Canonical configs already provide container_url/container_id.
            or "Missing required field: prompt_learning.task_app_url" in err
        ):
            continue
        normalized.append(err)
    return normalized


def _normalize_canonical_mipro_aliases(config_data: dict[str, Any]) -> None:
    """Normalize canonical MIPRO field aliases before strict validation (mutates in-place)."""
    sections: list[dict[str, Any]] = []
    if isinstance(config_data.get("prompt_learning"), dict):
        sections.append(config_data["prompt_learning"])

    for section in sections:
        mipro = section.get("mipro")
        if isinstance(mipro, dict):
            if mipro.get("online_pool") is None and mipro.get("online_train_seeds") is not None:
                mipro["online_pool"] = mipro.get("online_train_seeds")
            if section.get("online_pool") is None and mipro.get("online_pool") is not None:
                section["online_pool"] = mipro.get("online_pool")
        if section.get("online_pool") is None and section.get("online_train_seeds") is not None:
            section["online_pool"] = section.get("online_train_seeds")


def _normalize_gepa_aliases(config_data: dict[str, Any]) -> None:
    """Normalize GEPA proposer alias fields to canonical synth proposer settings (mutates in-place).

    Canonical/new benchmark configs may send:
    - prompt_learning.gepa.proposer_backend = "rlm" | "agent"
    - prompt_learning.gepa.context_override = {...}

    proposer_backend='prompt' is removed and rejected.
    proposer_backend='rlm' and 'agent' pass through; rust_backend GEPA adapters
    dispatch execution.
    """
    sections: list[dict[str, Any]] = []
    if isinstance(config_data.get("prompt_learning"), dict):
        sections.append(config_data["prompt_learning"])

    for section in sections:
        gepa = section.get("gepa")
        if not isinstance(gepa, dict):
            continue

        top_level_legacy_rollout_keys = (
            "seed_checkpoint",
            "seed_checkpoints",
            "seed_checkpoint_refs",
        )
        if any(section.get(key) is not None for key in top_level_legacy_rollout_keys):
            raise click.ClickException(
                "INVALID_CONFIG_NAMESPACE: prompt_learning.seed_checkpoint* keys are no longer "
                "supported for GEPA. Use prompt_learning.gepa.rollout_checkpoint*."
            )

        # Hard cutover: legacy checkpoint namespace is no longer supported.
        legacy_checkpoint_keys = (
            "checkpoint",
            "seed_checkpoint",
            "seed_checkpoints",
            "seed_checkpoint_refs",
        )
        detected_legacy = [key for key in legacy_checkpoint_keys if gepa.get(key) is not None]
        if detected_legacy:
            raise click.ClickException(
                "INVALID_CONFIG_NAMESPACE: prompt_learning.gepa.checkpoint and "
                "seed_checkpoint* keys are no longer supported. "
                "Use prompt_learning.gepa.job_checkpoint and prompt_learning.gepa.rollout_checkpoint*."
            )

        # Canonical UX alias: map [prompt_learning.gepa.throughput]
        # into strict GEPA rollout shape before Rust strict validation.
        throughput_cfg = gepa.get("throughput")
        if isinstance(throughput_cfg, dict):
            max_concurrent_rollouts = throughput_cfg.get("max_concurrent_rollouts")
            if isinstance(max_concurrent_rollouts, int):
                rollout_cfg = gepa.get("rollout")
                if not isinstance(rollout_cfg, dict):
                    rollout_cfg = {}
                    gepa["rollout"] = rollout_cfg
                rollout_cfg.setdefault("max_concurrent", max_concurrent_rollouts)
            # Strip alias section so strict validators don't reject unknown fields.
            gepa.pop("throughput", None)

        # Accept new top-level context_override alias used by benchmark runners.
        if (
            gepa.get("context_override") is not None
            and gepa.get("baseline_context_override") is None
        ):
            gepa["baseline_context_override"] = gepa.get("context_override")

        proposer_backend_raw = gepa.get("proposer_backend")
        if proposer_backend_raw is None:
            continue

        proposer_backend = str(proposer_backend_raw).strip().lower()
        if proposer_backend == "prompt":
            raise click.ClickException(
                "prompt_learning.gepa.proposer_backend='prompt' is removed. "
                "Use proposer_backend='rlm' or proposer_backend='agent'."
            )

        # proposer_backend='rlm' and 'agent' are supported by rust_backend gepa_adapter.
        # Pass through; no normalization needed.
        if proposer_backend in {"rlm", "agent"}:
            continue

        raise click.ClickException(
            f"Invalid prompt_learning.gepa.proposer_backend='{proposer_backend_raw}'. "
            "Expected one of: rlm, agent."
        )


_SUPPORTED_POLICY_MODELS: dict[str, set[str]] = {
    "openai": {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.3-codex",
    },
    "groq": {
        "gpt-oss-20b",
        "gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "qwen3-32b",
    },
    "google": {
        "gemini-2.5-pro",
        "gemini-2.5-pro-gt200k",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    },
}


def _normalize_provider(provider: Any) -> str:
    return str(provider or "").strip().lower()


def _normalize_model_name(model: Any, provider: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    if "/" in raw:
        _prefix, rest = raw.split("/", 1)
        if rest.strip():
            raw = rest.strip()
    return raw.lower()


def _validate_model_choice(*, provider: Any, model: Any, label: str, allow_nano: bool) -> None:
    provider_name = _normalize_provider(provider)
    if not provider_name:
        raise click.ClickException(f"Missing required field: {label}.provider")
    if provider_name not in _SUPPORTED_POLICY_MODELS:
        raise click.ClickException(f"Unsupported provider: {provider_name}")

    model_name = _normalize_model_name(model, provider_name)
    if not model_name:
        raise click.ClickException(f"Missing required field: {label}.model")
    if provider_name == "openai" and model_name == "gpt-5-pro":
        raise click.ClickException("gpt-5-pro is too expensive for prompt-learning jobs")

    try:
        normalize_model_identifier(model_name)
    except Exception:
        pass

    if model_name not in _SUPPORTED_POLICY_MODELS[provider_name]:
        provider_label = {
            "openai": "OpenAI",
            "groq": "Groq",
            "google": "Google",
        }[provider_name]
        raise click.ClickException(f"Unsupported {provider_label} model: {model}")
    if not allow_nano and model_name.endswith("-nano"):
        raise click.ClickException("Nano models are NOT allowed for proposal/mutation models")


def _validate_prompt_learning_models(config_data: dict[str, Any]) -> None:
    pl = config_data.get("prompt_learning")
    if not isinstance(pl, dict):
        return

    policy = pl.get("policy")
    if isinstance(policy, dict):
        _validate_model_choice(
            provider=policy.get("provider"),
            model=policy.get("model"),
            label="prompt_learning.policy",
            allow_nano=True,
        )

    gepa = pl.get("gepa")
    if isinstance(gepa, dict):
        mutation = gepa.get("mutation")
        if isinstance(mutation, dict) and (
            mutation.get("llm_model") is not None or mutation.get("llm_provider") is not None
        ):
            _validate_model_choice(
                provider=mutation.get("llm_provider"),
                model=mutation.get("llm_model"),
                label="prompt_learning.gepa.mutation",
                allow_nano=False,
            )

    mipro = pl.get("mipro")
    if isinstance(mipro, dict) and (
        mipro.get("meta_model") is not None or mipro.get("meta_model_provider") is not None
    ):
        _validate_model_choice(
            provider=mipro.get("meta_model_provider"),
            model=mipro.get("meta_model"),
            label="prompt_learning.mipro",
            allow_nano=False,
        )


def validate_prompt_learning_config(config_data: dict[str, Any], config_path: Path) -> None:
    """Validate prompt learning config using Rust core."""
    _raise_if_legacy_sections(config_data, config_path)
    _raise_if_forbidden_policy_fields(config_data, config_path)
    normalized_for_validation = deepcopy(config_data)
    _normalize_canonical_mipro_aliases(normalized_for_validation)
    _normalize_gepa_aliases(normalized_for_validation)
    _normalize_kind_mode_aliases(normalized_for_validation)
    _normalize_container_fields(normalized_for_validation)
    _validate_prompt_learning_models(normalized_for_validation)
    try:
        validation_result = _validate_unknown_fields(
            normalized_for_validation,
            config_path=config_path,
        )
        for warning_msg in _filter_known_gepa_alias_warnings(list(validation_result.warnings)):
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
        for info_msg in validation_result.info:
            warnings.warn(f"Info: {info_msg}", UserWarning, stacklevel=3)
    except Exception as e:
        logger.debug("Unknown field validation skipped: %s", e)

    normalized_for_strict = normalized_for_validation
    errors = synth_ai_py.validate_prompt_learning_config_strict(normalized_for_strict)
    errors = _normalize_strict_errors(normalized_for_strict, list(errors))
    errors = _normalize_supported_model_errors(normalized_for_strict, list(errors))
    if errors:
        _raise_validation_errors(list(errors), config_path)


def validate_prompt_learning_config_from_file(config_path: Path, algorithm: str) -> None:
    """Load a config file and validate prompt learning settings."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")

    with config_path.open("rb") as fh:
        config = _toml.load(fh)
    if algorithm not in ("gepa", "mipro"):
        raise click.ClickException(f"Unknown optimization algorithm: {algorithm}")

    validate_prompt_learning_config(config, config_path)


__all__ = [
    "reject_legacy_policy_optimization",
    "validate_prompt_learning_config",
    "validate_prompt_learning_config_from_file",
    "ConfigValidationError",
]
