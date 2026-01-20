import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import click
from pydantic import ValidationError

try:
    _models_module = cast(Any, importlib.import_module("synth_ai.sdk.api.models.supported"))
    UnsupportedModelError = cast(type[Exception], _models_module.UnsupportedModelError)
    ensure_allowed_model = cast(Callable[..., None], _models_module.ensure_allowed_model)
    normalize_model_identifier = cast(
        Callable[[str], str], _models_module.normalize_model_identifier
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load supported model helpers") from exc

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

from synth_ai.core.config.resolver import ConfigResolver
from synth_ai.core.telemetry import log_info
from synth_ai.sdk.localapi.auth import ensure_localapi_auth

from .configs import PromptLearningConfig
from .utils import ensure_api_base


@dataclass(slots=True)
class PromptLearningBuildResult:
    payload: dict[str, Any]
    task_url: str


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "invalid value")
        lines.append(f"{loc or '<root>'}: {msg}")
    details = "\n".join(f"  - {line}" for line in lines) or "  - Invalid configuration"
    return f"Config validation failed ({path}):\n{details}"


def build_prompt_learning_payload(
    *,
    config_path: Path,
    task_url: str | None,
    overrides: dict[str, Any],
    allow_experimental: bool | None = None,
) -> PromptLearningBuildResult:
    """Build payload for prompt learning job (MIPRO or GEPA)."""
    ctx: dict[str, Any] = {"config_path": str(config_path), "task_url": task_url}
    log_info("build_prompt_learning_payload invoked", ctx=ctx)
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
        if not pl_cfg.gepa.evaluation:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
            )
        train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "seeds", None
        )
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train_seeds: [prompt_learning.gepa.evaluation] must have 'train_seeds' or 'seeds' field"
            )
        val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "validation_seeds", None
        )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing val_seeds: [prompt_learning.gepa.evaluation] must have 'val_seeds' or 'validation_seeds' field"
            )

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("TASK_APP_URL")
    config_task_url = (pl_cfg.task_app_url or "").strip() or None

    # For prompt learning, prefer config value over env if config is explicitly set
    # This allows TOML files to specify task_app_url without env var interference
    # But CLI override always wins
    if cli_task_url:
        # CLI override takes precedence
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
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
            "task_app_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=True,
        )
    assert final_task_url is not None  # required=True guarantees non-None

    # Get task_app_api_key from config or environment
    # Note: task_app_api_key is not a field on PromptLearningConfig, use getattr
    config_api_key = (getattr(pl_cfg, "task_app_api_key", None) or "").strip() or None
    cli_api_key = overrides.get("task_app_api_key")
    env_api_key = ensure_localapi_auth()
    task_app_api_key = ConfigResolver.resolve(
        "task_app_api_key",
        cli_value=cli_api_key,
        env_value=env_api_key,
        config_value=config_api_key,
        required=True,
    )

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    # ASSERT: MIPRO fields exist in Pydantic model
    if pl_cfg.algorithm == "mipro":
        assert pl_cfg.mipro is not None, "pl_cfg.mipro is None for MIPRO algorithm"
        bootstrap_in_model = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
        online_in_model = getattr(pl_cfg.mipro, "online_pool", None)
        assert bootstrap_in_model is not None, (
            f"pl_cfg.mipro.bootstrap_train_seeds is None! pl_cfg.mipro keys: {dir(pl_cfg.mipro) if pl_cfg.mipro else 'N/A'}"
        )
        assert online_in_model is not None, (
            f"pl_cfg.mipro.online_pool is None! pl_cfg.mipro keys: {dir(pl_cfg.mipro) if pl_cfg.mipro else 'N/A'}"
        )

    # Ensure task_app_url is set (task_app_api_key is resolved by backend from ENVIRONMENT_API_KEY)
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url

        # GEPA: Extract train_seeds from nested structure for backwards compatibility
        # Backend checks for train_seeds at top level before parsing nested structure
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

            # Add train_seeds to top level for backwards compatibility
            if train_seeds and not pl_section.get("train_seeds"):
                pl_section["train_seeds"] = train_seeds
            if train_seeds and not pl_section.get("evaluation_seeds"):
                pl_section["evaluation_seeds"] = train_seeds

        # MIPRO: CRITICAL - Ensure bootstrap_train_seeds and online_pool are ALWAYS in mipro section
        # Handle Pydantic model serialization - mipro might be a model object, not a dict
        if pl_cfg.algorithm == "mipro":
            mipro_section = pl_section.get("mipro", {})

            # ASSERT: Check what we got from to_dict()
            assert pl_cfg.mipro is not None, "pl_cfg.mipro is None"
            bootstrap_before_convert = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
            online_before_convert = getattr(pl_cfg.mipro, "online_pool", None)
            assert bootstrap_before_convert is not None, (
                f"bootstrap_train_seeds is None in model! mipro_section type: {type(mipro_section)}, keys: {list(mipro_section.keys()) if isinstance(mipro_section, dict) else 'N/A'}"
            )
            assert online_before_convert is not None, (
                f"online_pool is None in model! mipro_section type: {type(mipro_section)}, keys: {list(mipro_section.keys()) if isinstance(mipro_section, dict) else 'N/A'}"
            )

            # Convert Pydantic model to dict if needed
            if hasattr(mipro_section, "model_dump"):
                mipro_section = mipro_section.model_dump(mode="python")
            elif hasattr(mipro_section, "dict"):
                mipro_section = mipro_section.dict()

            if not isinstance(mipro_section, dict):
                mipro_section = {}

            # ASSERT: After conversion, check if fields are present
            bootstrap_after_convert = mipro_section.get("bootstrap_train_seeds")
            online_after_convert = mipro_section.get("online_pool")
            if bootstrap_after_convert is None:
                raise AssertionError(
                    f"bootstrap_train_seeds missing after conversion! mipro_section keys: {list(mipro_section.keys())}, bootstrap_before_convert: {bootstrap_before_convert}"
                )
            if online_after_convert is None:
                raise AssertionError(
                    f"online_pool missing after conversion! mipro_section keys: {list(mipro_section.keys())}, online_before_convert: {online_before_convert}"
                )

            # CRITICAL: Get fields from Pydantic model FIRST (most reliable)
            # These fields MUST be present - get them from the source of truth
            if not pl_cfg.mipro:
                raise ValueError(
                    "MIPRO config missing: pl_cfg.mipro is None. "
                    "Ensure [prompt_learning.mipro] section exists in TOML."
                )

            bootstrap_from_model = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
            online_from_model = getattr(pl_cfg.mipro, "online_pool", None)
            test_from_model = getattr(pl_cfg.mipro, "test_pool", None)
            reference_from_model = getattr(pl_cfg.mipro, "reference_pool", None)

            # FORCE these fields into mipro_section (model is source of truth)
            # Use model values if present, otherwise keep existing dict values
            assert bootstrap_from_model is not None, (
                f"bootstrap_from_model is None! pl_cfg.mipro: {pl_cfg.mipro}"
            )
            assert online_from_model is not None, (
                f"online_from_model is None! pl_cfg.mipro: {pl_cfg.mipro}"
            )

            mipro_section["bootstrap_train_seeds"] = bootstrap_from_model
            mipro_section["online_pool"] = online_from_model

            if test_from_model is not None:
                mipro_section["test_pool"] = test_from_model
            elif not mipro_section.get("test_pool") and pl_section.get("test_pool"):
                mipro_section["test_pool"] = pl_section["test_pool"]

            if reference_from_model is not None:
                mipro_section["reference_pool"] = reference_from_model
            elif not mipro_section.get("reference_pool") and pl_section.get("reference_pool"):
                mipro_section["reference_pool"] = pl_section["reference_pool"]

            # ASSERT: Fields are now in mipro_section
            assert mipro_section.get("bootstrap_train_seeds") is not None, (
                f"bootstrap_train_seeds STILL missing after forcing! mipro_section keys: {list(mipro_section.keys())}"
            )
            assert mipro_section.get("online_pool") is not None, (
                f"online_pool STILL missing after forcing! mipro_section keys: {list(mipro_section.keys())}"
            )

            # CRITICAL: Validate fields are present BEFORE override merge
            # If they're missing here, we'll check overrides after they're extracted
            # For now, just ensure they're in mipro_section from TOML
            if not mipro_section.get("bootstrap_train_seeds") and bootstrap_from_model is None:
                raise ValueError(
                    f"MIPRO config missing bootstrap_train_seeds in TOML. "
                    f"pl_cfg.mipro.bootstrap_train_seeds={bootstrap_from_model}, "
                    f"mipro_section keys={list(mipro_section.keys())}, "
                    f"pl_section keys={list(pl_section.keys())[:10]}. "
                    f"Ensure [prompt_learning.mipro] has bootstrap_train_seeds or provide override."
                )
            if not mipro_section.get("online_pool") and online_from_model is None:
                raise ValueError(
                    f"MIPRO config missing online_pool in TOML. "
                    f"pl_cfg.mipro.online_pool={online_from_model}, "
                    f"mipro_section keys={list(mipro_section.keys())}, "
                    f"pl_section keys={list(pl_section.keys())[:10]}. "
                    f"Ensure [prompt_learning.mipro] has online_pool or provide override."
                )

            # Extract env_name from mipro section to top-level (backend expects it there)
            mipro_env_name = mipro_section.get("env_name")
            if (
                mipro_env_name
                and not pl_section.get("env_name")
                and not pl_section.get("task_app_id")
            ):
                pl_section["env_name"] = mipro_env_name

            # CRITICAL: Update mipro section back to config_dict IMMEDIATELY
            # This ensures fields are present before override merge
            pl_section["mipro"] = mipro_section
            config_dict["prompt_learning"] = pl_section

            # ASSERT: Fields are in config_dict before override merge
            assert (
                config_dict.get("prompt_learning", {}).get("mipro", {}).get("bootstrap_train_seeds")
                is not None
            ), (
                f"bootstrap_train_seeds missing from config_dict before override merge! config_dict keys: {list(config_dict.keys())}"
            )
            assert (
                config_dict.get("prompt_learning", {}).get("mipro", {}).get("online_pool")
                is not None
            ), (
                f"online_pool missing from config_dict before override merge! config_dict keys: {list(config_dict.keys())}"
            )
    else:
        config_dict["prompt_learning"] = {
            "task_app_url": final_task_url,
        }

    # Build payload matching backend API format
    # Extract nested overrides if present, otherwise use flat overrides directly
    # The experiment queue passes flat overrides like {"prompt_learning.policy.model": "..."}
    # But some SDK code passes nested like {"overrides": {"prompt_learning.policy.model": "..."}}
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    # Remove non-override keys (backend, task_url, metadata, auto_start)
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start")
    }

    # ASSERT: Check MIPRO fields BEFORE override merge
    if pl_cfg.algorithm == "mipro":
        pre_merge_mipro = config_dict.get("prompt_learning", {}).get("mipro", {})
        assert pre_merge_mipro.get("bootstrap_train_seeds") is not None, (
            f"bootstrap_train_seeds missing BEFORE override merge! pre_merge_mipro keys: {list(pre_merge_mipro.keys()) if isinstance(pre_merge_mipro, dict) else 'NOT DICT'}"
        )
        assert pre_merge_mipro.get("online_pool") is not None, (
            f"online_pool missing BEFORE override merge! pre_merge_mipro keys: {list(pre_merge_mipro.keys()) if isinstance(pre_merge_mipro, dict) else 'NOT DICT'}"
        )

    # CRITICAL: Merge overrides into config_dict BEFORE sending to backend
    # This ensures early validation in backend sees merged values
    # Use the same deep_update logic used throughout core utilities.
    if config_overrides:
        from synth_ai.core.dict_utils import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)

        # ASSERT: Check MIPRO fields AFTER override merge
        if pl_cfg.algorithm == "mipro":
            post_merge_mipro = config_dict.get("prompt_learning", {}).get("mipro", {})
            assert post_merge_mipro.get("bootstrap_train_seeds") is not None, (
                f"bootstrap_train_seeds missing AFTER override merge! post_merge_mipro keys: {list(post_merge_mipro.keys()) if isinstance(post_merge_mipro, dict) else 'NOT DICT'}, overrides: {list(config_overrides.keys())[:5]}"
            )
            assert post_merge_mipro.get("online_pool") is not None, (
                f"online_pool missing AFTER override merge! post_merge_mipro keys: {list(post_merge_mipro.keys()) if isinstance(post_merge_mipro, dict) else 'NOT DICT'}, overrides: {list(config_overrides.keys())[:5]}"
            )

    # After merging overrides (or if no overrides), re-run MIPRO reorganization to ensure
    # fields like bootstrap_train_seeds and online_pool are in the right place
    # This must run AFTER override merge so we catch fields from overrides
    pl_section_in_dict = config_dict.get("prompt_learning", {})
    if pl_cfg.algorithm == "mipro" and isinstance(pl_section_in_dict, dict):
        mipro_section = pl_section_in_dict.get("mipro", {})
        if not isinstance(mipro_section, dict):
            mipro_section = {}

        # CRITICAL: After override merge, ensure fields are in mipro section
        # Check both top-level and nested locations (overrides may have added them anywhere)
        if not mipro_section.get("bootstrap_train_seeds") and pl_section_in_dict.get(
            "bootstrap_train_seeds"
        ):
            mipro_section["bootstrap_train_seeds"] = pl_section_in_dict["bootstrap_train_seeds"]

        if not mipro_section.get("online_pool") and pl_section_in_dict.get("online_pool"):
            mipro_section["online_pool"] = pl_section_in_dict["online_pool"]

        if not mipro_section.get("test_pool") and pl_section_in_dict.get("test_pool"):
            mipro_section["test_pool"] = pl_section_in_dict["test_pool"]

        if not mipro_section.get("reference_pool") and pl_section_in_dict.get("reference_pool"):
            mipro_section["reference_pool"] = pl_section_in_dict["reference_pool"]

        # ASSERT: Fields should be in mipro_section after reorganization
        assert mipro_section.get("bootstrap_train_seeds") is not None, (
            f"bootstrap_train_seeds missing after reorganization! mipro_section keys: {list(mipro_section.keys())}, pl_section keys: {list(pl_section_in_dict.keys())[:10]}"
        )
        assert mipro_section.get("online_pool") is not None, (
            f"online_pool missing after reorganization! mipro_section keys: {list(mipro_section.keys())}, pl_section keys: {list(pl_section_in_dict.keys())[:10]}"
        )

        # Update mipro section back to config_dict
        pl_section_in_dict["mipro"] = mipro_section
        config_dict["prompt_learning"] = pl_section_in_dict

        # ASSERT: Fields are in config_dict after reorganization
        assert (
            config_dict.get("prompt_learning", {}).get("mipro", {}).get("bootstrap_train_seeds")
            is not None
        ), "bootstrap_train_seeds missing from config_dict after reorganization!"
        assert (
            config_dict.get("prompt_learning", {}).get("mipro", {}).get("online_pool") is not None
        ), "online_pool missing from config_dict after reorganization!"

        # CRITICAL: Verify MIPRO required fields are present after merge
        # This is the final check before sending to backend (runs whether or not overrides were applied)
        final_mipro_section = pl_section_in_dict.get("mipro", {})

        # DEBUG: Log what we found for troubleshooting
        bootstrap_seeds = final_mipro_section.get("bootstrap_train_seeds")
        online_pool_val = final_mipro_section.get("online_pool")

        if not bootstrap_seeds or not online_pool_val:
            # Log debug info before raising error
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                "MIPRO config validation failed:\n"
                f"  bootstrap_train_seeds in mipro section: {bootstrap_seeds is not None}\n"
                f"  online_pool in mipro section: {online_pool_val is not None}\n"
                f"  mipro_section keys: {list(final_mipro_section.keys()) if isinstance(final_mipro_section, dict) else 'NOT A DICT'}\n"
                f"  pl_section keys: {list(pl_section_in_dict.keys()) if isinstance(pl_section_in_dict, dict) else 'NOT A DICT'}\n"
                f"  config_overrides keys: {list(config_overrides.keys()) if config_overrides else 'NO OVERRIDES'}"
            )

        if not bootstrap_seeds:
            raise ValueError(
                "MIPRO config missing bootstrap_train_seeds after applying overrides. "
                "Ensure it's set in TOML at [prompt_learning.mipro] or [prompt_learning] level, "
                "or provided via override 'prompt_learning.mipro.bootstrap_train_seeds'. "
                f"Current mipro_section keys: {list(final_mipro_section.keys()) if isinstance(final_mipro_section, dict) else 'NOT A DICT'}"
            )
        if not online_pool_val:
            raise ValueError(
                "MIPRO config missing online_pool after applying overrides. "
                "Ensure it's set in TOML at [prompt_learning.mipro] or [prompt_learning] level, "
                "or provided via override 'prompt_learning.mipro.online_pool'. "
                f"Current mipro_section keys: {list(final_mipro_section.keys()) if isinstance(final_mipro_section, dict) else 'NOT A DICT'}"
            )

    # CRITICAL: Final validation - ensure MIPRO fields are in config_body before sending
    pl_section_final = config_dict.get("prompt_learning", {})
    if pl_cfg.algorithm == "mipro":
        mipro_final = pl_section_final.get("mipro", {})
        if not isinstance(mipro_final, dict):
            mipro_final = {}

        # If fields are still missing, FORCE them from the Pydantic model one last time
        if pl_cfg.mipro:
            if not mipro_final.get("bootstrap_train_seeds"):
                bootstrap_val = getattr(pl_cfg.mipro, "bootstrap_train_seeds", None)
                if bootstrap_val is not None:
                    mipro_final["bootstrap_train_seeds"] = bootstrap_val

            if not mipro_final.get("online_pool"):
                online_val = getattr(pl_cfg.mipro, "online_pool", None)
                if online_val is not None:
                    mipro_final["online_pool"] = online_val

            if not mipro_final.get("test_pool"):
                test_val = getattr(pl_cfg.mipro, "test_pool", None)
                if test_val is not None:
                    mipro_final["test_pool"] = test_val

            if not mipro_final.get("reference_pool"):
                ref_val = getattr(pl_cfg.mipro, "reference_pool", None)
                if ref_val is not None:
                    mipro_final["reference_pool"] = ref_val

        # Update back to config_dict
        pl_section_final["mipro"] = mipro_final
        config_dict["prompt_learning"] = pl_section_final

        # ASSERT: Fields are in config_dict after final check
        assert (
            config_dict.get("prompt_learning", {}).get("mipro", {}).get("bootstrap_train_seeds")
            is not None
        ), (
            f"bootstrap_train_seeds missing from config_dict after final check! mipro_final keys: {list(mipro_final.keys())}"
        )
        assert (
            config_dict.get("prompt_learning", {}).get("mipro", {}).get("online_pool") is not None
        ), (
            f"online_pool missing from config_dict after final check! mipro_final keys: {list(mipro_final.keys())}"
        )

        # FINAL ASSERTION: These fields MUST be present
        assert mipro_final.get("bootstrap_train_seeds") is not None, (
            f"CRITICAL: bootstrap_train_seeds missing from config_body! mipro_final keys: {list(mipro_final.keys())}, pl_cfg.mipro.bootstrap_train_seeds: {getattr(pl_cfg.mipro, 'bootstrap_train_seeds', None) if pl_cfg.mipro else 'N/A'}"
        )
        assert mipro_final.get("online_pool") is not None, (
            f"CRITICAL: online_pool missing from config_body! mipro_final keys: {list(mipro_final.keys())}, pl_cfg.mipro.online_pool: {getattr(pl_cfg.mipro, 'online_pool', None) if pl_cfg.mipro else 'N/A'}"
        )

    # ASSERT: Verify critical overrides are reflected in config_body
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
                assert actual_budget == expected_budget, (
                    f"Rollout budget mismatch: config_body has {actual_budget} but override specifies {expected_budget}. "
                    f"This indicates the override wasn't applied correctly."
                )

        # Check model override
        model_key = "prompt_learning.policy.model"
        if model_key in config_overrides:
            expected_model = config_overrides[model_key]
            policy_section = pl_section_in_dict.get("policy", {})
            actual_model = policy_section.get("model") if isinstance(policy_section, dict) else None
            if actual_model is not None:
                assert actual_model == expected_model, (
                    f"Model mismatch: config_body has {actual_model} but override specifies {expected_model}. "
                    f"This indicates the override wasn't applied correctly."
                )

        # Check provider override
        provider_key = "prompt_learning.policy.provider"
        if provider_key in config_overrides:
            expected_provider = config_overrides[provider_key]
            policy_section = pl_section_in_dict.get("policy", {})
            actual_provider = (
                policy_section.get("provider") if isinstance(policy_section, dict) else None
            )
            if actual_provider is not None:
                assert actual_provider == expected_provider, (
                    f"Provider mismatch: config_body has {actual_provider} but override specifies {expected_provider}. "
                    f"This indicates the override wasn't applied correctly."
                )

    # FINAL CHECK: Ensure config_body has correct structure for backend
    # Backend expects: {"prompt_learning": {...}} (full TOML structure)
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    # CRITICAL: Final validation - check MIPRO fields are in config_body before sending
    if pl_cfg.algorithm == "mipro":
        pl_section_final = config_dict.get("prompt_learning", {})
        mipro_section_final = (
            pl_section_final.get("mipro", {}) if isinstance(pl_section_final, dict) else {}
        )

        # ASSERT: Fields MUST be present before building payload
        assert mipro_section_final.get("bootstrap_train_seeds") is not None, (
            f"CRITICAL ASSERTION FAILED: bootstrap_train_seeds missing from config_body before sending to backend! "
            f"mipro_section keys: {list(mipro_section_final.keys()) if isinstance(mipro_section_final, dict) else 'NOT A DICT'}, "
            f"pl_section keys: {list(pl_section_final.keys()) if isinstance(pl_section_final, dict) else 'NOT A DICT'}, "
            f"pl_cfg.mipro.bootstrap_train_seeds: {getattr(pl_cfg.mipro, 'bootstrap_train_seeds', None) if pl_cfg.mipro else 'N/A'}"
        )
        assert mipro_section_final.get("online_pool") is not None, (
            f"CRITICAL ASSERTION FAILED: online_pool missing from config_body before sending to backend! "
            f"mipro_section keys: {list(mipro_section_final.keys()) if isinstance(mipro_section_final, dict) else 'NOT A DICT'}, "
            f"pl_section keys: {list(pl_section_final.keys()) if isinstance(pl_section_final, dict) else 'NOT A DICT'}, "
            f"pl_cfg.mipro.online_pool: {getattr(pl_cfg.mipro, 'online_pool', None) if pl_cfg.mipro else 'N/A'}"
        )

        if not mipro_section_final.get("bootstrap_train_seeds"):
            import json

            raise ValueError(
                "CRITICAL: bootstrap_train_seeds missing from config_body before sending to backend. "
                f"mipro_section keys: {list(mipro_section_final.keys()) if isinstance(mipro_section_final, dict) else 'NOT A DICT'}, "
                f"pl_section keys: {list(pl_section_final.keys()) if isinstance(pl_section_final, dict) else 'NOT A DICT'}, "
                f"config_overrides: {json.dumps(list(config_overrides.keys()) if config_overrides else [])}"
            )
        if not mipro_section_final.get("online_pool"):
            import json

            raise ValueError(
                "CRITICAL: online_pool missing from config_body before sending to backend. "
                f"mipro_section keys: {list(mipro_section_final.keys()) if isinstance(mipro_section_final, dict) else 'NOT A DICT'}, "
                f"pl_section keys: {list(pl_section_final.keys()) if isinstance(pl_section_final, dict) else 'NOT A DICT'}, "
                f"config_overrides: {json.dumps(list(config_overrides.keys()) if config_overrides else [])}"
            )

    payload: dict[str, Any] = {
        "algorithm": pl_cfg.algorithm,
        "config_body": config_dict,
        "overrides": config_overrides,
        "metadata": overrides.get("metadata", {}),
        "auto_start": overrides.get("auto_start", True),
    }

    # CRITICAL DEBUG: Print MIPRO section structure before sending
    if pl_cfg.algorithm == "mipro":
        import json

        mipro_debug = config_dict.get("prompt_learning", {}).get("mipro", {})
        print("\n🔍 DEBUG: MIPRO section in config_body before sending:")
        print(f"  Type: {type(mipro_debug)}")
        print(
            f"  Keys: {list(mipro_debug.keys()) if isinstance(mipro_debug, dict) else 'NOT A DICT'}"
        )
        print(
            f"  bootstrap_train_seeds present: {mipro_debug.get('bootstrap_train_seeds') is not None}"
        )
        print(f"  online_pool present: {mipro_debug.get('online_pool') is not None}")
        if isinstance(mipro_debug, dict):
            print(f"  bootstrap_train_seeds value: {mipro_debug.get('bootstrap_train_seeds')}")
            print(f"  online_pool value: {mipro_debug.get('online_pool')}")
        # Print full mipro section (truncated)
        mipro_json = json.dumps(mipro_debug, indent=2, default=str)
        print(f"  Full mipro section (first 500 chars):\n{mipro_json[:500]}")
        print("🔍 END DEBUG\n")

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
        task_url: Override for task_app_url
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
        ...             "task_app_url": "https://tunnel.example.com",
        ...             "policy": {"model": "gpt-4o-mini", "provider": "openai"},
        ...             "gepa": {...},
        ...         }
        ...     },
        ...     task_url=None,
        ...     overrides={},
        ... )
    """
    ctx: dict[str, Any] = {"source": source_label}
    log_info("build_prompt_learning_payload_from_mapping invoked", ctx=ctx)
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
        if not pl_cfg.gepa.evaluation:
            raise click.ClickException(
                "GEPA config missing: [prompt_learning.gepa.evaluation] section is required"
            )
        train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "seeds", None
        )
        if not train_seeds:
            raise click.ClickException(
                "GEPA config missing train_seeds: [prompt_learning.gepa.evaluation] must have 'train_seeds' or 'seeds' field"
            )
        val_seeds = getattr(pl_cfg.gepa.evaluation, "val_seeds", None) or getattr(
            pl_cfg.gepa.evaluation, "validation_seeds", None
        )
        if not val_seeds:
            raise click.ClickException(
                "GEPA config missing val_seeds: [prompt_learning.gepa.evaluation] must have 'val_seeds' or 'validation_seeds' field"
            )

    cli_task_url = overrides.get("task_url") or task_url
    env_task_url = os.environ.get("TASK_APP_URL")
    config_task_url = (pl_cfg.task_app_url or "").strip() or None

    # Resolve task_app_url with same precedence as file-based builder
    if cli_task_url:
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=cli_task_url,
            env_value=None,
            config_value=config_task_url,
            required=True,
        )
    elif config_task_url:
        final_task_url = config_task_url
    else:
        final_task_url = ConfigResolver.resolve(
            "task_app_url",
            cli_value=None,
            env_value=env_task_url,
            config_value=None,
            required=True,
        )
    assert final_task_url is not None

    # Get task_app_api_key from config or environment
    # Note: task_app_api_key is not a field on PromptLearningConfig, use getattr
    config_api_key = (getattr(pl_cfg, "task_app_api_key", None) or "").strip() or None
    cli_api_key = overrides.get("task_app_api_key")
    env_api_key = ensure_localapi_auth()
    task_app_api_key = ConfigResolver.resolve(
        "task_app_api_key",
        cli_value=cli_api_key,
        env_value=env_api_key,
        config_value=config_api_key,
        required=True,
    )

    # Build config dict for backend
    config_dict = pl_cfg.to_dict()

    # Ensure task_app_url is set (task_app_api_key is resolved by backend from ENVIRONMENT_API_KEY)
    pl_section = config_dict.get("prompt_learning", {})
    if isinstance(pl_section, dict):
        pl_section["task_app_url"] = final_task_url

        # GEPA: Extract train_seeds from nested structure
        if pl_cfg.algorithm == "gepa" and pl_cfg.gepa:
            train_seeds = None
            if pl_cfg.gepa.evaluation:
                train_seeds = getattr(pl_cfg.gepa.evaluation, "train_seeds", None) or getattr(
                    pl_cfg.gepa.evaluation, "seeds", None
                )

            if train_seeds and not pl_section.get("train_seeds"):
                pl_section["train_seeds"] = train_seeds
            if train_seeds and not pl_section.get("evaluation_seeds"):
                pl_section["evaluation_seeds"] = train_seeds
    else:
        config_dict["prompt_learning"] = {
            "task_app_url": final_task_url,
        }

    # Build payload matching backend API format
    config_overrides = overrides.get("overrides", {}) if "overrides" in overrides else overrides
    config_overrides = {
        k: v
        for k, v in config_overrides.items()
        if k not in ("backend", "task_url", "metadata", "auto_start")
    }

    # Merge overrides into config_dict
    if config_overrides:
        from synth_ai.core.dict_utils import deep_update as _deep_update

        _deep_update(config_dict, config_overrides)

    # Final validation
    if "prompt_learning" not in config_dict:
        raise ValueError(
            f"config_dict must have 'prompt_learning' key. Found keys: {list(config_dict.keys())}"
        )

    payload: dict[str, Any] = {
        "algorithm": pl_cfg.algorithm,
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
    "build_prompt_learning_payload",
    "build_prompt_learning_payload_from_mapping",
]
