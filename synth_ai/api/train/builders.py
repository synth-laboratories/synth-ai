from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from synth_ai.api.models.supported import (
    UnsupportedModelError,
    ensure_allowed_model,
    normalize_model_identifier,
)
from synth_ai.learning.sft.config import prepare_sft_job_payload

from .supported_algos import (
    AlgorithmValidationError,
    ensure_model_supported_for_algorithm,
    validate_algorithm_config,
)
from .utils import TrainError, ensure_api_base, load_toml


@dataclass(slots=True)
class RLBuildResult:
    payload: dict[str, Any]
    task_url: str
    idempotency: str | None


@dataclass(slots=True)
class SFTBuildResult:
    payload: dict[str, Any]
    train_file: Path
    validation_file: Path | None


def build_rl_payload(
    *,
    config_path: Path,
    task_url: str,
    overrides: dict[str, Any],
    idempotency: str | None,
    allow_experimental: bool | None = None,
) -> RLBuildResult:
    data = load_toml(config_path)
    try:
        spec = validate_algorithm_config(data.get("algorithm"), expected_family="rl")
    except AlgorithmValidationError as exc:
        raise click.ClickException(str(exc)) from exc
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}

    final_task_url = (
        overrides.get("task_url")
        or task_url
        or (services.get("task_url") if isinstance(services, dict) else None)
        or ""
    ).strip()
    if not final_task_url:
        raise click.ClickException(
            "Task app URL required (provide --task-url or set services.task_url in TOML)"
        )

    raw_source = model_cfg.get("source") if isinstance(model_cfg, dict) else ""
    model_source = str(raw_source or "").strip()
    raw_base = model_cfg.get("base") if isinstance(model_cfg, dict) else ""
    model_base = str(raw_base or "").strip()
    override_model = (overrides.get("model") or "").strip()
    if override_model:
        model_source = override_model
        model_base = ""
    if bool(model_source) == bool(model_base):
        details = (
            f"Config: {config_path}\n"
            f"[model].source={model_source!r} | [model].base={model_base!r}"
        )
        hint = (
            "Set exactly one: [model].base for a base model (e.g. 'Qwen/Qwen3-1.7B') "
            "or [model].source for a fine-tuned model id. Also remove any conflicting "
            "'[policy].model' entries."
        )
        raise click.ClickException(
            "Invalid model config: exactly one of [model].source or [model].base is required.\n"
            + details
            + "\nHint: "
            + hint
        )

    try:
        if model_source:
            model_source = normalize_model_identifier(model_source)
        if model_base:
            model_base = normalize_model_identifier(model_base, allow_finetuned_prefixes=False)
    except UnsupportedModelError as exc:
        raise click.ClickException(str(exc)) from exc

    base_model_for_training: str | None = None
    if model_source:
        base_model_for_training = ensure_allowed_model(
            model_source,
            allow_finetuned_prefixes=True,
            allow_experimental=allow_experimental,
        )
    elif model_base:
        base_model_for_training = ensure_allowed_model(
            model_base,
            allow_finetuned_prefixes=False,
            allow_experimental=allow_experimental,
        )
    if base_model_for_training:
        try:
            ensure_model_supported_for_algorithm(base_model_for_training, spec)
        except AlgorithmValidationError as exc:
            raise click.ClickException(str(exc)) from exc

    # Force TOML services.task_url to the effective endpoint to avoid split URLs
    try:
        if isinstance(data.get("services"), dict):
            data["services"]["task_url"] = final_task_url
        else:
            data["services"] = {"task_url": final_task_url}
    except Exception:
        pass

    payload: dict[str, Any] = {
        "job_type": "rl",
        "compute": data.get("compute", {}),
        "data": {
            "endpoint_base_url": final_task_url.rstrip("/"),
            "config": data,
        },
        "tags": {"source": "train-cli"},
    }
    if model_source:
        payload["data"]["model"] = model_source
    if model_base:
        payload["data"]["base_model"] = model_base

    backend = overrides.get("backend")
    if backend:
        payload.setdefault("metadata", {})["backend_base_url"] = ensure_api_base(str(backend))

    return RLBuildResult(payload=payload, task_url=final_task_url, idempotency=idempotency)


def build_sft_payload(
    *,
    config_path: Path,
    dataset_override: Path | None,
    allow_experimental: bool | None,
) -> SFTBuildResult:
    data = load_toml(config_path)
    try:
        spec = validate_algorithm_config(data.get("algorithm"), expected_family="sft")
    except AlgorithmValidationError as exc:
        raise TrainError(str(exc)) from exc
    job_cfg = data.get("job") if isinstance(data.get("job"), dict) else {}
    data_cfg = data.get("data") if isinstance(data.get("data"), dict) else {}
    hp_cfg = data.get("hyperparameters") if isinstance(data.get("hyperparameters"), dict) else {}
    train_cfg = data.get("training") if isinstance(data.get("training"), dict) else {}
    compute_cfg = data.get("compute") if isinstance(data.get("compute"), dict) else {}

    raw_dataset = (
        dataset_override
        or (job_cfg.get("data") if isinstance(job_cfg, dict) else None)
        or (job_cfg.get("data_path") if isinstance(job_cfg, dict) else None)
    )
    if not raw_dataset:
        raise TrainError("Dataset not specified; pass --dataset or set [job].data")
    dataset_path = Path(raw_dataset)
    # Resolve relative paths from current working directory, not config directory
    dataset_path = (
        dataset_path if dataset_path.is_absolute() else (Path.cwd() / dataset_path)
    ).resolve()
    if not dataset_path.exists():
        raise TrainError(f"Dataset not found: {dataset_path}")

    validation_path = (
        data_cfg.get("validation_path")
        if isinstance(data_cfg, dict)
        else None
        if isinstance(data_cfg, dict) and isinstance(data_cfg.get("validation_path"), str)
        else None
    )
    validation_file = None
    if validation_path:
        vpath = Path(validation_path)
        # Resolve relative paths from current working directory, not config directory
        vpath = (vpath if vpath.is_absolute() else (Path.cwd() / vpath)).resolve()
        if not vpath.exists():
            click.echo(f"[WARN] Validation dataset {vpath} missing; continuing without validation")
        else:
            validation_file = vpath

    hp_block: dict[str, Any] = {
        "n_epochs": int(hp_cfg.get("n_epochs", 1) if isinstance(hp_cfg, dict) else 1),
    }
    for key in (
        "batch_size",
        "global_batch",
        "per_device_batch",
        "gradient_accumulation_steps",
        "sequence_length",
        "learning_rate",
        "warmup_ratio",
        "train_kind",
    ):
        if isinstance(hp_cfg, dict) and key in hp_cfg:
            hp_block[key] = hp_cfg[key]
    if isinstance(hp_cfg, dict) and isinstance(hp_cfg.get("parallelism"), dict):
        hp_block["parallelism"] = hp_cfg["parallelism"]

    compute_block = {
        k: compute_cfg[k]
        for k in ("gpu_type", "gpu_count", "nodes")
        if isinstance(compute_cfg, dict) and k in compute_cfg
    }

    effective = {
        "compute": compute_block,
        "data": {
            "topology": data_cfg.get("topology", {})
            if isinstance(data_cfg, dict) and isinstance(data_cfg.get("topology"), dict)
            else {}
        },
        "training": {
            k: v
            for k, v in (train_cfg.items() if isinstance(train_cfg, dict) else [])
            if k in ("mode", "use_qlora")
        },
    }

    validation_cfg = (
        train_cfg.get("validation")
        if isinstance(train_cfg, dict) and isinstance(train_cfg.get("validation"), dict)
        else None
    )
    if isinstance(validation_cfg, dict):
        hp_block.update(
            {
                "evaluation_strategy": validation_cfg.get("evaluation_strategy", "steps"),
                "eval_steps": int(validation_cfg.get("eval_steps", 0) or 0),
                "save_best_model_at_end": bool(validation_cfg.get("save_best_model_at_end", True)),
                "metric_for_best_model": validation_cfg.get("metric_for_best_model", "val.loss"),
                "greater_is_better": bool(validation_cfg.get("greater_is_better", False)),
            }
        )
        effective.setdefault("training", {})["validation"] = {
            "enabled": bool(validation_cfg.get("enabled", True))
        }

    raw_model = str(
        job_cfg.get("model") if isinstance(job_cfg, dict) else None or data.get("model") or ""
    ).strip()
    if not raw_model:
        raise TrainError("Model not specified; set [job].model or [model].base in the config")

    try:
        base_model = ensure_allowed_model(
            raw_model,
            allow_finetuned_prefixes=False,
            allow_experimental=allow_experimental,
        )
    except UnsupportedModelError as exc:
        raise TrainError(str(exc)) from exc
    try:
        ensure_model_supported_for_algorithm(base_model, spec)
    except AlgorithmValidationError as exc:
        raise TrainError(str(exc)) from exc

    try:
        payload = prepare_sft_job_payload(
            model=raw_model,
            training_file=None,
            hyperparameters=hp_block,
            metadata={"effective_config": effective},
            training_type="sft_offline",
            training_file_field="training_file_id",
            require_training_file=False,
            include_training_file_when_none=True,
            allow_finetuned_prefixes=False,
        )
    except UnsupportedModelError as exc:
        raise TrainError(str(exc)) from exc
    except ValueError as exc:
        raise TrainError(str(exc)) from exc

    return SFTBuildResult(payload=payload, train_file=dataset_path, validation_file=validation_file)


__all__ = [
    "RLBuildResult",
    "SFTBuildResult",
    "build_rl_payload",
    "build_sft_payload",
]
