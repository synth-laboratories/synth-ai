from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from .utils import ensure_api_base, load_toml, TrainError


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
) -> RLBuildResult:
    data = load_toml(config_path)
    services = data.get("services") if isinstance(data.get("services"), dict) else {}
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}

    final_task_url = (
        overrides.get("task_url") or task_url or services.get("task_url") or ""
    ).strip()
    if not final_task_url:
        raise click.ClickException(
            "Task app URL required (provide --task-url or set services.task_url in TOML)"
        )

    model_source = (model_cfg.get("source") or "").strip()
    model_base = (model_cfg.get("base") or "").strip()
    override_model = (overrides.get("model") or "").strip()
    if override_model:
        model_source = override_model
        model_base = ""
    if bool(model_source) == bool(model_base):
        raise click.ClickException(
            "Model section must specify exactly one of [model].source or [model].base"
        )

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
) -> SFTBuildResult:
    data = load_toml(config_path)
    job_cfg = data.get("job") if isinstance(data.get("job"), dict) else {}
    data_cfg = data.get("data") if isinstance(data.get("data"), dict) else {}
    hp_cfg = data.get("hyperparameters") if isinstance(data.get("hyperparameters"), dict) else {}
    train_cfg = data.get("training") if isinstance(data.get("training"), dict) else {}
    compute_cfg = data.get("compute") if isinstance(data.get("compute"), dict) else {}

    raw_dataset = dataset_override or job_cfg.get("data") or job_cfg.get("data_path")
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
        if isinstance(data_cfg.get("validation_path"), str)
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
        "n_epochs": int(hp_cfg.get("n_epochs", 1)),
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
        if key in hp_cfg:
            hp_block[key] = hp_cfg[key]
    if isinstance(hp_cfg.get("parallelism"), dict):
        hp_block["parallelism"] = hp_cfg["parallelism"]

    compute_block = {
        k: compute_cfg[k] for k in ("gpu_type", "gpu_count", "nodes") if k in compute_cfg
    }

    effective = {
        "compute": compute_block,
        "data": {
            "topology": data_cfg.get("topology", {})
            if isinstance(data_cfg.get("topology"), dict)
            else {}
        },
        "training": {k: v for k, v in train_cfg.items() if k in ("mode", "use_qlora")},
    }

    validation_cfg = (
        train_cfg.get("validation") if isinstance(train_cfg.get("validation"), dict) else None
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

    payload = {
        "model": job_cfg.get("model") or data.get("model"),
        "training_file_id": None,  # populated after upload
        "training_type": "sft_offline",
        "hyperparameters": hp_block,
        "metadata": {"effective_config": effective},
    }

    return SFTBuildResult(payload=payload, train_file=dataset_path, validation_file=validation_file)


__all__ = [
    "RLBuildResult",
    "SFTBuildResult",
    "build_rl_payload",
    "build_sft_payload",
]
