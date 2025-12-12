import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from synth_ai.cli.lib.prompts import ctx_print
from synth_ai.core.paths import is_hidden_path, validate_file_type

# Train config types: prompt optimization, reinforcement learning, supervised fine-tuning, ADAS, context learning
TrainType = Literal["prompt", "rl", "sft", "adas", "context_learning"]


def get_type(config: Dict[str, Any]) -> TrainType | None:
    if "context_learning" in config:
        return "context_learning"

    if "prompt_learning" in config:
        return "prompt"

    # Graph / ADAS jobs use a dedicated [graph] (or [adas]) section.
    if isinstance(config.get("graph"), dict) or isinstance(config.get("adas"), dict):
        return "adas"

    algorithm = config.get("algorithm")
    algo_type = None
    algo_method = None
    if isinstance(algorithm, dict):
        algo_type = str(algorithm.get("type") or "").lower()
        algo_method = str(algorithm.get("method") or "").lower()

    job = config.get("job")
    compute = config.get("compute")
    rollout = config.get("rollout")
    policy = config.get("policy")
    model = config.get("model")

    if algo_type == "online" or algo_method in {"policy_gradient", "ppo", "gspo"}:
        return "rl"
    if rollout and (policy or model):
        return "rl"

    if algo_type == "offline" or algo_method in {"sft", "supervised_finetune"}:
        return "sft"
    if job and compute:
        return "sft"

    return None


def validate_context_learning_cfg(cfg: Dict[str, Any]) -> None:
    section = cfg.get("context_learning")
    if not isinstance(section, dict):
        raise ValueError("[context_learning] section must be a dict")

    task_app_url = section.get("task_app_url") or section.get("task_url")
    if not task_app_url:
        raise ValueError("[context_learning].task_app_url is required")

    evaluation_seeds = section.get("evaluation_seeds")
    if evaluation_seeds is not None and not isinstance(evaluation_seeds, list):
        raise ValueError("[context_learning].evaluation_seeds must be a list when provided")

    env_section = section.get("environment")
    if env_section is not None and not isinstance(env_section, dict):
        raise ValueError("[context_learning].environment must be a dict when provided")

    algorithm_section = section.get("algorithm")
    if algorithm_section is not None and not isinstance(algorithm_section, dict):
        raise ValueError("[context_learning].algorithm must be a dict when provided")

    metadata = section.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("[context_learning].metadata must be a dict when provided")

    return None


def validate_po_cfg(cfg: Dict[str, Any]) -> None:
    pl_cfg = cfg.get("prompt_learning")
    if not pl_cfg:
        pl_cfg = cfg

    if not isinstance(pl_cfg, dict):
        raise ValueError("[prompt_learning] section must be a dict")

    algorithm = pl_cfg.get("algorithm")
    if not algorithm:
        raise ValueError("[prompt_learning].algorithm is required")

    if algorithm not in {"mipro", "gepa"}:
        raise ValueError("[prompt_learning].algorithm must be 'mipro' or 'gepa'")

    if not pl_cfg.get("task_app_url"):
        raise ValueError("[prompt_learning].task_app_url is required")

    if algorithm == "mipro":
        mipro = pl_cfg.get("mipro")
        if not isinstance(mipro, dict):
            raise ValueError("[prompt_learning].mipro section is required when algorithm is 'mipro'")
    elif algorithm == "gepa":
        gepa = pl_cfg.get("gepa")
        if not isinstance(gepa, dict):
            raise ValueError("[prompt_learning].gepa section is required when algorithm is 'gepa'")

    policy = pl_cfg.get("policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValueError("[prompt_learning].policy must be a dict")
        if not policy.get("model"):
            raise ValueError("[prompt_learning].policy.model is required")
        if not policy.get("provider"):
            raise ValueError("[prompt_learning].policy.provider is required")

    return None


def validate_sft_cfg(cfg: Dict[str, Any]) -> None:
    algorithm = cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("[algorithm] section is required")
    if algorithm.get("type") != "offline":
        raise ValueError("[algorithm].type must be 'offline'")
    method = algorithm.get("method")
    if method and method not in {"sft", "supervised_finetune"}:
        raise ValueError("[algorithm].method must be 'sft' or 'supervised_finetune'")
    if not algorithm.get("variety"):
        raise ValueError("[algorithm].variety is required")

    job = cfg.get("job")
    if not isinstance(job, dict):
        raise ValueError("[job] section is required")
    if not job.get("model"):
        raise ValueError("[job].model is required")
    if not (job.get("data") or job.get("data_path")):
        raise ValueError("[job].data or [job].data_path is required")

    compute = cfg.get("compute")
    if not isinstance(compute, dict):
        raise ValueError("[compute] section is required")
    if not compute.get("gpu_type"):
        raise ValueError("[compute].gpu_type is required")
    if not compute.get("gpu_count"):
        raise ValueError("[compute].gpu_count is required")

    return None


def validate_rl_cfg(cfg: Dict[str, Any]) -> None:
    algorithm = cfg.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("[algorithm] section is required")
    if algorithm.get("type") != "online":
        raise ValueError("[algorithm].type must be 'online'")
    method = algorithm.get("method")
    if method and method not in {"policy_gradient", "ppo", "gspo"}:
        raise ValueError("[algorithm].method must be policy_gradient/ppo/gspo")
    if not algorithm.get("variety"):
        raise ValueError("[algorithm].variety is required")

    policy = cfg.get("policy")
    model = cfg.get("model")

    section: Dict[str, Any]
    if isinstance(policy, dict):
        section = policy
        if not (policy.get("model_name") or policy.get("source")):
            raise ValueError("[policy].model_name or [policy].source is required")
    elif isinstance(model, dict):
        section = model
        if not (model.get("base") or model.get("source")):
            raise ValueError("[model].base or [model].source is required")
    else:
        raise ValueError("[policy] or [model] section is required")

    if not section.get("trainer_mode"):
        raise ValueError("trainer_mode is required")
    if not section.get("label"):
        raise ValueError("label is required")

    compute = cfg.get("compute")
    if not isinstance(compute, dict):
        raise ValueError("[compute] section is required")
    if not compute.get("gpu_type"):
        raise ValueError("[compute].gpu_type is required")
    if not compute.get("gpu_count"):
        raise ValueError("[compute].gpu_count is required")

    rollout = cfg.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError("[rollout] section is required")
    if not rollout.get("env_name"):
        raise ValueError("[rollout].env_name is required")
    if not rollout.get("policy_name"):
        raise ValueError("[rollout].policy_name is required")

    topology = cfg.get("topology") or compute.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("[topology] or [compute.topology] is required")

    training = cfg.get("training")
    if training:
        required_training_fields = (
            "num_epochs",
            "iterations_per_epoch",
            "max_turns",
            "batch_size",
            "group_size",
            "learning_rate",
        )
        for field in required_training_fields:
            if field not in training:
                raise ValueError(f"[training].{field} is required")

    evaluation = cfg.get("evaluation")
    if evaluation:
        required_eval_fields = ("instances", "every_n_iters", "seeds")
        for field in required_eval_fields:
            if field not in evaluation:
                raise ValueError(f"[evaluation].{field} is required")
    
    return None


def validate_adas_cfg(cfg: Dict[str, Any], *, path: Path) -> None:
    """Validate a graph/ADAS TOML config.

    Uses the SDK validator so backend and CLI stay in sync.
    """
    from synth_ai.sdk.api.train.graph_validators import validate_graph_job_section

    section = cfg.get("graph") or cfg.get("adas") or {}
    validate_graph_job_section(section, base_dir=path.parent.resolve())


def validate_train_cfg(path: Path, discovery: bool = False) -> TrainType:
    def print_pass():
        ctx_print("Check passed", not discovery)

    ctx_print("\nChecking if .toml file", not discovery)
    validate_file_type(path, ".toml")
    print_pass()

    ctx_print("\nChecking if TOML parses", not discovery)
    cfg = tomllib.loads(path.read_text())
    print_pass()

    ctx_print("\nChecking if train type is valid", not discovery)
    train_type = get_type(cfg)
    if not train_type:
        raise ValueError(
            "Unable to determine training config type; expected [algorithm] or [prompt_learning] sections."
        )
    print_pass()

    ctx_print(f"\nChecking if {train_type} config is valid", not discovery)
    match train_type:
        case "context_learning":
            validate_context_learning_cfg(cfg)
        case "prompt":
            validate_po_cfg(cfg)
        case "rl":
            validate_rl_cfg(cfg)
        case "sft":
            validate_sft_cfg(cfg)
        case "adas":
            validate_adas_cfg(cfg, path=path)
    print_pass()

    return train_type


def find_train_cfgs_in_cwd() -> List[Tuple[TrainType, str, str]]:
    cwd = Path.cwd().resolve()
    entries: List[Tuple[TrainType, str, str, float]] = []
    for path in cwd.rglob("*.toml"):
        if is_hidden_path(path, cwd):
            continue
        if not path.is_file():
            continue
        try:
            train_type = validate_train_cfg(path, discovery=True)
        except Exception:
            continue
        try:
            rel_path = path.relative_to(cwd)
        except ValueError:
            rel_path = path
        try:
            mtime = path.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime).isoformat(
                sep=" ",
                timespec="seconds",
            )
        except OSError:
            mtime = 0.0
            mtime_str = ""
        entries.append((train_type, str(rel_path), mtime_str, mtime))

    entries.sort(key=lambda entry: entry[3], reverse=True)
    return [(train_type, rel_path, mtime_str) for train_type, rel_path, mtime_str, _ in entries]
