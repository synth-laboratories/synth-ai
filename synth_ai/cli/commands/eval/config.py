"""Eval command configuration loading and normalization.

This module handles loading and resolving evaluation configuration from:
- TOML config files (legacy eval format or prompt_learning format)
- Command-line arguments (override config values)
- Environment variables (for API keys, etc.)

**Config File Formats:**

1. **Legacy Eval Format:**
    ```toml
    [eval]
    app_id = "banking77"
    url = "http://localhost:8103"
    env_name = "banking77"
    seeds = [0, 1, 2, 3, 4]
    
    [eval.policy_config]
    model = "gpt-4"
    provider = "openai"
    ```

2. **Prompt Learning Format:**
    ```toml
    [prompt_learning]
    task_app_id = "banking77"
    task_app_url = "http://localhost:8103"
    
    [prompt_learning.gepa]
    env_name = "banking77"
    
    [prompt_learning.gepa.evaluation]
    seeds = [0, 1, 2, 3, 4]
    ```

**See Also:**
    - `synth_ai.cli.commands.eval.core.eval_command()`: CLI entry point
    - `synth_ai.cli.commands.eval.runner.run_eval()`: Uses resolved config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from synth_ai.sdk.api.train.configs.prompt_learning import PromptLearningConfig
from synth_ai.sdk.api.train.utils import load_toml
from synth_ai.sdk.task.contracts import RolloutMode


SeedSet = Literal["seeds", "validation_seeds", "test_pool"]


@dataclass(slots=True)
class EvalRunConfig:
    """Configuration for evaluation runs.
    
    This dataclass holds all configuration needed to execute an evaluation
    against a task app. Values can come from TOML config files, CLI arguments,
    or environment variables.
    
    **Required Fields:**
        app_id: Task app identifier
        task_app_url: URL of running task app (or None to spawn locally)
        seeds: List of seeds/indices to evaluate
        
    **Optional Fields:**
        env_name: Environment name (usually matches app_id)
        policy_config: Model and provider configuration
        backend_url: Backend URL for trace capture (enables backend mode)
        concurrency: Number of parallel rollouts
        return_trace: Whether to include traces in responses
        
    **Example:**
        ```python
        config = EvalRunConfig(
            app_id="banking77",
            task_app_url="http://localhost:8103",
            backend_url="http://localhost:8000",
            env_name="banking77",
            seeds=[0, 1, 2, 3, 4],
            policy_config={"model": "gpt-4", "provider": "openai"},
            concurrency=5,
            return_trace=True,
        )
        ```
    """
    app_id: str
    task_app_url: str | None
    task_app_api_key: str | None
    env_name: str | None
    env_config: dict[str, Any] = field(default_factory=dict)
    policy_name: str | None = None
    policy_config: dict[str, Any] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)
    mode: RolloutMode = RolloutMode.EVAL
    return_trace: bool = False
    trace_format: str = "compact"
    concurrency: int = 1
    metadata: dict[str, str] = field(default_factory=dict)
    output_txt: Path | None = None
    output_json: Path | None = None
    judge_config: dict[str, Any] | None = None
    backend_url: str | None = None
    backend_api_key: str | None = None
    wait: bool = False
    poll_interval: float = 5.0
    traces_dir: Path | None = None
    config_path: Path | None = None
    timeout: float | None = None


def load_eval_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval config not found: {path}")
    return load_toml(path)


def _select_seed_pool(
    *,
    seeds: list[int] | None,
    validation_seeds: list[int] | None,
    test_pool: list[int] | None,
    seed_set: SeedSet,
) -> list[int]:
    if seed_set == "validation_seeds" and validation_seeds:
        return validation_seeds
    if seed_set == "test_pool" and test_pool:
        return test_pool
    if seeds:
        return seeds
    if validation_seeds:
        return validation_seeds
    if test_pool:
        return test_pool
    return []


def _from_prompt_learning(
    raw: dict[str, Any],
    *,
    seed_set: SeedSet,
) -> EvalRunConfig:
    pl_cfg = PromptLearningConfig.from_mapping(raw)
    gepa = pl_cfg.gepa
    mipro = pl_cfg.mipro

    eval_cfg = gepa.evaluation if gepa else None
    seeds = _select_seed_pool(
        seeds=eval_cfg.seeds if eval_cfg else None,
        validation_seeds=eval_cfg.validation_seeds if eval_cfg else None,
        test_pool=eval_cfg.test_pool if eval_cfg else None,
        seed_set=seed_set,
    )

    env_name = None
    env_config: dict[str, Any] = {}
    if gepa:
        env_name = gepa.env_name
        env_config = dict(gepa.env_config or {})
    elif mipro:
        env_name = mipro.env_name
        env_config = dict(mipro.env_config or {})

    policy_cfg: dict[str, Any] = {}
    if pl_cfg.policy:
        policy_cfg = {
            "model": pl_cfg.policy.model,
            "provider": pl_cfg.policy.provider,
        }
        if pl_cfg.policy.inference_url:
            policy_cfg["inference_url"] = pl_cfg.policy.inference_url

    app_id = pl_cfg.task_app_id or (env_name or "")
    judge_cfg = None
    if pl_cfg.judge:
        if isinstance(pl_cfg.judge, dict):
            judge_cfg = dict(pl_cfg.judge)
        else:
            judge_cfg = pl_cfg.judge.model_dump(mode="python")

    return EvalRunConfig(
        app_id=app_id,
        task_app_url=pl_cfg.task_app_url,
        task_app_api_key=pl_cfg.task_app_api_key,
        env_name=env_name,
        env_config=env_config,
        policy_name=pl_cfg.policy.policy_name if pl_cfg.policy else None,
        policy_config=policy_cfg,
        seeds=seeds,
        ops=[],
        concurrency=(gepa.rollout.max_concurrent if gepa and gepa.rollout else 1),
        judge_config=judge_cfg,
    )


def _from_legacy_eval(raw: dict[str, Any]) -> EvalRunConfig:
    eval_section = raw.get("eval", {})
    if not isinstance(eval_section, dict):
        eval_section = {}
    app_id = str(eval_section.get("app_id") or "").strip()
    model = str(eval_section.get("model") or "").strip()
    policy_cfg = dict(eval_section.get("policy_config") or {})
    if model and "model" not in policy_cfg:
        policy_cfg["model"] = model
    if "provider" not in policy_cfg and eval_section.get("provider"):
        policy_cfg["provider"] = eval_section.get("provider")
    return EvalRunConfig(
        app_id=app_id,
        task_app_url=eval_section.get("url") or eval_section.get("task_app_url"),
        task_app_api_key=eval_section.get("task_app_api_key"),
        env_name=eval_section.get("env_name"),
        env_config=dict(eval_section.get("env_config") or {}),
        policy_name=eval_section.get("policy_name"),
        policy_config=policy_cfg,
        seeds=list(eval_section.get("seeds") or []),
        ops=list(eval_section.get("ops") or []),
        return_trace=bool(eval_section.get("return_trace", False)),
        trace_format=str(eval_section.get("trace_format") or "compact"),
        concurrency=int(eval_section.get("concurrency") or 1),
        metadata=dict(eval_section.get("metadata") or {}),
    )


def resolve_eval_config(
    *,
    config_path: Path | None,
    cli_app_id: str | None,
    cli_model: str | None,
    cli_seeds: list[int] | None,
    cli_url: str | None,
    cli_env_file: str | None,
    cli_ops: list[str] | None,
    cli_return_trace: bool | None,
    cli_concurrency: int | None,
    cli_output_txt: Path | None,
    cli_output_json: Path | None,
    cli_backend_url: str | None,
    cli_wait: bool,
    cli_poll_interval: float | None,
    cli_traces_dir: Path | None,
    seed_set: SeedSet,
    metadata: dict[str, str],
) -> EvalRunConfig:
    """Resolve evaluation configuration from multiple sources.
    
    Loads configuration from TOML file (if provided) and merges with CLI arguments.
    CLI arguments take precedence over config file values.
    
    **Config File Formats:**
    - Legacy eval format: `[eval]` section
    - Prompt learning format: `[prompt_learning]` section
    
    **Precedence Order:**
    1. CLI arguments (highest priority)
    2. Config file values
    3. Default values
    
    Args:
        config_path: Path to TOML config file (optional)
        cli_app_id: App ID from CLI (overrides config)
        cli_model: Model name from CLI (overrides config)
        cli_seeds: Seeds list from CLI (overrides config)
        cli_url: Task app URL from CLI (overrides config)
        cli_backend_url: Backend URL from CLI (overrides config)
        cli_concurrency: Concurrency from CLI (overrides config)
        seed_set: Which seed pool to use ("seeds", "validation_seeds", "test_pool")
        metadata: Metadata key-value pairs for filtering
        
    Returns:
        Resolved EvalRunConfig with all values merged.
        
    Raises:
        FileNotFoundError: If config file is specified but doesn't exist.
        
    Example:
        ```python
        config = resolve_eval_config(
            config_path=Path("banking77_eval.toml"),
            cli_app_id="banking77",
            cli_seeds=[0, 1, 2],
            cli_url="http://localhost:8103",
            seed_set="seeds",
            metadata={},
        )
        ```
    """
    raw: dict[str, Any] = {}
    if config_path is not None:
        raw = load_eval_toml(config_path)

    if raw and ("prompt_learning" in raw or raw.get("algorithm") in {"gepa", "mipro"}):
        resolved = _from_prompt_learning(raw, seed_set=seed_set)
    else:
        resolved = _from_legacy_eval(raw)

    if cli_app_id:
        resolved.app_id = cli_app_id
    if cli_url:
        resolved.task_app_url = cli_url
    if cli_seeds:
        resolved.seeds = cli_seeds
    if cli_ops:
        resolved.ops = cli_ops
    if cli_return_trace is not None:
        resolved.return_trace = cli_return_trace
    if cli_concurrency is not None:
        resolved.concurrency = cli_concurrency
    if cli_output_txt is not None:
        resolved.output_txt = cli_output_txt
    if cli_output_json is not None:
        resolved.output_json = cli_output_json
    if cli_backend_url:
        resolved.backend_url = cli_backend_url
    if cli_wait:
        resolved.wait = True
    if cli_poll_interval is not None:
        resolved.poll_interval = cli_poll_interval
    if cli_traces_dir is not None:
        resolved.traces_dir = cli_traces_dir

    if cli_model:
        resolved.policy_config["model"] = cli_model
    if metadata:
        resolved.metadata = metadata

    if cli_env_file:
        # Store in metadata for logging; env loading handled in core.
        resolved.metadata.setdefault("env_file", cli_env_file)

    resolved.config_path = config_path

    return resolved


__all__ = ["EvalRunConfig", "resolve_eval_config", "SeedSet"]
