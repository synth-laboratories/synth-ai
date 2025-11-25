from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
import uuid
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from synth_ai.cli.lib.task_app_discovery import discover_eval_config_paths
from synth_ai.core.tracing_v3.session_tracer import SessionTracer
from synth_ai.sdk.task.config import EvalConfig

from .errors import (
    EvalCliError,
    EvalConfigNotFoundError,
    EvalConfigParseError,
    InvalidEvalConfigError,
    MetadataFilterFormatError,
    MetadataSQLExecutionError,
    MetadataSQLResultError,
    MissingEvalTableError,
    NoSeedsMatchedError,
    SeedParseError,
    TaskInfoUnavailableError,
    TomlUnavailableError,
)
from .validation import validate_eval_options

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback
    _toml = None  # type: ignore[assignment]

__all__ = ["command", "get_command", "format_eval_error"]

if TYPE_CHECKING:
    from synth_ai.cli.task_apps import AppChoice, TaskAppEntryType


@lru_cache(maxsize=1)
def _task_apps_module():
    from synth_ai.cli import task_apps as module  # local import to avoid circular deps

    return module


@click.command(
    "eval",
    help="Run one-off rollouts against a task app and print judge/eval summaries.",
)
@click.argument("app_id", type=str, required=False)
@click.option(
    "--config",
    type=click.Path(),
    default=None,
    help="Path to eval TOML (short schema). Auto-discovers the first matching file when omitted.",
)
@click.option(
    "--url",
    "task_app_url",
    type=str,
    default=None,
    help="Base URL of a running task app instead of spawning locally (requires --env-file for secrets).",
)
@click.option(
    "--seeds",
    default="0,1,2,3,4",
    help="Comma-separated seeds/indices to evaluate. Use negative numbers to wrap around the dataset.",
)
@click.option("--split", default="train", show_default=True, help="Dataset split to use")
@click.option(
    "--model",
    default=None,
    help="Model identifier. When omitted the CLI will prompt based on task metadata.",
)
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file(s) to load (API keys, etc.). Required when using --url or remote judges.",
)
@click.option(
    "--trace-db",
    default="traces/v3/synth_ai.db",
    show_default=True,
    help="SQLite/Turso URL for storing rollout traces set to 'none' to disable persistence.",
)
@click.option(
    "--metadata",
    multiple=True,
    help="Filter tasks by key=value metadata (e.g., --metadata difficulty=easy)",
)
@click.option(
    "--metadata-sql",
    default=None,
    help="SQLite query that returns seeds to evaluate (e.g., SELECT seed FROM tasks WHERE difficulty='easy' LIMIT 5)",
)
def eval_command(
    app_id: str | None,
    config: str | None,
    task_app_url: str | None,
    seeds: str,
    split: str,
    model: str | None,
    env_file: Sequence[str],
    trace_db: str,
    metadata: Sequence[str],
    metadata_sql: str | None,
) -> None:
    try:
        return _eval_command_impl(
            app_id=app_id,
            config=config,
            task_app_url=task_app_url,
            seeds=seeds,
            split=split,
            model=model,
            env_file=env_file,
            trace_db=trace_db,
            metadata=metadata,
            metadata_sql=metadata_sql,
        )
    except EvalCliError as exc:
        raise click.ClickException(format_eval_error(exc)) from exc


def _eval_command_impl(
    app_id: str | None,
    config: str | None,
    task_app_url: str | None,
    seeds: str,
    split: str,
    model: str | None,
    env_file: Sequence[str],
    trace_db: str,
    metadata: Sequence[str],
    metadata_sql: str | None,
) -> None:
    """Run rollouts against a task app and report judge statistics.

    By default the command spins up the selected task app in-process, executes the
    requested seeds, and prints aggregate scores (official and custom judges). When
    pointing at a remote `--url`, supply matching `--env-file` values so the CLI can
    forward authentication headers to the running service.
    """
    module = _task_apps_module()
    task_app_config_type = module.TaskAppConfig
    create_task_app = module.create_task_app
    select_app_choice = module._select_app_choice
    determine_env_files = module._determine_env_files
    load_env_files_into_process = module._load_env_files_into_process
    store_trace = getattr(module, "_store_trace", None)
    pearson = module._pearson
    judge_spec_cls = module.JudgeSpec
    session_tracer_cls = getattr(module, "SessionTracer", None)

    # Parse and validate TOML config

    cfg: dict[str, Any] = {}
    eval_cfg: EvalConfig | None = None
    config_path: Path | None = None

    if config:
        config_path = Path(config)
    else:
        auto_configs = discover_eval_config_paths()
        if auto_configs:
            config_path = auto_configs[0]
            click.echo(f"Using eval config: {config_path}")

    if config_path:
        if _toml is None:
            raise TomlUnavailableError()
        if not config_path.exists():
            raise EvalConfigNotFoundError(str(config_path))
        try:
            data = config_path.read_bytes()
            parsed = _toml.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                section = parsed.get("eval")
                if section is None:
                    cfg = dict(parsed)
                elif isinstance(section, dict):
                    cfg = dict(section)
                else:
                    raise MissingEvalTableError()
        except Exception as exc:
            raise EvalConfigParseError(path=str(config_path), detail=str(exc)) from exc

    if cfg:
        try:
            normalized_cfg = validate_eval_options(cfg)
            normalized_cfg_dict = dict(normalized_cfg)
            eval_cfg = EvalConfig.from_dict(normalized_cfg_dict)
            cfg = normalized_cfg_dict
            click.echo(f"✓ Config validated: {len(eval_cfg.seeds)} seeds, model={eval_cfg.model}")
        except (ValueError, TypeError) as validation_error:
            raise InvalidEvalConfigError(detail=str(validation_error)) from validation_error
    else:
        cfg = {}

    # CLI args override config
    if eval_cfg:
        app_id = app_id or eval_cfg.app_id
    else:
        app_id = app_id or (cfg.get("app_id") if isinstance(cfg.get("app_id"), str) else None)  # type: ignore

    metadata_filters: dict[str, str] = {}
    if eval_cfg:
        metadata_filters.update(eval_cfg.metadata)
    else:
        cfg_metadata = cfg.get("metadata")
        if isinstance(cfg_metadata, dict):
            for key, value in cfg_metadata.items():
                metadata_filters[str(key)] = str(value)
        elif isinstance(cfg_metadata, list):
            for item in cfg_metadata:
                if isinstance(item, str) and "=" in item:
                    key, value = item.split("=", 1)
                    metadata_filters[key.strip()] = value.strip()

    for item in metadata or ():
        if "=" not in item:
            raise MetadataFilterFormatError(entry=item)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise MetadataFilterFormatError(entry=item)
        metadata_filters[key] = value

    metadata_sql_query: str | None = None
    if eval_cfg and eval_cfg.metadata_sql:
        metadata_sql_query = eval_cfg.metadata_sql
    else:
        cfg_metadata_sql = cfg.get("metadata_sql")
        if isinstance(cfg_metadata_sql, dict):
            metadata_sql_query = cfg_metadata_sql.get("query") or cfg_metadata_sql.get("sql")
        elif isinstance(cfg_metadata_sql, str):
            metadata_sql_query = cfg_metadata_sql

    if metadata_sql:
        metadata_sql_query = metadata_sql
    if metadata_sql_query is not None:
        metadata_sql_query = str(metadata_sql_query)

    trace_db_url: str | None = None
    trace_db = (trace_db or "").strip()
    if trace_db and trace_db.lower() not in {"none", "off", "disable"}:
        if "://" in trace_db:
            trace_db_url = trace_db
        else:
            trace_path = Path(trace_db).expanduser()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_db_url = f"sqlite+aiosqlite:///{trace_path}"
    trace_tracer: SessionTracer | None = None
    if trace_db_url and session_tracer_cls is not None:
        trace_tracer = cast(SessionTracer, session_tracer_cls(db_url=trace_db_url, auto_save=True))

    # Determine selection params (CLI takes precedence; TOML only fills unset model/seeds/env)
    if cfg.get("model") and not model:
        model = str(cfg["model"])  # type: ignore[index]
    if cfg.get("seeds") and seeds == "0,1,2,3,4":
        val = cfg["seeds"]
        if isinstance(val, list):
            with contextlib.suppress(Exception):
                seeds = ",".join(str(int(x)) for x in val)
        elif isinstance(val, str):
            seeds = val
        elif isinstance(val, int):
            seeds = str(val)
    if cfg.get("env_file") and not env_file:
        ef = cfg["env_file"]
        if isinstance(ef, str):
            env_file = (ef,)  # type: ignore[assignment]
        elif isinstance(ef, list):
            env_file = tuple(str(x) for x in ef)  # type: ignore[assignment]

    choice_for_env: AppChoice | None = None
    entry: TaskAppEntryType | None = None
    if task_app_url is None:
        choice_for_env = select_app_choice(app_id, purpose="eval")
        entry = choice_for_env.ensure_entry()

    env_paths: list[Path] = []
    if entry is not None:
        original_env_path = choice_for_env.path if choice_for_env is not None else None
        env_paths = determine_env_files(entry, env_file, original_path=original_env_path)
    else:
        if not env_file:
            raise click.ClickException("--env-file is required when using --url")
        for candidate in env_file:
            p = Path(candidate).expanduser()
            if not p.exists():
                raise click.ClickException(f"Env file not found: {p}")
            env_paths.append(p)

    click.echo("Using env file(s): " + ", ".join(str(p) for p in env_paths))
    load_env_files_into_process([str(Path(p)) for p in env_paths])

    if task_app_url is None:
        config = entry.config_factory()  # type: ignore[union-attr]
        # Help the type checker; runtime check also enforced in server.run_task_app
        if not isinstance(config, task_app_config_type):
            raise click.ClickException(
                "Invalid task app: config_factory did not return TaskAppConfig"
            )
        app = create_task_app(config)

    # Determine supported models
    inference_meta: dict[str, Any] = {}
    supported: list[str] = []
    seen_models: set[str] = set()

    def _add_supported_model(candidate: Any) -> None:
        if not candidate:
            return
        text = str(candidate).strip()
        if not text or text in seen_models:
            return
        supported.append(text)
        seen_models.add(text)

    if task_app_url is None:
        try:
            if hasattr(config, "base_task_info") and config.base_task_info:
                inf_obj = getattr(config.base_task_info, "inference", None)
                if inf_obj is not None:
                    if hasattr(inf_obj, "model_dump"):
                        inference_meta = dict(inf_obj.model_dump(exclude_none=True))  # type: ignore[attr-defined]
                    elif isinstance(inf_obj, dict):
                        inference_meta = dict(inf_obj)
        except Exception:
            inference_meta = {}
    else:
        try:
            import httpx as _hx

            headers = {}
            api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
            if api_key:
                headers["X-API-Key"] = api_key
            with _hx.Client(base_url=task_app_url, headers=headers, timeout=15.0) as c:
                info = c.get("/info").json()
            inf = info.get("inference") if isinstance(info, dict) else None
            if isinstance(inf, dict):
                inference_meta = dict(inf)
        except Exception:
            inference_meta = {}

    default_model = inference_meta.get("model")
    if isinstance(default_model, str):
        _add_supported_model(default_model)

    models_field = inference_meta.get("models")
    if isinstance(models_field, list):
        for candidate in models_field:
            _add_supported_model(candidate)

    supported_models = inference_meta.get("supported_models")
    if isinstance(supported_models, list):
        for candidate in supported_models:
            _add_supported_model(candidate)

    providers = inference_meta.get("providers")
    if isinstance(providers, list):
        if "openai" in providers:
            _add_supported_model("gpt-5")
        if "groq" in providers:
            _add_supported_model("groq:llama-3.1-70b-versatile")

    _add_supported_model("synth:qwen-0.6b")

    selected_model = model
    if not selected_model:
        if not supported:
            raise click.ClickException(
                "No supported models; supply --model or add base_task_info.inference.model"
            )
        click.echo("Select model to evaluate:")
        for idx, m in enumerate(supported, start=1):
            click.echo(f"  {idx}) {m}")
        choice_idx = click.prompt("Enter choice", type=click.IntRange(1, len(supported)))
        selected_model = supported[choice_idx - 1]

    try:
        seed_values = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except Exception as exc:
        raise SeedParseError(value=seeds) from exc

    import httpx

    headers = {}
    api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    # Precompute optional policy overrides from TOML
    policy_overrides: dict[str, Any] = {}
    try:
        # Accept [eval.policy] table or top-level keys for convenience
        if isinstance(cfg.get("policy"), dict):
            policy_overrides.update(dict(cfg["policy"]))
        # Back-compat: allow temperature/max_tokens at top level
        for k in (
            "temperature",
            "max_tokens",
            "reasoning_effort",
            "system_hint",
            "tool_choice",
            "inference_url",
        ):
            if k in cfg and k not in policy_overrides:
                policy_overrides[k] = cfg.get(k)
    except Exception:
        policy_overrides = {}

    raw_concurrency = cfg.get("concurrency")
    try:
        concurrency_limit = int(raw_concurrency) if raw_concurrency is not None else 1
    except Exception:
        concurrency_limit = 1
    if concurrency_limit <= 0:
        concurrency_limit = 1
    concurrency_limit = min(concurrency_limit, max(1, len(seed_values)))

    judge_specs: list[Any] = []

    def _register_judge(name_hint: str | None, judge_cfg: dict[str, Any]) -> None:
        if not judge_cfg:
            return
        judge_module = judge_cfg.get("module")
        judge_path = judge_cfg.get("path")
        judge_callable_name = judge_cfg.get("callable") or judge_cfg.get("function")
        if judge_module and judge_path:
            raise click.ClickException("Judge config cannot set both 'module' and 'path'")
        if not judge_module and not judge_path:
            raise click.ClickException("Judge config requires 'module' or 'path'")
        try:
            if judge_module:
                module = importlib.import_module(str(judge_module))
            else:
                path = Path(str(judge_path)).expanduser()
                if not path.exists():
                    raise click.ClickException(f"Judge module path not found: {path}")
                spec = importlib.util.spec_from_file_location(
                    f"_eval_judge_{path.stem}", path
                )
                if not spec or not spec.loader:
                    raise click.ClickException(f"Failed to load judge module from {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(f"Unable to load judge module: {exc}") from exc

        if judge_callable_name:
            try:
                judge_fn = getattr(module, str(judge_callable_name))
            except AttributeError as exc:
                raise click.ClickException(
                    f"Judge callable '{judge_callable_name}' not found in module"
                ) from exc
        else:
            if hasattr(module, "judge"):
                judge_fn = module.judge
            else:
                raise click.ClickException("Judge module must expose 'judge' callable")

        if not callable(judge_fn):
            raise click.ClickException("Judge callable is not callable")

        judge_kwargs = {
            k: v
            for k, v in judge_cfg.items()
            if k not in {"module", "path", "callable", "function", "name"}
        }
        display_name = str(
            judge_cfg.get("name")
            or name_hint
            or f"judge{len(judge_specs) + 1}"
        )
        judge_specs.append(judge_spec_cls(display_name, judge_fn, judge_kwargs))

    raw_judge_cfg = cfg.get("judge")
    if isinstance(raw_judge_cfg, dict) and raw_judge_cfg:
        direct_keys = {"module", "path", "callable", "function", "name"}
        has_direct_keys = any(key in raw_judge_cfg for key in direct_keys)
        nested_candidates = [
            (key, value)
            for key, value in raw_judge_cfg.items()
            if isinstance(value, dict)
        ]
        if has_direct_keys and not nested_candidates:
            _register_judge(None, raw_judge_cfg)
        else:
            for sub_name, sub_cfg in nested_candidates:
                _register_judge(sub_name, sub_cfg)

    raw_judges_list = cfg.get("judges")
    if isinstance(raw_judges_list, list):
        for _index, entry in enumerate(raw_judges_list, start=1):
            if isinstance(entry, dict):
                _register_judge(entry.get("name") or f"judge{len(judge_specs) + 1}", entry)

    records: list[dict[str, Any]] = []

    successes = 0
    failures = 0
    # Aggregate outcome stats across successful seeds
    outcome_sum: float = 0.0
    outcome_count: int = 0
    outcome_correct: int = 0

    def _build_task_rows(taskset: Any) -> dict[int, dict[str, Any]]:
        rows: dict[int, dict[str, Any]] = {}
        if not isinstance(taskset, dict):
            return rows

        scenario_ids = taskset.get("scenario_ids") or []
        loop_ids = taskset.get("loop_ids") or []
        thread_ids = taskset.get("thread_ids") or []
        difficulty_map = taskset.get("difficulty_map") or {}

        max_len = max(len(scenario_ids), len(loop_ids), len(thread_ids))
        for seed in range(max_len):
            scenario_id = scenario_ids[seed] if seed < len(scenario_ids) else None
            loop_id = loop_ids[seed] if seed < len(loop_ids) else None
            thread_id = thread_ids[seed] if seed < len(thread_ids) else None
            difficulty = None
            if isinstance(difficulty_map, dict):
                if scenario_id and scenario_id in difficulty_map:
                    difficulty = difficulty_map.get(scenario_id)
                elif str(seed) in difficulty_map:
                    difficulty = difficulty_map.get(str(seed))

            rows[seed] = {
                "seed": seed,
                "scenario_id": scenario_id,
                "loop_id": loop_id,
                "thread_id": thread_id,
                "difficulty": difficulty,
            }
        return rows

    def _apply_metadata_filters(
        rows: dict[int, dict[str, Any]], seeds_list: list[int], filters: dict[str, str]
    ) -> list[int]:
        if not filters:
            return seeds_list
        filtered: list[int] = []
        for seed in seeds_list:
            row = rows.get(seed)
            if not row:
                continue
            include = True
            for key, expected in filters.items():
                actual = row.get(key)
                if actual is None:
                    include = False
                    break
                if str(actual).lower() != expected.lower():
                    include = False
                    break
            if include:
                filtered.append(seed)
        return filtered

    def _apply_metadata_sql(
        rows: dict[int, dict[str, Any]], seeds_list: list[int], query: str
    ) -> list[int]:
        """Return seeds that satisfy an arbitrary SQL query.

        The query is executed against an in-memory SQLite table named `tasks`
        with columns (seed INTEGER, scenario_id TEXT, loop_id TEXT, thread_id TEXT, difficulty TEXT).
        Any rows whose `seed` value (or first column if `seed` is absent) appear in the result set are retained.
        """
        if not query:
            return seeds_list
        conn = sqlite3.connect(":memory:")
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE TABLE tasks (seed INTEGER, scenario_id TEXT, loop_id TEXT, thread_id TEXT, difficulty TEXT)"
            )
            insert_stmt = (
                "INSERT INTO tasks (seed, scenario_id, loop_id, thread_id, difficulty) VALUES (?,?,?,?,?)"
            )
            for seed in seeds_list:
                row = rows.get(seed, {})
                cur.execute(
                    insert_stmt,
                    [
                        seed,
                        row.get("scenario_id"),
                        row.get("loop_id"),
                        row.get("thread_id"),
                        row.get("difficulty"),
                    ],
                )

            result = cur.execute(query)
            fetched = result.fetchall()
            if not fetched:
                return []
            description = result.description or []
            col_names = [col[0] for col in description]
            seeds_out: list[int] = []
            for entry in fetched:
                value = entry[col_names.index("seed")] if "seed" in col_names else entry[0]
                try:
                    seeds_out.append(int(value))
                except Exception as exc:
                    raise MetadataSQLResultError(
                        query=query,
                        detail="non-integer value returned",
                    ) from exc
            seeds_set = set(seeds_out)
            return [seed for seed in seeds_list if seed in seeds_set]
        except sqlite3.Error as exc:
            raise MetadataSQLExecutionError(query=query, detail=str(exc)) from exc
        finally:
            conn.close()

    async def _run_eval() -> None:
        nonlocal successes, failures, outcome_sum, outcome_count, outcome_correct, records, seed_values

        if trace_tracer is not None and trace_tracer.db is None:
            await trace_tracer.initialize()

        if task_app_url is None:
            transport = httpx.ASGITransport(app=app)  # type: ignore[name-defined]
            async_client = httpx.AsyncClient(
                transport=cast(Any, transport),
                base_url="http://eval.local",
                timeout=300.0,
                follow_redirects=True,
                headers=headers,
            )
        else:
            async_client = httpx.AsyncClient(
                base_url=task_app_url,
                timeout=300.0,
                follow_redirects=True,
                headers=headers,
            )

        try:
            taskset_payload: dict[str, Any] | None = None
            try:
                task_info_response = await async_client.get("/task_info")
            except Exception:
                task_info_response = None
            if task_info_response is not None and task_info_response.status_code == 200:
                with contextlib.suppress(Exception):
                    payload_json = task_info_response.json()
                if isinstance(payload_json, dict) and "taskset" in payload_json:
                    taskset_payload = payload_json.get("taskset")
                    if not isinstance(taskset_payload, dict):
                        taskset_payload = None
                elif isinstance(payload_json, dict):
                    taskset_payload = payload_json

            available_seeds = list(seed_values)
            if metadata_sql_query or metadata_filters:
                if not taskset_payload:
                    raise TaskInfoUnavailableError()
                rows = _build_task_rows(taskset_payload)
                if metadata_sql_query:
                    available_seeds = _apply_metadata_sql(rows, available_seeds, metadata_sql_query)
                if metadata_filters:
                    available_seeds = _apply_metadata_filters(rows, available_seeds, metadata_filters)
                if not available_seeds:
                    raise NoSeedsMatchedError()
                seed_values = available_seeds

            semaphore = asyncio.Semaphore(concurrency_limit)

            async def _run_seed(seed_val: int) -> None:
                nonlocal successes, failures, outcome_sum, outcome_count, outcome_correct, records
                # Read env_name and policy_name from config if available
                env_name = cfg.get("env_name") or (cfg.get("env", {}).get("env_name") if isinstance(cfg.get("env"), dict) else None)
                policy_name = cfg.get("policy_name") or (cfg.get("policy", {}).get("policy_name") if isinstance(cfg.get("policy"), dict) else None)
                env_config_overrides = cfg.get("env_config", {}) if isinstance(cfg.get("env_config"), dict) else {}
                policy_config_overrides = cfg.get("policy_config", {}) if isinstance(cfg.get("policy_config"), dict) else {}
                
                # Debug: print config parsing
                if seed_val == 0:
                    click.echo(f"[DEBUG] env_name from config: {env_name}")
                    click.echo(f"[DEBUG] policy_name from config: {policy_name}")
                
                # Generate default ops sequence if not provided
                max_llm_calls = policy_config_overrides.get("max_llm_calls", 10)
                ops_list = cfg.get("ops", [])
                if not ops_list:
                    # Generate default "agent, env" pairs for max_llm_calls
                    ops_list = ["agent", "env"] * int(max_llm_calls)
                
                body = {
                    "run_id": str(uuid.uuid4()),
                    "env": {"config": {"split": split, "index": seed_val, **env_config_overrides}, "seed": seed_val},
                    "policy": {
                        "policy_name": policy_name or selected_model,
                        "config": {"model": selected_model, **policy_overrides, **policy_config_overrides},
                    },
                    "ops": ops_list,
                    "record": {
                        "return_trace": cfg.get("return_trace", True),
                        "trace_format": cfg.get("trace_format", "structured"),
                    },
                    "mode": "eval",  # RolloutMode.EVAL: use inference URLs as-is, no transformations
                }
                if env_name:
                    env_section = body.get("env")
                    if isinstance(env_section, dict):
                        env_section["env_name"] = env_name
                    else:
                        body["env"] = {"env_name": env_name}

                # Debug: print the body being sent
                if seed_val == 0:
                    click.echo(f"[DEBUG] rollout body env: {body['env']}")
                    click.echo(f"[DEBUG] rollout body policy: {body['policy']}")
                    click.echo(f"[DEBUG] rollout body mode: {body.get('mode', 'NOT SET')}")
                rollout_elapsed: float | None = None
                rollout_start = time.perf_counter()
                try:
                    import logging
                    _log = logging.getLogger(__name__)
                    _log.info(f"[EVAL_BODY_DEBUG] Sending body with mode={body.get('mode')}")
                    async with semaphore:
                        response = await async_client.post("/rollout", json=body)
                    rollout_elapsed = time.perf_counter() - rollout_start
                except Exception as exc:
                    failures += 1
                    click.echo(f"seed={seed_val} error={exc}")
                    return

                ok = 200 <= response.status_code < 300
                if ok:
                    successes += 1
                else:
                    failures += 1

                summary = [f"seed={seed_val}", f"status={response.status_code}"]
                data: Any
                try:
                    data = response.json()
                except Exception:
                    data = None
                
                # Debug: print validation errors
                if response.status_code == 422 and data:
                    click.echo(f"[DEBUG] 422 Validation Error: {data}")

                metrics: dict[str, Any] | None = None
                completion: str | None = None
                prompt_index: int | None = None
                prompt_text: str | None = None
                task_id: str | None = None
                task_split: str | None = None
                task_rubric_id: str | None = None

                trace_namespace: dict[str, Any] | None = None
                session_trace_dict: dict[str, Any] | None = None

                if isinstance(data, dict):
                    import logging
                    _logger = logging.getLogger(__name__)
                    _logger.info(f"[EVAL_DEBUG] Response data keys: {list(data.keys())}")
                    if "detail" in data:
                        _logger.error(f"[EVAL_DEBUG] Task app returned error: {data['detail']}")
                    trace_namespace = data.get("trace")
                    _logger.info(f"[EVAL_DEBUG] trace_namespace type: {type(trace_namespace)}, value: {trace_namespace if not isinstance(trace_namespace, dict) else 'dict with keys: ' + str(list(trace_namespace.keys()) if trace_namespace else 'None')}")
                    if not isinstance(trace_namespace, dict):
                        raise RuntimeError(
                            "The 'synth-ai eval' command requires trace payloads in rollout responses. "
                            "Ensure the rollout request includes 'trace_format': 'structured' and 'return_trace': true, "
                            "and that task app tracing is enabled (TASKAPP_TRACING_ENABLED=1). "
                            "Note: This is specific to the eval command - general rollout endpoints don't require traces."
                        )
                    # Handle both "compact" and "full" trace formats:
                    # - compact: trace_namespace contains {session_id, metadata, ...}
                    # - full: trace_namespace IS the full session_trace dict
                    session_trace_dict = trace_namespace.get("session_trace")
                    if not isinstance(session_trace_dict, dict):
                        # If no session_trace key, assume "full" format where trace itself is the session_trace
                        if "session_id" in trace_namespace:
                            session_trace_dict = trace_namespace
                        else:
                            raise RuntimeError(
                                "The 'synth-ai eval' command requires 'session_trace' in the trace payload or a valid full trace format. "
                                "Ensure the task app is using tracing_v3 and returning structured trace data."
                            )
                    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else None
                    if metrics:
                        mean_return = metrics.get("mean_return") or metrics.get("total_reward")
                        outcome = metrics.get("outcome_score")
                        if mean_return is not None:
                            summary.append(f"mean_return={mean_return}")
                        if outcome is not None:
                            summary.append(f"outcome={outcome}")
                            try:
                                val = float(outcome)
                                outcome_sum += val
                                outcome_count += 1
                                if val >= 0.5:
                                    outcome_correct += 1
                            except Exception:
                                pass
                    trajs = (
                        data.get("trajectories")
                        if isinstance(data.get("trajectories"), list)
                        else None
                    )
                    if trajs:
                        first = trajs[0] if trajs else None
                        steps = first.get("steps") if isinstance(first, dict) else None
                        if isinstance(steps, list) and steps:
                            step0 = steps[0]
                            tool_calls = step0.get("tool_calls") or step0.get("tools") or []
                            if isinstance(tool_calls, list):
                                summary.append(f"tool_calls={len(tool_calls)}")
                            obs = step0.get("obs") if isinstance(step0, dict) else None
                            if isinstance(obs, dict):
                                idx_val = obs.get("prompt_index")
                                if isinstance(idx_val, int):
                                    prompt_index = idx_val
                                prompt_raw = obs.get("prompt")
                                if isinstance(prompt_raw, str):
                                    prompt_text = prompt_raw
                                if task_id is None:
                                    candidate_id = obs.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = obs.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = obs.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                        final = first.get("final") if isinstance(first, dict) else None
                        if isinstance(final, dict):
                            final_obs = final.get("observation")
                            if isinstance(final_obs, dict):
                                comp_val = final_obs.get("completion")
                                if isinstance(comp_val, str):
                                    completion = comp_val
                                if task_id is None:
                                    candidate_id = final_obs.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = final_obs.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = final_obs.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                            final_info = final.get("info")
                            if isinstance(final_info, dict):
                                if task_id is None:
                                    candidate_id = final_info.get("task_id")
                                    if isinstance(candidate_id, str) and candidate_id:
                                        task_id = candidate_id
                                if task_split is None:
                                    candidate_split = final_info.get("task_split")
                                    if isinstance(candidate_split, str) and candidate_split:
                                        task_split = candidate_split
                                if task_rubric_id is None:
                                    candidate_rid = final_info.get("task_rubric_id")
                                    if isinstance(candidate_rid, str) and candidate_rid:
                                        task_rubric_id = candidate_rid
                    if task_id:
                        summary.append(f"task_id={task_id}")
                    click.echo(" ".join(summary))
                    with contextlib.suppress(Exception):
                        click.echo(json.dumps(data, indent=2))
                else:
                    click.echo(" ".join(summary))

                official_score = None
                if isinstance(metrics, dict):
                    for key in ("mean_return", "total_reward", "outcome_score"):
                        val = metrics.get(key)
                        if isinstance(val, int | float):
                            official_score = float(val)
                            break
                if official_score is None and isinstance(data, dict):
                    try:
                        reward_val = data["trajectories"][0]["steps"][0].get("reward")
                        if isinstance(reward_val, int | float):
                            official_score = float(reward_val)
                    except Exception:
                        pass

                if official_score is not None:
                    if official_score < 0.0:
                        official_score = 0.0
                    elif official_score > 1.0:
                        official_score = min(1.0, official_score)

                judge_scores: dict[str, float | None] = {}
                judges_timings: dict[str, float | None] = {}
                timings: dict[str, Any] = {
                    "rollout_s": rollout_elapsed,
                    "judges": judges_timings,
                }
                if judge_specs:
                    for spec in judge_specs:
                        score_value: float | None = None
                        judge_elapsed: float | None = None
                        # Run judges for all tasks (text-based and trajectory-based)
                        # Text-based tasks have completion, trajectory-based tasks use response
                        judge_payload = {
                            "seed": seed_val,
                            "prompt_index": prompt_index,
                            "prompt": prompt_text,
                            "completion": completion,
                            "metrics": metrics,
                            "response": data,
                            "trace": trace_namespace,
                        }
                        try:
                            judge_start = time.perf_counter()
                            result = spec.fn(judge_payload, **spec.kwargs)
                            judge_elapsed = time.perf_counter() - judge_start
                            if isinstance(result, int | float):
                                score_value = float(result)
                        except Exception as exc:
                            if judge_elapsed is None:
                                judge_elapsed = time.perf_counter() - judge_start
                            click.echo(f"seed={seed_val} judge[{spec.name}]_error={exc}")
                        judges_timings[spec.name] = judge_elapsed
                        judge_scores[spec.name] = score_value

                if trace_tracer is not None and trace_namespace:
                    storage_metadata = {
                        "eval_seed": seed_val,
                        "prompt_index": prompt_index,
                        "task_id": task_id,
                        "task_split": task_split,
                        "task_rubric_id": task_rubric_id,
                        "official_score": official_score,
                        "judge_scores": judge_scores,
                        "model": selected_model,
                        "prompt": prompt_text,
                        "completion": completion,
                    }
                    if store_trace is not None:
                        await store_trace(trace_tracer, trace_namespace, storage_metadata)

                records.append(
                    {
                        "seed": seed_val,
                        "prompt_index": prompt_index,
                        "task_id": task_id,
                        "task_split": task_split,
                        "task_rubric_id": task_rubric_id,
                        "official_score": official_score,
                        "judge_scores": judge_scores,
                        "timings": timings,
                    }
                )

            await asyncio.gather(*[_run_seed(seed_val) for seed_val in seed_values])
        finally:
            await async_client.aclose()

    try:
        asyncio.run(_run_eval())
    finally:
        if trace_tracer is not None and trace_tracer.db is not None:
            asyncio.run(trace_tracer.db.close())

    click.echo(
        f"Eval complete: {successes} ok, {failures} failed; model={selected_model}, split={split}"
    )

    if outcome_count > 0:
        mean_outcome = outcome_sum / float(outcome_count)
        frac_right = outcome_correct / float(outcome_count)
        click.echo(
            f"Outcome summary: correct={outcome_correct}/{outcome_count} ({frac_right:.2%}), mean_outcome={mean_outcome:.3f}"
        )

    if records:
        judge_specs = judge_specs or []  # ensure iterable
        official_scores = [
            r["official_score"] for r in records if r["official_score"] is not None
        ]
        if official_scores:
            click.echo(f"  Official mean: {sum(official_scores) / len(official_scores):.3f}")
        else:
            click.echo("  Official mean: n/a")

        for spec in judge_specs:
            spec_scores = [
                record["judge_scores"].get(spec.name)
                for record in records
                if record["judge_scores"].get(spec.name) is not None
            ]
            if spec_scores:
                mean_spec = sum(spec_scores) / len(spec_scores)
                click.echo(f"  [{spec.name}] mean: {mean_spec:.3f}")
            else:
                click.echo(f"  [{spec.name}] mean: n/a")

            paired = [
                (
                    record["official_score"],
                    record["judge_scores"].get(spec.name),
                )
                for record in records
                if record["official_score"] is not None
                and record["judge_scores"].get(spec.name) is not None
            ]
            if len(paired) >= 2:
                corr = pearson(
                    [p[0] for p in paired if p[0] is not None],
                    [p[1] for p in paired if p[1] is not None],
                )
                if corr is not None:
                    click.echo(f"    Pearson r: {corr:.3f}")
                else:
                    click.echo("    Pearson r: undefined (zero variance)")
            else:
                click.echo("    Pearson r: n/a (need ≥2 paired scores)")

        header = ["Seed", "Prompt", "Official"]
        header.extend(spec.name for spec in judge_specs)
        rows: list[list[str]] = []
        for record in sorted(records, key=lambda r: (r["seed"], r.get("prompt_index") or -1)):
            seed_val = str(record["seed"])
            prompt_idx = (
                str(record["prompt_index"])
                if record["prompt_index"] is not None
                else "-"
            )
            official_val = (
                f"{record['official_score']:.3f}"
                if record["official_score"] is not None
                else "-"
            )
            row = [seed_val, prompt_idx, official_val]
            for spec in judge_specs:
                score_val = record["judge_scores"].get(spec.name)
                row.append(f"{score_val:.3f}" if isinstance(score_val, int | float) else "-")
            rows.append(row)

        widths = [len(col) for col in header]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        click.echo("")
        click.echo("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(header)))
        click.echo("  ".join("-" * widths[idx] for idx in range(len(header))))
        for row in rows:
            click.echo("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))



command = eval_command


def get_command() -> click.Command:
    """Return the Click command implementing task-app evaluation."""
    return command


def format_eval_error(err: EvalCliError) -> str:
    if isinstance(err, TomlUnavailableError):
        hint = err.hint or "Install tomli or use Python 3.11+."
        return f"TOML parser not available. {hint}"
    if isinstance(err, EvalConfigNotFoundError):
        return f"Eval config not found: {err.path}"
    if isinstance(err, EvalConfigParseError):
        return f"Failed to parse TOML '{err.path}': {err.detail}"
    if isinstance(err, MissingEvalTableError):
        return "Config must contain an [eval] table."
    if isinstance(err, InvalidEvalConfigError):
        return f"Invalid eval config: {err.detail}"
    if isinstance(err, SeedParseError):
        return f"Unable to parse seeds from '{err.value}'. Provide comma-separated integers."
    if isinstance(err, MetadataFilterFormatError):
        return f"Metadata filter '{err.entry}' must be key=value."
    if isinstance(err, TaskInfoUnavailableError):
        return "Task metadata filters require the task app to expose /task_info metadata."
    if isinstance(err, NoSeedsMatchedError):
        hint = err.hint or "Adjust the metadata filters or seed list."
        return f"No seeds match the provided metadata filters. {hint}"
    if isinstance(err, MetadataSQLExecutionError):
        return f"Failed to execute metadata SQL query '{err.query}': {err.detail}"
    if isinstance(err, MetadataSQLResultError):
        return f"metadata SQL query '{err.query}' must return integer seed values ({err.detail})"
    return str(err)
