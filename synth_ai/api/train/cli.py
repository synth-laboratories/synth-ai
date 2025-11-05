from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import click

try:
    _config_module = cast(
        Any, importlib.import_module("synth_ai.config.base_url")
    )
    get_backend_from_env = cast(Callable[[], str], _config_module.get_backend_from_env)
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load backend configuration helpers") from exc

from synth_ai.streaming import (
    CLIHandler,
    JobStreamer,
    LossCurveHandler,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)

from .builders import build_prompt_learning_payload, build_rl_payload, build_sft_payload
from .config_finder import discover_configs, prompt_for_config
from .env_resolver import KeySpec, resolve_env
from .task_app import check_task_app_health
from .utils import (
    REPO_ROOT,
    TrainError,
    ensure_api_base,
    http_get,
    http_post,
    limit_jsonl_examples,
    mask_value,
    post_multipart,
    preview_json,
    sleep,
    validate_sft_jsonl,
)

# Constants for prompt learning event types
_PROMPT_LEARNING_EVENT_BEST_PROMPT = "prompt.learning.best.prompt"
_PROMPT_LEARNING_EVENT_FINAL_RESULTS = "prompt.learning.final.results"
_PROMPT_LEARNING_EVENT_VALIDATION_SCORED = "prompt.learning.validation.scored"
_PROMPT_LEARNING_EVENT_GEPA_COMPLETE = "prompt.learning.gepa.complete"
_PROMPT_LEARNING_EVENT_MIPRO_COMPLETE = "prompt.learning.mipro.complete"

# Constants for formatting
_MAX_TEXT_REPLACEMENTS_DISPLAY = 3  # Max number of text replacements to show in output
_RESULTS_FILE_MAX_EVENTS = 10000  # Max events to fetch for results file generation


def _format_text_replacements(obj: dict[str, Any] | None, max_display: int = _MAX_TEXT_REPLACEMENTS_DISPLAY) -> list[str]:
    """Extract and format text replacements from a candidate object.
    
    Args:
        obj: Candidate object dictionary containing text_replacements
        max_display: Maximum number of replacements to display
        
    Returns:
        List of formatted lines showing role and replacement text
    """
    lines = []
    if not obj or not isinstance(obj, dict):
        return lines
    
    text_replacements = obj.get("text_replacements", [])
    if not text_replacements or not isinstance(text_replacements, list):
        return lines
    
    for replacement in text_replacements[:max_display]:
        if isinstance(replacement, dict):
            new_text = replacement.get("new_text", "")
            role = replacement.get("apply_to_role", "system")
            if new_text:
                lines.append(f"  [{role.upper()}]: {new_text}")
                lines.append("")
    
    return lines


def _discover_dataset_candidates(
    config_path: Path, limit: int = 50, timeout: float = 10.0
) -> list[Path]:
    root = config_path.parent
    parent = root.parent
    cwd = Path.cwd()

    search_dirs: list[Path] = [
        root,
        root / "datasets",
        parent,
        parent / "datasets",
        parent / "ft_data",
        cwd,
        cwd / "datasets",
        cwd / "ft_data",
        REPO_ROOT / "datasets",
        REPO_ROOT / "ft_data",
        REPO_ROOT / "traces",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()
    start = time.monotonic()
    timed_out = False
    for directory in search_dirs:
        if timed_out or time.monotonic() - start > timeout:
            timed_out = True
            break
        if not directory.exists() or not directory.is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
            if time.monotonic() - start > timeout:
                timed_out = True
                break
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.stat().st_size == 0:
                continue
            candidates.append(resolved)
            if len(candidates) >= limit:
                return candidates
    return candidates


def prompt_for_dataset(config_path: Path) -> Path:
    candidates = _discover_dataset_candidates(config_path)
    while True:
        if candidates:
            click.echo("Select dataset JSONL file:")
            for idx, candidate in enumerate(candidates, start=1):
                click.echo(f"  {idx}) {candidate}")
            click.echo("  m) Enter path manually")
            click.echo("  0) Abort")
            choice = click.prompt("Choice", default="m").strip().lower()
            if choice == "0":
                raise click.ClickException("Aborted by user")
            if choice in {"m", "manual"}:
                selected = _prompt_manual_dataset()
            else:
                try:
                    idx = int(choice)
                except ValueError:
                    click.echo("Invalid selection; try again")
                    continue
                if idx < 1 or idx > len(candidates):
                    click.echo("Invalid selection; try again")
                    continue
                selected = candidates[idx - 1]
        else:
            selected = _prompt_manual_dataset()

        if selected.exists() and selected.suffix == ".jsonl":
            return selected.resolve()
        click.echo("File not found or not a .jsonl; please try again.")


def _prompt_manual_dataset() -> Path:
    manual = click.prompt("Enter dataset JSONL path", type=str).strip()
    return Path(manual).expanduser()


def _default_backend() -> str:
    """Resolve backend URL with proper production default."""
    # Check explicit override first
    explicit = os.getenv("BACKEND_BASE_URL", "").strip()
    if explicit:
        return explicit
    # Use standard resolution logic
    base, _ = get_backend_from_env()
    return f"{base}/api" if not base.endswith("/api") else base


_DEFAULT_SFT_HIDDEN_EVENTS = {
    "sft.created",
    "sft.pricing.check.requested",
    "sft.pricing.check.allowed",
    "sft.stage",
    "snapshot.fetch",
    "hatchet.preflight",
    "hatchet.submission.attempt",
    "hatchet.submission.result",
    "sft.running",
    "sft.status",
    "sft.worker.alive",
    "sft.dispatch.selected",
    "sft.config.prepared",
    "sft.strategy.selected",
    "sft.training.args",
}

_DEFAULT_RL_HIDDEN_SUBSTRINGS = {"modal", "hatchet"}

_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS = {
    "prompt.learning.policy.tokens",
}


def _build_stream_components(
    stream_format: str,
    *,
    hidden_event_types: set[str] | None = None,
    hidden_event_substrings: set[str] | None = None,
) -> tuple[StreamConfig, list]:
    """Return stream configuration and handlers for the requested format."""
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "sft.progress",
                "sft.training.started",
                "sft.training.finish",
                "sft.validation.summary",
                "rl.train.step",
                "rl.train.started",
                "rl.train.completed",
                "workflow.completed",
                "workflow.failed",
            },
            metric_names={"train.loss"},
        )
        handlers = [LossCurveHandler()]
    else:
        config = StreamConfig.default()
        handlers = [
            CLIHandler(
                hidden_event_types=hidden_event_types or set(),
                hidden_event_substrings=hidden_event_substrings or set(),
            )
        ]
    return config, handlers


@click.command("train")
@click.option(
    "--config",
    "config_paths",
    multiple=True,
    type=click.Path(),
    help="Path to training TOML (repeatable)",
)
@click.option("--type", "train_type", type=click.Choice(["auto", "rl", "sft", "prompt_learning"]), default="auto")
@click.option(
    "--env-file",
    "env_files",
    multiple=True,
    type=click.Path(),
    help=".env file(s) to preload (skips selection prompt)",
)
@click.option("--task-url", default=None, help="Override task app base URL (RL only)")
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(),
    default=None,
    help="Override dataset JSONL path (SFT)",
)
@click.option("--backend", default=_default_backend, help="Backend base URL")
@click.option("--model", default=None, help="Override model identifier")
@click.option(
    "--allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=True,
    default=None,
    help="Allow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option(
    "--no-allow-experimental",
    "allow_experimental",
    is_flag=True,
    flag_value=False,
    help="Disallow experimental models (overrides SDK_EXPERIMENTAL env)",
)
@click.option("--idempotency", default=None, help="Idempotency-Key header for job creation")
@click.option("--dry-run", is_flag=True, hidden=True, help="Deprecated: no-op")
@click.option("--poll/--no-poll", default=True, help="Poll job status until terminal state")
@click.option(
    "--poll-timeout", default=3600.0, type=float, help="Maximum seconds to poll before timing out"
)
@click.option("--poll-interval", default=5.0, type=float, help="Seconds between poll attempts")
@click.option(
    "--stream-format",
    type=click.Choice(["cli", "chart"]),
    default="cli",
    show_default=True,
    help="Streaming output style (cli = line updates, chart = live loss panel)",
)
@click.option(
    "--examples",
    "examples_limit",
    type=int,
    default=None,
    help="Limit SFT training to the first N examples",
)
def train_command(
    config_paths: tuple[str, ...],
    train_type: str,
    env_files: tuple[str, ...],
    task_url: str | None,
    dataset_path: str | None,
    backend: str,
    model: str | None,
    allow_experimental: bool | None,
    idempotency: str | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
    examples_limit: int | None,
) -> None:
    """Interactive launcher for RL / SFT / Prompt Learning jobs."""

    candidates = discover_configs(
        list(config_paths), requested_type=train_type if train_type != "auto" else None
    )
    selection = prompt_for_config(
        candidates,
        requested_type=train_type if train_type != "auto" else None,
        allow_autoselect=bool(config_paths),
    )

    effective_type = train_type if train_type != "auto" else selection.train_type
    if effective_type not in {"rl", "sft", "prompt_learning"}:
        effective_type = click.prompt(
            "Detected config type is ambiguous. Enter type", type=click.Choice(["rl", "sft", "prompt_learning"])
        )

    cfg_path = selection.path
    click.echo(f"Using config: {cfg_path} ({effective_type})")

    required_keys: list[KeySpec] = []
    if effective_type == "rl" or effective_type == "prompt_learning":
        required_keys.append(KeySpec("SYNTH_API_KEY", "Synth API key for backend"))
        required_keys.append(
            KeySpec(
                "ENVIRONMENT_API_KEY",
                "Environment API key for task app",
                allow_modal_secret=True,
                modal_secret_pattern="env",
            )
        )
        required_keys.append(
            KeySpec(
                "TASK_APP_URL",
                "Task app base URL",
                secret=False,
                allow_modal_app=True,
                optional=bool(task_url),
            )
        )
    else:  # sft
        required_keys.append(KeySpec("SYNTH_API_KEY", "Synth API key for backend"))

    env_path, env_values = resolve_env(
        config_path=cfg_path,
        explicit_env_paths=env_files,
        required_keys=required_keys,
    )

    missing_keys = [
        spec.name
        for spec in required_keys
        if not spec.optional and not (env_values.get(spec.name) or os.environ.get(spec.name))
    ]
    if missing_keys:
        try:
            _task_apps_module = cast(
                Any, importlib.import_module("synth_ai.cli.task_apps")
            )
            _interactive_fill_env = cast(
                Callable[[Path], Path | None], _task_apps_module._interactive_fill_env
            )
        except Exception as exc:  # pragma: no cover - protective fallback
            raise click.ClickException(f"Unable to prompt for env values: {exc}") from exc

        target_dir = cfg_path.parent
        generated = _interactive_fill_env(target_dir / ".env")
        if generated is None:
            raise click.ClickException("Required environment values missing; aborting.")
        env_path, env_values = resolve_env(
            config_path=cfg_path,
            explicit_env_paths=(str(generated),),
            required_keys=required_keys,
        )
    click.echo(f"Using env file: {env_path}")

    synth_key = env_values.get("SYNTH_API_KEY") or os.environ.get("SYNTH_API_KEY")
    if not synth_key:
        raise click.ClickException("SYNTH_API_KEY required")

    backend_base = ensure_api_base(backend)
    click.echo(f"Backend base: {backend_base} (key {mask_value(synth_key)})")

    if effective_type == "rl":
        handle_rl(
            cfg_path=cfg_path,
            backend_base=backend_base,
            synth_key=synth_key,
            task_url_override=task_url,
            model_override=model,
            idempotency=idempotency,
            allow_experimental=allow_experimental,
            dry_run=dry_run,
            poll=poll,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            stream_format=stream_format,
        )
    elif effective_type == "prompt_learning":
        handle_prompt_learning(
            cfg_path=cfg_path,
            backend_base=backend_base,
            synth_key=synth_key,
            task_url_override=task_url,
            allow_experimental=allow_experimental,
            dry_run=dry_run,
            poll=poll,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            stream_format=stream_format,
        )
    else:
        dataset_override_path = Path(dataset_path).expanduser().resolve() if dataset_path else None
        handle_sft(
            cfg_path=cfg_path,
            backend_base=backend_base,
            synth_key=synth_key,
            dataset_override=dataset_override_path,
            allow_experimental=allow_experimental,
            dry_run=dry_run,
            poll=poll,
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            stream_format=stream_format,
            examples_limit=examples_limit,
        )


def _wait_for_training_file(
    backend_base: str, api_key: str, file_id: str, *, timeout: float = 10.0
) -> None:
    """Wait for training file to be visible after upload.
    
    Reduced from 120s to 10s because:
    - POST response already confirms file is uploaded
    - Backend now forces read-your-writes consistency  
    - By job creation time, replica lag has resolved
    - Quick sanity check only, not critical path
    """
    url = f"{backend_base.rstrip('/')}/files/{file_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0.0
    interval = 2.0
    first_check = True
    while True:
        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = {}
            status = str(
                data.get("status") or data.get("state") or data.get("storage_state") or "ready"
            ).lower()
            if first_check:
                click.echo(f"File uploaded successfully (id={file_id}, status={status})")
                first_check = False
            if status in {"ready", "uploaded", "stored", "complete"}:
                click.echo(f"✓ Training file ready (status={status})")
                return
            # Show progress for processing states
            if status in {"processing", "pending", "validating"}:
                click.echo(
                    f"  Waiting for file processing... (status={status}, {elapsed:.0f}s elapsed)"
                )
        elif resp.status_code == 404:
            # Keep polling; object may not be visible yet
            if first_check:
                click.echo(f"Waiting for file {file_id} to become visible...")
                first_check = False
        elif resp.status_code in {401, 403}:
            # Auth errors won't resolve by polling - fail immediately
            try:
                error_body = resp.json()
            except json.JSONDecodeError:
                error_body = resp.text[:400]
            click.echo("\n[ERROR] Authentication failed when checking training file:")
            click.echo(f"  URL: {url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {error_body}")
            click.echo(f"  API key: {mask_value(api_key)}")
            raise click.ClickException(
                f"Authentication error ({resp.status_code}). "
                "Check that your SYNTH_API_KEY is valid and has permission to access this organization's files."
            )
        else:
            # Other errors - show details but keep polling
            try:
                error_body = resp.json()
            except json.JSONDecodeError:
                error_body = resp.text[:400]
            click.echo(f"[WARN] Unexpected response checking file {file_id}:")
            click.echo(f"  URL: {url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {error_body}")

        if elapsed >= timeout:
            raise click.ClickException(
                f"Training file {file_id} not ready after {timeout:.0f}s (last status: {resp.status_code})"
            )
        sleep(interval)
        elapsed += interval


def handle_rl(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    task_url_override: str | None,
    model_override: str | None,
    idempotency: str | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
) -> None:
    overrides: dict[str, Any] = {
        "backend": backend_base,
        "task_url": task_url_override,
        "model": model_override,
    }
    build = build_rl_payload(
        config_path=cfg_path,
        task_url=task_url_override or os.environ.get("TASK_APP_URL", ""),
        overrides=overrides,
        idempotency=idempotency,
        allow_experimental=allow_experimental,
    )

    # Backend-side verification: try ALL org environment keys against /health and /task_info
    verify_url = f"{backend_base}/rl/verify_task_app"
    verify_headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    try:
        vresp = http_post(
            verify_url, headers=verify_headers, json_body={"endpoint_base_url": build.task_url}
        )
        try:
            parsed_json = vresp.json()
        except json.JSONDecodeError:
            parsed_json = None

        if isinstance(parsed_json, Mapping):
            vjs: dict[str, Any] = dict(parsed_json)
        else:
            vjs = {
                "status": vresp.status_code,
                "text": (vresp.text or "")[:400],
            }
            if parsed_json is not None:
                vjs["body"] = parsed_json
    except Exception as _ve:
        raise click.ClickException(
            f"Task app verification call failed: {type(_ve).__name__}: {_ve}"
        ) from _ve
    if vresp.status_code is not None and vresp.status_code >= 400:
        click.echo("Task app verification error:\n" + preview_json(vjs, limit=800))
        raise click.ClickException(f"Verification failed with status {vresp.status_code}")
    if not bool(vjs.get("any_ok")):
        click.echo("Task app verification failed; no auth combination succeeded. Full report:")
        click.echo(preview_json(vjs, limit=1200))
        raise click.ClickException("Task app verification failed (auth)")
    else:
        # Print concise summary
        try:
            cands = vjs.get("candidates_first15") or []
            attempts_raw = vjs.get("attempts")
            attempts: list[Mapping[str, Any]] = (
                [a for a in attempts_raw if isinstance(a, Mapping)]
                if isinstance(attempts_raw, list)
                else []
            )
            statuses = [attempt.get("status") for attempt in attempts]
            click.echo(f"Verification OK (candidates={cands}, statuses={statuses})")
        except (KeyError, ValueError, AttributeError):
            # Parsing verification summary failed, but verification itself succeeded
            click.echo("Verification OK")

    env_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not env_key:
        raise click.ClickException("ENVIRONMENT_API_KEY required for RL flow")

    click.echo("Performing task app health check…")
    health = check_task_app_health(build.task_url, env_key)
    if not health.ok:
        click.echo(f"Task app health check failed: {health.detail}")
        raise click.ClickException("Aborting due to failing health check")
    else:
        click.echo("Task app healthy")

    create_url = f"{backend_base}/rl/jobs"
    headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    if build.idempotency:
        headers["Idempotency-Key"] = build.idempotency

    click.echo(f"POST {create_url}")
    click.echo("Payload preview:\n" + preview_json(build.payload, limit=800))

    resp = http_post(create_url, headers=headers, json_body=build.payload)
    try:
        js = resp.json()
    except json.JSONDecodeError as e:
        click.echo(f"⚠️  Failed to parse JSON response: {e}")
        js = {"status": resp.status_code, "text": resp.text[:400]}
    click.echo(f"Response {resp.status_code}: {preview_json(js, limit=400)}")
    if resp.status_code not in (200, 201):
        raise click.ClickException("Job creation failed")
    job_id = js.get("job_id") or js.get("id")
    if not job_id:
        raise click.ClickException("Response missing job id")

    if not poll:
        click.echo(f"Created job {job_id} (polling disabled)")
        return

    click.echo("\n=== Streaming Job Progress ===")
    
    # Enable metrics for prompt learning
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [LossCurveHandler()]
        click.echo("Using live chart (metric=gepa.transformation.mean_score)")
    else:
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [CLIHandler(hidden_event_substrings=_DEFAULT_RL_HIDDEN_SUBSTRINGS)]
    
    streamer = JobStreamer(
        base_url=backend_base,
        api_key=synth_key,
        job_id=job_id,
        endpoints=StreamEndpoints.rl(job_id),
        config=config,
        handlers=handlers,
        interval_seconds=poll_interval,
        timeout_seconds=poll_timeout,
    )
    final_status = asyncio.run(streamer.stream_until_terminal())
    click.echo(f"Final status: {final_status.get('status', 'unknown')}")
    click.echo(preview_json(final_status, limit=600))


def handle_sft(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    dataset_override: Path | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
    examples_limit: int | None,
) -> None:
    dataset_path = dataset_override

    while True:
        try:
            build = build_sft_payload(
                config_path=cfg_path,
                dataset_override=dataset_path,
                allow_experimental=allow_experimental,
            )
            break
        except TrainError as exc:
            click.echo(str(exc))
            dataset_path = prompt_for_dataset(cfg_path)

    limited_path: Path | None = None

    try:
        if examples_limit is not None:
            limited_path = limit_jsonl_examples(build.train_file, examples_limit)
            click.echo(
                f"Using first {examples_limit} examples from {build.train_file} -> {limited_path}"
            )
            build.train_file = limited_path

        click.echo("Validating training dataset…")
        validate_sft_jsonl(build.train_file)
        if build.validation_file and build.validation_file.suffix == ".jsonl":
            click.echo("Validating validation dataset…")
            validate_sft_jsonl(build.validation_file)

        upload_url = f"{backend_base.rstrip('/')}/files"
        click.echo("\n=== Uploading Training Data ===")
        click.echo(f"Dataset: {build.train_file}")
        click.echo(f"Destination: {upload_url}")
        resp = post_multipart(
            upload_url, api_key=synth_key, file_field="file", file_path=build.train_file
        )
        js = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if resp.status_code is not None and resp.status_code >= 400 or "id" not in js:
            click.echo("\n[ERROR] Training file upload failed:")
            click.echo(f"  URL: {upload_url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {js or resp.text[:400]}")
            click.echo(f"  File: {build.train_file}")
            raise click.ClickException(
                f"Training file upload failed with status {resp.status_code}"
            )
        train_file_id = js["id"]
        click.echo(f"✓ Training file uploaded (id={train_file_id})")
        val_file_id = None
        if build.validation_file:
            click.echo(f"Uploading validation dataset: {build.validation_file}")
            vresp = post_multipart(
                upload_url,
                api_key=synth_key,
                file_field="file",
                file_path=build.validation_file,
            )
            vjs = (
                vresp.json()
                if vresp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            if vresp.status_code is not None and vresp.status_code < 400 and "id" in vjs:
                val_file_id = vjs["id"]
                click.echo(f"✓ Validation file uploaded (id={val_file_id})")
            else:
                click.echo(
                    f"[WARN] Validation upload failed ({vresp.status_code}): {vjs or vresp.text[:200]}"
                )
        payload = dict(build.payload)
        payload["training_file_id"] = train_file_id
        if val_file_id:
            payload.setdefault("metadata", {}).setdefault("effective_config", {}).setdefault(
                "data", {}
            )["validation_files"] = [val_file_id]

        click.echo("\n=== Checking File Processing Status ===")
        try:
            _wait_for_training_file(backend_base, synth_key, train_file_id)
        except click.ClickException as exc:
            click.echo(f"[WARN] File readiness check failed: {exc}")
            click.echo("Proceeding anyway - backend will validate file during job creation...")

        click.echo("\n=== Creating Training Job ===")
        click.echo("Job payload preview:")
        click.echo(preview_json(payload, limit=800))

        create_url = f"{backend_base}/learning/jobs"
        headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
        click.echo(f"\nPOST {create_url}")
        resp = http_post(create_url, headers=headers, json_body=payload)
        js = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        if resp.status_code not in (200, 201):
            click.echo("\n[ERROR] Job creation failed:")
            click.echo(f"  URL: {create_url}")
            click.echo(f"  Status: {resp.status_code}")
            click.echo(f"  Response: {preview_json(js, limit=600)}")
            raise click.ClickException(f"Job creation failed with status {resp.status_code}")
        job_id = js.get("job_id") or js.get("id")
        if not job_id:
            raise click.ClickException("Response missing job id")
        click.echo(f"✓ Job created (id={job_id})")

        click.echo("\n=== Starting Training Job ===")
        start_url = f"{backend_base}/learning/jobs/{job_id}/start"
        click.echo(f"POST {start_url}")
        start_resp = http_post(start_url, headers=headers, json_body={})
        if start_resp.status_code not in (200, 201):
            click.echo(f"[WARN] Job start returned status {start_resp.status_code}")
        else:
            click.echo("✓ Job started")

        if not poll:
            click.echo(f"Started job {job_id} (polling disabled)")
            return

        click.echo("\n=== Streaming Job Progress ===")
        config, handlers = _build_stream_components(
            stream_format, hidden_event_types=_DEFAULT_SFT_HIDDEN_EVENTS
        )
        if stream_format == "chart":
            click.echo("Using live loss chart (metric=train.loss)")
        streamer = JobStreamer(
            base_url=backend_base,
            api_key=synth_key,
            job_id=job_id,
            endpoints=StreamEndpoints.learning(job_id),
            config=config,
            handlers=handlers,
            interval_seconds=poll_interval,
            timeout_seconds=poll_timeout,
        )
        final_status = asyncio.run(streamer.stream_until_terminal())
        status = final_status.get('status') if isinstance(final_status, dict) else 'unknown'
        click.echo(f"Final status: {status}")
        click.echo(preview_json(final_status, limit=600))
    finally:
        if limited_path is not None:
            with contextlib.suppress(OSError):
                limited_path.unlink(missing_ok=True)
            # Clean up empty parent directory if possible
            with contextlib.suppress(OSError):
                limited_path.parent.rmdir()


def _save_prompt_learning_results_locally(
    *,
    backend_base: str,
    api_key: str,
    job_id: str,
    config_path: Path,
) -> None:
    """Fetch events and generate results file locally after prompt learning completes."""
    from datetime import datetime
    
    try:
        # Fetch all events
        url = f"{backend_base}/prompt-learning/online/jobs/{job_id}/events?limit={_RESULTS_FILE_MAX_EVENTS}"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = http_get(url, headers=headers, timeout=30.0)
        
        if resp.status_code != 200:
            click.echo(f"⚠️  Could not fetch events to generate results file (status={resp.status_code})")
            return
        
        data = resp.json()
        # Validate response structure
        if not isinstance(data, dict):
            click.echo(f"⚠️  Unexpected response type: {type(data).__name__}")
            return
        
        events = data.get("events", [])
        if not isinstance(events, list):
            click.echo(f"⚠️  Events field is not a list: {type(events).__name__}")
            return
        
        if not events:
            return
        
        # Extract key data from events
        best_score = None
        best_prompt = None
        baseline_score = None
        attempted_candidates = []
        optimized_candidates = []
        mipro_topk_candidates = []  # Collect MIPRO top-K candidates
        
        for event in events:
            if not isinstance(event, dict):
                continue  # Skip malformed events
            
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            if not isinstance(event_data, dict):
                event_data = {}  # Fallback to empty dict for safety
            
            if event_type == _PROMPT_LEARNING_EVENT_BEST_PROMPT:
                best_score = event_data.get("best_score")
                best_prompt = event_data.get("best_prompt")
            elif event_type == _PROMPT_LEARNING_EVENT_FINAL_RESULTS:
                attempted_candidates = event_data.get("attempted_candidates", [])
                optimized_candidates = event_data.get("optimized_candidates", [])
            elif event_type == _PROMPT_LEARNING_EVENT_VALIDATION_SCORED:
                # Check if this is the baseline by checking for is_baseline flag or baseline in message
                is_baseline = event_data.get("is_baseline", False)
                if not is_baseline:
                    msg = event.get("message", "")
                    is_baseline = "baseline" in msg.lower()
                if is_baseline:
                    baseline_score = event_data.get("accuracy")
            elif event_type == _PROMPT_LEARNING_EVENT_GEPA_COMPLETE and best_score is None:
                best_score = event_data.get("best_score")
            elif event_type == _PROMPT_LEARNING_EVENT_MIPRO_COMPLETE:
                # MIPRO completion event includes best_prompt and best_score
                if best_score is None:
                    best_score = event_data.get("best_score")
                if best_prompt is None:
                    best_prompt = event_data.get("best_prompt")
            elif event_type == "mipro.topk.evaluated":
                # Extract MIPRO top-K candidate data
                rank = event_data.get("rank")
                train_score = event_data.get("train_score")
                test_score = event_data.get("test_score")
                if rank is not None and train_score is not None and test_score is not None:
                    mipro_topk_candidates.append({
                        "rank": rank,
                        "train_score": train_score,
                        "test_score": test_score,
                        "lift_absolute": event_data.get("lift_absolute"),
                        "lift_percent": event_data.get("lift_percent"),
                        "instruction_text": event_data.get("instruction_text", ""),
                        "demo_indices": event_data.get("demo_indices", []),
                    })
            elif event_type == "mipro.baseline.test":
                # Extract baseline test score
                if baseline_score is None:
                    baseline_score = event_data.get("test_score")
        
        # Check if we have any results to display (best_prompt, best_score, or candidates)
        has_results = bool(attempted_candidates or optimized_candidates or best_prompt or best_score is not None)
        if not has_results:
            return
        
        # Determine algorithm name from events
        algorithm_name = "PROMPT LEARNING"
        for event in events:
            if isinstance(event, dict):
                event_type = event.get("type", "")
                if "gepa" in event_type.lower():
                    algorithm_name = "GEPA"
                    break
                elif "mipro" in event_type.lower():
                    algorithm_name = "MIPRO"
                    break
        
        # Generate formatted report
        lines = []
        lines.append("=" * 80)
        lines.append(f"{algorithm_name} PROMPT LEARNING RESULTS")
        lines.append("=" * 80)
        lines.append(f"Job ID: {job_id}")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        if baseline_score is not None:
            lines.append(f"📊 Baseline Score: {baseline_score:.4f} ({baseline_score*100:.1f}%)")
        if best_score is not None:
            lines.append(f"🏆 Best Score:     {best_score:.4f} ({best_score*100:.1f}%)")
        if baseline_score is not None and best_score is not None:
            improvement = ((best_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
            lines.append(f"📈 Improvement:    {improvement:+.1f}% relative ({(best_score - baseline_score)*100:+.1f} pp absolute)")
        lines.append("=" * 80)
        lines.append("")
        
        # Add best prompt if available
        if best_prompt and isinstance(best_prompt, dict):
            lines.append("🏆 BEST PROMPT")
            lines.append("-" * 80)
            sections = best_prompt.get("sections", [])
            if not isinstance(sections, list):
                sections = []
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                role = sec.get("role", "unknown")
                content = sec.get("content", "")
                lines.append(f"\n[{role.upper()}]:")
                lines.append(content)
            lines.append("")
        
        # Add optimized candidates
        if optimized_candidates and isinstance(optimized_candidates, list):
            lines.append("=" * 80)
            lines.append(f"✨ TOP OPTIMIZED CANDIDATES ({len(optimized_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, cand in enumerate(optimized_candidates):
                if not isinstance(cand, dict):
                    continue
                candidate_score = cand.get("score") or {}
                accuracy = candidate_score.get("accuracy", 0.0)
                prompt_length = candidate_score.get("prompt_length", 0)
                payload_kind = cand.get("payload_kind", "unknown")
                
                # Try score.instance_scores first, then cand.instance_scores (explicit check)
                instance_scores = (
                    candidate_score.get('instance_scores') 
                    if 'instance_scores' in candidate_score 
                    else cand.get('instance_scores')
                )
                n_eval = len(instance_scores) if instance_scores and isinstance(instance_scores, list) else 0
                
                lines.append(f"[{idx+1}] Accuracy: {accuracy:.4f} | Length: {prompt_length} | Type: {payload_kind} | N: {n_eval}")
                lines.append("-" * 80)
                
                obj = cand.get("object")
                if obj and isinstance(obj, dict) and payload_kind == "transformation":
                    # For transformations, text_replacements are nested in data
                    data_obj = obj.get("data", {})
                    replacement_lines = _format_text_replacements(data_obj)
                    lines.extend(replacement_lines)
                lines.append("")
        
        # Add MIPRO top-K candidates
        if mipro_topk_candidates and isinstance(mipro_topk_candidates, list):
            # Sort by rank
            mipro_topk_candidates.sort(key=lambda x: x.get("rank", 999))
            lines.append("=" * 80)
            lines.append(f"🎯 TOP-K CANDIDATES ({len(mipro_topk_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for cand in mipro_topk_candidates:
                rank = cand.get("rank", 0)
                train_score = cand.get("train_score", 0.0)
                test_score = cand.get("test_score", 0.0)
                lift_abs = cand.get("lift_absolute")
                lift_pct = cand.get("lift_percent")
                instruction_text = cand.get("instruction_text", "")
                demo_indices = cand.get("demo_indices", [])
                
                lift_str = ""
                if lift_abs is not None and lift_pct is not None:
                    lift_str = f" | Lift: {lift_abs:+.3f} ({lift_pct:+.1f}%)"
                
                lines.append(f"[Rank {rank}] Train: {train_score:.4f} ({train_score*100:.1f}%) | Test: {test_score:.4f} ({test_score*100:.1f}%){lift_str}")
                lines.append("-" * 80)
                
                if instruction_text:
                    lines.append(f"Instruction: {instruction_text[:200]}{'...' if len(instruction_text) > 200 else ''}")
                if demo_indices:
                    lines.append(f"Demo Indices: {demo_indices}")
                lines.append("")
        
        # Add all proposal candidates
        if attempted_candidates and isinstance(attempted_candidates, list):
            lines.append("=" * 80)
            lines.append(f"💡 ALL PROPOSAL CANDIDATES ({len(attempted_candidates)})")
            lines.append("=" * 80)
            lines.append("")
            
            for idx, cand in enumerate(attempted_candidates):
                if not isinstance(cand, dict):
                    continue
                accuracy = cand.get('accuracy', 0.0)
                prompt_length = cand.get('prompt_length', 0)
                tool_rate = cand.get('tool_call_rate', 0.0)
                instance_scores = cand.get('instance_scores', [])
                n_eval = len(instance_scores) if instance_scores else 0
                
                lines.append(f"[{idx+1}] Accuracy: {accuracy:.4f} | Length: {prompt_length} | Tool Rate: {tool_rate:.2f} | N: {n_eval}")
                lines.append("-" * 80)
                
                obj = cand.get("object")
                if obj and isinstance(obj, dict):
                    # For proposals, text_replacements are at top level of object
                    replacement_lines = _format_text_replacements(obj)
                    lines.extend(replacement_lines)
                lines.append("")
        
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Determine save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try to save in config directory first
        output_dir = config_path.parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"gepa_results_{job_id}_{timestamp}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        click.echo(f"\n📄 Results saved locally to: {output_file}")
        
    except (PermissionError, OSError) as e:
        click.echo(f"⚠️  Could not save results file locally: {e}")
    except Exception as e:
        click.echo(f"⚠️  Unexpected error saving results file: {e}")


def handle_prompt_learning(
    *,
    cfg_path: Path,
    backend_base: str,
    synth_key: str,
    task_url_override: str | None,
    allow_experimental: bool | None,
    dry_run: bool,
    poll: bool,
    poll_timeout: float,
    poll_interval: float,
    stream_format: str,
) -> None:
    """Handle prompt learning job creation (MIPRO or GEPA)."""
    import os
    
    overrides: dict[str, Any] = {
        "backend": backend_base,
    }
    
    build = build_prompt_learning_payload(
        config_path=cfg_path,
        task_url=None,  # Force using TOML only
        overrides=overrides,
        allow_experimental=allow_experimental,
    )
    
    env_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not env_key:
        raise click.ClickException("ENVIRONMENT_API_KEY required for prompt learning flow")
    
    click.echo("Performing task app health check…")
    health = check_task_app_health(build.task_url, env_key)
    if not health.ok:
        click.echo(f"Task app health check failed: {health.detail}")
        raise click.ClickException("Aborting due to failing health check")
    else:
        click.echo("Task app healthy")
    
    create_url = f"{backend_base}/prompt-learning/online/jobs"
    headers = {"Authorization": f"Bearer {synth_key}", "Content-Type": "application/json"}
    
    click.echo(f"POST {create_url}")
    click.echo("Payload preview:\n" + preview_json(build.payload, limit=800))
    
    resp = http_post(create_url, headers=headers, json_body=build.payload)
    try:
        js = resp.json()
    except json.JSONDecodeError as e:
        click.echo(f"⚠️  Failed to parse JSON response: {e}")
        js = {"status": resp.status_code, "text": resp.text[:400]}
    click.echo(f"Response {resp.status_code}: {preview_json(js, limit=400)}")
    if resp.status_code not in (200, 201):
        raise click.ClickException("Job creation failed")
    job_id = js.get("job_id") or js.get("id")
    if not job_id:
        raise click.ClickException("Response missing job id")
    
    if not poll:
        click.echo(f"Created job {job_id} (polling disabled)")
        return
    
    click.echo("\n=== Streaming Job Progress ===")
    
    # Custom config for prompt learning to enable metrics
    if stream_format == "chart":
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [LossCurveHandler()]
        click.echo("Using live loss chart (metric=gepa.transformation.mean_score)")
    else:
        # Enable metrics for CLI mode too
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        handlers = [CLIHandler(
            hidden_event_types=_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
            hidden_event_substrings=_DEFAULT_RL_HIDDEN_SUBSTRINGS,
        )]
    
    streamer = JobStreamer(
        base_url=backend_base,
        api_key=synth_key,
        job_id=job_id,
        endpoints=StreamEndpoints.prompt_learning(job_id),
        config=config,
        handlers=handlers,
        interval_seconds=poll_interval,
        timeout_seconds=poll_timeout,
    )
    final_status = asyncio.run(streamer.stream_until_terminal())
    click.echo(f"Final status: {final_status.get('status', 'unknown')}")
    click.echo(preview_json(final_status, limit=600))
    
    # Save results file locally
    _save_prompt_learning_results_locally(
        backend_base=backend_base,
        api_key=synth_key,
        job_id=job_id,
        config_path=cfg_path,
    )


def register(cli: click.Group) -> None:
    cli.add_command(train_command)
