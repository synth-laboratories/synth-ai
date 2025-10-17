from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
from synth_ai.config.base_url import get_backend_from_env

from .builders import build_rl_payload, build_sft_payload
from .config_finder import discover_configs, prompt_for_config
from .env_resolver import KeySpec, resolve_env
from .pollers import RLJobPoller, SFTJobPoller
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


def _discover_dataset_candidates(config_path: Path, limit: int = 50) -> list[Path]:
    search_dirs: list[Path] = [
        config_path.parent,
        config_path.parent / "datasets",
        REPO_ROOT / "traces",
        REPO_ROOT / "datasets",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
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


@click.command("train")
@click.option(
    "--config",
    "config_paths",
    multiple=True,
    type=click.Path(),
    help="Path to training TOML (repeatable)",
)
@click.option("--type", "train_type", type=click.Choice(["auto", "rl", "sft"]), default="auto")
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
    examples_limit: int | None,
) -> None:
    """Interactive launcher for RL / SFT jobs."""

    candidates = discover_configs(
        list(config_paths), requested_type=train_type if train_type != "auto" else None
    )
    selection = prompt_for_config(
        candidates,
        requested_type=train_type if train_type != "auto" else None,
        allow_autoselect=bool(config_paths),
    )

    effective_type = train_type if train_type != "auto" else selection.train_type
    if effective_type not in {"rl", "sft"}:
        effective_type = click.prompt(
            "Detected config type is ambiguous. Enter type", type=click.Choice(["rl", "sft"])
        )

    cfg_path = selection.path
    click.echo(f"Using config: {cfg_path} ({effective_type})")

    required_keys: list[KeySpec] = []
    if effective_type == "rl":
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
            from synth_ai.cli.task_apps import _interactive_fill_env
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
            examples_limit=examples_limit,
        )


def _wait_for_training_file(
    backend_base: str, api_key: str, file_id: str, *, timeout: float = 120.0
) -> None:
    url = f"{backend_base}/learning/files/{file_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0.0
    interval = 2.0
    first_check = True
    while True:
        resp = http_get(url, headers=headers, timeout=30.0)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
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
            except Exception:
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
            except Exception:
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
            vjs = vresp.json()
        except Exception:
            vjs = {"status": vresp.status_code, "text": (vresp.text or "")[:400]}
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
            attempts = vjs.get("attempts") or []
            statuses = [a.get("status") for a in attempts]
            click.echo(f"Verification OK (candidates={cands}, statuses={statuses})")
        except Exception:
            pass

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
    except Exception:
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

    poller = RLJobPoller(backend_base, synth_key, interval=poll_interval, timeout=poll_timeout)
    outcome = poller.poll_job(job_id)
    click.echo(f"Final status: {outcome.status}")
    click.echo(preview_json(outcome.payload, limit=600))


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

        upload_url = f"{backend_base}/learning/files"
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
            raise click.ClickException(f"Training file {train_file_id} not ready: {exc}") from exc

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

        poller = SFTJobPoller(backend_base, synth_key, interval=poll_interval, timeout=poll_timeout)
        outcome = poller.poll_job(job_id)
        click.echo(f"Final status: {outcome.status}")
        click.echo(preview_json(outcome.payload, limit=600))
    finally:
        if limited_path is not None:
            try:
                limited_path.unlink(missing_ok=True)
                limited_path.parent.rmdir()
            except Exception:
                pass


def register(cli: click.Group) -> None:
    cli.add_command(train_command)
