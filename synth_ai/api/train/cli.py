from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import click

from .builders import build_rl_payload, build_sft_payload
from .pollers import RLJobPoller, SFTJobPoller
from .task_app import check_task_app_health
from .utils import (
    REPO_ROOT,
    TrainError,
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
    with contextlib.suppress(OSError):
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        named = [p for p in candidates if "sft_dataset_" in p.name]
        if named:
            return named[:limit]
    return candidates[:limit]


def prompt_for_dataset(config_path: Path) -> Path:
    candidates = _discover_dataset_candidates(config_path)
    while True:
        if candidates:
            click.echo("Select dataset JSONL file:")
            for idx, candidate in enumerate(candidates, start=1):
                marker = " <- most recent" if idx == 1 else ""
                click.echo(f"  {idx}) {candidate}{marker}")
            click.echo("  m) Enter path manually")
            click.echo("  0) Abort")
            choice = click.prompt("Choice", default="1").strip().lower()
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
