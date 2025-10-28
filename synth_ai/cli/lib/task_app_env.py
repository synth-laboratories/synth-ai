from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import click
from click.exceptions import Abort
from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT
from synth_ai.task.apps import TaskAppEntry

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env_files_into_process(paths: Sequence[str]) -> None:
    """Load key/value pairs from .env-style files into the current process."""

    for path_str in paths:
        try:
            content = Path(path_str).expanduser().read_text()
        except Exception:
            continue
        for line in content.splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            val = value.strip().strip('"').strip("'")
            if not key:
                continue
            current = os.environ.get(key, "")
            if not current.strip():
                os.environ[key] = val


def _collect_env_candidates(base_dir: Path) -> list[Path]:
    cwd = Path.cwd()
    candidates: list[Path] = []

    candidates.extend(sorted(cwd.glob("**/*.env")))

    repo_candidates = sorted(REPO_ROOT.glob("**/*.env"))
    for candidate in repo_candidates:
        if candidate not in candidates:
            candidates.append(candidate)

    if base_dir not in (cwd, REPO_ROOT):
        base_candidates = sorted(base_dir.glob("**/*.env"))
        for candidate in base_candidates:
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def determine_env_files(entry: TaskAppEntry, user_env_files: Sequence[str]) -> list[Path]:
    """Resolve env file paths for a task app invocation."""

    resolved: list[Path] = []
    for candidate in user_env_files:
        path = Path(candidate).expanduser()
        if not path.exists():
            raise click.ClickException(f"Env file not found: {path}")
        resolved.append(path)
    if resolved:
        return resolved

    candidates = _collect_env_candidates(Path.cwd())
    if not candidates:
        raise click.ClickException("No env file found. Pass --env-file explicitly.")

    click.echo("Select env file to load:")
    for idx, path in enumerate(candidates, start=1):
        click.echo(f"  {idx}) {path.resolve()}")
    choice = click.prompt("Enter choice", type=click.IntRange(1, len(candidates)), default=1)
    selected = candidates[choice - 1]
    return [selected]


def resolve_env_paths_for_script(script_path: Path, explicit: Sequence[str]) -> list[Path]:
    """Resolve env files for a standalone Modal script."""

    if explicit:
        resolved = []
        for candidate in explicit:
            path = Path(candidate).expanduser()
            if not path.exists():
                raise click.ClickException(f"Env file not found: {path}")
            resolved.append(path)
        return resolved

    candidates = _collect_env_candidates(script_path.parent.resolve())
    if not candidates:
        created = interactive_create_env(script_path.parent)
        if created is None:
            raise click.ClickException("Env file required (--env-file) for this task app")
        return [created]

    click.echo("Select env file to load:")
    for idx, path in enumerate(candidates, start=1):
        click.echo(f"  {idx}) {path.resolve()}")
    choice = click.prompt("Enter choice", type=click.IntRange(1, len(candidates)), default=1)
    return [candidates[choice - 1]]


def ensure_port_free(port: int, host: str, *, force: bool) -> None:
    """Ensure a TCP port is not in use, optionally killing processes if --force."""

    import socket  # local import to avoid unnecessary dependency during CLI import

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        in_use = sock.connect_ex((host, port)) == 0
    if not in_use:
        return

    try:
        out = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [pid for pid in out.stdout.strip().splitlines() if pid]
    except FileNotFoundError:
        pids = []

    if not force:
        message = f"Port {port} appears to be in use"
        if pids:
            message += f" (PIDs: {', '.join(pids)})"
        raise click.ClickException(message)

    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception as exc:
            raise click.ClickException(f"Failed to terminate PID {pid}: {exc}") from exc

    time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        still_in_use = sock.connect_ex((host, port)) == 0

    if still_in_use:
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception as exc:
                raise click.ClickException(f"Failed to force terminate PID {pid}: {exc}") from exc
        time.sleep(0.5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((host, port)) == 0:
            raise click.ClickException(
                f"Port {port} is still in use after attempting to terminate processes."
            )


def save_to_env_file(env_path: Path, key: str, value: str) -> None:
    """Save or update a key/value pair in a .env file."""

    try:
        existing_lines = env_path.read_text().splitlines() if env_path.exists() else []
    except Exception as exc:
        raise click.ClickException(f"Failed to read {env_path}: {exc}") from exc

    env_path.parent.mkdir(parents=True, exist_ok=True)

    key_updated = False
    updated_lines: list[str] = []
    for line in existing_lines:
        if line.strip().startswith(f"{key}="):
            updated_lines.append(f"{key}={value}")
            key_updated = True
        else:
            updated_lines.append(line)

    if key_updated:
        env_path.write_text("\n".join(updated_lines) + "\n")
        click.echo(f"Updated {key} in {env_path}")
        return

    with env_path.open("a", encoding="utf-8") as handle:
        if existing_lines and existing_lines[-1].strip():
            handle.write("\n")
        handle.write(f"{key}={value}\n")
    click.echo(f"Saved {key} to {env_path}")


def persist_env_api_key(env_api_key: str, env_paths: Sequence[Path] | None) -> None:
    """Persist ENVIRONMENT_API_KEY to provided .env files (or demo directory .env)."""

    targets: list[Path] = []
    seen: set[Path] = set()
    for path in env_paths or ():
        try:
            resolved = Path(path).resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        targets.append(resolved)

    if not targets:
        demo_dir = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
        targets.append((demo_dir / ".env").resolve())

    for target in targets:
        save_to_env_file(target, "ENVIRONMENT_API_KEY", env_api_key)


def _load_dotenv_if_present(env_file: Path) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    with contextlib.suppress(Exception):
        load_dotenv(env_file, override=False)


def validate_required_env_keys() -> None:
    """Ensure ENVIRONMENT_API_KEY (and optional Groq key) are set, prompting if needed."""

    demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
    env_file = demo_base / ".env"

    if env_file.exists():
        _load_dotenv_if_present(env_file)

    env_api_key = os.environ.get("ENVIRONMENT_API_KEY", "").strip()
    if not env_api_key:
        env_api_key = click.prompt(
            "Please enter your RL Environment API key",
            type=str,
        ).strip()
        if not env_api_key:
            raise click.ClickException("RL Environment API key is required to start the server")
        os.environ["ENVIRONMENT_API_KEY"] = env_api_key
        save_to_env_file(env_file, "ENVIRONMENT_API_KEY", env_api_key)

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        click.echo("\nInference API key configuration:")
        click.echo("This workflow requires a Groq API key.")
        groq_api_key = click.prompt(
            "Groq API key (or press Enter to skip)",
            type=str,
            default="",
            show_default=False,
        ).strip()
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            save_to_env_file(env_file, "GROQ_API_KEY", groq_api_key)


def print_demo_next_steps_if_applicable() -> None:
    """Print helpful instructions when operating inside a demo directory."""

    try:
        from synth_ai.demos.demo_task_apps.core import load_demo_dir

        cwd = Path.cwd().resolve()
        demo_dir = load_demo_dir()

        if demo_dir and Path(demo_dir).resolve() == cwd and (cwd / "run_local_rollout_traced.py").exists():
            click.echo("\n" + "=" * 60)
            click.echo("Next step: Collect traced rollouts")
            click.echo("=" * 60)
            click.echo("\nIn another terminal, run:")
            click.echo(f"  cd {cwd}")
            click.echo("  uv run python run_local_rollout_traced.py")
            click.echo("\nRun this 5-10 times to collect diverse traces.")
            click.echo("=" * 60 + "\n")
    except Exception:
        pass


def _preview_secret(value: str) -> str:
    if len(value) <= 10:
        return value
    return f"{value[:6]}...{value[-4:]}"


def preflight_env_key(env_paths: Sequence[Path] | None = None, *, crash_on_failure: bool = False) -> None:
    """Ensure ENVIRONMENT_API_KEY exists and attempt to upload it to the backend."""

    raw_backend = (
        os.environ.get("BACKEND_BASE_URL")
        or os.environ.get("SYNTH_BASE_URL")
        or f"{PROD_BASE_URL_DEFAULT}/api"
    )
    backend_base = raw_backend.rstrip("/")
    if not backend_base.endswith("/api"):
        backend_base += "/api"

    synth_key = os.environ.get("SYNTH_API_KEY") or ""
    env_api_key = (
        os.environ.get("ENVIRONMENT_API_KEY")
        or os.environ.get("DEV_ENVIRONMENT_API_KEY")
        or ""
    ).strip()

    def _mint_key() -> str | None:
        try:
            from synth_ai.learning.rl.secrets import mint_environment_api_key

            key = mint_environment_api_key()
            os.environ["ENVIRONMENT_API_KEY"] = key
            os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", key)
            click.echo(f"[preflight] minted ENVIRONMENT_API_KEY ({_preview_secret(key)})")
            return key
        except Exception as exc:  # pragma: no cover - defensive fallback
            if crash_on_failure:
                raise click.ClickException(
                    f"[CRITICAL] Failed to mint ENVIRONMENT_API_KEY: {exc}"
                ) from exc
            click.echo(
                f"[WARN] Failed to mint ENVIRONMENT_API_KEY automatically ({exc}); proceeding without upload"
            )
            return None

    minted = False
    if not env_api_key:
        env_api_key = _mint_key() or ""
        minted = bool(env_api_key)

    if env_api_key and minted and env_paths:
        persist_env_api_key(env_api_key, env_paths)

    if not synth_key.strip():
        click.echo("[preflight] SYNTH_API_KEY not set; skipping backend preflight.")
        return

    if not env_api_key:
        click.echo("[preflight] ENVIRONMENT_API_KEY missing; continuing without verification.")
        return

    try:
        import base64

        import httpx
        from nacl.public import PublicKey, SealedBox
    except Exception:  # pragma: no cover - optional deps
        click.echo("[preflight] Optional crypto dependencies missing; skipping upload.")
        return

    try:
        with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {synth_key}"}) as client:
            click.echo(f"[preflight] backend={backend_base}")
            click.echo("[preflight] fetching public key…")
            rpk = client.get(f"{backend_base.rstrip('/')}/v1/crypto/public-key")
            if rpk.status_code != 200:
                click.echo(f"[preflight] public key fetch failed with {rpk.status_code}; skipping upload")
                return
            pk = (rpk.json() or {}).get("public_key")
            if not pk:
                click.echo("[preflight] no public key returned; skipping upload")
                return

            pk_bytes = base64.b64decode(pk, validate=True)
            sealed_box = SealedBox(PublicKey(pk_bytes))
            ciphertext = sealed_box.encrypt(env_api_key.encode("utf-8"))
            ct_b64 = base64.b64encode(ciphertext).decode()
            payload = {"name": "ENVIRONMENT_API_KEY", "ciphertext_b64": ct_b64}

            click.echo(f"[preflight] posting to {backend_base.rstrip('/')}/v1/env-keys")
            response = client.post(f"{backend_base.rstrip('/')}/v1/env-keys", json=payload)
            if 200 <= response.status_code < 300:
                click.echo(
                    f"✅ ENVIRONMENT_API_KEY uploaded successfully ({_preview_secret(env_api_key)})"
                )
                try:
                    ver = client.get(f"{backend_base.rstrip('/')}/v1/env-keys/verify")
                    if ver.status_code == 200 and (ver.json() or {}).get("present"):
                        click.echo("✅ Key verified in backend")
                    else:
                        click.echo(
                            f"⚠️  Verification returned {ver.status_code}, but upload succeeded - proceeding"
                        )
                except Exception as verify_err:  # pragma: no cover - verification optional
                    click.echo(
                        f"⚠️  Verification check failed ({verify_err}), but upload succeeded - proceeding"
                    )
                return

            snippet = response.text[:400] if response.text else ""
            message = (
                f"ENVIRONMENT_API_KEY upload failed with status {response.status_code}"
                + (f" body={snippet}" if snippet else "")
            )
            if crash_on_failure:
                raise click.ClickException(f"[CRITICAL] {message}")
            click.echo(f"[WARN] {message}; proceeding anyway")
    except Exception as exc:  # pragma: no cover - network failures
        message = f"Backend preflight for ENVIRONMENT_API_KEY failed: {exc}"
        if crash_on_failure:
            raise click.ClickException(f"[CRITICAL] {message}") from exc
        click.echo(f"[WARN] {message}; proceeding anyway")


def load_env_values(paths: Sequence[Path], *, allow_empty: bool = False) -> dict[str, str]:
    """Load values from a sequence of env files, returning a merged dictionary."""

    values: dict[str, str] = {}
    for path in paths:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        for line in content.splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in values:
                values[key] = value
    if not allow_empty and not values:
        raise click.ClickException("No environment values found")
    os.environ.update({k: v for k, v in values.items() if k and v})
    return values


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return data


def interactive_fill_env(env_path: Path) -> Path | None:
    """Interactively collect credentials and write them to a .env file."""

    existing = _parse_env_file(env_path) if env_path.exists() else {}

    def _prompt(label: str, *, default: str = "", required: bool) -> str | None:
        while True:
            try:
                value = click.prompt(
                    label,
                    default=default,
                    show_default=bool(default) or not required,
                ).strip()
            except (Abort, EOFError, KeyboardInterrupt):
                click.echo("Aborted env creation.")
                return None
            if value or not required:
                return value
            click.echo("This field is required.")

    env_default = existing.get("ENVIRONMENT_API_KEY", "").strip()
    env_api_key = _prompt("ENVIRONMENT_API_KEY", default=env_default, required=True)
    if env_api_key is None:
        return None

    synth_default = existing.get("SYNTH_API_KEY", "").strip()
    openai_default = existing.get("OPENAI_API_KEY", "").strip()
    synth_key = _prompt("SYNTH_API_KEY (optional)", default=synth_default, required=False) or ""
    openai_key = _prompt("OPENAI_API_KEY (optional)", default=openai_default, required=False) or ""

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "\n".join(
            [
                f"ENVIRONMENT_API_KEY={env_api_key}",
                f"SYNTH_API_KEY={synth_key}",
                f"OPENAI_API_KEY={openai_key}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote credentials to {env_path}")
    return env_path


def interactive_create_env(target_dir: Path) -> Path | None:
    """Create a .env file for the provided directory if one does not exist."""

    env_path = (target_dir / ".env").resolve()
    if env_path.exists():
        existing = _parse_env_file(env_path)
        env_api = (existing.get("ENVIRONMENT_API_KEY") or "").strip()
        if env_api:
            return env_path
        click.echo(f"Existing {env_path} is missing ENVIRONMENT_API_KEY. Let's update it.")
        return interactive_fill_env(env_path)

    click.echo("No .env found for this task app. Let's create one.")
    return interactive_fill_env(env_path)


def ensure_env_values(env_paths: list[Path], fallback_dir: Path) -> None:
    """Ensure required env values are present, prompting to create .env if needed."""

    if (os.environ.get("ENVIRONMENT_API_KEY") or "").strip():
        return

    target = env_paths[0] if env_paths else (fallback_dir / ".env").resolve()
    result = interactive_fill_env(target)
    if result is None:
        raise click.ClickException("ENVIRONMENT_API_KEY required to continue")

    load_env_values([result])
    if not (os.environ.get("ENVIRONMENT_API_KEY") or "").strip():
        raise click.ClickException("Failed to load ENVIRONMENT_API_KEY from generated .env")
