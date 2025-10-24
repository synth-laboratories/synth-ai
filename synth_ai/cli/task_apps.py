"""Task app CLI commands powered by shared utilities."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import click
from synth_ai._utils.task_app_discovery import AppChoice, select_app_choice
from synth_ai._utils.task_app_env import (
    ensure_env_credentials,
    ensure_port_free,
    hydrate_user_environment,
    preflight_env_key,
)
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry
from synth_ai.task.server import run_task_app

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maybe_use_demo_dir() -> Path | None:
    """Change into the configured demo directory, if present."""

    try:
        from synth_ai.demos.core import load_demo_dir
    except Exception:
        return None

    demo_dir = load_demo_dir()
    if not demo_dir:
        return None

    demo_path = Path(demo_dir)
    if not demo_path.is_dir():
        raise click.ClickException(
            f"Demo directory not found: {demo_dir}\nRun 'synth-ai setup' to create a demo."
        )

    resolved = demo_path.resolve()
    if Path.cwd().resolve() != resolved:
        os.chdir(resolved)
        click.echo(f"Using demo directory: {resolved}\n")

    os.environ["SYNTH_DEMO_DIR"] = str(resolved)
    return resolved


def _apply_tracing_configuration(trace_dir: str | None, trace_db: str | None) -> None:
    """Configure tracing-related environment variables and directories."""

    if not trace_dir and not trace_db:
        if os.getenv("TASKAPP_TRACING_ENABLED"):
            click.echo("Tracing enabled via environment variables")
        return

    os.environ["TASKAPP_TRACING_ENABLED"] = "1"
    demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())

    if trace_dir:
        dir_path = Path(trace_dir).expanduser()
        if not dir_path.is_absolute():
            dir_path = (demo_base / dir_path).resolve()
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise click.ClickException(f"Failed to create trace directory {dir_path}: {exc}") from exc
        os.environ["TASKAPP_SFT_OUTPUT_DIR"] = str(dir_path)
        click.echo(f"Tracing enabled. SFT JSONL will be written to {dir_path}")

    if trace_db:
        db_path = Path(trace_db).expanduser()
        if not db_path.is_absolute():
            db_path = (demo_base / db_path).resolve()
        db_url = f"sqlite+aiosqlite:///{db_path}"
        os.environ["SQLD_DB_PATH"] = str(db_path)
        os.environ["TURSO_LOCAL_DB_URL"] = db_url
        os.environ["TASKAPP_TRACE_DB_PATH"] = str(db_path)
        click.echo(f"Tracing DB path set to {db_path}")

        try:
            from synth_ai.tracing_v3.config import CONFIG as TRACE_CONFIG

            new_db_url = os.getenv("TURSO_LOCAL_DB_URL") or TRACE_CONFIG.db_url
            TRACE_CONFIG.db_url = new_db_url
            if new_db_url:
                click.echo(f"Tracing DB URL resolved to {new_db_url}")
        except Exception:
            pass


def _modal_command_prefix(modal_cli: str) -> list[str]:
    """Return the command prefix used to invoke the Modal CLI."""

    if modal_cli == "modal" and importlib.util.find_spec("modal") is not None:
        return [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]

    modal_path = shutil.which(modal_cli)
    if modal_path:
        return [modal_path]

    if modal_cli == "modal":
        raise click.ClickException(
            "Modal CLI not found. Install the 'modal' package or pass --modal-cli with an explicit path."
        )
    raise click.ClickException(f"Modal CLI not found (looked for '{modal_cli}')")


def _build_modal_app_wrapper(original_script: Path) -> tuple[Path, Path]:
    """Generate a temporary wrapper that loads a Modal app with local mounts."""

    source_dir = original_script.parent.resolve()
    repo_root = REPO_ROOT
    temp_root = Path(tempfile.mkdtemp(prefix="synth_modal_app_"))

    wrapper_source = textwrap.dedent(
        f"""
        from importlib import util as _util
        from pathlib import Path as _Path
        import sys as _sys

        _source_dir = _Path({str(source_dir)!r}).resolve()
        _module_path = _source_dir / {original_script.name!r}
        _package_name = _source_dir.name
        _repo_root = _Path({str(repo_root)!r}).resolve()
        _synth_dir = _repo_root / "synth_ai"

        for _path in (str(_source_dir), str(_source_dir.parent), str(_repo_root)):
            if _path not in _sys.path:
                _sys.path.insert(0, _path)

        _spec = _util.spec_from_file_location("_synth_modal_target", str(_module_path))
        if _spec is None or _spec.loader is None:
            raise SystemExit("Unable to load modal task app from {original_script}")
        _module = _util.module_from_spec(_spec)
        _sys.modules.setdefault("_synth_modal_target", _module)
        _spec.loader.exec_module(_module)

        try:
            from modal import App as _ModalApp
            from modal import Image as _ModalImage
        except Exception:
            _ModalApp = None  # type: ignore[assignment]
            _ModalImage = None  # type: ignore[assignment]

        def _apply_local_mounts(image):
            if _ModalImage is None or not isinstance(image, _ModalImage):
                return image
            mounts = [
                (str(_source_dir), f"/root/{{_package_name}}"),
                (str(_synth_dir), "/root/synth_ai"),
            ]
            for local_path, remote_path in mounts:
                try:
                    image = image.add_local_dir(local_path, remote_path=remote_path)
                except Exception:
                    pass
            return image

        if hasattr(_module, "image"):
            _module.image = _apply_local_mounts(getattr(_module, "image"))

        _candidate = getattr(_module, "app", None)
        if _ModalApp is None or not isinstance(_candidate, _ModalApp):
            candidate_modal_app = getattr(_module, "modal_app", None)
            if _ModalApp is not None and isinstance(candidate_modal_app, _ModalApp):
                _candidate = candidate_modal_app
                setattr(_module, "app", _candidate)

        if _ModalApp is not None and not isinstance(_candidate, _ModalApp):
            raise SystemExit(
                "Modal task app must expose an 'app = modal.App(...)' (or modal_app) attribute."
            )

        for remote_path in ("/root/synth_ai", f"/root/{{_package_name}}"):
            if remote_path not in _sys.path:
                _sys.path.insert(0, remote_path)

        globals().update({{k: v for k, v in vars(_module).items() if not k.startswith("__")}})
        app = getattr(_module, "app")
        """
    ).strip()

    wrapper_path = temp_root / "__modal_wrapper__.py"
    wrapper_path.write_text(wrapper_source + "\n", encoding="utf-8")
    return wrapper_path, temp_root


def _stream_modal_output(proc: subprocess.Popen[str]) -> str | None:
    """Relay Modal CLI stdout to the terminal and capture deployment URLs."""

    assert proc.stdout is not None
    task_app_url: str | None = None
    for line in proc.stdout:
        click.echo(line, nl=False)
        if task_app_url is None and ("modal.run" in line and "=>" in line):
            parts = line.split("=>")
            if len(parts) >= 2:
                candidate = parts[-1].strip()
                if candidate:
                    task_app_url = candidate
                    click.echo(f"\n✓ Task app URL: {task_app_url}\n")
    return task_app_url


def _run_modal_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
) -> str | None:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        task_app_url = _stream_modal_output(proc)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        return task_app_url
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"modal command failed with exit code {exc.returncode}") from exc
    return None


def _prepare_modal_env(require_synth: bool = False) -> None:
    _prepare_runtime_env(require_synth=require_synth, perform_preflight=True)


def _run_modal_script(
    script_path: Path,
    modal_cli: str,
    command: str,
    *,
    modal_name: str | None = None,
    dry_run: bool = False,
) -> str | None:
    _prepare_modal_env(require_synth=False)

    proc_env = os.environ.copy()
    pythonpath_entries: list[str] = [str(script_path.parent.resolve()), str(REPO_ROOT)]
    if (script_path.parent / "__init__.py").exists():
        pythonpath_entries.insert(1, str(script_path.parent.parent.resolve()))
    existing_pp = proc_env.get("PYTHONPATH")
    if existing_pp:
        pythonpath_entries.append(existing_pp)
    proc_env["PYTHONPATH"] = os.pathsep.join(list(dict.fromkeys(pythonpath_entries)))

    wrapper_info: tuple[Path, Path] | None = None
    target_script = script_path
    if command in {"serve", "deploy"}:
        wrapper_path, temp_root = _build_modal_app_wrapper(script_path)
        wrapper_info = (wrapper_path, temp_root)
        target_script = wrapper_path
        proc_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), proc_env["PYTHONPATH"]])

    cmd = [*_modal_command_prefix(modal_cli), command, str(target_script)]
    if modal_name and command == "deploy":
        cmd.extend(["--name", modal_name])

    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        if wrapper_info is not None:
            wrapper_path, temp_root = wrapper_info
            with contextlib.suppress(Exception):
                wrapper_path.unlink(missing_ok=True)
            shutil.rmtree(temp_root, ignore_errors=True)
        return None

    task_app_url: str | None = None
    try:
        task_app_url = _run_modal_subprocess(cmd, env=proc_env)
    finally:
        if wrapper_info is not None:
            wrapper_path, temp_root = wrapper_info
            with contextlib.suppress(Exception):
                wrapper_path.unlink(missing_ok=True)
            shutil.rmtree(temp_root, ignore_errors=True)
    return task_app_url


def _write_modal_entrypoint(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    override_name: str | None,
    *,
    original_path: Path | None = None,
    inline_secret_values: dict[str, str] | None = None,
) -> Path:
    modal_name = override_name or modal_cfg.app_name

    remote_file_str: str | None = None
    if original_path:
        try:
            mount_map: list[tuple[Path, Path]] = [
                (Path(local).resolve(), Path(remote)) for (local, remote) in modal_cfg.extra_local_dirs
            ]
            orig = Path(original_path).resolve()
            for local_src, remote_dst in mount_map:
                with contextlib.suppress(Exception):
                    relative = orig.relative_to(local_src)
                    remote_file_str = str((remote_dst / relative).resolve())
                    break
        except Exception:
            remote_file_str = None

    module_name = entry.config_factory.__module__
    guaranteed_file_str: str | None = None
    if original_path:
        guaranteed_file_str = str(
            (Path("/opt/synth_ai_repo/__local_task_app__") / Path(original_path).stem).with_suffix(".py")
        )

    pip_packages = list(modal_cfg.pip_packages)
    synth_pkg = "synth-ai"
    try:
        import synth_ai as _host_synth

        host_ver = getattr(_host_synth, "__version__", None)
        if host_ver:
            synth_pkg = f"synth-ai=={host_ver}"
    except Exception:
        pass
    if not any(str(pkg).startswith("synth-ai") for pkg in pip_packages):
        pip_packages.insert(0, synth_pkg)
    if not any(str(pkg).startswith("synth-ai") for pkg in pip_packages):
        pip_packages.insert(0, synth_pkg)

    local_dirs = [(str(Path(src)), dst) for src, dst in modal_cfg.extra_local_dirs]
    try:
        import synth_ai as _host_synth

        host_synth_dir = Path(_host_synth.__file__).resolve().parent
        candidate = (str(host_synth_dir), "/opt/synth_ai_repo/synth_ai")
        if candidate not in local_dirs:
            local_dirs.insert(0, candidate)
    except Exception:
        pass

    if original_path:
        discovered_dir = str(Path(original_path).resolve().parent)
        mount_dst = "/opt/synth_ai_repo/__local_task_app__"
        if (discovered_dir, mount_dst) not in local_dirs:
            local_dirs.append((discovered_dir, mount_dst))

    secret_names = list(modal_cfg.secret_names)
    volume_mounts = [(name, mount) for name, mount in modal_cfg.volume_mounts]
    inline_secret_values = {k: v for k, v in (inline_secret_values or {}).items() if v}

    script = f"""from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path as _Path
import fnmatch
sys.path.insert(0, '/opt/synth_ai_repo')

from modal import App, Image, Secret, Volume, asgi_app

ENTRY_ID = {entry.app_id!r}
MODAL_APP_NAME = {modal_name!r}
MODULE_NAME = {module_name!r}
MODULE_FILE = {guaranteed_file_str or remote_file_str!r}
INLINE_SECRET_VALUES = {inline_secret_values!r}

image = Image.debian_slim(python_version={modal_cfg.python_version!r})

pip_packages = {pip_packages!r}
if pip_packages:
    image = image.pip_install(*pip_packages)

local_dirs = {local_dirs!r}


def _copy_tree_filtered(src_dir: str) -> str:
    src = _Path(src_dir)
    temp_dir = _Path(tempfile.mkdtemp(prefix='synth_mount_'))

    exclude_dirs = {{".cache", ".git", "__pycache__"}}
    exclude_globs = ['*.db', '*.db-journal', '*-wal', '*-shm']

    for root, dirs, files in os.walk(src):
        rel_root = _Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        target_dir = temp_dir / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if any(fnmatch.fnmatch(name, pat) for pat in exclude_globs):
                continue
            src_file = _Path(root) / name
            dst_file = target_dir / name
            try:
                shutil.copy2(src_file, dst_file)
            except Exception:
                continue
    return str(temp_dir)


for local_src, remote_dst in local_dirs:
    safe_src = _copy_tree_filtered(local_src)
    image = image.add_local_dir(safe_src, remote_dst)

secrets = {secret_names!r}
secret_objs = [Secret.from_name(name) for name in secrets]

if INLINE_SECRET_VALUES:
    secret_objs.append(Secret.from_dict(INLINE_SECRET_VALUES))

volume_mounts = {volume_mounts!r}
volume_map = {{}}
for vol_name, mount_path in volume_mounts:
    volume_map[mount_path] = Volume.from_name(vol_name, create_if_missing=True)

app = App(MODAL_APP_NAME)


@app.function(
    image=image,
    timeout={modal_cfg.timeout},
    memory={modal_cfg.memory},
    cpu={modal_cfg.cpu},
    min_containers={modal_cfg.min_containers},
    max_containers={modal_cfg.max_containers},
    secrets=secret_objs,
    volumes=volume_map,
)
@asgi_app()
def fastapi_app():
    import importlib as _importlib
    import os as _os
    import sys as _sys

    for key in list(_sys.modules.keys()):
        if key == 'synth_ai' or key.startswith('synth_ai.'):
            _sys.modules.pop(key, None)

    _importlib.invalidate_caches()

    try:
        if MODULE_FILE and _os.path.exists(MODULE_FILE):
            spec = importlib.util.spec_from_file_location(MODULE_NAME or 'task_app_module', MODULE_FILE)
            if not spec or not spec.loader:
                raise RuntimeError('Failed to prepare spec for: ' + str(MODULE_FILE))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME or 'task_app_module'] = mod
            spec.loader.exec_module(mod)
        else:
            try:
                importlib.import_module(MODULE_NAME)
            except Exception:
                fallback_file = '/opt/synth_ai_repo/__local_task_app__/' + (MODULE_NAME.split('.')[-1] if MODULE_NAME else 'task_app') + '.py'
                if os.path.exists(fallback_file):
                    spec = importlib.util.spec_from_file_location(MODULE_NAME or 'task_app_module', fallback_file)
                    if not spec or not spec.loader:
                        raise RuntimeError('Failed to prepare fallback spec for: ' + str(fallback_file))
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[MODULE_NAME or 'task_app_module'] = mod
                    spec.loader.exec_module(mod)
                else:
                    raise
    except Exception as exc:
        raise RuntimeError('Task app import failed: ' + str(exc))

    from synth_ai.task.apps import registry as _registry
    from synth_ai.task.server import create_task_app as _create_task_app

    entry = _registry.get(ENTRY_ID)
    cfg = entry.modal
    if cfg is None:
        raise RuntimeError('Modal configuration missing for task app ' + ENTRY_ID)
    config = entry.config_factory()
    return _create_task_app(config)
"""

    with tempfile.NamedTemporaryFile("w", suffix=f"_{entry.app_id}_modal.py", delete=False) as tmp:
        tmp.write(script)
        tmp.flush()
        name = tmp.name
    return Path(name)


def _prepare_runtime_env(*, require_synth: bool, perform_preflight: bool) -> None:
    """Load persisted credentials into the process and ensure required keys exist."""

    hydrate_user_environment(override=False)
    ensure_env_credentials(require_synth=require_synth)
    if perform_preflight:
        preflight_env_key(crash_on_failure=True)


def _run_modal_with_entry(
    entry: TaskAppEntry,
    modal_cfg: ModalDeploymentConfig,
    modal_cli: str,
    modal_name: str | None,
    command: str,
    *,
    dry_run: bool = False,
    original_path: Path | None = None,
) -> str | None:
    _prepare_modal_env(require_synth=False)

    inline_secret_values: dict[str, str] = {}
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if env_key:
        inline_secret_values["ENVIRONMENT_API_KEY"] = env_key
        inline_secret_values.setdefault("DEV_ENVIRONMENT_API_KEY", env_key)
    aliases = (os.environ.get("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    if aliases:
        inline_secret_values["ENVIRONMENT_API_KEY_ALIASES"] = aliases
    for vendor_key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        value = (os.environ.get(vendor_key) or "").strip()
        if value:
            inline_secret_values[vendor_key] = value

    if inline_secret_values:
        preview = inline_secret_values.get("ENVIRONMENT_API_KEY", "")
        shown = f"{preview[:6]}...{preview[-4:]}" if preview and len(preview) > 10 else preview
        click.echo(f"[deploy] inline ENVIRONMENT_API_KEY prepared ({shown})")
    else:
        click.echo("[deploy] no inline ENVIRONMENT_API_KEY found; relying on Modal secrets")

    script_path = _write_modal_entrypoint(
        entry,
        modal_cfg,
        modal_name,
        original_path=original_path,
        inline_secret_values=inline_secret_values,
    )

    cmd = [*_modal_command_prefix(modal_cli), command, str(script_path)]
    if modal_name and command == "deploy":
        cmd.extend(["--name", modal_name])

    proc_env = os.environ.copy()
    pythonpath_entries: list[str] = [str(REPO_ROOT)]
    if original_path is not None:
        pythonpath_entries.insert(0, str(Path(original_path).resolve().parent))
    existing_pp = proc_env.get("PYTHONPATH")
    if existing_pp:
        pythonpath_entries.append(existing_pp)
    proc_env["PYTHONPATH"] = os.pathsep.join(list(dict.fromkeys(pythonpath_entries)))

    if dry_run:
        click.echo("Dry run: " + " ".join(cmd))
        script_path.unlink(missing_ok=True)
        return None

    try:
        return _run_modal_subprocess(cmd, env=proc_env)
    finally:
        script_path.unlink(missing_ok=True)


def _resolve_trace_options(
    trace_dir: str | None,
    trace_db: str | None,
) -> tuple[str | None, str | None]:
    demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
    if trace_dir is None:
        click.echo(
            "\nTracing captures rollout data (actions, rewards, model outputs) to a local SQLite DB."
        )
        click.echo("This data can be exported to JSONL for supervised fine-tuning (SFT).")
        enable_tracing = click.confirm("Enable tracing?", default=True)
        if enable_tracing:
            default_trace_dir = str((demo_base / "traces").resolve())
            trace_dir = click.prompt(
                "Trace directory", type=str, default=default_trace_dir, show_default=True
            )
    if trace_dir and trace_db is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_trace_db = (Path(trace_dir).expanduser().resolve() / f"task_app_traces_{timestamp}.db")
        trace_db = click.prompt(
            "Trace DB path",
            type=str,
            default=str(default_trace_db),
            show_default=True,
        )
    return trace_dir, trace_db


def _serve_cli(
    app_id: str | None,
    host: str,
    port: int | None,
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
    *,
    allow_demo_dir: bool = True,
    choice: AppChoice | None = None,
) -> None:
    if allow_demo_dir:
        _maybe_use_demo_dir()

    if port is None:
        port = click.prompt("Port to serve on", type=int, default=8001)

    trace_dir, trace_db = _resolve_trace_options(trace_dir, trace_db)

    if choice is None:
        choice = select_app_choice(app_id, purpose="serve")

    entry = choice.ensure_entry()

    _prepare_runtime_env(require_synth=True, perform_preflight=True)

    if trace_dir or trace_db:
        _apply_tracing_configuration(trace_dir, trace_db)

    ensure_port_free(port, host, force=force)

    run_task_app(
        entry.config_factory,
        host=host,
        port=port,
        reload=reload_flag,
    )


# ---------------------------------------------------------------------------
# Click command group
# ---------------------------------------------------------------------------


@click.group(name="task-app", help="Utilities for serving and deploying Synth task apps.")
def task_app_group() -> None:
    """Container for task app subcommands."""


# ---------------------------------------------------------------------------
# Public registration helper
# ---------------------------------------------------------------------------


def register(cli: click.Group) -> None:
    """Attach task app commands to an external Click group."""

    importlib.import_module(".task_app_list", __package__)
    importlib.import_module(".task_app_deploy", __package__)
    importlib.import_module(".task_app_modal_serve", __package__)
    serve_module = importlib.import_module(".task_app_serve", __package__)

    cli.add_command(serve_module.serve_command)
    cli.add_command(task_app_group, name="task-app")
    cli.add_command(task_app_group.commands["deploy"], name="deploy")
    cli.add_command(task_app_group.commands["modal-serve"], name="modal-serve")


__all__ = [
    "task_app_group",
    "register",
]
