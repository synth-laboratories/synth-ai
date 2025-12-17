"""Queue management commands for running Celery workers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import click

# Clear config cache if env vars are set (must happen before other imports)
if os.getenv("EXPERIMENT_QUEUE_DB_PATH") or os.getenv("EXPERIMENT_QUEUE_TRAIN_CMD"):
    from synth_ai.cli.local.experiment_queue import config as queue_config

    queue_config.reset_config_cache()


def _require_celery_binary() -> str:
    celery_path = shutil.which("celery")
    if not celery_path:
        # Check if we're in a virtual environment and celery is installed there
        venv_bin = os.environ.get("VIRTUAL_ENV")
        if venv_bin:
            venv_celery = Path(venv_bin) / "bin" / "celery"
            if venv_celery.exists():
                celery_path = str(venv_celery)
        # Also check common uv venv locations
        if not celery_path:
            import sys
            if hasattr(sys, "executable"):
                venv_base = Path(sys.executable).parent.parent
                uv_celery = venv_base / "bin" / "celery"
                if uv_celery.exists():
                    celery_path = str(uv_celery)
    if not celery_path:
        raise click.ClickException(
            "Celery executable not found on PATH. Install it with `uv pip install celery` or ensure it is available."
        )
    return celery_path


def _background_log_file() -> Path:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "experiment_queue_worker.log"


def _worker_lock_file() -> Path:
    """Return path to lock file for ensuring only one worker runs."""
    lock_dir = Path("logs")
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / "experiment_queue_worker.lock"


def _kill_all_existing_workers() -> int:
    """Kill ALL existing Celery workers for experiment queue.
    
    Returns the number of workers killed.
    """
    killed = 0
    try:
        import psutil  # type: ignore[import-untyped]
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue
                cmdline_str = ' '.join(cmdline)
                # Check if this is a Celery worker for our experiment queue
                if ('celery' in cmdline_str.lower() and 
                    'synth_ai.experiment_queue' in cmdline_str):
                    click.echo(f"Killing existing worker (PID: {proc.info['pid']})", err=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except ImportError:
        # Fallback to pgrep/pkill if psutil not available
        import subprocess
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'celery.*synth_ai.experiment_queue'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        click.echo(f"Killing existing worker (PID: {pid})", err=True)
                        try:
                            subprocess.run(['kill', '-TERM', pid], timeout=3)
                            import time as time_module  # Import here to avoid F823 false positive
                            time_module.sleep(1)
                            subprocess.run(['kill', '-9', pid], timeout=1)
                            killed += 1
                        except Exception:
                            pass
        except Exception:
            pass
    
    # Remove lock file if it exists
    lock_file = _worker_lock_file()
    if lock_file.exists():
        lock_file.unlink(missing_ok=True)
    
    if killed > 0:
        import time as time_module
        time_module.sleep(2)  # Give processes time to fully terminate
    
    return killed


def _get_running_workers() -> list[dict]:
    """Get all running experiment queue workers.
    
    Returns list of dicts with 'pid', 'cmdline', 'db_path', 'broker_url' keys.
    """
    workers = []
    try:
        import psutil  # type: ignore[import-untyped]
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue
                cmdline_str = ' '.join(cmdline)
                # Check if this is a Celery worker for our experiment queue
                if ('celery' in cmdline_str.lower() and 
                    'synth_ai.experiment_queue' in cmdline_str):
                    env = proc.environ()
                    workers.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline_str,
                        'db_path': env.get('EXPERIMENT_QUEUE_DB_PATH', 'N/A'),
                        'broker_url': env.get('EXPERIMENT_QUEUE_BROKER_URL', 'N/A'),
                        'result_backend_url': env.get('EXPERIMENT_QUEUE_RESULT_BACKEND_URL', 'N/A'),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except ImportError:
        # Fallback to pgrep if psutil not available
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'celery.*synth_ai.experiment_queue'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        workers.append({
                            'pid': int(pid.strip()),
                            'cmdline': 'N/A (psutil not available)',
                            'db_path': 'N/A',
                            'broker_url': 'N/A',
                            'result_backend_url': 'N/A',
                        })
        except Exception:
            pass
    
    return workers


def _check_existing_worker() -> bool:
    """Check if a worker is already running by checking lock file."""
    lock_file = _worker_lock_file()
    if not lock_file.exists():
        return False
    
    try:
        # Read PID from lock file
        pid = int(lock_file.read_text().strip())
        # Check if process is still running
        import psutil  # type: ignore[import-untyped]
        try:
            proc = psutil.Process(pid)
            # Check if it's actually a celery worker
            cmdline = ' '.join(proc.cmdline())
            if 'celery' in cmdline.lower() and 'synth_ai.experiment_queue' in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process doesn't exist or we can't access it - remove stale lock
            lock_file.unlink(missing_ok=True)
            return False
    except (ValueError, FileNotFoundError):
        # Invalid lock file - remove it
        lock_file.unlink(missing_ok=True)
        return False
    
    return False


@click.command("status")
def status_cmd() -> None:
    """Show status of experiment queue workers."""
    workers = _get_running_workers()
    
    # Get current config
    db_path = os.getenv("EXPERIMENT_QUEUE_DB_PATH", "NOT SET")
    broker_url = os.getenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0 (default)")
    result_backend_url = os.getenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1 (default)")
    
    click.echo("=" * 60)
    click.echo("Experiment Queue Status")
    click.echo("=" * 60)
    click.echo()
    
    click.echo("Configuration:")
    click.echo(f"  EXPERIMENT_QUEUE_DB_PATH: {db_path}")
    click.echo(f"  EXPERIMENT_QUEUE_BROKER_URL: {broker_url}")
    click.echo(f"  EXPERIMENT_QUEUE_RESULT_BACKEND_URL: {result_backend_url}")
    click.echo()
    
    if not workers:
        click.echo("❌ No workers running")
        click.echo()
        click.echo("Start a worker with: synth-ai queue start")
        return
    
    click.echo(f"✅ {len(workers)} worker(s) running:")
    click.echo()
    for i, worker in enumerate(workers, 1):
        click.echo(f"Worker {i}:")
        click.echo(f"  PID: {worker['pid']}")
        click.echo(f"  Database: {worker['db_path']}")
        click.echo(f"  Broker: {worker['broker_url']}")
        click.echo(f"  Result Backend: {worker['result_backend_url']}")
        if i < len(workers):
            click.echo()
    
    # Check for database path mismatches
    if db_path != "NOT SET":
        mismatches = [w for w in workers if w['db_path'] != db_path and w['db_path'] != 'N/A']
        if mismatches:
            click.echo()
            click.echo("⚠️  WARNING: Some workers are using different database paths!")
            for worker in mismatches:
                click.echo(f"  Worker PID {worker['pid']}: {worker['db_path']} (expected: {db_path})")


@click.command("stop")
def stop_cmd() -> None:
    """Stop all running experiment queue workers."""
    workers = _get_running_workers()
    if not workers:
        click.echo("No experiment queue workers running.")
        return
    
    click.echo(f"Found {len(workers)} worker(s) to stop...")
    killed = _kill_all_existing_workers()
    
    if killed > 0:
        click.echo(f"✅ Stopped {killed} worker(s)")
    else:
        click.echo("⚠️  No workers were stopped (they may have already exited)")


@click.group("queue", invoke_without_command=True)
@click.pass_context
def queue_group(ctx: click.Context) -> None:
    """Manage the experiment queue Celery worker.
    
    Use subcommands to start, stop, or check status of the worker.
    """
    if ctx.invoked_subcommand is None:
        # Default to status if no subcommand
        ctx.invoke(status_cmd)


@queue_group.command("start")
@click.option(
    "--concurrency",
    default=None,
    type=int,
    help="Worker concurrency (default: 5, or EXPERIMENT_QUEUE_WORKER_CONCURRENCY env var).",
)
@click.option(
    "--loglevel",
    default="info",
    show_default=True,
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    help="Logging level for Celery worker and task queue (debug/info/warning/error/critical).",
)
@click.option(
    "--pool",
    default="prefork",
    show_default=True,
    type=click.Choice(["solo", "prefork", "threads", "gevent", "eventlet"]),
    help="Worker pool type (default 'prefork' for Redis broker).",
)
@click.option(
    "--background/--foreground",
    default=True,
    show_default=True,
    help="Run worker in background (default) or foreground.",
)
@click.option(
    "--beat/--no-beat",
    default=True,
    show_default=True,
    help="Run Celery Beat scheduler in the same process (for periodic queue checks).",
)
@click.option(
    "--local/--no-local",
    default=None,
    help="Use local backend (localhost:8000) instead of production. "
    "Sets EXPERIMENT_QUEUE_LOCAL=true. Default: use production backend.",
)
@click.option(
    "--extra",
    multiple=True,
    help="Extra arguments forwarded to celery worker (use multiple times).",
)
def start_cmd(
    concurrency: int | None,
    loglevel: str,
    pool: str,
    background: bool,
    beat: bool,
    local: bool | None,
    extra: tuple[str, ...],
) -> None:
    """Start the experiment queue Celery worker.
    
    The periodic queue check task runs every 5 seconds to dispatch queued jobs.
    Use --beat to run Beat scheduler in the same process (default: enabled).
    
    Backend URL Configuration:
    - Default: Production backend (https://api.usesynth.ai/api)
    - Use --local flag to connect to local backend (http://localhost:8000/api)
    - Override with EXPERIMENT_QUEUE_BACKEND_URL env var for custom URL
    
    Database Configuration:
    - Uses EXPERIMENT_QUEUE_DB_PATH if set, otherwise defaults to ~/.synth_ai/experiment_queue.db
    
    Concurrency:
    - Defaults to 5 (or EXPERIMENT_QUEUE_WORKER_CONCURRENCY env var), allowing up to 5
      experiments/jobs to run in parallel. Override with --concurrency flag.
    
    Logging:
    - Use --loglevel to control verbosity (debug/info/warning/error/critical)
    - Controls both Celery worker logs and task queue internal logs (poller, etc.)
    - Default: info (shows INFO and above)
    - Use debug for verbose output including API polling details
    
    Examples:
        # Start worker with production backend (default)
        synth-ai queue start
        
        # Start worker with local backend
        synth-ai queue start --local
        
        # Start worker with verbose logging (debug level)
        synth-ai queue start --loglevel debug
        
        # Start worker with custom backend URL
        EXPERIMENT_QUEUE_BACKEND_URL=http://localhost:8000/api synth-ai queue start
    """
    # Determine concurrency: flag > env var > default
    if concurrency is None:
        concurrency = int(os.getenv("EXPERIMENT_QUEUE_WORKER_CONCURRENCY", "5"))
    # CRITICAL: Kill ALL existing workers to ensure only one instance
    killed = _kill_all_existing_workers()
    if killed > 0:
        click.echo(f"Killed {killed} existing worker(s)", err=True)
    
    # CRITICAL: Clear config cache before starting worker to ensure fresh config
    from synth_ai.cli.local.experiment_queue import config as queue_config
    queue_config.reset_config_cache()
    
    # Load config to get database path (uses EXPERIMENT_QUEUE_DB_PATH if set, otherwise default)
    config = queue_config.load_config()
    db_path_resolved = config.sqlite_path.resolve()
    
    # Log which path is being used
    db_path_env = os.getenv("EXPERIMENT_QUEUE_DB_PATH")
    if db_path_env:
        click.echo(f"Using database from EXPERIMENT_QUEUE_DB_PATH: {db_path_resolved}", err=True)
    else:
        click.echo(f"Using default database path: {db_path_resolved}", err=True)
    
    # Ensure parent directory exists
    db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL: Verify this is the ONLY database path being used
    # Check for any other workers using different database paths
    try:
        import psutil  # type: ignore[import-untyped]
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue
                cmdline_str = ' '.join(cmdline)
                if ('celery' in cmdline_str.lower() and 
                    'synth_ai.experiment_queue' in cmdline_str):
                    # Check env vars of this process
                    env = proc.environ()
                    other_db = env.get('EXPERIMENT_QUEUE_DB_PATH')
                    if other_db and Path(other_db).resolve() != db_path_resolved:
                        raise click.ClickException(
                            f"Another worker is using different database: {other_db} "
                            f"(this worker: {db_path_resolved}). "
                            f"All workers must use the same database path."
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass  # psutil not available, skip check
    
    celery_bin = _require_celery_binary()
    base_cmd = [
        celery_bin,
        "-A",
        "synth_ai.cli.local.experiment_queue.celery_app",
        "worker",
        "--loglevel",
        loglevel,
        "--concurrency",
        str(concurrency),
        "--pool",
        pool,
    ]
    if beat:
        base_cmd.append("--beat")
    if extra:
        base_cmd.extend(extra)

    # Lock file check is now redundant since we kill all workers above
    # But keep it as a safety check
    if _check_existing_worker():
        lock_file = _worker_lock_file()
        # This shouldn't happen since we killed workers above, but be safe
        click.echo("WARNING: Lock file exists but worker should be killed", err=True)
        lock_file.unlink(missing_ok=True)
    
    env = os.environ.copy()
    # Ensure EXPERIMENT_QUEUE_DB_PATH is explicitly set in worker environment
    # Use the resolved path from config (either from env var or default)
    env["EXPERIMENT_QUEUE_DB_PATH"] = str(db_path_resolved)
    
    # Set Python logging level for task queue loggers (poller, etc.)
    # This controls our custom loggers, while --loglevel controls Celery's logger
    env["EXPERIMENT_QUEUE_LOG_LEVEL"] = loglevel.upper()
    
    # Set EXPERIMENT_QUEUE_LOCAL if --local flag is provided
    if local is True:
        env["EXPERIMENT_QUEUE_LOCAL"] = "true"
        click.echo("Using local backend (localhost:8000)", err=True)
    elif local is False:
        # Explicitly unset if --no-local is used
        env.pop("EXPERIMENT_QUEUE_LOCAL", None)
        click.echo("Using production backend (api.usesynth.ai)", err=True)
    else:
        # Use existing env var or default (production)
        if "EXPERIMENT_QUEUE_LOCAL" not in env:
            click.echo("Using production backend (api.usesynth.ai)", err=True)

    # Create lock file with current PID before starting worker
    lock_file = _worker_lock_file()
    lock_file.write_text(str(os.getpid()))
    
    def cleanup_lock():
        """Remove lock file on exit."""
        from contextlib import suppress
        with suppress(Exception):
            lock_file.unlink(missing_ok=True)
    
    import atexit
    atexit.register(cleanup_lock)

    if background:
        log_path = _background_log_file()
        with open(log_path, "ab") as handle:
            proc = subprocess.Popen(
                base_cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        click.echo(f"Started experiment queue worker (pid={proc.pid}) in background.")
        click.echo(f"Logs streaming to {log_path}")
    else:
        click.echo("Starting experiment queue worker (foreground)...")
        subprocess.run(base_cmd, check=True, env=env)


# Register subcommands (already registered via decorators, but ensure they're all added)
queue_group.add_command(stop_cmd, name="stop")
queue_group.add_command(status_cmd, name="status")


def register(cli: click.Group) -> None:
    """Register queue command with CLI."""
    cli.add_command(queue_group, name="queue")
