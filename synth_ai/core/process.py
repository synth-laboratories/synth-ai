import os
import signal
import socket
import time
from collections.abc import Iterable
from typing import Any

__all__ = [
    "ensure_local_port_available",
    "popen_capture",
    "popen_stream",
    "popen_stream_capture",
    "should_filter_log_line",
    "get_subprocess_env",
]

# Import log filter to avoid duplication
from synth_ai.core.log_filter import should_filter_log_line


def get_subprocess_env(extra_env: dict[str, Any] | None = None) -> dict[str, str]:
    """Get environment dict for subprocesses with RUST_LOG set to suppress noisy logs.
    
    Always includes RUST_LOG="codex_otel::otel_event_manager=warn" unless overridden.
    """
    env = os.environ.copy()
    # Set RUST_LOG to suppress noisy codex_otel logs by default
    if "RUST_LOG" not in env:
        env["RUST_LOG"] = "codex_otel::otel_event_manager=warn"
    if extra_env:
        env.update(extra_env)
    return env


def popen_capture(
    cmd: list[str], cwd: str | None = None, env: dict[str, Any] | None = None
) -> tuple[int, str]:
    """Execute a subprocess and capture combined stdout/stderr."""
    import subprocess

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=get_subprocess_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out, _ = proc.communicate()
        return int(proc.returncode or 0), out or ""
    except Exception as exc:
        return 1, str(exc)


def popen_stream(
    cmd: list[str], cwd: str | None = None, env: dict[str, Any] | None = None
) -> int:
    """Stream subprocess output line-by-line to stdout for real-time feedback."""
    import subprocess
    import threading

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=get_subprocess_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        print(f"Failed to launch {' '.join(cmd)}: {exc}")
        return 1

    def _pump(stdout) -> None:
        try:
            for line in stdout:
                if not should_filter_log_line(line):
                    print(line.rstrip())
        except Exception:
            pass

    if proc.stdout is not None:
        t = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=1.0)
    else:
        proc.wait()
    return int(proc.returncode or 0)


def popen_stream_capture(
    cmd: list[str], cwd: str | None = None, env: dict[str, Any] | None = None
) -> tuple[int, str]:
    """Stream subprocess output to stdout and also capture it into a buffer."""
    import subprocess
    import threading

    buf_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=get_subprocess_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        print(f"Failed to launch {' '.join(cmd)}: {exc}")
        return 1, ""

    def _pump(stdout) -> None:
        try:
            for line in stdout:
                line = line.rstrip()
                if not should_filter_log_line(line):
                    print(line)
                    buf_lines.append(line)
        except Exception:
            pass

    if proc.stdout is not None:
        t = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        t.start()
        proc.wait()
        t.join(timeout=1.0)
    else:
        proc.wait()
    return int(proc.returncode or 0), "\n".join(buf_lines)


def _list_process_ids(port: int) -> list[int]:
    try:
        import subprocess

        out = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not out.stdout:
            return []
        result: list[int] = []
        for token in out.stdout.strip().splitlines():
            token = token.strip()
            if token.isdigit():
                result.append(int(token))
        return result
    except Exception:
        return []


def _terminate_pids(pids: Iterable[int], *, aggressive: bool) -> bool:
    terminated_any = False
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated_any = True
        except Exception as exc:
            print(f"Failed to terminate PID {pid}: {exc}")
    if terminated_any:
        time.sleep(1.0)

    if aggressive and pids:
        still_running = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            still_running.append(pid)
        if still_running:
            for pid in still_running:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception as exc:
                    print(f"Failed to force terminate PID {pid}: {exc}")
            time.sleep(0.5)
    return terminated_any


def ensure_local_port_available(host: str, port: int, *, force: bool = False) -> bool:
    """Ensure ``host:port`` is free before starting a local server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        in_use = sock.connect_ex((host, port)) == 0
    if not in_use:
        return True

    print(f"Port {port} on {host} is already in use.")
    pids = _list_process_ids(port)

    if pids:
        print("Found processes using this port:")
        for pid in pids:
            print(f"  PID {pid}")
    else:
        print("Could not automatically identify the owning process.")

    if not force:
        try:
            choice = input(f"Stop the existing process on port {port}? [y/N]: ").strip().lower() or "n"
        except Exception:
            choice = "n"
        if not choice.startswith("y"):
            print("Aborting; stop the running server and try again.")
            return False
    else:
        print("Attempting to terminate the existing process...")

    if pids:
        _terminate_pids(pids, aggressive=force)
    else:
        print("Unable to determine owning process. Please stop it manually and retry.")
        return False

    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) != 0:
                print("Port is now available.")
                return True
        time.sleep(0.5)

    print("Port still in use after terminating processes.")
    return False
