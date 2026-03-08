#!/usr/bin/env python3
"""Transparent stdio wrapper for debugging Codex MCP startup.

This preserves the child's MCP stdio stream while logging stdin, stdout, and stderr
to files under /tmp so we can inspect Codex's handshake behavior.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path


LOG_DIR = Path("/tmp/codex_synth_mcp")
ENV_PATH = Path("/Users/joshpurtell/Documents/GitHub/synth-ai/.env")
REPO_ROOT = Path("/Users/joshpurtell/Documents/GitHub/synth-ai")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stderr_pump(src, dst, log_file) -> None:
    while True:
        chunk = src.read(4096)
        if not chunk:
            break
        log_file.write(b"\n[" + _timestamp().encode("ascii") + b"] STDERR\n")
        log_file.write(chunk)
        log_file.flush()
        dst.buffer.write(chunk)
        dst.buffer.flush()


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session = str(int(time.time() * 1000))
    stdin_log = (LOG_DIR / f"{session}.stdin.log").open("ab")
    stdout_log = (LOG_DIR / f"{session}.stdout.log").open("ab")
    stderr_log = (LOG_DIR / f"{session}.stderr.log").open("ab")
    meta_log = (LOG_DIR / f"{session}.meta.log").open("a", encoding="utf-8")

    _load_env_file(ENV_PATH)
    meta_log.write(f"{_timestamp()} start\n")
    meta_log.write(f"cwd={REPO_ROOT}\n")
    meta_log.write(
        "command=uv --directory /Users/joshpurtell/Documents/GitHub/synth-ai run --quiet synth-ai-mcp-managed-research\n"
    )
    meta_log.write(
        f"SYNTH_API_KEY_present={'SYNTH_API_KEY' in os.environ and bool(os.environ['SYNTH_API_KEY'])}\n"
    )
    meta_log.write(
        f"SYNTH_BACKEND_URL={os.environ.get('SYNTH_BACKEND_URL', '')}\n"
    )
    meta_log.flush()

    proc = subprocess.Popen(
        [
            "uv",
            "--directory",
            str(REPO_ROOT),
            "run",
            "--quiet",
            "synth-ai-mcp-managed-research",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )

    stderr_thread = threading.Thread(
        target=_stderr_pump,
        args=(proc.stderr, sys.stderr, stderr_log),
        daemon=True,
    )
    stderr_thread.start()

    try:
        while True:
            read_fds = [sys.stdin.buffer, proc.stdout]
            ready, _, _ = select.select(read_fds, [], [], 0.5)

            if sys.stdin.buffer in ready:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    if proc.stdin:
                        proc.stdin.close()
                else:
                    stdin_log.write(b"\n[" + _timestamp().encode("ascii") + b"] STDIN\n")
                    stdin_log.write(chunk)
                    stdin_log.flush()
                    if proc.stdin:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()

            if proc.stdout in ready:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                stdout_log.write(b"\n[" + _timestamp().encode("ascii") + b"] STDOUT\n")
                stdout_log.write(chunk)
                stdout_log.flush()
                os.write(sys.stdout.fileno(), chunk)

            if proc.poll() is not None and not ready:
                break
    finally:
        rc = proc.wait(timeout=5) if proc.poll() is None else proc.returncode
        meta_log.write(f"{_timestamp()} exit rc={rc}\n")
        meta_log.flush()
        stdin_log.close()
        stdout_log.close()
        stderr_log.close()
        meta_log.close()

    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
