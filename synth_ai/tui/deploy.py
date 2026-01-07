"""Deploy a LocalAPI task app with Cloudflare tunnel and streaming logs.

Usage:
    python -m synth_ai.tui.deploy /path/to/localapi.py [--deployment-id ID]

Outputs NDJSON (newline-delimited JSON) to stdout:
    {"type":"status","status":"starting","message":"...","deployment_id":"..."}
    {"type":"log","source":"app","message":"...","timestamp":1234567890.123}
    {"type":"log","source":"uvicorn","level":"INFO","message":"...","timestamp":...}
    {"type":"status","status":"ready","url":"https://...","port":8001,"deployment_id":"..."}
    {"type":"log","source":"cloudflare","message":"...","timestamp":...}

The process stays alive to keep the tunnel open. Kill it to tear down.
"""

import argparse
import asyncio
import importlib.util
import inspect
import io
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TextIO


class LogAggregator:
    """Aggregates logs from all sources and outputs as NDJSON to stdout.

    Thread-safe queue that collects log entries from multiple sources
    (app stdout, uvicorn, cloudflare) and outputs them as newline-delimited JSON.
    Also persists logs to separate files for deploy status vs runtime logs.
    """

    def __init__(self, deployment_id: str, persist_dir: Path | None = None):
        self.deployment_id = deployment_id
        self.queue: queue.Queue[dict | None] = queue.Queue()
        self._running = True
        self._deploy_handle: TextIO | None = None
        self._runtime_handle: TextIO | None = None
        self._original_stdout = sys.__stdout__

        # Set up file persistence
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            timestamp = _timestamp_for_filename()
            self._deploy_handle = open(persist_dir / f"{deployment_id}_deploy_{timestamp}.jsonl", "a", encoding="utf-8")
            self._runtime_handle = open(persist_dir / f"{deployment_id}_serve_{timestamp}.jsonl", "a", encoding="utf-8")

        # Start writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"log-writer-{deployment_id}"
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        """Background thread that drains the queue and outputs to stdout."""
        while self._running or not self.queue.empty():
            try:
                entry = self.queue.get(timeout=0.1)
                if entry is None:
                    continue

                # Add deployment_id to entry
                entry["deployment_id"] = self.deployment_id
                line = json.dumps(entry)

                # Output to stdout for TUI (use original stdout to avoid recursion)
                self._original_stdout.write(line + "\n")
                self._original_stdout.flush()

                # Persist to files
                if self._deploy_handle and entry.get("type") == "status":
                    self._deploy_handle.write(line + "\n")
                    self._deploy_handle.flush()
                if self._runtime_handle and entry.get("type") == "log":
                    self._runtime_handle.write(line + "\n")
                    self._runtime_handle.flush()

            except queue.Empty:
                continue
            except Exception:
                # Don't crash the writer thread on errors
                pass

    def put(self, entry: dict) -> None:
        """Add a log entry to the queue."""
        self.queue.put(entry)

    def log(self, source: str, message: str, level: str = "INFO") -> None:
        """Convenience method to log a message."""
        self.put({
            "type": "log",
            "source": source,
            "message": message,
            "level": level,
            "timestamp": time.time(),
        })

    def status(self, status: str, **kwargs) -> None:
        """Convenience method to output a status message."""
        self.put({
            "type": "status",
            "status": status,
            "timestamp": time.time(),
            **kwargs,
        })

    def stop(self) -> None:
        """Stop the writer thread and close resources."""
        self._running = False
        self.queue.put(None)  # Wake up the writer thread
        self._writer_thread.join(timeout=2.0)
        if self._deploy_handle:
            self._deploy_handle.close()
        if self._runtime_handle:
            self._runtime_handle.close()


class StreamCapture(io.TextIOWrapper):
    """Captures writes to stdout/stderr and forwards to log queue.

    Wraps sys.stdout/stderr to intercept print() calls and other writes,
    forwarding them both to the original stream and to the log aggregator.
    """

    def __init__(
        self,
        original_stream: TextIO,
        aggregator: LogAggregator,
        source: str
    ):
        self.original = original_stream
        self.aggregator = aggregator
        self.source = source
        self._buffer = ""

    def write(self, data: str) -> int:
        """Intercept writes and forward to aggregator."""
        if data:
            # Buffer until we get a newline (to handle partial writes)
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                if line.strip():  # Ignore empty lines
                    self.aggregator.log(self.source, line.rstrip())

        # Also write to original stream
        return self.original.write(data)

    def flush(self) -> None:
        """Flush both buffer and original stream."""
        # Flush any remaining buffer content
        if self._buffer.strip():
            self.aggregator.log(self.source, self._buffer.rstrip())
            self._buffer = ""
        self.original.flush()

    def fileno(self) -> int:
        """Return the file descriptor of the original stream."""
        return self.original.fileno()

    def isatty(self) -> bool:
        """Check if original stream is a tty."""
        return self.original.isatty()

    @property
    def encoding(self) -> str:
        """Return encoding of original stream."""
        return getattr(self.original, "encoding", "utf-8")


class QueueLogHandler(logging.Handler):
    """Logging handler that forwards log records to the aggregator queue.

    Attaches to uvicorn loggers to capture server logs.
    """

    def __init__(self, aggregator: LogAggregator, source: str = "uvicorn"):
        super().__init__()
        self.aggregator = aggregator
        self.source = source
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the aggregator."""
        try:
            message = self.format(record)
            self.aggregator.put({
                "type": "log",
                "source": self.source,
                "level": record.levelname,
                "message": message,
                "timestamp": record.created,
            })
        except Exception:
            # Don't crash on logging errors
            pass


def _setup_uvicorn_logging(aggregator: LogAggregator) -> None:
    """Configure uvicorn loggers to use our aggregator."""
    handler = QueueLogHandler(aggregator, source="uvicorn")
    handler.setLevel(logging.INFO)

    # Configure all uvicorn loggers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        # Prevent duplicate logs by not propagating to root
        logger.propagate = False


def _validate_localapi(module: ModuleType, path: Path) -> str | None:
    """
    Validate a LocalAPI module before deployment.

    Returns None if valid, or an error message if invalid.
    """
    from fastapi import FastAPI

    # Check app exists and is FastAPI
    if not hasattr(module, "app"):
        return "LocalAPI must define an 'app' variable"

    app = module.app
    if not isinstance(app, FastAPI):
        return f"'app' must be a FastAPI instance, got {type(app).__name__}"

    # Check required routes exist
    routes = {route.path for route in app.routes}
    if "/health" not in routes:
        return "LocalAPI missing /health endpoint (use create_local_api() to create your app)"
    if "/rollout" not in routes:
        return "LocalAPI missing /rollout endpoint (use create_local_api() to create your app)"

    # Check get_dataset_size is implemented
    if hasattr(module, "get_dataset_size"):
        get_dataset_size = module.get_dataset_size
        if callable(get_dataset_size):
            try:
                result = get_dataset_size()
                if not isinstance(result, int):
                    return f"get_dataset_size() must return an int, got {type(result).__name__}"
                if result <= 0:
                    return f"get_dataset_size() must return a positive number, got {result}"
            except NotImplementedError as e:
                return f"get_dataset_size() not implemented: {e}"
            except Exception as e:
                return f"get_dataset_size() failed: {e}"

    # Check get_sample is implemented
    if hasattr(module, "get_sample"):
        get_sample = module.get_sample
        if callable(get_sample):
            try:
                # Try calling with seed=0 to catch NotImplementedError
                result = get_sample(0)
                if not isinstance(result, dict):
                    return f"get_sample() must return a dict, got {type(result).__name__}"
            except NotImplementedError as e:
                return f"get_sample() not implemented: {e}"
            except Exception as e:
                return f"get_sample(0) failed: {e}"

    # Check score_response is implemented
    if hasattr(module, "score_response"):
        score_response = module.score_response
        if callable(score_response):
            # Check function signature
            sig = inspect.signature(score_response)
            params = list(sig.parameters.keys())
            if len(params) < 2:
                return f"score_response() must accept (response, sample), got {params}"

            try:
                # Try calling with dummy data to catch NotImplementedError
                result = score_response("test response", {"input": "test", "expected": "test"})
                if not isinstance(result, (int, float)):
                    return f"score_response() must return a number, got {type(result).__name__}"
            except NotImplementedError as e:
                return f"score_response() not implemented: {e}"
            except Exception:
                # Other errors might be OK - could be due to dummy data not matching expected format
                pass

    return None


def _timestamp_for_filename() -> str:
    now = datetime.now()
    return f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a LocalAPI with streaming logs.")
    parser.add_argument("localapi_path", help="Path to localapi.py")
    parser.add_argument("--deployment-id", dest="deployment_id", default=None, help="Optional deployment identifier")
    return parser.parse_args()


async def deploy_localapi(localapi_path: str, deployment_id: str | None = None) -> None:
    """Deploy a LocalAPI file with streaming logs."""
    # Use provided deployment ID or generate one
    path = Path(localapi_path).resolve()
    if deployment_id is None:
        deployment_id = f"{path.stem}_{_timestamp_for_filename()}"

    # Set up log directory
    persist_dir = Path.home() / ".synth-ai" / "tui" / "logs"

    # Create log aggregator
    aggregator = LogAggregator(deployment_id, persist_dir)

    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Check for local mode (skip Cloudflare tunnel, use localhost directly)
        LOCAL_MODE = os.environ.get("SYNTH_LOCAL_MODE", "").lower() in ("1", "true", "yes")

        aggregator.status("starting", message="Initializing deployment...")

        # Check for API key early
        if not os.environ.get("SYNTH_API_KEY"):
            aggregator.status("error", error="SYNTH_API_KEY not set - run 'synth auth' or export SYNTH_API_KEY")
            return

        from synth_ai.sdk.localapi.auth import ensure_localapi_auth
        from synth_ai.sdk.task import run_server_background
        from synth_ai.sdk.tunnels import (
            TunnelBackend,
            TunneledLocalAPI,
            wait_for_health_check,
            acquire_port,
            PortConflictBehavior,
        )

        # Get environment API key
        aggregator.log("app", "Getting environment API key...")
        try:
            # In local mode, use localhost backend for auth; otherwise use env vars or production
            backend_base = "http://localhost:8000" if LOCAL_MODE else None
            synth_api_key = os.environ.get("SYNTH_API_KEY")
            env_api_key = ensure_localapi_auth(
                backend_base=backend_base,
                synth_api_key=synth_api_key,
            )
        except Exception as e:
            aggregator.status("error", error=f"Failed to get environment API key: {e}")
            return

        if not path.exists():
            aggregator.status("error", error=f"File not found: {localapi_path}")
            return

        # Redirect stdout/stderr BEFORE importing user module
        # This captures any print() statements during module load
        sys.stdout = StreamCapture(original_stdout, aggregator, "app")
        sys.stderr = StreamCapture(original_stderr, aggregator, "app")

        # Import the user's localapi module
        aggregator.log("app", f"Loading module: {path.name}")
        try:
            spec = importlib.util.spec_from_file_location("localapi", path)
            if spec is None or spec.loader is None:
                aggregator.status("error", error=f"Could not load module from: {localapi_path}")
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except SyntaxError as e:
            aggregator.status("error", error=f"Syntax error in {path.name}: {e}")
            return
        except ImportError as e:
            aggregator.status("error", error=f"Import error in {path.name}: {e}")
            return
        except Exception as e:
            aggregator.status("error", error=f"Failed to load {path.name}: {e}")
            return

        # Validate the module
        aggregator.log("app", "Validating LocalAPI module...")
        validation_error = _validate_localapi(module, path)
        if validation_error:
            aggregator.status("error", error=validation_error)
            return

        app = module.app

        # Set up uvicorn logging BEFORE starting server
        _setup_uvicorn_logging(aggregator)

        # Start server on available port
        port = acquire_port(8001, on_conflict=PortConflictBehavior.FIND_NEW)
        aggregator.log("app", f"Starting server on port {port}...")
        try:
            run_server_background(app, port)
        except Exception as e:
            aggregator.status("error", error=f"Failed to start server: {e}")
            return

        # Wait for health check
        aggregator.log("app", "Waiting for health check...")
        try:
            await wait_for_health_check("localhost", port, api_key=env_api_key, timeout=30.0)
        except Exception as e:
            aggregator.status("error", error=f"Health check failed: {e}")
            return

        aggregator.log("app", "Health check passed")

        # Create tunnel (or use localhost in local mode)
        if LOCAL_MODE:
            url = f"http://localhost:{port}"
            aggregator.log("app", f"Local mode enabled, using {url}")
        else:
            aggregator.log("cloudflare", "Creating Cloudflare tunnel...")
            try:
                tunnel = await TunneledLocalAPI.create(
                    local_port=port,
                    backend=TunnelBackend.CloudflareManagedTunnel,
                    env_api_key=env_api_key,
                    progress=False,
                )
                url = tunnel.url
                aggregator.log("cloudflare", f"Tunnel created: {url}")

                # Start background task to stream cloudflare logs
                if hasattr(tunnel, 'process') and tunnel.process:
                    asyncio.create_task(
                        _stream_cloudflare_logs(tunnel.process, aggregator)
                    )
            except Exception as e:
                aggregator.status("error", error=f"Failed to create tunnel: {e}")
                return

        # Output ready status
        aggregator.status("ready", url=url, port=port)

        # Keep alive until killed
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            aggregator.status("stopped", message="Deployment stopped")

    finally:
        # Restore original streams
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        aggregator.stop()


async def _stream_cloudflare_logs(proc, aggregator: LogAggregator) -> None:
    """Stream logs from cloudflared subprocess."""
    import select

    try:
        while proc.poll() is None:
            # Check if there's data to read (non-blocking)
            if proc.stdout:
                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        decoded = line.decode("utf-8", errors="replace").strip()
                        if decoded:
                            aggregator.log("cloudflare", decoded)
            await asyncio.sleep(0.05)
    except Exception:
        # Don't crash on streaming errors
        pass


def main() -> None:
    args = _parse_args()
    asyncio.run(deploy_localapi(args.localapi_path, args.deployment_id))


if __name__ == "__main__":
    main()
