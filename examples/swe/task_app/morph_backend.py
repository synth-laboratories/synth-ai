"""Utility classes for running swe-mini environments on Morph Cloud."""

from __future__ import annotations

import contextlib
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict

_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - optional dependency
    from morphcloud.api import MorphCloudClient
except Exception as exc:  # pragma: no cover - optional dependency
    MorphCloudClient = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def _quote_env_var(key: str, value: str) -> str:
    """Return a safe shell export statement."""
    return f"export {key}={shlex.quote(value)}"


def _now() -> float:
    return time.time()


@dataclass
class MorphSandboxBackend:
    """Thin wrapper around Morph Cloud instances for command execution.

    The API mirrors the subset consumed by :class:`MiniSweEnvironmentWrapper`:
    we expose an ``execute`` method that matches the mini-swe environment shape.
    """

    snapshot_id: str | None = None
    image_id: str | None = None
    cwd: str = "/workspace"
    env: Dict[str, str] | None = None
    metadata: Dict[str, str] | None = None
    vcpus: int = 4
    memory_mb: int = 8192
    disk_mb: int = 65536
    startup_timeout: int = 600

    _client: MorphCloudClient = field(init=False)
    _instance: Any = field(init=False, default=None)
    _last_exec: Dict[str, Any] = field(init=False, default_factory=dict)
    _started_at: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if MorphCloudClient is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "morphcloud package is required for Morph environments. "
                "Install with `pip install morphcloud`."
            ) from _IMPORT_ERROR

        api_key = os.getenv("MORPH_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set MORPH_API_KEY before using the Morph backend.")

        # Normalise metadata/env early to avoid shared references.
        self.metadata = {str(k): str(v) for k, v in (self.metadata or {}).items()}
        self.env = {str(k): str(v) for k, v in (self.env or {}).items()}
        self.cwd = self.cwd or "/workspace"

        self._client = MorphCloudClient()

    # Public API -----------------------------------------------------------------

    def execute(self, command: str, timeout: int | None = None) -> Dict[str, Any]:
        """Execute ``command`` inside the Morph instance."""
        if not command.strip():
            command = "true"

        instance = self._ensure_instance()

        script_parts = []
        for key, value in self.env.items():
            script_parts.append(_quote_env_var(key, value))
        if self.cwd:
            script_parts.append(f"cd {shlex.quote(self.cwd)}")
        script_parts.append(command)

        script = " && ".join(script_parts)
        if timeout:
            wrapped = f"timeout {int(timeout)}s bash -lc {shlex.quote(script)}"
        else:
            wrapped = script

        shell_cmd = f"bash -lc {shlex.quote(wrapped)}"
        started = _now()
        result = instance.exec(shell_cmd)
        duration = _now() - started

        payload = {
            "output": (result.stdout or ""),
            "stderr": (result.stderr or ""),
            "returncode": getattr(result, "exit_code", None),
            "duration": duration,
        }
        self._last_exec = payload
        return payload

    def close(self) -> None:
        """Stops the Morph instance if one is running."""
        instance = getattr(self, "_instance", None)
        if not instance:
            return
        try:
            instance.stop()
        except Exception:  # pragma: no cover - best-effort shutdown
            pass
        finally:
            self._instance = None

    # Internal helpers -----------------------------------------------------------

    def _ensure_instance(self):
        instance = getattr(self, "_instance", None)
        if instance is not None:
            return instance

        snapshot_id = (
            self.snapshot_id
            or os.getenv("SWE_MINI_MORPH_SNAPSHOT_ID")
            or os.getenv("MORPH_SNAPSHOT_ID")
        )
        metadata = dict(self.metadata)

        if snapshot_id:
            instance = self._client.instances.start(snapshot_id=snapshot_id, metadata=metadata or None)
        else:
            image_id = (
                self.image_id
                or os.getenv("SWE_MINI_MORPH_IMAGE_ID")
                or os.getenv("MORPH_IMAGE_ID")
                or "morphvm-minimal"
            )
            snapshot = self._client.snapshots.create(
                image_id=image_id,
                vcpus=self.vcpus,
                memory=self.memory_mb,
                disk_size=self.disk_mb,
            )
            instance = self._client.instances.start(snapshot_id=snapshot.id, metadata=metadata or None)
            self.snapshot_id = snapshot.id

        self._instance = instance
        self._started_at = _now()
        self._wait_until_ready(instance)
        self._ensure_cwd(instance)
        return instance

    def _wait_until_ready(self, instance) -> None:
        deadline = _now() + float(self.startup_timeout)
        while True:
            try:
                instance.wait_until_ready()
                break
            except Exception as exc:  # pragma: no cover - SDK may raise while polling
                if _now() > deadline:
                    raise TimeoutError(f"Morph instance did not become ready within {self.startup_timeout}s") from exc
                time.sleep(5.0)

    def _ensure_cwd(self, instance) -> None:
        if not self.cwd:
            return
        try:
            instance.exec(f"bash -lc {shlex.quote(f'mkdir -p {self.cwd}')}")
        except Exception as exc:  # pragma: no cover - surface friendly error
            raise RuntimeError(f"Failed to create remote workspace {self.cwd!r}: {exc}") from exc

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.close()
