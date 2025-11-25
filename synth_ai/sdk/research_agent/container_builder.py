from __future__ import annotations

import asyncio
import base64
import contextlib
import fnmatch
import io
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Tuple

from synth_ai.sdk.research_agent.container_spec import ContainerSpec
from synth_ai.sdk.research_agent.defaults import DEFAULT_BACKEND


class ContainerBackend(ABC):
    """Abstract base for container execution backends."""

    @abstractmethod
    async def provision(self, spec: ContainerSpec) -> str:
        """Provision a new container and return its id/handle."""

    @abstractmethod
    async def execute(
        self,
        container_id: str,
        command: str,
        *,
        env: Dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> Dict[str, str | int]:
        """Execute a command in the container."""

    @abstractmethod
    async def collect_artifacts(self, container_id: str, patterns: Iterable[str]) -> Dict[str, bytes]:
        """Pull artifacts that match any of the glob patterns."""

    @abstractmethod
    async def destroy(self, container_id: str) -> None:
        """Tear down container resources."""


class DockerBackend(ContainerBackend):
    """Docker implementation using docker-py."""

    def __init__(self, *, client=None):
        self._client = client
        self._containers: Dict[str, Tuple[object, ContainerSpec]] = {}

    def _ensure_client(self):
        if self._client is None:
            try:
                import docker  # type: ignore
            except ImportError as exc:
                raise RuntimeError("docker SDK is not installed. Add docker>=7.0.0 to dependencies.") from exc
            self._client = docker.from_env()
        return self._client

    async def provision(self, spec: ContainerSpec) -> str:
        spec.validate()
        client = self._ensure_client()
        context_bytes = spec.build_context()
        image_tag = f"research-agent:{int(time.time())}"

        def _build():
            return client.images.build(
                fileobj=io.BytesIO(context_bytes),
                custom_context=True,
                rm=True,
                nocache=True,
                tag=image_tag,
                buildargs=spec.build_args,
            )

        loop = asyncio.get_running_loop()
        image, _ = await loop.run_in_executor(None, _build)

        container = client.containers.create(
            image=image.id,
            command="sleep infinity",
            environment={**spec.env_vars, **spec.secrets},
            tty=True,
            detach=True,
            working_dir=str(spec.workdir),
        )
        container.start()
        self._containers[container.id] = (container, spec)
        return container.id

    async def execute(
        self,
        container_id: str,
        command: str,
        *,
        env: Dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> Dict[str, str | int]:
        container, spec = self._containers[container_id]
        if env is None:
            env = {}
        exec_env = {**spec.env_vars, **spec.secrets, **env}
        workdir_str = str(workdir or spec.workdir)

        def _run():
            result = container.exec_run(  # type: ignore[attr-defined]
                cmd=["bash", "-lc", command],
                environment=exec_env,
                workdir=workdir_str,
                demux=True,
            )
            stdout, stderr = result.output
            return {
                "exit_code": result.exit_code,
                "stdout": (stdout or b"").decode(),
                "stderr": (stderr or b"").decode(),
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)

    async def collect_artifacts(self, container_id: str, patterns: Iterable[str]) -> Dict[str, bytes]:
        container, spec = self._containers[container_id]
        loop = asyncio.get_running_loop()

        def _pull():
            try:
                stream, _ = container.get_archive(str(spec.artifacts_dir))  # type: ignore[attr-defined]
            except Exception:
                return {}
            tar_bytes = b"".join(stream)
            collected: Dict[str, bytes] = {}
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    relative_name = str(Path(member.name).name)
                    if not any(fnmatch.fnmatch(relative_name, pat) for pat in patterns):
                        continue
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        collected[relative_name] = file_obj.read()
            return collected

        return await loop.run_in_executor(None, _pull)

    async def destroy(self, container_id: str) -> None:
        container, _ = self._containers.pop(container_id, (None, None))
        if container is None:
            return

        def _stop():
            with contextlib.suppress(Exception):
                container.kill()  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                container.remove(force=True)  # type: ignore[attr-defined]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stop)


class ModalBackend(ContainerBackend):
    """Modal implementation using modal SDK. Returns artifacts inline from execute()."""

    def __init__(self):
        self._runs: Dict[str, Dict[str, object]] = {}

    def _write_build_context(self, spec: ContainerSpec) -> tempfile.TemporaryDirectory:
        """Materialize Dockerfile + overlay files to a temp dir for Modal build."""
        temp_dir = tempfile.TemporaryDirectory()
        ctx = Path(temp_dir.name)
        (ctx / "Dockerfile").write_text(spec.to_dockerfile())

        overlay_root = ctx / "overlay_files"
        for rel_path, content in spec.rendered_overlay_files().items():
            target = overlay_root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        for rel_path, content in spec.files.items():
            if not str(rel_path).startswith("/"):
                continue
            data = content.encode() if isinstance(content, str) else content
            target = ctx / str(rel_path).lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        return temp_dir

    async def provision(self, spec: ContainerSpec) -> str:
        spec.validate()
        try:
            import modal  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime import guard
            raise RuntimeError("modal SDK is not installed. Add modal>=1.1.1 to dependencies.") from exc

        ctx_dir = self._write_build_context(spec)
        loop = asyncio.get_running_loop()

        def _build_image():
            return modal.Image.from_dockerfile(
                path=ctx_dir.name,
                build_args=spec.build_args,
                force_build=True,
            )

        image = await loop.run_in_executor(None, _build_image)

        # Combine env_vars and secrets into a Modal Secret
        # Modal function decorator doesn't accept 'env' parameter directly
        # Environment variables must be passed via secrets
        combined_env: dict[str, str | None] = {**spec.env_vars, **spec.secrets}
        secret_obj = None
        if combined_env:
            secret_obj = modal.Secret.from_dict(combined_env)

        app = modal.App(f"oneshot-research-{int(time.time())}")

        workdir_str = str(spec.workdir)
        
        @app.function(
            image=image,
            timeout=60 * 60,
            secrets=[secret_obj] if secret_obj else [],
        )
        def run_task(command: str, patterns: list[str], artifacts_dir: str = "/app/artifacts") -> Dict:
            """Execute the agent and pull artifacts matching patterns."""
            import glob
            import os
            import subprocess

            result = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                cwd=workdir_str,
            )

            artifacts: Dict[str, str] = {}
            for pat in patterns:
                for path in glob.glob(os.path.join(artifacts_dir, pat)):
                    if not os.path.isfile(path):
                        continue
                    name = os.path.basename(path)
                    with open(path, "rb") as f:
                        artifacts[name] = base64.b64encode(f.read()).decode()

            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "artifacts": artifacts,
            }

        container_id = str(uuid.uuid4())
        self._runs[container_id] = {
            "app": app,
            "function": run_task,
            "result": None,
            "ctx_dir": ctx_dir,
            "patterns": tuple(spec.result_matchers()),
        }
        return container_id

    async def execute(
        self,
        container_id: str,
        command: str,
        *,
        env: Dict[str, str] | None = None,
        workdir: Path | None = None,
    ) -> Dict[str, str | int]:
        run_info = self._runs.get(container_id)
        if not run_info:
            raise ValueError(f"Unknown container_id: {container_id}")
        app = run_info["app"]
        run_fn = run_info["function"]
        patterns = list(run_info["patterns"])

        loop = asyncio.get_running_loop()

        def _call():
            with app.run():
                return run_fn.call(command, patterns)

        result = await loop.run_in_executor(None, _call)
        run_info["result"] = result
        return {
            "exit_code": result.get("exit_code", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }

    async def collect_artifacts(self, container_id: str, patterns: Iterable[str]) -> Dict[str, bytes]:
        run_info = self._runs.get(container_id)
        if not run_info:
            return {}
        result = run_info.get("result") or {}
        artifacts: Dict[str, bytes] = {}
        encoded = result.get("artifacts") or {}  # type: ignore[misc]
        for name, b64 in encoded.items():
            try:
                artifacts[name] = base64.b64decode(b64)
            except Exception:
                continue
        return artifacts

    async def destroy(self, container_id: str) -> None:
        info = self._runs.pop(container_id, None)
        if not info:
            return
        ctx_dir = info.get("ctx_dir")
        if ctx_dir and hasattr(ctx_dir, "cleanup"):
            with contextlib.suppress(Exception):
                ctx_dir.cleanup()  # type: ignore[call-arg]


def get_backend(name: str = DEFAULT_BACKEND) -> ContainerBackend:
    """Resolve backend by name."""
    normalized = (name or DEFAULT_BACKEND).lower()
    if normalized == "docker":
        return DockerBackend()
    if normalized == "modal":
        return ModalBackend()
    raise ValueError(f"Unsupported container backend: {name}")
