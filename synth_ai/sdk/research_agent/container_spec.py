from __future__ import annotations

import io
import textwrap
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from synth_ai.sdk.research_agent.defaults import (
    DEFAULT_BACKEND,
    DEFAULT_BASE_IMAGE,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_PACKAGES,
    DEFAULT_PYTHON_VERSION,
    DEFAULT_RESULT_PATTERNS,
    DEFAULT_SYNTH_PIP_SPEC,
)


def _load_box_bootstrap() -> str:
    """Load the canonical box_bootstrap.sh content from regenerate_bootstrap."""
    try:
        mod = import_module("scripts.regenerate_bootstrap")
        content = getattr(mod, "bootstrap_content", None)
        if content:
            return content
    except Exception:
        pass
    return "#!/bin/bash\nset -euo pipefail\necho 'box_bootstrap.sh missing; ensure regenerate_bootstrap.py is available.'\nexit 1\n"


@dataclass
class ContainerSpec:
    """Declarative container configuration for running the research agent."""

    repo_url: str
    repo_branch: str = "main"
    repo_commit: Optional[str] = None

    agent_instructions: str = DEFAULT_INSTRUCTIONS
    base_image: str = DEFAULT_BASE_IMAGE
    python_version: str = DEFAULT_PYTHON_VERSION
    apt_packages: List[str] = field(default_factory=lambda: list(DEFAULT_PACKAGES))

    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str | bytes] = field(default_factory=dict)
    preflight_script: Optional[str] = None
    postflight_script: Optional[str] = None

    artifacts_dir: Path = Path("/app/artifacts")
    result_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_RESULT_PATTERNS))
    workdir: Path = Path("/app/repo")

    backend: str = DEFAULT_BACKEND
    overlay_dir: Optional[Path] = None
    bootstrap_content: str = field(default_factory=_load_box_bootstrap)
    synth_pip_spec: str = DEFAULT_SYNTH_PIP_SPEC

    def validate(self) -> None:
        """Lightweight validation before provisioning."""
        if not self.repo_url:
            raise ValueError("repo_url is required")
        if not self.agent_instructions:
            raise ValueError("agent_instructions is required")
        if not self.base_image:
            raise ValueError("base_image is required")
        if not self.artifacts_dir.is_absolute():
            raise ValueError("artifacts_dir must be absolute")

    @property
    def build_args(self) -> Dict[str, str]:
        """Build args passed to Docker/Modal image creation."""
        return {
            "GIT_URL": self.repo_url,
            "GIT_BRANCH": self.repo_branch,
            "GIT_COMMIT": self.repo_commit or "",
            "PYTHON_VERSION": self.python_version,
            "SYNTH_PIP_SPEC": self.synth_pip_spec or "",
        }

    def to_dockerfile(self) -> str:
        """Render a Dockerfile that mirrors the existing OneShot bootstrap."""
        package_line = " ".join(sorted(set(self.apt_packages)))
        return textwrap.dedent(
            f"""
            FROM {self.base_image}

            ARG GIT_URL
            ARG GIT_BRANCH
            ARG GIT_COMMIT
            ARG PYTHON_VERSION="{self.python_version}"

            ENV DEBIAN_FRONTEND=noninteractive
            ENV PIP_DISABLE_PIP_VERSION_CHECK=1
            ENV PIP_NO_PYTHON_VERSION_WARNING=1
            ENV PIP_BREAK_SYSTEM_PACKAGES=1
            ENV PYTHONWARNINGS=ignore

            RUN apt-get update && \\
                apt-get install -y --no-install-recommends {package_line} && \\
                ln -sf /usr/bin/python3 /usr/bin/python && \\
                mkdir -p {self.artifacts_dir} /app/overlay_files && \\
                apt-get clean

            # Install uv for fast Python installs
            RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \\
                ln -sf /root/.local/bin/uv /usr/local/bin/uv || true

            WORKDIR /app
            RUN git clone --branch "$GIT_BRANCH" "$GIT_URL" repo && \\
                cd repo && if [ -n "$GIT_COMMIT" ]; then git checkout "$GIT_COMMIT"; fi && \\
                python3 -m venv /app/repo/.venv && \\
                . /app/repo/.venv/bin/activate && \\
                pip install --no-cache-dir --upgrade pip && \\
                if [ -n "$SYNTH_PIP_SPEC" ]; then \\
                  pip install --no-cache-dir "$SYNTH_PIP_SPEC"; \\
                else \\
                  pip install --no-cache-dir -e .; \\
                fi
            ENV VIRTUAL_ENV="/app/repo/.venv"
            ENV PATH="/app/repo/.venv/bin:${{PATH}}"

            COPY overlay_files/ /app/

            WORKDIR {self.workdir}
            """
        ).strip() + "\n"

    def rendered_overlay_files(self) -> Dict[str, bytes]:
        """Overlay files placed into the build context / runtime container."""
        files: Dict[str, bytes] = {}

        if self.overlay_dir and self.overlay_dir.exists():
            for path in self.overlay_dir.rglob("*"):
                if path.is_dir():
                    continue
                rel_path = path.relative_to(self.overlay_dir)
                files.setdefault(str(rel_path), path.read_bytes())

        if self.agent_instructions:
            files["LM_INSTRUCTIONS.md"] = self.agent_instructions.encode()

        if self.preflight_script:
            files["overlay_hidden_pre/preflight.sh"] = self.preflight_script.encode()
            files["pre_flight.sh"] = self.preflight_script.encode()
        if self.postflight_script:
            files["overlay_hidden_post/postflight.sh"] = self.postflight_script.encode()

        for relative_path, content in self.files.items():
            data = content.encode() if isinstance(content, str) else content
            files[str(relative_path)] = data

        files.setdefault("box_bootstrap.sh", self.bootstrap_content.encode())

        return files

    def build_context(self) -> bytes:
        """Create a tar build context for Docker builds."""
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            dockerfile_bytes = self.to_dockerfile().encode()
            dockerfile_info = tarfile.TarInfo("Dockerfile")
            dockerfile_info.size = len(dockerfile_bytes)
            dockerfile_info.mtime = int(time.time())
            tar.addfile(dockerfile_info, io.BytesIO(dockerfile_bytes))

            for rel_path, content in self.rendered_overlay_files().items():
                overlay_path = Path("overlay_files") / rel_path
                executable = overlay_path.name.endswith(".sh")
                info = tarfile.TarInfo(str(overlay_path))
                info.size = len(content)
                info.mtime = int(time.time())
                if executable:
                    info.mode = 0o755
                tar.addfile(info, io.BytesIO(content))

            for rel_path, content in self.files.items():
                if str(rel_path).startswith("/"):
                    data = content.encode() if isinstance(content, str) else content
                    target = Path(str(rel_path).lstrip("/"))
                    info = tarfile.TarInfo(str(target))
                    info.size = len(data)
                    info.mtime = int(time.time())
                    if target.name.endswith(".sh"):
                        info.mode = 0o755
                    tar.addfile(info, io.BytesIO(data))

        buf.seek(0)
        return buf.read()

    def result_matchers(self) -> Iterable[str]:
        """Expose patterns for callers that only need read-only access."""
        return tuple(self.result_patterns)
