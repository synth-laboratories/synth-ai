from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_MANIFEST_ENV = "SYNTH_WORKSPACE_MANIFEST"
WORKSPACE_ROOT_ENV = "SYNTH_WORKSPACE_ROOT"
MANIFEST_BASENAME = "synth-workspace.toml"
REPO_ENV_BY_NAME = {
    "backend": "BACKEND_ROOT",
    "evals": "EVALS_ROOT",
    "frontend": "FRONTEND_ROOT",
    "managed-research": "MANAGED_RESEARCH_ROOT",
    "synth-ai": "SYNTH_AI_ROOT",
    "synth-dev": "SYNTH_DEV_ROOT",
}


@dataclass(frozen=True, slots=True)
class WorkspaceResolution:
    workspace_root: Path
    manifest_path: Path | None
    repo_paths: Mapping[str, Path]

    def scoped_env(self) -> dict[str, str]:
        env = {WORKSPACE_ROOT_ENV: str(self.workspace_root)}
        if self.manifest_path is not None:
            env[WORKSPACE_MANIFEST_ENV] = str(self.manifest_path)
        for repo_name, env_key in REPO_ENV_BY_NAME.items():
            repo_path = self.repo_paths.get(repo_name)
            if repo_path is not None:
                env[env_key] = str(repo_path)
        return env

    def repo(self, repo_name: str, *, required: bool = False) -> Path | None:
        path = self.repo_paths.get(repo_name)
        if required and path is None:
            raise RuntimeError(f"workspace repo is not configured: {repo_name}")
        if required and path is not None and not path.exists():
            raise RuntimeError(f"workspace repo does not exist: {repo_name} at {path}")
        return path


def resolve_workspace(
    *,
    manifest_path: str | Path | None = None,
    workspace_root: str | Path | None = None,
    repo_overrides: Mapping[str, str | Path] | None = None,
) -> WorkspaceResolution:
    resolved_manifest = _resolve_manifest_path(manifest_path, workspace_root)
    manifest_payload = _read_manifest(resolved_manifest)
    resolved_root = _resolve_workspace_root(
        workspace_root=workspace_root,
        manifest_path=resolved_manifest,
        manifest_payload=manifest_payload,
    )
    repo_paths = _repo_paths_from_sources(
        workspace_root=resolved_root,
        manifest_path=resolved_manifest,
        manifest_payload=manifest_payload,
        repo_overrides=repo_overrides or {},
    )
    return WorkspaceResolution(
        workspace_root=resolved_root,
        manifest_path=resolved_manifest,
        repo_paths=repo_paths,
    )


def _resolve_manifest_path(
    manifest_path: str | Path | None,
    workspace_root: str | Path | None,
) -> Path | None:
    explicit = _path_from_value(manifest_path)
    if explicit is not None:
        return explicit
    env_manifest = _path_from_value(os.environ.get(WORKSPACE_MANIFEST_ENV))
    if env_manifest is not None:
        return env_manifest
    root = _path_from_value(workspace_root) or _path_from_value(os.environ.get(WORKSPACE_ROOT_ENV))
    if root is None:
        return None
    candidate = root / MANIFEST_BASENAME
    return candidate.resolve() if candidate.is_file() else None


def _resolve_workspace_root(
    *,
    workspace_root: str | Path | None,
    manifest_path: Path | None,
    manifest_payload: Mapping[str, Any],
) -> Path:
    explicit_root = _path_from_value(workspace_root) or _path_from_value(
        os.environ.get(WORKSPACE_ROOT_ENV)
    )
    if explicit_root is not None:
        return explicit_root
    manifest_root = manifest_payload.get("workspace_root")
    if isinstance(manifest_root, str) and manifest_root.strip():
        root = Path(manifest_root.strip()).expanduser()
        if not root.is_absolute() and manifest_path is not None:
            root = manifest_path.parent / root
        return root.resolve()
    if manifest_path is not None:
        return manifest_path.parent.resolve()
    return Path.cwd().resolve()


def _repo_paths_from_sources(
    *,
    workspace_root: Path,
    manifest_path: Path | None,
    manifest_payload: Mapping[str, Any],
    repo_overrides: Mapping[str, str | Path],
) -> Mapping[str, Path]:
    paths: dict[str, Path] = {}
    repos = manifest_payload.get("repos")
    if isinstance(repos, Mapping):
        base = manifest_path.parent if manifest_path is not None else workspace_root
        for repo_name, raw_path in repos.items():
            if isinstance(raw_path, str) and raw_path.strip():
                paths[str(repo_name)] = _resolve_repo_path(raw_path, base=base)
    for repo_name, env_key in REPO_ENV_BY_NAME.items():
        raw_env = os.environ.get(env_key)
        if raw_env and repo_name not in paths:
            paths[repo_name] = Path(raw_env).expanduser().resolve()
    for repo_name, raw_path in repo_overrides.items():
        paths[str(repo_name)] = Path(raw_path).expanduser().resolve()
    for repo_name in REPO_ENV_BY_NAME:
        paths.setdefault(repo_name, (workspace_root / repo_name).resolve())
    return paths


def _resolve_repo_path(raw_path: str, *, base: Path) -> Path:
    path = Path(raw_path.strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _read_manifest(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"workspace manifest must be a TOML table: {path}")
    return payload


def _path_from_value(value: str | Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


__all__ = [
    "MANIFEST_BASENAME",
    "REPO_ENV_BY_NAME",
    "WORKSPACE_MANIFEST_ENV",
    "WORKSPACE_ROOT_ENV",
    "WorkspaceResolution",
    "resolve_workspace",
]
