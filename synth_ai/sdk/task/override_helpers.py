"""Agent-aware context override applicator for task apps.

This module provides batteries-included helpers for task apps to apply context
overrides (AGENTS.md, skills, preflight scripts, env vars) sent by GEPA during
unified optimization.

Key features:
- Agent-specific path handling (Codex vs OpenCode)
- Safe defaults (workspace-local only by default)
- Path traversal protection
- Size limits and script timeouts
- Structured application results for GEPA feedback

Example:
    from synth_ai.sdk.task.override_helpers import apply_context_overrides

    # In your task app rollout handler:
    results = await apply_context_overrides(
        overrides=request.context_overrides,
        workspace_dir=sandbox_dir,
        agent="codex",  # or "opencode"
    )

Note:
    This helper applies file artifacts and preflight scripts in the workspace,
    but it only validates env vars. Task apps must inject validated env vars
    into the agent process environment themselves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

from synth_ai.data.artifacts import (
    ApplicationErrorType,
    ApplicationStatus,
    ContextOverride,
    ContextOverrideStatus,
    OverrideApplicationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Limits
# =============================================================================

# Safe limits for overrides
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB per file
MAX_TOTAL_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB total
MAX_FILES_PER_OVERRIDE = 20
MAX_ENV_VARS = 50
MAX_ENV_VAR_VALUE_LENGTH = 10 * 1024  # 10 KB per env var value
PREFLIGHT_SCRIPT_TIMEOUT_SECONDS = 60


class AgentType(str, Enum):
    """Supported agent types for context overrides."""

    CODEX = "codex"
    OPENCODE = "opencode"


# =============================================================================
# Agent-Specific Path Configuration
# =============================================================================

# Codex CLI reads skills from:
#   1. Built-in defaults
#   2. ~/.codex/skills.yaml (global)
#   3. .codex/skills.yaml (workspace, wins)
CODEX_SKILLS_WORKSPACE_PATH = ".codex/skills.yaml"
CODEX_SKILLS_GLOBAL_PATH = "~/.codex/skills.yaml"

# OpenCode reads skills from:
#   1. .opencoderc or .opencode/skills.yaml (workspace)
#   2. ~/.opencode/config.yaml (global, only if load_global_skills: true)
OPENCODE_SKILLS_WORKSPACE_PATH = ".opencode/skills.yaml"
OPENCODE_SKILLS_GLOBAL_PATH = "~/.opencode/skills.yaml"


def get_agent_skills_path(agent: str | AgentType, global_: bool = False) -> str:
    """Get the canonical skills file path for an agent.

    Args:
        agent: Agent type ("codex" or "opencode")
        global_: If True, return the global path; otherwise workspace-local

    Returns:
        Relative path for workspace-local, or absolute path for global.

    Example:
        path = get_agent_skills_path("codex")
        # ".codex/skills.yaml"

    Note:
        Workspace-local paths are relative to the task app sandbox root.
    """
    agent_str = agent.value if isinstance(agent, AgentType) else agent.lower()

    if agent_str == "codex":
        return CODEX_SKILLS_GLOBAL_PATH if global_ else CODEX_SKILLS_WORKSPACE_PATH
    elif agent_str == "opencode":
        return OPENCODE_SKILLS_GLOBAL_PATH if global_ else OPENCODE_SKILLS_WORKSPACE_PATH
    else:
        # Default to a generic path
        return f".{agent_str}/skills.yaml"


# =============================================================================
# Path Safety Validation
# =============================================================================


def is_path_safe(path: str, workspace_dir: Path, allow_global: bool = False) -> tuple[bool, str]:
    """Check if a path is safe to write to.

    Args:
        path: Path to validate (relative or absolute)
        workspace_dir: Workspace root directory
        allow_global: If True, allow writing to global paths (~/...)

    Returns:
        Tuple of (is_safe, error_message). If is_safe is True, error_message is empty.

    Example:
        ok, error_message = is_path_safe("AGENTS.md", Path("/tmp/sandbox"))
        # ok == True

    Note:
        Absolute paths are only allowed if they resolve within workspace_dir,
        unless allow_global is True and the path is under the home directory.
    """
    # Normalize path
    path = path.strip()

    # Check for empty path
    if not path:
        return False, "Empty path"

    # Check for path traversal attempts
    if ".." in path:
        return False, f"Path traversal not allowed: {path}"

    # Check for absolute paths
    if path.startswith("/") and not path.startswith(str(workspace_dir)):
        if allow_global and path.startswith(os.path.expanduser("~")):
            return True, ""
        return False, f"Absolute paths outside workspace not allowed: {path}"

    # Check for home directory paths
    if path.startswith("~"):
        if not allow_global:
            return False, f"Global paths require explicit opt-in: {path}"
        return True, ""

    # Resolve and check the final path is within workspace
    try:
        if path.startswith("/"):
            resolved = Path(path).resolve()
        else:
            resolved = (workspace_dir / path).resolve()

        if not str(resolved).startswith(str(workspace_dir.resolve())):
            return False, f"Path escapes workspace: {path} -> {resolved}"
    except (ValueError, OSError) as e:
        return False, f"Invalid path: {path} ({e})"

    return True, ""


# =============================================================================
# File Application
# =============================================================================


def _apply_file_artifact(
    path: str,
    content: str,
    workspace_dir: Path,
    allow_global: bool = False,
) -> dict[str, Any]:
    """Apply a single file artifact.

    Args:
        path: Relative path within workspace (or global path if allow_global)
        content: File content to write
        workspace_dir: Workspace root directory
        allow_global: If True, allow writing to global paths

    Returns:
        Dict with status, bytes_written, error (if any), and error_type when failed.

    Note:
        File writes are size-checked and path-validated before writing.
    """
    result: dict[str, Any] = {"path": path}

    # Validate path safety
    is_safe, error_msg = is_path_safe(path, workspace_dir, allow_global)
    if not is_safe:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = error_msg
        result["error_type"] = ApplicationErrorType.PATH_TRAVERSAL.value
        return result

    # Check size limit
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"File too large: {len(content_bytes)} > {MAX_FILE_SIZE_BYTES} bytes"
        result["error_type"] = ApplicationErrorType.SIZE_LIMIT.value
        return result

    # Resolve final path
    if path.startswith("~"):
        final_path = Path(os.path.expanduser(path))
    elif path.startswith("/"):
        final_path = Path(path)
    else:
        final_path = workspace_dir / path

    # Create parent directories
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Failed to create directory: {e}"
        result["error_type"] = ApplicationErrorType.PERMISSION.value
        return result

    # Write file
    try:
        final_path.write_bytes(content_bytes)
        result["status"] = ApplicationStatus.APPLIED.value
        result["bytes_written"] = len(content_bytes)
        logger.debug(f"Applied file artifact: {path} ({len(content_bytes)} bytes)")
    except OSError as e:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Failed to write file: {e}"
        result["error_type"] = ApplicationErrorType.PERMISSION.value

    return result


# =============================================================================
# Preflight Script Execution
# =============================================================================


async def _run_preflight_script(
    script_content: str,
    workspace_dir: Path,
    timeout: int = PREFLIGHT_SCRIPT_TIMEOUT_SECONDS,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a preflight bash script.

    Args:
        script_content: Bash script content (should start with #!/bin/bash)
        workspace_dir: Working directory for the script
        timeout: Timeout in seconds
        env_vars: Additional environment variables to set

    Returns:
        Dict with status, exit_code, stdout, stderr, and duration_ms.

    Note:
        Stdout/stderr are truncated to 10,000 characters. A timeout results in
        a failed status and a TIMEOUT error_type.
    """
    import time

    result: dict[str, Any] = {}

    # Validate script has shebang
    if not script_content.strip().startswith("#!"):
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = "Preflight script must start with a shebang (e.g., #!/bin/bash)"
        result["error_type"] = ApplicationErrorType.VALIDATION.value
        return result

    # Check size
    if len(script_content.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Script too large: {len(script_content)} bytes"
        result["error_type"] = ApplicationErrorType.SIZE_LIMIT.value
        return result

    # Write script to temp file
    script_path = workspace_dir / ".synth_preflight.sh"
    try:
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)
    except OSError as e:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Failed to write script: {e}"
        result["error_type"] = ApplicationErrorType.PERMISSION.value
        return result

    # Build environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    # Run script
    start_time = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            str(script_path),
            cwd=str(workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        duration_ms = (time.perf_counter() - start_time) * 1000

        result["status"] = (
            ApplicationStatus.APPLIED.value
            if proc.returncode == 0
            else ApplicationStatus.FAILED.value
        )
        result["exit_code"] = proc.returncode
        result["stdout"] = stdout.decode("utf-8", errors="replace")[:10000]  # Cap output
        result["stderr"] = stderr.decode("utf-8", errors="replace")[:10000]
        result["duration_ms"] = round(duration_ms, 2)

        if proc.returncode != 0:
            result["error"] = f"Script exited with code {proc.returncode}"
            result["error_type"] = ApplicationErrorType.RUNTIME.value

    except TimeoutError:
        duration_ms = (time.perf_counter() - start_time) * 1000
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Script timed out after {timeout}s"
        result["error_type"] = ApplicationErrorType.TIMEOUT.value
        result["duration_ms"] = round(duration_ms, 2)
        # Kill the process
        with contextlib.suppress(Exception):
            proc.kill()  # type: ignore

    except Exception as e:
        result["status"] = ApplicationStatus.FAILED.value
        result["error"] = f"Script execution failed: {e}"
        result["error_type"] = ApplicationErrorType.RUNTIME.value

    finally:
        # Clean up script file
        with contextlib.suppress(Exception):
            script_path.unlink(missing_ok=True)

    return result


# =============================================================================
# Environment Variables
# =============================================================================


def _validate_env_vars(
    env_vars: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[OverrideApplicationError]]:
    """Validate environment variables.

    Args:
        env_vars: Dict of env var name -> value

    Returns:
        Tuple of (results dict, list of errors).

    Note:
        This function validates names and sizes only. Task apps must still inject
        validated env vars into the agent process environment.
    """
    results: dict[str, dict[str, Any]] = {}
    errors: list[OverrideApplicationError] = []

    if len(env_vars) > MAX_ENV_VARS:
        errors.append(
            OverrideApplicationError(
                error_type=ApplicationErrorType.SIZE_LIMIT,
                message=f"Too many env vars: {len(env_vars)} > {MAX_ENV_VARS}",
            )
        )

    for key, value in env_vars.items():
        # Validate key (alphanumeric + underscore, must start with letter/underscore)
        if not key or not key[0].isalpha() and key[0] != "_":
            results[key] = {
                "status": ApplicationStatus.FAILED.value,
                "error": "Invalid env var name (must start with letter or underscore)",
            }
            continue

        # Validate value size
        if len(value) > MAX_ENV_VAR_VALUE_LENGTH:
            results[key] = {
                "status": ApplicationStatus.FAILED.value,
                "error": f"Value too large: {len(value)} > {MAX_ENV_VAR_VALUE_LENGTH}",
            }
            continue

        results[key] = {"status": ApplicationStatus.APPLIED.value}

    return results, errors


# =============================================================================
# Main Entry Point
# =============================================================================


async def apply_context_overrides(
    overrides: list[ContextOverride] | None,
    workspace_dir: Path | str,
    agent: str | AgentType = AgentType.OPENCODE,
    allow_global: bool = False,
    override_bundle_id: str | None = None,
) -> list[ContextOverrideStatus]:
    """Apply context overrides to a workspace.

    This is the main entry point for task apps to apply GEPA-generated overrides.

    Args:
        overrides: List of context overrides to apply. If None or empty, returns
            an empty list.
        workspace_dir: Workspace directory (sandbox root). Created if missing.
        agent: Agent type for path resolution ("codex" or "opencode").
        allow_global: If True, allow writing to global paths (~/...).
        override_bundle_id: Optional bundle ID for traceability.

    Returns:
        List of ContextOverrideStatus with per-target results.

    Example:
        results = await apply_context_overrides(
            overrides=request.context_overrides,
            workspace_dir=sandbox_dir,
            agent="codex",
        )

    Raises:
        OSError: If the workspace directory cannot be created or accessed.

    Note:
        Env vars are validated and included in the returned status, but they are
        not injected into the agent environment by this helper.
    """
    if not overrides:
        return []

    workspace_dir = Path(workspace_dir)
    if not workspace_dir.exists():
        workspace_dir.mkdir(parents=True, exist_ok=True)

    results: list[ContextOverrideStatus] = []

    for idx, override in enumerate(overrides):
        status = ContextOverrideStatus(
            override_id=override.override_id or f"{override_bundle_id or 'ov'}_{idx}",
            overall_status=ApplicationStatus.APPLIED,
            errors=[],
            file_artifacts={},
            preflight_script=None,
            env_vars={},
        )

        has_failures = False

        # Apply file artifacts
        if override.file_artifacts:
            if len(override.file_artifacts) > MAX_FILES_PER_OVERRIDE:
                status.errors.append(
                    OverrideApplicationError(
                        error_type=ApplicationErrorType.SIZE_LIMIT,
                        message=f"Too many files: {len(override.file_artifacts)} > {MAX_FILES_PER_OVERRIDE}",
                    )
                )
                has_failures = True

            for path, content in override.file_artifacts.items():
                result = _apply_file_artifact(path, content, workspace_dir, allow_global)
                status.file_artifacts[path] = result
                if result.get("status") == ApplicationStatus.FAILED.value:
                    has_failures = True

        # Run preflight script
        if override.preflight_script:
            script_result = await _run_preflight_script(
                override.preflight_script,
                workspace_dir,
                env_vars=override.env_vars,
            )
            status.preflight_script = script_result
            if script_result.get("status") == ApplicationStatus.FAILED.value:
                has_failures = True

        # Validate env vars (actual application happens at agent execution time)
        if override.env_vars:
            env_results, env_errors = _validate_env_vars(override.env_vars)
            status.env_vars = env_results
            status.errors.extend(env_errors)
            if any(r.get("status") == ApplicationStatus.FAILED.value for r in env_results.values()):
                has_failures = True

        # Set overall status
        if has_failures:
            if any(
                r.get("status") == ApplicationStatus.APPLIED.value
                for r in status.file_artifacts.values()
            ):
                status.overall_status = ApplicationStatus.PARTIAL
            else:
                status.overall_status = ApplicationStatus.FAILED
        else:
            status.overall_status = ApplicationStatus.APPLIED

        results.append(status)

    return results


def get_applied_env_vars(overrides: list[ContextOverride] | None) -> dict[str, str]:
    """Extract all env vars from overrides for injection into agent process.

    Args:
        overrides: List of context overrides

    Returns:
        Merged dict of env vars (later overrides win).

    Example:
        env = os.environ.copy()
        env.update(get_applied_env_vars(request.context_overrides))
        subprocess.run(agent_cmd, env=env)

    Note:
        Later overrides take precedence when keys overlap.
    """
    if not overrides:
        return {}

    merged: dict[str, str] = {}
    for override in overrides:
        if override.env_vars:
            merged.update(override.env_vars)
    return merged


__all__ = [
    "apply_context_overrides",
    "get_applied_env_vars",
    "get_agent_skills_path",
    "is_path_safe",
    "AgentType",
    "CODEX_SKILLS_WORKSPACE_PATH",
    "CODEX_SKILLS_GLOBAL_PATH",
    "OPENCODE_SKILLS_WORKSPACE_PATH",
    "OPENCODE_SKILLS_GLOBAL_PATH",
]
