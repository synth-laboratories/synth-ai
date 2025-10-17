from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.environments import get_environment
from synth_ai.environments.environment.tools import EnvToolCall

from .shared import summarise_history
from .tools import TOOLS_SCHEMA

logger = logging.getLogger(__name__)


def _environment_type_from_config(config: dict[str, Any]) -> str:
    value = (config or {}).get("environment_class") or os.getenv(
        "SWE_MINI_ENVIRONMENT_CLASS", "local"
    )
    return str(value).strip() or "local"


def _environment_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(config or {}).get("environment_kwargs") or {}
    if not kwargs and (raw := os.getenv("SWE_MINI_ENVIRONMENT_KWARGS")):
        try:
            kwargs = json.loads(raw)
        except Exception:  # pragma: no cover - environment var malformed
            logger.warning("Failed to parse SWE_MINI_ENVIRONMENT_KWARGS; ignoring")
            kwargs = {}
    if not isinstance(kwargs, dict):
        logger.warning("environment_kwargs must be a mapping, got %r", type(kwargs))
        kwargs = {}
    return kwargs


def _default_submit_command() -> str:
    return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"


@dataclass
class MiniSweEnvironmentState:
    """Serializable environment state used for snapshots."""

    task: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    step_idx: int = 0
    submitted: bool = False
    submission_success: bool | None = None


class MiniSweEnvironmentWrapper:
    """Wrapper around mini-swe-agent environments exposing Synth task-app semantics."""

    name = "swe-mini"

    def __init__(
        self,
        *,
        task: dict[str, Any],
        env_config: dict[str, Any] | None = None,
        submit_command: str | None = None,
    ) -> None:
        self.task = dict(task)
        self.env_config = dict(env_config or {})
        self.submit_command = submit_command or _default_submit_command()
        self.environment_type = _environment_type_from_config(self.env_config)
        kwargs = _environment_kwargs_from_config(self.env_config)

        self.instance_id = str(
            self.task.get("instance_id") or f"swe-mini-{uuid.uuid4().hex[:8]}"
        )
        self.metadata = dict(self.task.get("metadata") or {})
        self.repo_url = self._resolve_repo_url(self.metadata)
        self.base_commit = (
            self.metadata.get("base_commit")
            or self.metadata.get("environment_setup_commit")
            or None
        )
        self._local_workspace_dir: Path | None = None
        self._remote_workspace: str | None = None
        self._cleanup_workspace = False

        if self.environment_type == "local":
            workspace = self._prepare_local_workspace(kwargs)
            kwargs.setdefault("cwd", str(workspace))
            kwargs.setdefault("timeout", int(self.env_config.get("timeout", 60)))
            # Merge custom env vars with defaults expected by mini-swe
            merged_env = dict(kwargs.get("env") or {})
            merged_env.setdefault("PAGER", "cat")
            merged_env.setdefault("MANPAGER", "cat")
            merged_env.setdefault("LESS", "-R")
            merged_env.setdefault("PIP_PROGRESS_BAR", "off")
            merged_env.setdefault("TQDM_DISABLE", "1")
            merged_env.setdefault("GIT_TERMINAL_PROMPT", "0")
            kwargs["env"] = merged_env
            self._local_workspace_dir = workspace
            self._cleanup_workspace = True
        else:
            remote_cwd = kwargs.get("cwd")
            if not remote_cwd:
                base_remote = os.getenv("SWE_MINI_REMOTE_WORKSPACE_BASE", "/workspace")
                remote_cwd = f"{base_remote.rstrip('/')}/{self.instance_id}"
                kwargs["cwd"] = remote_cwd
            self._remote_workspace = kwargs["cwd"]
            timeout = self.env_config.get("timeout")
            if timeout and "timeout" not in kwargs:
                kwargs["timeout"] = int(timeout)
            if self.repo_url and "image" not in kwargs:
                image = self.metadata.get("image_name") or os.getenv("SWE_MINI_DOCKER_IMAGE")
                if image:
                    kwargs["image"] = image
            if self.environment_type in {"docker", "bubblewrap"}:
                remote_env = dict(kwargs.get("env") or {})
                remote_env.setdefault("GIT_TERMINAL_PROMPT", "0")
                kwargs["env"] = remote_env

        logger.info(
            "Initialising mini-swe environment: type=%s kwargs=%s",
            self.environment_type,
            kwargs,
        )
        self.env = get_environment(
            {
                "environment_class": self.environment_type,
                **kwargs,
            },
            default_type="local",
        )

        if self.environment_type != "local":
            self._bootstrap_remote_workspace()

        self.state = MiniSweEnvironmentState(task=self.task)
        self.last_result: dict[str, Any] | None = None
        self.last_submission: dict[str, Any] | None = None

    async def initialize(self) -> dict[str, Any]:
        """Return initial observation."""
        logger.info(
            "Mini-swe task initialised: instance=%s",
            self.task.get("instance_id"),
        )
        return self._build_response(observation=self._build_observation(None), step_idx=0)

    async def terminate(self) -> dict[str, Any]:
        """Terminate the environment, returning the final observation."""
        logger.info(
            "Terminating mini-swe environment instance=%s submitted=%s",
            self.task.get("instance_id"),
            self.state.submitted,
        )
        response = self._build_response(
            observation=self._build_observation(self.last_result),
            step_idx=self.state.step_idx,
        )
        self._cleanup_workspaces()
        return response

    def _cleanup_workspaces(self) -> None:
        if self._cleanup_workspace and self._local_workspace_dir:
            with contextlib.suppress(Exception):
                shutil.rmtree(self._local_workspace_dir)
            self._local_workspace_dir = None
            self._cleanup_workspace = False
        if (
            self._remote_workspace
            and os.getenv("SWE_MINI_CLEANUP_REMOTE_WORKSPACE", "1") not in {"0", "false", "False"}
        ):
            with contextlib.suppress(Exception):
                self.env.execute(f"rm -rf {shlex.quote(self._remote_workspace)}")
        self._remote_workspace = None

    def _resolve_repo_url(self, metadata: dict[str, Any]) -> str | None:
        candidates = [
            metadata.get("repo_url"),
            metadata.get("repo"),
            metadata.get("repository"),
        ]
        for value in candidates:
            if not value:
                continue
            repo = str(value).strip()
            if not repo:
                continue
            if repo.startswith("http://") or repo.startswith("https://"):
                url = repo
            else:
                repo = repo.removesuffix(".git")
                url = f"https://github.com/{repo}.git"
            if not url.endswith(".git"):
                url = f"{url}.git"
            return url
        return None

    def _prepare_local_workspace(self, kwargs: dict[str, Any]) -> Path:
        if not self.repo_url:
            fallback = Path(kwargs.get("cwd") or self.env_config.get("cwd") or os.getcwd())
            fallback.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "No repo URL provided for swe-mini instance %s; using cwd=%s",
                self.instance_id,
                fallback,
            )
            return fallback

        root = Path(
            os.getenv("SWE_MINI_LOCAL_WORKSPACE_ROOT")
            or Path.home() / ".cache" / "synth-ai" / "swe-mini" / "workspaces"
        )
        workspace = root / self.instance_id
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        workspace.parent.mkdir(parents=True, exist_ok=True)

        self._run_local_cmd(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-tags",
                self.repo_url,
                str(workspace),
            ],
            description="clone repository",
        )
        if self.base_commit:
            self._run_local_cmd(
                ["git", "-C", str(workspace), "checkout", self.base_commit],
                description="checkout base commit",
            )
        self._run_local_cmd(
            ["git", "-C", str(workspace), "reset", "--hard"],
            description="reset working tree",
        )
        self._run_local_cmd(
            ["git", "-C", str(workspace), "clean", "-ffd"],
            description="clean working tree",
        )
        logger.info(
            "Prepared local workspace for %s at %s (repo=%s, commit=%s)",
            self.instance_id,
            workspace,
            self.repo_url,
            self.base_commit,
        )
        return workspace

    def _bootstrap_remote_workspace(self) -> None:
        if not self.repo_url or not self._remote_workspace:
            logger.warning(
                "Skipping remote workspace bootstrap for instance %s (repo=%s workspace=%s)",
                self.instance_id,
                self.repo_url,
                self._remote_workspace,
            )
            return

        workspace = self._remote_workspace.rstrip("/")
        base_dir = os.path.dirname(workspace) or "/"
        self._execute_bootstrap_command(f"mkdir -p {shlex.quote(base_dir)}")
        self._execute_bootstrap_command(f"rm -rf {shlex.quote(workspace)}")
        clone_cmd = (
            f"git clone --filter=blob:none --no-tags {shlex.quote(self.repo_url)} {shlex.quote(workspace)}"
        )
        self._execute_bootstrap_command(clone_cmd, timeout=900, description="clone repository")
        if self.base_commit:
            checkout_cmd = (
                f"cd {shlex.quote(workspace)} && git checkout {shlex.quote(self.base_commit)}"
            )
            self._execute_bootstrap_command(checkout_cmd, timeout=300, description="checkout commit")
        self._execute_bootstrap_command(
            f"cd {shlex.quote(workspace)} && git reset --hard",
            description="reset working tree",
        )
        self._execute_bootstrap_command(
            f"cd {shlex.quote(workspace)} && git clean -ffd",
            description="clean working tree",
        )
        logger.info(
            "Prepared remote workspace for %s at %s (repo=%s, commit=%s)",
            self.instance_id,
            workspace,
            self.repo_url,
            self.base_commit,
        )

    def _run_local_cmd(
        self, args: list[str], *, cwd: Path | None = None, description: str | None = None
    ) -> None:
        logger.debug(
            "Preparing workspace %s: running local command %s",
            self.instance_id,
            " ".join(args),
        )
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            desc = description or "command"
            raise RuntimeError(
                f"Failed to {desc} (cmd={' '.join(args)}): {proc.stdout or ''}{proc.stderr or ''}"
            )

    def _execute_bootstrap_command(
        self, command: str, *, timeout: int | None = None, description: str | None = None
    ) -> None:
        logger.debug(
            "Preparing workspace %s: running remote command %s",
            self.instance_id,
            command,
        )
        result = self.env.execute(command, timeout=timeout)
        if result.get("returncode"):
            desc = description or command
            raise RuntimeError(
                f"Failed to {desc}: rc={result.get('returncode')} output={result.get('output')}"
            )

    def _normalize_tool_call(self, tool_call: EnvToolCall | dict[str, Any]) -> EnvToolCall:
        if isinstance(tool_call, EnvToolCall):
            return tool_call
        tool = tool_call.get("tool") or tool_call.get("tool_name")
        if not tool:
            raise ValueError(f"Tool call missing tool name: {tool_call}")
        args = tool_call.get("args") or tool_call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return EnvToolCall(tool=str(tool), args=dict(args))

    async def step(self, tool_calls: list[EnvToolCall] | list[dict[str, Any]]) -> dict[str, Any]:
        """Execute run_command or submit_patch tool calls."""
        if not tool_calls:
            raise ValueError("MiniSweEnvironmentWrapper.step requires at least one tool call")

        responses: list[dict[str, Any]] = []
        for raw_call in tool_calls:
            call = self._normalize_tool_call(raw_call)
            tool = call.tool
            if tool == "run_command":
                responses.append(self._run_command(call))
            elif tool == "submit_patch":
                responses.append(self._submit(call))
            else:
                raise ValueError(f"Unsupported tool '{tool}' for swe-mini environment")

        last_result = responses[-1] if responses else None
        self.last_result = last_result
        observation = self._build_observation(last_result)
        done = bool(self.state.submitted)
        reward = 0.0
        if done:
            reward = 1.0 if self.state.submission_success else 0.0
        return self._build_response(
            observation=observation,
            step_idx=self.state.step_idx,
            done=done,
            reward=reward,
            info={"responses": responses},
        )

    def _run_command(self, call: EnvToolCall) -> dict[str, Any]:
        command = str(call.args.get("command") or "").strip()
        if not command:
            raise ValueError("run_command requires a non-empty 'command' argument")
        timeout = call.args.get("timeout")
        timeout = int(timeout) if timeout is not None else None

        started_at = time.time()
        result = self.env.execute(command, timeout=timeout)
        duration = time.time() - started_at

        record = {
            "command": command,
            "returncode": result.get("returncode"),
            "stdout": result.get("output") or "",
            "duration": duration,
            "timestamp": started_at,
        }
        self.state.history.append(record)
        self.state.step_idx += 1
        logger.info(
            "Executed command step=%s rc=%s",
            self.state.step_idx,
            record["returncode"],
        )
        return record

    def _submit(self, call: EnvToolCall) -> dict[str, Any]:
        if self.state.submitted:
            logger.info("Submit called again; ignoring additional submission.")
            return {
                "submitted": True,
                "command": None,
                "returncode": 0,
                "stdout": "",
                "submission_success": self.state.submission_success,
                "evaluation": self.last_submission,
            }
        command = str(call.args.get("command") or self.submit_command)
        result = self.env.execute(command)
        record = {
            "command": command,
            "returncode": result.get("returncode"),
            "stdout": result.get("output") or "",
            "duration": 0.0,
            "timestamp": time.time(),
        }
        self.state.history.append(record)
        self.state.step_idx += 1
        diff = self._extract_submission_diff(record["stdout"])

        evaluation: dict[str, Any] | None = None
        submission_success = False
        if record["returncode"] == 0 and diff is not None:
            evaluation = self._evaluate_submission(diff)
            submission_success = bool(evaluation.get("resolved")) if evaluation else False
        else:
            evaluation = {
                "completed": False,
                "resolved": False,
                "error": "submit command failed or diff unavailable",
                "returncode": record["returncode"],
            }

        self.state.submitted = True
        self.state.submission_success = submission_success
        self.last_submission = evaluation

        logger.info(
            "Submission command executed rc=%s resolved=%s",
            record["returncode"],
            submission_success,
        )

        return {
            **record,
            "submitted": True,
            "submission_success": submission_success,
            "diff": diff,
            "evaluation": evaluation,
        }

    def _extract_submission_diff(self, stdout: str) -> str | None:
        if stdout is None:
            return None
        lines = stdout.splitlines()
        if not lines:
            return ""
        first = lines[0].strip()
        sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        if first.startswith(sentinel):
            lines = lines[1:]
        diff = "\n".join(lines).strip("\n")
        return diff

    def _evaluate_submission(self, diff: str) -> dict[str, Any]:
        metadata = dict(self.task.get("metadata") or {})
        instance = dict(metadata.get("raw_instance") or {})
        instance_id = instance.setdefault("instance_id", self.task.get("instance_id"))

        required_fields = ["repo", "base_commit", "test_patch", "version"]
        missing = [field for field in required_fields if not instance.get(field)]
        if missing:
            msg = (
                "Cannot run SWE-bench evaluation; task metadata missing required fields "
                f"{missing}. Ensure the dataset preserves full SWE-bench records."
            )
            logger.error(msg)
            return {"completed": False, "resolved": False, "error": msg}

        try:
            from swebench.harness.constants import (
                KEY_INSTANCE_ID,
                KEY_MODEL,
                KEY_PREDICTION,
            )
        except Exception as exc:  # pragma: no cover - dependency missing
            msg = (
                "SWE-bench harness is required for official scoring. "
                "Install swebench with evaluation extras."
            )
            logger.exception("Failed to import swebench harness constants: %s", exc)
            return {"completed": False, "resolved": False, "error": f"{msg} ({exc})"}

        backend = self._resolve_evaluation_backend(metadata)

        image_name = str(metadata.get("image_name") or "")
        namespace = metadata.get("namespace") or self._namespace_from_image(image_name) or "swebench"
        instance_image_tag = metadata.get("instance_image_tag") or self._image_tag_from_name(image_name) or "latest"
        env_image_tag = metadata.get("env_image_tag") or "latest"

        model_name = metadata.get("submission_model_name") or metadata.get("model_name") or "synth-ai-agent"
        run_id = f"swe_mini_eval_{uuid.uuid4().hex[:12]}"
        eval_timeout = self._resolve_eval_timeout(metadata)
        rm_image = self._to_bool(metadata.get("eval_rm_image") or os.getenv("SWE_MINI_EVAL_RM_IMAGE", "false"))
        force_rebuild = self._to_bool(metadata.get("eval_force_rebuild") or os.getenv("SWE_MINI_EVAL_FORCE_REBUILD", "false"))

        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: model_name,
            KEY_PREDICTION: diff or "",
        }

        # Ensure log root exists so downstream collection succeeds.
        with contextlib.suppress(Exception):
            from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

            Path(RUN_EVALUATION_LOG_DIR).mkdir(parents=True, exist_ok=True)

        if backend == "modal_harness":
            evaluation_payload = self._run_modal_harness(
                instance=instance,
                prediction=prediction,
                run_id=run_id,
                eval_timeout=eval_timeout,
                model_name=model_name,
            )
        elif backend == "swe_rex":
            evaluation_payload = self._run_swe_rex(
                instance=instance,
                prediction=prediction,
                run_id=run_id,
                eval_timeout=eval_timeout,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
                model_name=model_name,
            )
        else:
            evaluation_payload = self._run_local_harness(
                instance=instance,
                prediction=prediction,
                run_id=run_id,
                eval_timeout=eval_timeout,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
                rm_image=rm_image,
                force_rebuild=force_rebuild,
                model_name=model_name,
            )

        evaluation_payload = dict(evaluation_payload or {})
        evaluation_payload.setdefault("backend", backend)
        evaluation_payload.setdefault("run_id", run_id)
        evaluation_payload.setdefault("model_name", model_name)
        evaluation_payload.setdefault("instance_id", instance_id)

        artifacts = self._collect_evaluation_artifacts(
            run_id=run_id,
            model_name=model_name,
            instance_id=instance_id,
        )
        # Merge artifact data without clobbering explicit error/resolution flags.
        merged = {**artifacts, **evaluation_payload}
        if artifacts.get("completed"):
            merged["completed"] = True
        else:
            merged.setdefault("completed", False)
        if artifacts.get("resolved"):
            merged["resolved"] = True
        else:
            merged.setdefault("resolved", False)
        merged.setdefault("log_dir", artifacts.get("log_dir"))
        merged.setdefault("report_path", artifacts.get("report_path"))
        merged.setdefault("test_output_path", artifacts.get("test_output_path"))
        if artifacts.get("report") and not merged.get("report"):
            merged["report"] = artifacts["report"]
        if artifacts.get("error") and not merged.get("error"):
            merged["error"] = artifacts["error"]
        return merged

    def _resolve_evaluation_backend(self, metadata: dict[str, Any]) -> str:
        raw = (
            metadata.get("evaluation_backend")
            or self.env_config.get("evaluation_backend")
            or os.getenv("SWE_MINI_EVALUATION_BACKEND")
            or "local"
        )
        backend = str(raw).strip().lower()
        mapping = {
            "": "local",
            "local": "local",
            "docker": "local",
            "modal": "modal_harness",
            "modal_harness": "modal_harness",
            "modal-harness": "modal_harness",
            "modal-harnesses": "modal_harness",
            "swe_rex": "swe_rex",
            "swe-rex": "swe_rex",
            "swerex": "swe_rex",
        }
        return mapping.get(backend, "local")

    def _resolve_eval_timeout(self, metadata: dict[str, Any]) -> int:
        raw = (
            metadata.get("evaluation_timeout")
            or self.env_config.get("evaluation_timeout")
            or os.getenv("SWE_MINI_EVALUATION_TIMEOUT")
            or 3600
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 3600
        return max(1, value)

    def _run_local_harness(
        self,
        *,
        instance: dict[str, Any],
        prediction: dict[str, Any],
        run_id: str,
        eval_timeout: int,
        namespace: str,
        instance_image_tag: str,
        env_image_tag: str,
        rm_image: bool,
        force_rebuild: bool,
        model_name: str,
    ) -> dict[str, Any]:
        try:
            from swebench.harness.run_evaluation import run_instance
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception as exc:  # pragma: no cover - dependency missing
            msg = (
                "SWE-bench harness is required for official scoring. "
                "Install swebench with evaluation extras."
            )
            logger.exception("Failed to import swebench harness: %s", exc)
            return {"completed": False, "resolved": False, "error": f"{msg} ({exc})", "backend": "local"}

        try:
            import docker
        except Exception as exc:  # pragma: no cover - dependency missing
            msg = "Docker SDK for Python is required to run local SWE-bench evaluation."
            logger.exception("Failed to import docker SDK: %s", exc)
            return {"completed": False, "resolved": False, "error": f"{msg} ({exc})", "backend": "local"}

        instance_id = str(instance["instance_id"])
        try:
            test_spec = make_test_spec(
                instance,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
            )
        except Exception as exc:
            logger.exception("Failed to build SWE-bench test spec for %s: %s", instance_id, exc)
            return {"completed": False, "resolved": False, "error": f"Failed to build test spec: {exc}", "backend": "local"}

        client = None
        result: dict[str, Any] = {}
        try:
            client = docker.from_env()
            result = run_instance(
                test_spec,
                prediction,
                rm_image,
                force_rebuild,
                client,
                run_id,
                int(eval_timeout),
                rewrite_reports=False,
            )
        except Exception as exc:
            logger.exception("Error while running SWE-bench evaluation for %s: %s", instance_id, exc)
            return {"completed": False, "resolved": False, "error": f"Evaluation failed: {exc}", "backend": "local"}
        finally:
            with contextlib.suppress(Exception):
                if client is not None:
                    client.close()

        payload = {
            "completed": bool(result.get("completed")),
            "resolved": bool(result.get("resolved")),
            "backend": "local",
        }
        return payload

    def _run_modal_harness(
        self,
        *,
        instance: dict[str, Any],
        prediction: dict[str, Any],
        run_id: str,
        eval_timeout: int,
        model_name: str,
    ) -> dict[str, Any]:
        try:
            from swebench.harness.modal_eval import run_instances_modal
        except Exception as exc:  # pragma: no cover - dependency missing
            msg = (
                "SWE-bench modal extras are required for the modal_harness backend. "
                "Install swebench[modal] inside the Modal deployment."
            )
            logger.exception("Failed to import swebench modal harness: %s", exc)
            return {"completed": False, "resolved": False, "error": f"{msg} ({exc})", "backend": "modal_harness"}

        instance_id = str(instance["instance_id"])
        predictions = {instance_id: dict(prediction)}
        dataset = [instance]
        try:
            run_instances_modal(
                predictions,
                dataset,
                dataset,
                run_id,
                int(eval_timeout),
            )
        except Exception as exc:
            logger.exception("Modal SWE-bench evaluation failed for %s: %s", instance_id, exc)
            return {"completed": False, "resolved": False, "error": f"Modal evaluation failed: {exc}", "backend": "modal_harness"}

        # run_instances_modal writes reports to RUN_EVALUATION_LOG_DIR; we rely on artifact collection.
        return {"backend": "modal_harness"}

    def _run_swe_rex(
        self,
        *,
        instance: dict[str, Any],
        prediction: dict[str, Any],
        run_id: str,
        eval_timeout: int,
        namespace: str,
        instance_image_tag: str,
        env_image_tag: str,
        model_name: str,
    ) -> dict[str, Any]:
        try:
            from swerex.deployment.config import ModalDeploymentConfig
            from swerex.runtime.abstract import Command, ReadFileRequest, WriteFileRequest
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            msg = (
                "SWE-ReX backend requires the swe-rex package. "
                "Install swe-rex (pip install swe-rex[modal]) to enable this backend."
            )
            logger.exception("Failed to import swe-rex: %s", exc)
            return {"completed": False, "resolved": False, "error": f"{msg} ({exc})", "backend": "swe_rex"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected swe-rex import failure: %s", exc)
            return {"completed": False, "resolved": False, "error": f"swe-rex import failed: {exc}", "backend": "swe_rex"}

        image_spec = (
            instance.get("swe_rex_image")
            or self.env_config.get("swe_rex_image")
            or os.getenv("SWE_REX_MODAL_IMAGE")
            or "ghcr.io/swe-agent/swe-rex-modal:latest"
        )
        install_pipx = self._to_bool(
            instance.get("swe_rex_install_pipx")
            or self.env_config.get("swe_rex_install_pipx")
            or os.getenv("SWE_REX_INSTALL_PIPX", "true")
        )
        modal_kwargs_raw = (
            instance.get("swe_rex_modal_kwargs")
            or self.env_config.get("swe_rex_modal_kwargs")
            or os.getenv("SWE_REX_MODAL_SANDBOX_KWARGS")
        )
        modal_kwargs: dict[str, Any] = {}
        if isinstance(modal_kwargs_raw, (dict, list)):
            modal_kwargs = dict(modal_kwargs_raw or {})
        elif isinstance(modal_kwargs_raw, str) and modal_kwargs_raw.strip():
            try:
                modal_kwargs = dict(json.loads(modal_kwargs_raw))
            except Exception as exc:  # pragma: no cover - user input parsing
                logger.warning("Failed to parse SWE_REX_MODAL_SANDBOX_KWARGS=%s: %s", modal_kwargs_raw, exc)

        deployment_config = ModalDeploymentConfig(
            image=image_spec,
            runtime_timeout=float(
                instance.get("swe_rex_runtime_timeout")
                or self.env_config.get("swe_rex_runtime_timeout")
                or os.getenv("SWE_REX_RUNTIME_TIMEOUT", 900)
            ),
            deployment_timeout=float(
                instance.get("swe_rex_deployment_timeout")
                or self.env_config.get("swe_rex_deployment_timeout")
                or os.getenv("SWE_REX_DEPLOYMENT_TIMEOUT", 3600)
            ),
            modal_sandbox_kwargs=modal_kwargs,
            install_pipx=bool(install_pipx),
        )

        remote_root = (
            instance.get("swe_rex_workdir")
            or self.env_config.get("swe_rex_workdir")
            or os.getenv("SWE_REX_REMOTE_WORKDIR")
            or "/root/swebench_eval"
        )
        remote_root = str(remote_root).rstrip("/")
        dataset_remote_path = f"{remote_root}/dataset.json"
        predictions_remote_path = f"{remote_root}/predictions.json"

        environment_forward_raw = (
            instance.get("swe_rex_forward_env")
            or self.env_config.get("swe_rex_forward_env")
            or os.getenv("SWE_REX_FORWARD_ENV")
        )
        forward_env: dict[str, str] | None = None
        if isinstance(environment_forward_raw, dict):
            forward_env = {str(k): str(v) for k, v in environment_forward_raw.items()}
        elif isinstance(environment_forward_raw, str) and environment_forward_raw.strip():
            try:
                parsed = json.loads(environment_forward_raw)
                if isinstance(parsed, dict):
                    forward_env = {str(k): str(v) for k, v in parsed.items()}
            except Exception as exc:  # pragma: no cover - parsing failure
                logger.warning("Failed to parse SWE_REX_FORWARD_ENV=%s: %s", environment_forward_raw, exc)

        # Build coroutine for the async swe-rex flow.
        coro = self._run_swe_rex_async(
            deployment_config=deployment_config,
            remote_root=remote_root,
            dataset_remote_path=dataset_remote_path,
            predictions_remote_path=predictions_remote_path,
            forward_env=forward_env,
            instance=instance,
            prediction=prediction,
            run_id=run_id,
            eval_timeout=eval_timeout,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
            model_name=model_name,
            Command=Command,
            WriteFileRequest=WriteFileRequest,
            ReadFileRequest=ReadFileRequest,
        )
        try:
            return self._run_coroutine_blocking(coro)
        except Exception as exc:  # pragma: no cover - remote execution failure
            logger.exception("SWE-ReX evaluation failed for %s: %s", instance.get("instance_id"), exc)
            return {"completed": False, "resolved": False, "error": f"SWE-ReX evaluation failed: {exc}", "backend": "swe_rex"}

    async def _run_swe_rex_async(
        self,
        *,
        deployment_config,
        remote_root: str,
        dataset_remote_path: str,
        predictions_remote_path: str,
        forward_env: dict[str, str] | None,
        instance: dict[str, Any],
        prediction: dict[str, Any],
        run_id: str,
        eval_timeout: int,
        namespace: str,
        instance_image_tag: str,
        env_image_tag: str,
        model_name: str,
        Command,
        WriteFileRequest,
        ReadFileRequest,
    ) -> dict[str, Any]:
        deployment = deployment_config.get_deployment()
        await deployment.start()
        try:
            runtime = deployment.runtime
            instance_id = str(instance["instance_id"])
            safe_model = prediction["model_name_or_path"].replace("/", "__")

            # Ensure working directory exists.
            mkdir_resp = await runtime.execute(
                Command(command=["mkdir", "-p", remote_root], timeout=60, shell=False)
            )
            if mkdir_resp.exit_code not in (0, None):
                logger.warning("Failed to ensure remote directory %s (exit=%s)", remote_root, mkdir_resp.exit_code)

            # Upload dataset & predictions.
            dataset_blob = json.dumps([instance], ensure_ascii=False)
            predictions_blob = json.dumps({instance_id: prediction}, ensure_ascii=False)
            await runtime.write_file(WriteFileRequest(path=dataset_remote_path, content=dataset_blob))
            await runtime.write_file(WriteFileRequest(path=predictions_remote_path, content=predictions_blob))

            eval_cmd = [
                "python",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                dataset_remote_path,
                "--split",
                "test",
                "--instance_ids",
                instance_id,
                "--predictions_path",
                predictions_remote_path,
                "-id",
                run_id,
                "--modal",
                "true",
                "--timeout",
                str(eval_timeout),
                "--namespace",
                namespace,
                "--instance_image_tag",
                instance_image_tag,
                "--env_image_tag",
                env_image_tag,
                "--max_workers",
                "1",
            ]

            command_timeout = max(eval_timeout + 900, 1200)
            response = await runtime.execute(
                Command(
                    command=eval_cmd,
                    timeout=command_timeout,
                    cwd=remote_root,
                    env=forward_env,
                    shell=False,
                    merge_output_streams=True,
                )
            )
            command_output = (response.stdout or "") + (response.stderr or "")
            exit_code = response.exit_code if response.exit_code is not None else -1

            # Retrieve artifacts back to local disk.
            artifacts = {}
            try:
                from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

                local_log_dir = Path(RUN_EVALUATION_LOG_DIR) / run_id / safe_model / instance_id
                local_log_dir.mkdir(parents=True, exist_ok=True)

                remote_log_dir = f"{remote_root}/logs/run_evaluation/{run_id}/{safe_model}/{instance_id}"
                for filename in ("report.json", "test_output.txt", "run_instance.log", "patch.diff"):
                    remote_path = f"{remote_log_dir}/{filename}"
                    try:
                        content = await runtime.read_file(ReadFileRequest(path=remote_path))
                    except Exception:
                        continue
                    if getattr(content, "content", None):
                        (local_log_dir / filename).write_text(content.content)

                artifacts = {
                    "log_dir": str(local_log_dir),
                }
            except Exception as exc:  # pragma: no cover - best effort artifact copy
                logger.warning("Failed to copy SWE-ReX artifacts locally: %s", exc)

            payload = {
                "backend": "swe_rex",
                "command_exit_code": exit_code,
                "command_output": command_output[-4000:] if command_output else "",
                "artifacts": artifacts,
            }
            if exit_code == 0:
                payload.setdefault("completed", True)
            return payload
        finally:
            with contextlib.suppress(Exception):
                await deployment.stop()

    def _collect_evaluation_artifacts(
        self,
        *,
        run_id: str,
        model_name: str,
        instance_id: str,
    ) -> dict[str, Any]:
        try:
            from swebench.harness.constants import (
                LOG_REPORT,
                LOG_TEST_OUTPUT,
                RUN_EVALUATION_LOG_DIR,
            )
        except Exception:  # pragma: no cover - dependency missing
            return {
                "completed": False,
                "resolved": False,
                "log_dir": None,
                "report_path": None,
                "test_output_path": None,
            }

        log_model = model_name.replace("/", "__")
        log_dir = Path(RUN_EVALUATION_LOG_DIR) / run_id / log_model / instance_id
        payload: dict[str, Any] = {
            "log_dir": str(log_dir),
            "report_path": None,
            "test_output_path": None,
            "report": None,
            "completed": False,
            "resolved": False,
        }

        if not log_dir.exists():
            return payload

        report_path = log_dir / LOG_REPORT
        if report_path.exists():
            payload["report_path"] = str(report_path)
            try:
                report_blob = json.loads(report_path.read_text())
                per_instance = report_blob.get(instance_id)
                if per_instance is not None:
                    payload["report"] = per_instance
                    payload["completed"] = True
                    payload["resolved"] = bool(per_instance.get("resolved"))
            except Exception as exc:  # pragma: no cover - log parsing failure
                logger.exception("Failed to parse SWE-bench report for %s: %s", instance_id, exc)
                payload["error"] = f"Failed to parse report.json: {exc}"

        test_output_path = log_dir / LOG_TEST_OUTPUT
        if test_output_path.exists():
            payload["test_output_path"] = str(test_output_path)

        return payload

    @staticmethod
    def _run_coroutine_blocking(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result: dict[str, Any] = {}
            error: dict[str, Exception] = {}

            def runner():
                try:
                    result["value"] = asyncio.run(coro)
                except Exception as exc:  # pragma: no cover - propagate to caller
                    error["exc"] = exc

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join()
            if error:
                raise error["exc"]
            return result.get("value")

        return asyncio.run(coro)

    @staticmethod
    def _namespace_from_image(image_name: str) -> str | None:
        if not image_name:
            return None
        parts = image_name.split("/")
        if len(parts) >= 2:
            return parts[-2] if parts[0].endswith(".io") else parts[0]
        return None

    @staticmethod
    def _image_tag_from_name(image_name: str) -> str | None:
        if not image_name or ":" not in image_name:
            return None
        return image_name.rsplit(":", 1)[-1] or None

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False  # pragma: no cover - defensive default

    def _build_observation(self, last_result: dict[str, Any] | None) -> dict[str, Any]:
        trimmed_history = summarise_history(self.state.history)
        observation = {
            "task": self.task,
            "step_idx": self.state.step_idx,
            "history": trimmed_history,
            "submitted": self.state.submitted,
            "submission_success": self.state.submission_success,
            "tools": TOOLS_SCHEMA,
        }
        if last_result is not None:
            observation["last"] = last_result
        if self.last_submission is not None:
            observation["submission_result"] = self.last_submission
        return observation

    def _build_response(
        self,
        *,
        observation: dict[str, Any],
        step_idx: int,
        done: bool = False,
        reward: float | None = None,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = {
            "observation": observation,
            "step_idx": step_idx,
            "done": bool(done),
        }
        if reward is not None:
            response["reward"] = reward
        if info is not None:
            response["info"] = info
        return response

    def state_dict(self) -> dict[str, Any]:
        return {
            "task": self.state.task,
            "history": self.state.history,
            "step_idx": self.state.step_idx,
            "submitted": self.state.submitted,
            "submission_success": self.state.submission_success,
            "last_result": self.last_result,
            "last_submission": self.last_submission,
            "environment_type": self.environment_type,
            "env_config": self.env_config,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state = MiniSweEnvironmentState(
            task=payload["task"],
            history=payload.get("history", []),
            step_idx=int(payload.get("step_idx", 0)),
            submitted=bool(payload.get("submitted", False)),
            submission_success=payload.get("submission_success"),
        )
        self.last_result = payload.get("last_result")
        self.last_submission = payload.get("last_submission")
        self.environment_type = payload.get("environment_type", self.environment_type)
        self.env_config = payload.get("env_config", self.env_config)

    async def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "env_config": self.env_config,
                "submit_command": self.submit_command,
            },
            "state": self.state_dict(),
        }

    @classmethod
    async def deserialize(cls, payload: dict[str, Any]) -> MiniSweEnvironmentWrapper:
        config = payload.get("config", {}) or {}
        wrapper = cls(
            task=payload["state"]["task"],
            env_config=config.get("env_config"),
            submit_command=config.get("submit_command"),
        )
        wrapper.load_state_dict(payload["state"])
        return wrapper


__all__ = ["MiniSweEnvironmentWrapper"]
