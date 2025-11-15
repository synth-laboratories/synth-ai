import os
from pathlib import Path
from typing import Any, Literal, Optional, cast

from pydantic import BaseModel
from synth_ai.utils import log_error, log_event
from synth_ai.utils.apps import validate_modal_app, validate_task_app
from synth_ai.utils.paths import get_bin_path


class LocalDeployCfg(BaseModel):
    task_app_path: Path
    env_api_key: str
    trace: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def create(
        cls,
        *,
        task_app_path: Path,
        env_api_key: str,
        trace: bool = True,
        host: str = "127.0.0.1",
        port: int = 8000
    ) -> "LocalDeployCfg":
        ctx = {
            "task_app": str(task_app_path),
            "trace": bool(trace),
            "host": str(host),
            "port": int(port),
        }
        log_event("info", "creating LocalDeployCfg", ctx=ctx)
        try:
            validate_task_app(task_app_path)
            cfg = cls(
                task_app_path=task_app_path,
                env_api_key=env_api_key,
                trace=trace,
                host=host,
                port=port,
            )
            log_event("info", "LocalDeployCfg created", ctx=ctx)
            return cfg
        except Exception as exc:
            log_error("LocalDeployCfg creation failed", ctx={**ctx, "error": type(exc).__name__})
            raise

    @classmethod
    def create_from_dict(
        cls,
        data: dict[str, Any]
    ) -> "LocalDeployCfg":
        path = Path(data["task_app_path"])
        ctx = {
            "task_app": str(path),
            "trace": bool(data.get("trace", True)),
            "host": str(data.get("host", "127.0.0.1")),
            "port": int(data.get("port", 8000)),
        }
        log_event("info", "creating LocalDeployCfg from dict", ctx=ctx)
        validate_task_app(path)
        try:
            cfg = cls(
                task_app_path=path,
                env_api_key=str(data.get("env_api_key", "")),
                trace=bool(data.get("trace", True)),
                host=str(data.get("host", "127.0.0.1")),
                port=int(data.get("port", 8000)),
            )
            log_event("info", "LocalDeployCfg from dict created", ctx=ctx)
            return cfg
        except (KeyError, TypeError, ValueError) as err:
            log_error("LocalDeployCfg from dict failed", ctx={**ctx, "error": type(err).__name__})
            raise ValueError(f"Invalid local deploy configuration: {err}") from err
        

class ModalDeployCfg(BaseModel):
    task_app_path: Path
    modal_app_path: Path
    modal_bin_path: Path
    synth_api_key: str
    env_api_key: str
    cmd_arg: Literal["deploy", "serve"] = "deploy"
    task_app_name: Optional[str] = None
    dry_run: bool = False

    @classmethod
    def create(
        cls,
        *,
        task_app_path: Path,
        modal_app_path: Path,
        synth_api_key: str,
        env_api_key: str,
        modal_bin_path: Path | None = None,
        cmd_arg: Literal["deploy", "serve"] = "deploy",
        task_app_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> "ModalDeployCfg":
        ctx = {
            "task_app": str(task_app_path),
            "modal_app": str(modal_app_path),
            "cmd_arg": cmd_arg,
            "dry_run": dry_run,
            "task_app_name": task_app_name,
            "modal_cli": str(modal_bin_path) if modal_bin_path else None,
        }
        log_event("info", "creating ModalDeployCfg", ctx=ctx)
        modal_bin_path = modal_bin_path or get_bin_path("modal")
        if modal_bin_path is None:
            log_error("ModalDeployCfg creation failed", ctx={**ctx, "error": "ModalCLINotFound"})
            raise ValueError("Modal CLI not found; install `modal` or pass --modal-cli with its path.")
        try:
            cfg = cls(
                task_app_path=validate_task_app(task_app_path),
                modal_app_path=validate_modal_app(modal_app_path),
                modal_bin_path=modal_bin_path,
                synth_api_key=synth_api_key,
                env_api_key=env_api_key,
                cmd_arg=cmd_arg,
                task_app_name=task_app_name,
                dry_run=dry_run,
            )
            log_event("info", "ModalDeployCfg created", ctx=ctx)
            return cfg
        except Exception as exc:
            log_error("ModalDeployCfg creation failed", ctx={**ctx, "error": type(exc).__name__})
            raise
    
    @classmethod
    def create_from_kwargs(cls, **kwargs: Any) -> "ModalDeployCfg":
        synth_api_key = kwargs.get("synth_api_key")
        if not synth_api_key:
            raise ValueError("synth_api_key must be provided")
        env_api_key = kwargs.get("env_api_key")
        if not env_api_key:
            raise ValueError("env_api_key must be provided")

        cmd_arg = str(kwargs.get("cmd_arg", "deploy")).strip().lower()
        if cmd_arg not in {"deploy", "serve"}:
            raise ValueError("`--modal-mode` must be either 'deploy' or 'serve'.")

        dry_run = bool(kwargs.get("dry_run", False))
        if dry_run and os.getenv("CTX") == "mcp":
            log_error("ModalDeployCfg create_from_kwargs blocked", ctx={"reason": "dry_run_mcp"})
            raise ValueError("`synth-ai deploy --runtime modal --dry-run` cannot be used by MCP")
        if dry_run and cmd_arg == "serve":
            log_error("ModalDeployCfg create_from_kwargs blocked", ctx={"reason": "dry_run_serve"})
            raise ValueError("`synth-ai deploy --runtime modal --modal-mode serve` cannot be used with `--dry-run`")

        modal_bin_path_arg = kwargs.get("modal_bin_path")
        modal_bin_path = Path(str(modal_bin_path_arg)).expanduser() if modal_bin_path_arg else get_bin_path("modal")
        if modal_bin_path is None or not modal_bin_path.exists():
            log_error(
                "ModalDeployCfg create_from_kwargs failed",
                ctx={"error": "ModalCLINotFound", "provided_modal_cli": str(modal_bin_path_arg or "")},
            )
            raise ValueError('Modal binary not found via shutil.which("modal"). Install `modal` or pass --modal-cli with its path.')
        modal_bin_path = modal_bin_path.resolve()

        task_app_name = kwargs.get("task_app_name")
        if task_app_name is not None:
            task_app_name = str(task_app_name).strip() or None

        literal_cmd = cast(Literal["deploy", "serve"], cmd_arg)

        ctx = {
            "task_app": str(kwargs.get("task_app_path")),
            "modal_app": str(kwargs.get("modal_app_path")),
            "cmd_arg": literal_cmd,
            "dry_run": dry_run,
            "task_app_name": task_app_name,
            "modal_cli": str(modal_bin_path) if modal_bin_path else None,
        }
        log_event("info", "creating ModalDeployCfg from kwargs", ctx=ctx)
        try:
            cfg = cls(
                task_app_path=validate_task_app(kwargs.get("task_app_path")),
                modal_app_path=validate_modal_app(kwargs.get("modal_app_path")),
                modal_bin_path=modal_bin_path,
                synth_api_key=synth_api_key,
                env_api_key=env_api_key,
                cmd_arg=literal_cmd,
                task_app_name=task_app_name,
                dry_run=dry_run,
            )
            log_event("info", "ModalDeployCfg from kwargs created", ctx=ctx)
            return cfg
        except Exception as exc:
            log_error("ModalDeployCfg from kwargs failed", ctx={**ctx, "error": type(exc).__name__})
            raise


class CloudflareTunnelDeployCfg(BaseModel):
    task_app_path: Path
    env_api_key: str
    host: str = "127.0.0.1"
    port: int = 8000
    mode: Literal["quick", "managed"] = "quick"
    tunnel_token: Optional[str] = None
    subdomain: Optional[str] = None
    trace: bool = True

    @classmethod
    def create(
        cls,
        *,
        task_app_path: Path,
        env_api_key: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        mode: Literal["quick", "managed"] = "quick",
        subdomain: Optional[str] = None,
        trace: bool = True,
    ) -> "CloudflareTunnelDeployCfg":
        validate_task_app(task_app_path)
        return cls(
            task_app_path=task_app_path,
            env_api_key=env_api_key,
            host=host,
            port=port,
            mode=mode,
            subdomain=subdomain,
            trace=trace,
        )
