import os
from pathlib import Path
from typing import Any, Literal, Optional, cast

from pydantic import BaseModel

from synth_ai.core.paths import get_bin_path
from synth_ai.core.telemetry import log_error, log_info


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
        ctx: dict[str, Any] = {
            "task_app_path": str(task_app_path),
            "trace": trace,
            "host": host,
            "port": port,
        }
        log_info("creating LocalDeployCfg", ctx=ctx)
        try:
            cfg = cls(
                task_app_path=task_app_path,
                env_api_key=env_api_key,
                trace=trace,
                host=host,
                port=port,
            )
            return cfg
        except Exception as exc:
            ctx["error"] = type(exc).__name__
            log_error("LocalDeployCfg creation failed", ctx=ctx)
            raise

    @classmethod
    def create_from_dict(
        cls,
        data: dict[str, Any]
    ) -> "LocalDeployCfg":
        path = Path(data["task_app_path"])
        trace = bool(data.get("trace", True))
        host = str(data.get("host", "127.0.0.1"))
        port = int(data.get("port", 8000))
        ctx: dict[str, Any] = {
            "task_app_path": str(path),
            "trace": trace,
            "host": host,
            "port": port,
        }
        log_info("creating LocalDeployCfg from dict", ctx=ctx)
        env_api_key = data.get("env_api_key")
        if not env_api_key or not isinstance(env_api_key, str):
            raise ValueError("env_api_key is required in local deploy configuration")

        try:
            cfg = cls(
                task_app_path=path,
                env_api_key=env_api_key,
                trace=trace,
                host=host,
                port=port,
            )
            return cfg
        except (KeyError, TypeError, ValueError) as err:
            ctx["error"] = type(err).__name__
            log_error("LocalDeployCfg from dict failed", ctx=ctx)
            raise ValueError(f"Invalid local deploy configuration: {err}") from err
        

class ModalDeployCfg(BaseModel):
    task_app_path: Path
    modal_app_path: Path
    modal_bin_path: Path
    synth_api_key: str
    env_api_key: str
    cmd_arg: Literal["deploy", "serve"] = "deploy"
    modal_app_name: Optional[str] = None
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
        modal_app_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> "ModalDeployCfg":
        ctx: dict[str, Any] = {
            "task_app_path": str(task_app_path),
            "modal_app_path": str(modal_app_path),
            "cmd_arg": cmd_arg,
            "dry_run": dry_run,
            "modal_app_name": modal_app_name,
            "modal_bin_path": str(modal_bin_path) if modal_bin_path else None,
        }
        log_info("creating ModalDeployCfg", ctx=ctx)
        modal_bin_path = modal_bin_path or get_bin_path("modal")
        if modal_bin_path is None:
            ctx["error"] = "ModalCLINotFound"
            log_error("ModalDeployCfg creation failed", ctx=ctx)
            raise ValueError("Modal CLI not found; install `modal` or pass --modal-cli with its path.")
        ctx["modal_bin_path"] = str(modal_bin_path)
        try:
            cfg = cls(
                task_app_path=task_app_path,
                modal_app_path=modal_app_path,
                modal_bin_path=modal_bin_path,
                synth_api_key=synth_api_key,
                env_api_key=env_api_key,
                cmd_arg=cmd_arg,
                modal_app_name=modal_app_name,
                dry_run=dry_run,
            )
            return cfg
        except Exception as exc:
            ctx["error"] = type(exc).__name__
            log_error("ModalDeployCfg creation failed", ctx=ctx)
            raise

    @classmethod
    def create_from_kwargs(cls, **kwargs: Any) -> "ModalDeployCfg":
        ctx: dict[str, Any] = {**kwargs}
        log_info("creating ModalDeployCfg from kwargs", ctx=ctx)

        synth_api_key = kwargs.get("synth_api_key")
        if not synth_api_key or not isinstance(synth_api_key, str):
            raise ValueError("synth_api_key must be provided as a string")
        env_api_key = kwargs.get("env_api_key")
        if not env_api_key or not isinstance(env_api_key, str):
            raise ValueError("env_api_key must be provided as a string")

        cmd_arg = str(kwargs.get("modal_mode", "deploy")).strip().lower()
        if cmd_arg not in {"deploy", "serve"}:
            raise ValueError("`--modal-mode` must be either 'deploy' or 'serve'.")

        dry_run = bool(kwargs.get("dry_run", False))
        if dry_run and os.getenv("CTX") == "mcp":
            ctx["error"] = "dry_run_mcp"
            log_error("ModalDeployCfg create_from_kwargs blocked", ctx=ctx)
            raise ValueError("`synth-ai deploy --runtime modal --dry-run` cannot be used by MCP")
        if dry_run and cmd_arg == "serve":
            ctx["error"] = "dry_run_serve"
            log_error("ModalDeployCfg create_from_kwargs blocked", ctx=ctx)
            raise ValueError("`synth-ai deploy --runtime modal --modal-mode serve` cannot be used with `--dry-run`")

        modal_bin_path_arg = kwargs.get("modal_cli")
        modal_bin_path = Path(str(modal_bin_path_arg)).expanduser() if modal_bin_path_arg else get_bin_path("modal")
        if modal_bin_path is None or not modal_bin_path.exists():
            ctx["error"] = "ModalCLINotFound"
            log_error("ModalDeployCfg create_from_kwargs failed", ctx=ctx)
            raise ValueError('Modal binary not found via shutil.which("modal"). Install `modal` or pass --modal-cli with its path.')
        modal_bin_path = modal_bin_path.resolve()

        modal_app_name = kwargs.get("name")
        if modal_app_name is not None:
            modal_app_name = str(modal_app_name).strip() or None

        literal_cmd = cast(Literal["deploy", "serve"], cmd_arg)

        task_app_path = kwargs.get("task_app_path")
        modal_app_path = kwargs.get("modal_app")

        if not task_app_path:
            raise ValueError("task_app_path is required")
        if not modal_app_path:
            raise ValueError("modal_app is required")

        if not isinstance(task_app_path, Path):
            task_app_path = Path(task_app_path)
        if not isinstance(modal_app_path, Path):
            modal_app_path = Path(modal_app_path)

        try:
            cfg = cls(
                task_app_path=task_app_path,
                modal_app_path=modal_app_path,
                modal_bin_path=modal_bin_path,
                synth_api_key=synth_api_key,
                env_api_key=env_api_key,
                cmd_arg=literal_cmd,
                modal_app_name=modal_app_name,
                dry_run=dry_run,
            )
            return cfg
        except Exception as exc:
            ctx["error"] = type(exc).__name__
            log_error("ModalDeployCfg from kwargs failed", ctx=ctx)
            raise


class CFDeployCfg(BaseModel):
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
    ) -> "CFDeployCfg":
        return cls(
            task_app_path=task_app_path,
            env_api_key=env_api_key,
            host=host,
            port=port,
            mode=mode,
            subdomain=subdomain,
            trace=trace,
        )
