import os
from pathlib import Path
from typing import Any, Literal, Optional, cast

from pydantic import BaseModel
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
        validate_task_app(task_app_path)
        return cls(
            task_app_path=task_app_path,
            env_api_key=env_api_key,
            trace=trace,
            host=host,
            port=port
        )

    @classmethod
    def create_from_dict(
        cls,
        data: dict[str, Any]
    ) -> "LocalDeployCfg":
        path = Path(data["task_app_path"])
        validate_task_app(path)
        try:
            return cls(
                task_app_path=path,
                env_api_key=str(data.get("env_api_key", '')),
                trace=bool(data.get("trace", True)),
                host=str(data.get("host", "127.0.0.1")),
                port=int(data.get("port", 8000))
            )
        except (KeyError, TypeError, ValueError) as err:
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
        modal_bin_path = modal_bin_path or get_bin_path("modal")
        if modal_bin_path is None:
            raise ValueError("Modal CLI not found; install `modal` or pass --modal-cli with its path.")
        return cls(
            task_app_path=validate_task_app(task_app_path),
            modal_app_path=validate_modal_app(modal_app_path),
            modal_bin_path=modal_bin_path,
            synth_api_key=synth_api_key,
            env_api_key=env_api_key,
            cmd_arg=cmd_arg,
            task_app_name=task_app_name,
            dry_run=dry_run
        )
    
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
            raise ValueError("`synth-ai deploy --runtime modal --dry-run` cannot be used by MCP")
        if dry_run and cmd_arg == "serve":
            raise ValueError("`synth-ai deploy --runtime modal --modal-mode serve` cannot be used with `--dry-run`")

        modal_bin_path_arg = kwargs.get("modal_bin_path")
        modal_bin_path = Path(str(modal_bin_path_arg)).expanduser() if modal_bin_path_arg else get_bin_path("modal")
        if modal_bin_path is None or not modal_bin_path.exists():
            raise ValueError('Modal binary not found via shutil.which("modal"). Install `modal` or pass --modal-cli with its path.')
        modal_bin_path = modal_bin_path.resolve()

        task_app_name = kwargs.get("task_app_name")
        if task_app_name is not None:
            task_app_name = str(task_app_name).strip() or None

        literal_cmd = cast(Literal["deploy", "serve"], cmd_arg)

        return cls(
            task_app_path=validate_task_app(kwargs.get("task_app_path")),
            modal_app_path=validate_modal_app(kwargs.get("modal_app_path")),
            modal_bin_path=modal_bin_path,
            synth_api_key=synth_api_key,
            env_api_key=env_api_key,
            cmd_arg=literal_cmd,
            task_app_name=task_app_name,
            dry_run=dry_run,
        )
