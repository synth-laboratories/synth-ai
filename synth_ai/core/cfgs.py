from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

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
        port: int = 8000,
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
    def create_from_dict(cls, data: dict[str, Any]) -> "LocalDeployCfg":
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
