from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel


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
        try:
            cfg = cls(
                task_app_path=task_app_path,
                env_api_key=env_api_key,
                trace=trace,
                host=host,
                port=port,
            )
            return cfg
        except Exception:
            raise

    @classmethod
    def create_from_dict(cls, data: dict[str, Any]) -> "LocalDeployCfg":
        path = Path(data["task_app_path"])
        trace = bool(data.get("trace", True))
        host = str(data.get("host", "127.0.0.1"))
        port = int(data.get("port", 8000))
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
