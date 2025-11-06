from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel
from synth_ai.utils.app import validate_task_app


class LocalDeployCfg(BaseModel):
    task_app_path: Path
    env_api_key: str
    trace: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def create(
        cls,
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
    cmd_arg: Literal["deploy", "serve"] = "deploy"
    task_app_name: Optional[str] = None
    dry_run: bool = False
