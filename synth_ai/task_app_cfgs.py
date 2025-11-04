from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel


class LocalTaskAppConfig(BaseModel):
    task_app_path: Path
    trace: bool = True
    host: str = "127.0.0.1"
    port: int = 8000



class ModalTaskAppConfig(BaseModel):
    task_app_path: Path
    modal_app_path: Path
    modal_bin_path: Path
    cmd_arg: Literal["deploy", "serve"] = "deploy"
    task_app_name: Optional[str] = None
    dry_run: bool = False
