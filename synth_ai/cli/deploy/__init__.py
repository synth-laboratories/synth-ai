"""Deploy command package - imports from deploy.py module."""
from __future__ import annotations

# Import from the deploy.py module file (using importlib to avoid conflicts)
# This package exists for backwards compatibility
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from click import Command

try:
    # Import the deploy.py module directly by file path to avoid package/module conflict
    deploy_module_path = Path(__file__).parent.parent / "deploy.py"
    if deploy_module_path.exists():
        spec = importlib.util.spec_from_file_location("synth_ai.cli.deploy_module", deploy_module_path)
        if spec and spec.loader:
            deploy_module = importlib.util.module_from_spec(spec)
            sys.modules["synth_ai.cli.deploy_module"] = deploy_module
            spec.loader.exec_module(deploy_module)
            command: Command | None = getattr(deploy_module, "deploy_cmd", None)  # type: ignore[assignment]
            deploy_cmd: Command | None = command  # type: ignore[assignment]
        else:
            raise ImportError("Could not load deploy.py")
    else:
        raise ImportError("deploy.py not found")
    
    get_command: None = None  # Not used in current implementation
    
    __all__: list[str] = [
        "command",
        "deploy_cmd",
    ]
except Exception:
    # If deploy.py doesn't exist or fails to import, provide a stub
    command: Command | None = None  # type: ignore[assignment]
    deploy_cmd: Command | None = None  # type: ignore[assignment]
    get_command: None = None
    
    __all__: list[str] = []
