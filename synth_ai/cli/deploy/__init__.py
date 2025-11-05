"""Deploy command package - imports from parent deploy module."""
from __future__ import annotations

# Import from the parent deploy module (synth_ai.cli.deploy, not synth_ai.cli.deploy.core)
# This package exists for backwards compatibility
try:
    from ..deploy import deploy_cmd as command
    
    # Alias for backwards compatibility
    deploy_cmd = command
    get_command = None  # Not used in current implementation
    
    __all__ = [
        "command",
        "deploy_cmd",
    ]
except ImportError:
    # If deploy.py doesn't exist, provide a stub
    command = None
    deploy_cmd = None
    get_command = None
    
    __all__ = []
