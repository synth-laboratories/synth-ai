"""Backward-compatible tunnel helpers."""

from __future__ import annotations

import sys
import types

from synth_ai.core.tunnels import (  # noqa: F401
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    cleanup_all,
    find_available_port,
    is_port_available,
    kill_port,
    open_quick_tunnel_with_dns_verification,
    track_process,
    tracked_processes,
    wait_for_health_check,
)

__all__ = [
    "PortConflictBehavior",
    "TunnelBackend",
    "TunneledLocalAPI",
    "acquire_port",
    "cleanup_all",
    "find_available_port",
    "is_port_available",
    "kill_port",
    "open_quick_tunnel_with_dns_verification",
    "track_process",
    "tracked_processes",
    "wait_for_health_check",
]

# Backward-compatible submodule aliases (e.g., synth_ai.sdk.tunnels.tunneled_api)
_submodules: dict[str, dict[str, object]] = {
    "synth_ai.sdk.tunnels.tunneled_api": {
        "TunnelBackend": TunnelBackend,
        "TunneledLocalAPI": TunneledLocalAPI,
    },
    "synth_ai.sdk.tunnels.ports": {
        "acquire_port": acquire_port,
        "find_available_port": find_available_port,
        "is_port_available": is_port_available,
        "kill_port": kill_port,
        "PortConflictBehavior": PortConflictBehavior,
    },
    "synth_ai.sdk.tunnels.cleanup": {
        "cleanup_all": cleanup_all,
        "track_process": track_process,
        "tracked_processes": tracked_processes,
    },
    "synth_ai.sdk.tunnels.rust": {
        "open_quick_tunnel_with_dns_verification": open_quick_tunnel_with_dns_verification,
    },
}

for name, attrs in _submodules.items():
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
