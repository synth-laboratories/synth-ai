"""Container server config re-exports.

Prefer this module over synth_ai.sdk.container._impl.server.* moving forward.
"""

from __future__ import annotations

from synth_ai.sdk.container._impl.server import (
    ContainerConfig,
    ProxyConfig,
    RubricBundle,
    create_container,
    run_container,
)

create_container = create_container
run_container = run_container

__all__ = [
    "ContainerConfig",
    "ContainerConfig",
    "ProxyConfig",
    "RubricBundle",
    "create_container",
    "create_container",
    "run_container",
    "run_container",
]
