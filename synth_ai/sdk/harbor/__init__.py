"""Harbor SDK - Upload and manage Harbor deployments.

This module provides the SDK for uploading deployments to Harbor and
using them via the unified task app interface.

## Quick Start

### Upload a Deployment

```python
from synth_ai.sdk.harbor import HarborBuildSpec, upload_harbor_deployment

# Define your deployment
spec = HarborBuildSpec(
    name="my-agent-v1",
    dockerfile_path="./Dockerfile",
    context_dir=".",
    entrypoint="python run_rollout.py --input /tmp/rollout.json --output /tmp/result.json",
    limits={"timeout_s": 600, "memory_mb": 8192},
    metadata={"agent_type": "codex", "benchmark": "engine-bench"},
)

# Upload and wait for build
result = upload_harbor_deployment(spec, wait_for_ready=True)
print(f"Deployment ready: {result.deployment_id}")
```

### Use with LocalAPIConfig

```python
from synth_ai.sdk.localapi import LocalAPIConfig
from synth_ai.sdk.harbor import HarborDeploymentRef

# Reference an existing deployment
config = LocalAPIConfig(
    task_app_code=my_task_app,
    execution_backend="harbor",
    harbor=HarborDeploymentRef(deployment_id="abc-123"),
)
```

## Classes

- `HarborBuildSpec`: User-facing spec for defining deployments
- `HarborLimits`: Resource limits (timeout, CPU, memory, disk)
- `HarborDeploymentRef`: Reference to existing deployment
- `HarborDeploymentResult`: Result of upload operation

## Functions

- `upload_harbor_deployment()`: Upload a deployment (sync)
- `upload_harbor_deployment_async()`: Upload a deployment (async)
"""

from .build_spec import (
    HarborBuildSpec,
    HarborDeploymentRef,
    HarborDeploymentResult,
    HarborLimits,
)
from .packager import HarborPackager
from .uploader import (
    HarborAPIError,
    HarborDeploymentUploader,
    upload_harbor_deployment,
    upload_harbor_deployment_async,
)

__all__ = [
    # Build spec
    "HarborBuildSpec",
    "HarborLimits",
    "HarborDeploymentRef",
    "HarborDeploymentResult",
    # Packager
    "HarborPackager",
    # Uploader
    "HarborDeploymentUploader",
    "HarborAPIError",
    "upload_harbor_deployment",
    "upload_harbor_deployment_async",
]
