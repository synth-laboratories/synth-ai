from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
from synth_ai.core.http.request import HttpRequest
from synth_ai.sdk.research.container_pools import (
    ContainerPoolsAPI,
    HarborBundleError,
    build_docker_context_archive,
)


class _Transport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.requests: list[HttpRequest] = []

    def execute(self, request: HttpRequest) -> Any:
        self.requests.append(request)
        return next(self.responses)


def test_container_pool_control_plane_routes_and_contracts() -> None:
    transport = _Transport(
        [
            {"pool_id": "pool one", "backend": "arbitrary", "status": "active"},
            {"release_id": "release/one", "source_kind": "docker_context"},
            {"bound": True},
            {"rollout_id": "rollout one", "status": "queued"},
            {"operation_id": "operation-1", "status": "completed"},
            {"operation_id": "operation-1", "state": "completed"},
        ]
    )
    api = ContainerPoolsAPI(transport)  # type: ignore[arg-type]

    pool = api.get("pool one")
    release = api.get_release(pool.pool_id, "release/one")
    bound = api.bind_release(pool.pool_id, release.release_id, bind_timeout_seconds=45)
    rollout = api.create_rollout(pool.pool_id, task_id="task-1", seed=7)
    mutation = api.mutate_deployment(
        pool.pool_id,
        "task-1",
        project_id="11111111-1111-1111-1111-111111111111",
        operation="update",
        idempotency_key="update-1",
        expected_revision="revision-1",
        payload={"image": "example:v2"},
    )
    reconciled = api.find_deployment_operation(
        pool.pool_id,
        "task-1",
        project_id="11111111-1111-1111-1111-111111111111",
        idempotency_key="update-1",
    )

    assert pool.pool_id == "pool one"
    assert release.release_id == "release/one"
    assert bound == {"bound": True}
    assert rollout.rollout_id == "rollout one"
    assert mutation == {"operation_id": "operation-1", "status": "completed"}
    assert reconciled == {"operation_id": "operation-1", "state": "completed"}
    assert [request.path for request in transport.requests] == [
        "/v1/pools/pool%20one",
        "/v1/pools/pool%20one/runtime_image_releases/release%2Fone",
        "/v1/pools/pool%20one/runtime_image_releases/release%2Fone/bind",
        "/v1/pools/pool%20one/rollouts",
        "/v1/pools/pool%20one/deployments/task-1/operations",
        "/v1/pools/pool%20one/deployments/task-1/operations",
    ]
    assert transport.requests[2].timeout_seconds == 45
    assert transport.requests[3].body == {"task_id": "task-1", "seed": 7}
    assert transport.requests[4].body == {
        "project_id": "11111111-1111-1111-1111-111111111111",
        "operation": "update",
        "idempotency_key": "update-1",
        "payload": {"image": "example:v2"},
        "expected_revision": "revision-1",
    }
    assert transport.requests[5].query == {
        "project_id": "11111111-1111-1111-1111-111111111111",
        "idempotency_key": "update-1",
    }


def _archive_names(encoded: str) -> set[str]:
    raw = base64.b64decode(encoded)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        return {member.name for member in archive.getmembers() if member.isfile()}


def test_docker_context_packaging_honors_ignore_rules(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\nCOPY app.txt /app.txt\n")
    (tmp_path / ".dockerignore").write_text("ignored.txt\n")
    (tmp_path / "app.txt").write_text("payload")
    (tmp_path / "ignored.txt").write_text("do not upload")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("do not upload")

    packaged = build_docker_context_archive(tmp_path)

    assert _archive_names(packaged.archive_base64) == {
        ".dockerignore",
        "Dockerfile",
        "app.txt",
    }
    assert packaged.ignored_by_dockerignore == 2


def test_docker_context_packaging_rejects_credentials(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / ".env").write_text("SECRET=value")

    with pytest.raises(HarborBundleError, match="looks like a credential"):
        build_docker_context_archive(tmp_path)
