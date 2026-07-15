"""CloudDeployment namespace — durable service deployments on persistent VMs.

The service lane of the substrate split: DevEnvironments are disposable
sandboxes; CloudDeployments are retained-by-default service stacks (exe).
Rows advance requested → provisioning → vm_ready → deploying → running,
driven by the backend reconciler; failed rows can be retried with deploy.
`running` means the deployed service answers its declared health path through
the VM's HTTPS proxy.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator, List, Literal, Mapping, TypedDict

from synth_ai.managed_research.models.cloud_deployment_claims import (
    ClaimHeartbeat,
    ClaimProjection,
    CloudDeploymentClaim,
)
from synth_ai.managed_research.sdk._base import _ClientNamespace

_RETRYABLE_FAILURE_STATES = frozenset({"failed"})
_TERMINAL_STATES = frozenset({"retired"})
CLOUD_SLOT_IDENTITIES = ("slot1-cloud", "slot2-cloud")
CloudSlotIdentity = Literal["slot1-cloud", "slot2-cloud"]


class CloudDeploymentProjectGitSource(TypedDict):
    """Immutable project-git source identity for a durable deployment."""

    kind: Literal["project_git"]
    source_commit_sha: str
    evidence_commit_sha: str
    instance_id: str


class CloudDeploymentService(TypedDict):
    """One topology-declared service available on a CloudDeployment."""

    service_id: str
    kind: str
    required: bool
    endpoint: str | None
    health_checks: list[dict[str, Any]]
    logs_supported: bool


class CloudDeploymentServices(TypedDict):
    """Declared service discovery projection (``cloud-deployment-services-v1``)."""

    schema_version: Literal["cloud-deployment-services-v1"]
    deployment_id: str
    vm_name: str
    service_url: str | None
    services: list[CloudDeploymentService]
    deployment_health: dict[str, Any]


class CloudDeploymentWorkspaceLiveState(TypedDict):
    """Observed Git state for one declared workspace repository."""

    available: bool
    head_commit_sha: str | None
    branch: str | None
    detached: bool
    dirty_path_count: int | None


class CloudDeploymentWorkspaceRepository(TypedDict):
    """Declared and observed source identity for one workspace repository."""

    repository: str
    path: str
    remote_repo: str
    declared_branch: str
    declared_source_commit_sha: str
    authority: str
    live: CloudDeploymentWorkspaceLiveState


class CloudDeploymentWorkspace(TypedDict):
    """Workspace/source proof projection (``cloud-deployment-workspace-v1``)."""

    schema_version: Literal["cloud-deployment-workspace-v1"]
    deployment_id: str
    vm_name: str
    workspace_root: str
    repositories: list[CloudDeploymentWorkspaceRepository]


class CloudDeploymentWorkspaceMaterialization(TypedDict):
    """Exact-source materialization receipt for a declared repository."""

    schema_version: Literal["cloud-deployment-workspace-materialization-v1"]
    deployment_id: str
    vm_name: str
    repository: str
    path: str
    branch: str
    source_commit_sha: str
    clean: bool
    detached: bool


class CloudDeploymentExecResult(TypedDict):
    """Bounded argv execution result (``cloud-deployment-exec-v1``)."""

    schema_version: Literal["cloud-deployment-exec-v1"]
    deployment_id: str
    vm_name: str
    working_directory: str
    argv_count: int
    timeout_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


class CloudDeploymentLogs(TypedDict):
    """Bounded declared-service logs result (``cloud-deployment-logs-v1``)."""

    schema_version: Literal["cloud-deployment-logs-v1"]
    deployment_id: str
    vm_name: str
    service_id: str
    tail: int
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


class CloudDeploymentsAPI(_ClientNamespace):
    def create(
        self,
        *,
        project_id: str,
        name: str,
        topology_id: str,
        topology_version: str | None = None,
        host_kind: str = "exe_dev",
        metadata: Mapping[str, Any] | None = None,
        source: CloudDeploymentProjectGitSource | Mapping[str, Any] | None = None,
        cloud_slot: CloudSlotIdentity | None = None,
    ) -> dict[str, Any]:
        return self._client.create_cloud_deployment(
            project_id=project_id,
            name=name,
            topology_id=topology_id,
            topology_version=topology_version,
            host_kind=host_kind,
            metadata=metadata,
            source=source,
            cloud_slot=cloud_slot,
        )

    def list(
        self,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> List[dict[str, Any]]:
        return self._client.list_cloud_deployments(
            project_id=project_id,
            limit=limit,
        )

    def get(self, *, deployment_id: str) -> dict[str, Any]:
        return self._client.get_cloud_deployment(deployment_id=deployment_id)

    def observe(
        self,
        *,
        deployment_id: str,
        fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Refresh substrate health, fenced when an active claim exists."""
        return self._client.observe_cloud_deployment(
            deployment_id=deployment_id,
            fencing_token=fencing_token,
        )

    def services(self, *, deployment_id: str) -> CloudDeploymentServices:
        """Discover topology-declared services and their routed endpoints."""
        return self._client.get_cloud_deployment_services(deployment_id=deployment_id)

    def workspace(self, *, deployment_id: str) -> CloudDeploymentWorkspace:
        """Inspect declared repositories and their live Git/source state."""
        return self._client.get_cloud_deployment_workspace(deployment_id=deployment_id)

    def materialize_workspace(
        self,
        *,
        deployment_id: str,
        repository: str,
        branch: str,
        source_commit_sha: str,
        fencing_token: int,
    ) -> CloudDeploymentWorkspaceMaterialization:
        """Materialize an exact declared source under an active fenced claim."""
        return self._client.materialize_cloud_deployment_workspace(
            deployment_id=deployment_id,
            repository=repository,
            branch=branch,
            source_commit_sha=source_commit_sha,
            fencing_token=fencing_token,
        )

    def exec(
        self,
        *,
        deployment_id: str,
        argv: Sequence[str],
        fencing_token: int,
        cwd: str | None = None,
        timeout_seconds: int = 300,
        max_output_bytes: int = 65_536,
    ) -> CloudDeploymentExecResult:
        """Execute argv (never caller shell text) under an active fenced claim."""
        return self._client.exec_cloud_deployment(
            deployment_id=deployment_id,
            argv=list(argv),
            fencing_token=fencing_token,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def logs(
        self,
        *,
        deployment_id: str,
        service_id: str,
        tail: int = 200,
    ) -> CloudDeploymentLogs:
        """Read bounded logs for a topology-declared service."""
        return self._client.get_cloud_deployment_logs(
            deployment_id=deployment_id,
            service_id=service_id,
            tail=tail,
        )

    def deploy(
        self,
        *,
        deployment_id: str,
        reason: str | None = None,
        fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Deploy (or retry) the stack; ``fencing_token`` is sent as ``X-Fencing-Token``.

        When a claim is active the backend refuses unfenced mutations
        (FencingTokenRequiredError) and superseded tokens
        (FencingTokenStaleError).
        """
        return self._client.deploy_cloud_deployment(
            deployment_id=deployment_id,
            reason=reason,
            fencing_token=fencing_token,
        )

    def retire(
        self,
        *,
        deployment_id: str,
        reason: str | None = None,
        delete_vm: bool = False,
        confirm_vm_name: str | None = None,
        fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Retire the deployment; ``fencing_token`` is sent as ``X-Fencing-Token``.

        When a claim is active the backend refuses unfenced mutations
        (FencingTokenRequiredError) and superseded tokens
        (FencingTokenStaleError).
        """
        return self._client.retire_cloud_deployment(
            deployment_id=deployment_id,
            reason=reason,
            delete_vm=delete_vm,
            confirm_vm_name=confirm_vm_name,
            fencing_token=fencing_token,
        )

    # ------------------------------------------------------------------
    # Claims (advisory TTL leases with integer fencing tokens)
    # ------------------------------------------------------------------

    def acquire_claim(
        self,
        *,
        deployment_id: str,
        holder: str,
        purpose: str,
        ttl_seconds: int,
    ) -> CloudDeploymentClaim:
        """Acquire the deployment's claim for ``holder``.

        Raises ClaimConflictError (409 ``claim_conflict:<holder>``) when another
        holder owns it; the error's ``holder`` attribute names the current owner.
        The returned claim carries the integer ``fencing_token`` to pass to
        ``deploy``/``retire``.

        Example::

            from synth_ai.managed_research import ManagedResearchClient

            client = ManagedResearchClient(
                api_key="sk-...",
                backend_base="https://staging.api.usesynth.ai",
            )
            claim = client.cloud_deployments.acquire_claim(
                deployment_id="cldep_123",
                holder="factory-worker-7",
                purpose="champion promotion",
                ttl_seconds=300,
            )
            client.cloud_deployments.deploy(
                deployment_id="cldep_123",
                fencing_token=claim.fencing_token,
            )
        """
        return CloudDeploymentClaim.from_wire(
            self._client.acquire_cloud_deployment_claim(
                deployment_id=deployment_id,
                holder=holder,
                purpose=purpose,
                ttl_seconds=ttl_seconds,
            )
        )

    def heartbeat_claim(
        self,
        *,
        deployment_id: str,
        claim_id: str,
    ) -> ClaimHeartbeat:
        """Renew the claim TTL; returns the new ``expires_at``.

        Raises ClaimExpiredError (410 ``claim_expired``) when the TTL already
        lapsed and ClaimSupersededError (409 ``claim_superseded``) when a newer
        claim replaced this one. The caller owns heartbeat cadence — the SDK
        never heartbeats in the background.
        """
        return ClaimHeartbeat.from_wire(
            self._client.heartbeat_cloud_deployment_claim(
                deployment_id=deployment_id,
                claim_id=claim_id,
            )
        )

    def release_claim(
        self,
        *,
        deployment_id: str,
        claim_id: str,
    ) -> CloudDeploymentClaim:
        """Release the claim (idempotent); returns the claim's final state."""
        return CloudDeploymentClaim.from_wire(
            self._client.release_cloud_deployment_claim(
                deployment_id=deployment_id,
                claim_id=claim_id,
            )
        )

    def get_claims(self, *, deployment_id: str) -> ClaimProjection:
        """Read the claim projection: active claim (or explicit none) + last fencing token issued."""
        return ClaimProjection.from_wire(
            self._client.get_cloud_deployment_claims(deployment_id=deployment_id)
        )

    @contextmanager
    def claim(
        self,
        *,
        deployment_id: str,
        holder: str,
        purpose: str,
        ttl_seconds: int,
    ) -> Iterator[CloudDeploymentClaim]:
        """Hold the deployment's claim for the duration of the ``with`` block.

        Acquires on entry and releases on exit — release runs even when the
        body raises, and nothing is swallowed: acquire conflicts, body
        exceptions, and release failures all propagate. Heartbeat is NOT
        automatic (no background threads or timers); the caller owns the
        heartbeat cadence and must call ``heartbeat_claim`` before the TTL
        lapses when the body outlives ``ttl_seconds``.

        Example::

            with client.cloud_deployments.claim(
                deployment_id="cldep_123",
                holder="factory-worker-7",
                purpose="champion promotion",
                ttl_seconds=300,
            ) as held:
                client.cloud_deployments.deploy(
                    deployment_id="cldep_123",
                    fencing_token=held.fencing_token,
                )
                # long work: caller heartbeats explicitly
                client.cloud_deployments.heartbeat_claim(
                    deployment_id="cldep_123",
                    claim_id=held.claim_id,
                )
        """
        held = self.acquire_claim(
            deployment_id=deployment_id,
            holder=holder,
            purpose=purpose,
            ttl_seconds=ttl_seconds,
        )
        try:
            yield held
        finally:
            self.release_claim(
                deployment_id=deployment_id,
                claim_id=held.claim_id,
            )

    def service_url(self, *, deployment_id: str) -> str | None:
        """HTTPS base URL of the deployed service, once a VM exists."""
        payload = self.get(deployment_id=deployment_id)
        url = str(payload.get("service_url") or "").strip()
        return url or None

    def wait_until_running(
        self,
        *,
        deployment_id: str,
        timeout_seconds: float = 1800.0,
        poll_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Poll until the deployment serves (state `running`).

        Raises RuntimeError with the deployment's failure_reason when it lands
        in a failed or terminal state, or TimeoutError when the budget runs out
        (the last payload is attached for diagnosis). A failed deployment can
        be retried with ``deploy`` after fixing the reported cause.
        """
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        payload = self.get(deployment_id=deployment_id)
        while True:
            state = str(payload.get("state") or "")
            if state == "running":
                return payload
            if state in _RETRYABLE_FAILURE_STATES:
                raise RuntimeError(
                    f"cloud deployment {deployment_id} reached retryable state "
                    f"{state}: {payload.get('failure_reason')}; fix the cause "
                    "then call deploy to retry"
                )
            if state in _TERMINAL_STATES:
                raise RuntimeError(
                    f"cloud deployment {deployment_id} reached terminal state "
                    f"{state}: {payload.get('failure_reason')}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"cloud deployment {deployment_id} not running after "
                    f"{timeout_seconds}s (state={state}, "
                    f"failure_reason={payload.get('failure_reason')})"
                )
            time.sleep(max(1.0, float(poll_seconds)))
            payload = self.get(deployment_id=deployment_id)


__all__ = [
    "CLOUD_SLOT_IDENTITIES",
    "CloudDeploymentExecResult",
    "CloudDeploymentLogs",
    "CloudDeploymentProjectGitSource",
    "CloudDeploymentService",
    "CloudDeploymentServices",
    "CloudDeploymentsAPI",
    "CloudDeploymentWorkspace",
    "CloudDeploymentWorkspaceLiveState",
    "CloudDeploymentWorkspaceMaterialization",
    "CloudDeploymentWorkspaceRepository",
    "CloudSlotIdentity",
]
