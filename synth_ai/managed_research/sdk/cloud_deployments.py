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
from typing import Any, List, Mapping

from synth_ai.managed_research.sdk._base import _ClientNamespace

_RETRYABLE_FAILURE_STATES = frozenset({"failed"})
_TERMINAL_STATES = frozenset({"retired"})


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
    ) -> dict[str, Any]:
        return self._client.create_cloud_deployment(
            project_id=project_id,
            name=name,
            topology_id=topology_id,
            topology_version=topology_version,
            host_kind=host_kind,
            metadata=metadata,
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

    def observe(self, *, deployment_id: str) -> dict[str, Any]:
        return self._client.observe_cloud_deployment(deployment_id=deployment_id)

    def deploy(
        self,
        *,
        deployment_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._client.deploy_cloud_deployment(
            deployment_id=deployment_id,
            reason=reason,
        )

    def retire(
        self,
        *,
        deployment_id: str,
        reason: str | None = None,
        delete_vm: bool = False,
        confirm_vm_name: str | None = None,
    ) -> dict[str, Any]:
        return self._client.retire_cloud_deployment(
            deployment_id=deployment_id,
            reason=reason,
            delete_vm=delete_vm,
            confirm_vm_name=confirm_vm_name,
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


__all__ = ["CloudDeploymentsAPI"]
