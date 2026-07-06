from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_ai.managed_research.mcp.server import ManagedResearchMcpServer

RUNNING_STATES = {"running"}
RETRYABLE_FAILURE_STATES = {"failed"}
TERMINAL_STATES = {"retired"}
PENDING_STATES = {"requested", "provisioning", "vm_ready", "deploying", "degraded"}
CLOUD_DEPLOYMENT_TOOLS = {
    "smr_list_cloud_deployments",
    "smr_create_cloud_deployment",
    "smr_get_cloud_deployment",
    "smr_observe_cloud_deployment",
    "smr_deploy_cloud_deployment",
    "smr_retire_cloud_deployment",
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env_or_arg(value: str | None, env_name: str) -> str:
    text = str(value or os.environ.get(env_name) or "").strip()
    if text:
        return text
    raise RuntimeError(f"{env_name} is required")


def _parse_metadata(values: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"metadata must be key=value, got: {value}")
        metadata[key.strip()] = raw
    return metadata


def _deployment_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"raw": payload}
    return {
        "deployment_id": payload.get("deployment_id"),
        "name": payload.get("name"),
        "state": payload.get("state"),
        "vm_name": payload.get("vm_name"),
        "service_url": payload.get("service_url"),
        "failure_reason": payload.get("failure_reason"),
        "health": payload.get("health"),
    }


def _health_probe(service_url: str, health_path: str) -> dict[str, Any]:
    url = urllib.parse.urljoin(service_url.rstrip("/") + "/", health_path.lstrip("/"))
    started = time.monotonic()
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            body = response.read(4000).decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 300,
                "url": url,
                "status": int(response.status),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "body_tail": body[-4000:],
            }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_receipt(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tool_registry_receipt(server: ManagedResearchMcpServer) -> dict[str, Any]:
    available_tools = set(server.available_tool_names())
    missing_tools = sorted(CLOUD_DEPLOYMENT_TOOLS - available_tools)
    missing_aliases = sorted(
        f"research_{name[4:]}"
        for name in CLOUD_DEPLOYMENT_TOOLS
        if f"research_{name[4:]}" not in available_tools
    )
    return {
        "missing_tools": missing_tools,
        "missing_aliases": missing_aliases,
    }


def _wait_for_running(
    *,
    server: ManagedResearchMcpServer,
    deployment_id: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    observations: list[dict[str, Any]] = []
    while True:
        payload = server.call_tool(
            "smr_get_cloud_deployment",
            {"deployment_id": deployment_id},
        )
        summary = _deployment_summary(payload)
        summary["observed_at"] = _utc_now()
        observations.append(summary)
        state = str(summary.get("state") or "")
        if state in RUNNING_STATES:
            return payload, observations
        if state in RETRYABLE_FAILURE_STATES:
            raise RuntimeError(
                "cloud deployment reached retryable failed state: "
                f"failure_reason={summary.get('failure_reason')}; "
                "fix the cause then call deploy to retry"
            )
        if state in TERMINAL_STATES:
            raise RuntimeError(
                "cloud deployment reached terminal state: "
                f"state={state} failure_reason={summary.get('failure_reason')}"
            )
        if state not in PENDING_STATES:
            raise RuntimeError(f"unknown cloud deployment state: {state}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"cloud deployment did not reach running before timeout: last={summary}"
            )
        time.sleep(max(1.0, poll_seconds))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CloudDeployment P8 live MCP proof from the synth-ai client "
            "worktree. The script creates only when --create is supplied."
        )
    )
    parser.add_argument("--backend-base", default=os.environ.get("SYNTH_BACKEND_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("SYNTH_API_KEY"))
    parser.add_argument("--project-id", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--registry-only", action="store_true")
    parser.add_argument(
        "--name",
        default=os.environ.get("DEPLOYMENT_NAME") or f"synth-dev-mcp-p8-{int(time.time())}",
    )
    parser.add_argument("--topology-id", default="synth-dev")
    parser.add_argument("--topology-version", default=None)
    parser.add_argument("--host-kind", default="exe_dev")
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--redeploy", action="store_true")
    parser.add_argument("--retire-at-end", action="store_true")
    parser.add_argument("--delete-vm", action="store_true")
    parser.add_argument("--confirm-vm-name", default=None)
    parser.add_argument("--retire-reason", default="mcp p8 proof complete")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt: dict[str, Any] = {
        "started_at": _utc_now(),
        "proof": "cloud_deployment_mcp_p8",
        "backend_base": args.backend_base,
        "deployment_id": args.deployment_id,
        "topology_id": args.topology_id,
        "host_kind": args.host_kind,
        "steps": [],
    }
    try:
        if args.registry_only:
            server = ManagedResearchMcpServer(
                api_key=args.api_key,
                backend_base=args.backend_base,
            )
            registry = _tool_registry_receipt(server)
            receipt["steps"].append({"step": "tool_registry", **registry})
            if registry["missing_tools"] or registry["missing_aliases"]:
                raise RuntimeError(
                    "missing CloudDeployment MCP tools="
                    f"{registry['missing_tools']} aliases={registry['missing_aliases']}"
                )
            receipt["completed_at"] = _utc_now()
            receipt["ok"] = True
            _write_receipt(args.output, receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0

        deployment_id = str(args.deployment_id or "").strip()
        if not deployment_id and not args.create:
            raise RuntimeError("--deployment-id or --create is required")
        if args.create and deployment_id:
            raise RuntimeError("--create cannot be combined with --deployment-id")
        if args.delete_vm and not args.retire_at_end:
            raise RuntimeError("--delete-vm requires --retire-at-end")
        if args.delete_vm and not str(args.confirm_vm_name or "").strip():
            raise RuntimeError("--delete-vm requires --confirm-vm-name")

        backend_base = _env_or_arg(args.backend_base, "SYNTH_BACKEND_BASE")
        api_key = _env_or_arg(args.api_key, "SYNTH_API_KEY")
        server = ManagedResearchMcpServer(api_key=api_key, backend_base=backend_base)
        registry = _tool_registry_receipt(server)
        receipt["steps"].append({"step": "tool_registry", **registry})
        if registry["missing_tools"] or registry["missing_aliases"]:
            raise RuntimeError(
                "missing CloudDeployment MCP tools="
                f"{registry['missing_tools']} aliases={registry['missing_aliases']}"
            )

        if args.create:
            project_id = _env_or_arg(args.project_id, "PROJECT_ID")
            metadata = {"proof": "cloud_deployment_mcp_p8"}
            metadata.update(_parse_metadata(args.metadata))
            deployment = server.call_tool(
                "smr_create_cloud_deployment",
                {
                    "project_id": project_id,
                    "name": args.name,
                    "topology_id": args.topology_id,
                    "topology_version": args.topology_version,
                    "host_kind": args.host_kind,
                    "metadata": metadata,
                },
            )
            deployment_id = str(deployment.get("deployment_id") or "")
            if not deployment_id:
                raise RuntimeError(f"create response did not include deployment_id: {deployment}")
            receipt["deployment_id"] = deployment_id
            receipt["steps"].append(
                {"step": "create", "deployment": _deployment_summary(deployment)}
            )
        else:
            deployment = server.call_tool(
                "smr_get_cloud_deployment",
                {"deployment_id": deployment_id},
            )
            receipt["steps"].append({"step": "load", "deployment": _deployment_summary(deployment)})

        running, observations = _wait_for_running(
            server=server,
            deployment_id=deployment_id,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        receipt["steps"].append(
            {
                "step": "wait_until_running",
                "deployment": _deployment_summary(running),
                "observations": observations,
            }
        )
        current_deployment = running

        service_url = str(running.get("service_url") or "").strip()
        if not service_url:
            raise RuntimeError("running deployment did not include service_url")
        health = _health_probe(service_url, args.health_path)
        receipt["steps"].append({"step": "health", "result": health})
        if not health["ok"]:
            raise RuntimeError(f"health probe failed: {health}")

        if args.redeploy:
            queued = server.call_tool(
                "smr_deploy_cloud_deployment",
                {"deployment_id": deployment_id, "reason": "mcp p8 proof idempotent redeploy"},
            )
            receipt["steps"].append(
                {"step": "redeploy_queue", "deployment": _deployment_summary(queued)}
            )
            redeployed, redeploy_observations = _wait_for_running(
                server=server,
                deployment_id=deployment_id,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
            receipt["steps"].append(
                {
                    "step": "redeploy_wait_until_running",
                    "deployment": _deployment_summary(redeployed),
                    "observations": redeploy_observations,
                }
            )
            current_deployment = redeployed

        if args.retire_at_end:
            vm_name = str(current_deployment.get("vm_name") or "").strip()
            confirmed_vm_name = str(args.confirm_vm_name or "").strip()
            if args.delete_vm and confirmed_vm_name != vm_name:
                raise RuntimeError(
                    "--confirm-vm-name must exactly match deployment vm_name "
                    f"for delete: expected {vm_name!r}, got {confirmed_vm_name!r}"
                )
            retired = server.call_tool(
                "smr_retire_cloud_deployment",
                {
                    "deployment_id": deployment_id,
                    "reason": args.retire_reason,
                    "delete_vm": bool(args.delete_vm),
                    "confirm_vm_name": confirmed_vm_name if args.delete_vm else None,
                },
            )
            receipt["steps"].append({"step": "retire", "deployment": _deployment_summary(retired)})

        receipt["completed_at"] = _utc_now()
        receipt["ok"] = True
        _write_receipt(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        receipt["completed_at"] = _utc_now()
        receipt["ok"] = False
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write_receipt(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
