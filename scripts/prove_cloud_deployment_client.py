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

from synth_ai.managed_research import ManagedResearchClient


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _env_or_arg(value: str | None, env_name: str) -> str:
    text = str(value or os.environ.get(env_name) or "").strip()
    if text:
        return text
    raise RuntimeError(f"{env_name} is required")


def _backend_base_default() -> str | None:
    return os.environ.get("SYNTH_BACKEND_BASE") or os.environ.get("SYNTH_BACKEND_URL")


def _backend_base_arg(value: str | None) -> str:
    text = str(value or _backend_base_default() or "").strip()
    if text:
        return text
    raise RuntimeError("SYNTH_BACKEND_BASE or SYNTH_BACKEND_URL is required")


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CloudDeployment P8 live SDK proof from the synth-ai client "
            "worktree. The script creates only when --create is supplied."
        )
    )
    parser.add_argument("--backend-base", default=_backend_base_default())
    parser.add_argument("--api-key", default=os.environ.get("SYNTH_API_KEY"))
    parser.add_argument("--project-id", default=os.environ.get("PROJECT_ID"))
    parser.add_argument("--deployment-id", default=None)
    parser.add_argument("--create", action="store_true")
    parser.add_argument(
        "--name",
        default=os.environ.get("DEPLOYMENT_NAME") or f"synth-dev-sdk-p8-{int(time.time())}",
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
    parser.add_argument("--retire-reason", default="sdk p8 proof complete")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt: dict[str, Any] = {
        "started_at": _utc_now(),
        "proof": "cloud_deployment_sdk_p8",
        "backend_base": args.backend_base,
        "deployment_id": args.deployment_id,
        "topology_id": args.topology_id,
        "host_kind": args.host_kind,
        "steps": [],
    }
    try:
        deployment_id = str(args.deployment_id or "").strip()
        if not deployment_id and not args.create:
            raise RuntimeError("--deployment-id or --create is required")
        if args.create and deployment_id:
            raise RuntimeError("--create cannot be combined with --deployment-id")
        if args.delete_vm and not args.retire_at_end:
            raise RuntimeError("--delete-vm requires --retire-at-end")
        if args.delete_vm and not str(args.confirm_vm_name or "").strip():
            raise RuntimeError("--delete-vm requires --confirm-vm-name")

        backend_base = _backend_base_arg(args.backend_base)
        api_key = _env_or_arg(args.api_key, "SYNTH_API_KEY")
        client = ManagedResearchClient(api_key=api_key, backend_base=backend_base)

        if args.create:
            project_id = _env_or_arg(args.project_id, "PROJECT_ID")
            metadata = {"proof": "cloud_deployment_sdk_p8"}
            metadata.update(_parse_metadata(args.metadata))
            deployment = client.cloud_deployments.create(
                project_id=project_id,
                name=args.name,
                topology_id=args.topology_id,
                topology_version=args.topology_version,
                host_kind=args.host_kind,
                metadata=metadata,
            )
            deployment_id = str(deployment.get("deployment_id") or "")
            if not deployment_id:
                raise RuntimeError(f"create response did not include deployment_id: {deployment}")
            receipt["deployment_id"] = deployment_id
            receipt["steps"].append(
                {"step": "create", "deployment": _deployment_summary(deployment)}
            )
        else:
            deployment = client.cloud_deployments.get(deployment_id=deployment_id)
            receipt["steps"].append({"step": "load", "deployment": _deployment_summary(deployment)})

        running = client.cloud_deployments.wait_until_running(
            deployment_id=deployment_id,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        receipt["steps"].append(
            {"step": "wait_until_running", "deployment": _deployment_summary(running)}
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
            queued = client.cloud_deployments.deploy(
                deployment_id=deployment_id,
                reason="sdk p8 proof idempotent redeploy",
            )
            receipt["steps"].append(
                {"step": "redeploy_queue", "deployment": _deployment_summary(queued)}
            )
            redeployed = client.cloud_deployments.wait_until_running(
                deployment_id=deployment_id,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
            receipt["steps"].append(
                {
                    "step": "redeploy_wait_until_running",
                    "deployment": _deployment_summary(redeployed),
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
            retired = client.cloud_deployments.retire(
                deployment_id=deployment_id,
                reason=args.retire_reason,
                delete_vm=bool(args.delete_vm),
                confirm_vm_name=confirmed_vm_name if args.delete_vm else None,
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
