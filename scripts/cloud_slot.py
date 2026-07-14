#!/usr/bin/env python3
"""cloud-slot — bring exe.dev cloud slots up, run code on them, take them down.

A thin, typed operator CLI over ``ManagedResearchClient.cloud_deployments`` for
the up / run / down loop against remote branches:

    uv run scripts/cloud_slot.py up   --name josh-dev --commit <sha>
    uv run scripts/cloud_slot.py run  --name josh-dev            # redeploy + reprove
    uv run scripts/cloud_slot.py ls                              # what sltop CLOUD shows
    uv run scripts/cloud_slot.py down --name josh-dev            # retire (keep VM)
    uv run scripts/cloud_slot.py down --name josh-dev --delete-vm --confirm-vm-name <vm>

Config (same env the sltop CLOUD scope reads, so what you bring up shows up there):
    SYNTH_BACKEND_URL (or SYNTH_BACKEND_BASE)  — backend API base
    SYNTH_API_KEY                              — authorization
    PROJECT_ID                                 — default project

There is one correct path per verb; failures fail closed with the backend's typed
failure class rather than a bare status code, and never blind-retry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from synth_ai.managed_research import ManagedResearchClient
from synth_ai.managed_research.errors import SmrApiError


# ---- config -----------------------------------------------------------------

def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _require(value: str | None, what: str) -> str:
    if value:
        return value
    raise CloudSlotError(f"{what} is required (set the env var or pass the flag)")


def _backend_base(arg: str | None) -> str:
    # SYNTH_BACKEND_URL first — the same var the sltop CLOUD scope keys off, so
    # `up` and the TUI point at one backend without extra wiring.
    base = arg or _first_env("SYNTH_BACKEND_URL", "SYNTH_BACKEND_BASE")
    return _require(base, "backend base (SYNTH_BACKEND_URL)").rstrip("/")


def _client(args: argparse.Namespace) -> ManagedResearchClient:
    base = _backend_base(getattr(args, "backend_base", None))
    api_key = _require(
        getattr(args, "api_key", None) or _first_env("SYNTH_API_KEY"),
        "SYNTH_API_KEY",
    )
    return ManagedResearchClient(api_key=api_key, backend_base=base)


class CloudSlotError(RuntimeError):
    """A fail-closed operator error with a human-facing reason."""


# ---- helpers ----------------------------------------------------------------

def _summary(payload: dict) -> dict:
    return {
        "deployment_id": payload.get("deployment_id"),
        "name": payload.get("name"),
        "state": payload.get("state"),
        "vm_name": payload.get("vm_name"),
        "service_url": payload.get("service_url"),
        "failure_reason": payload.get("failure_reason"),
    }


def _resolve_deployment(client: ManagedResearchClient, args: argparse.Namespace) -> dict:
    """Address a slot by --deployment-id or by unique --name."""
    deployment_id = str(getattr(args, "deployment_id", "") or "").strip()
    if deployment_id:
        return client.cloud_deployments.get(deployment_id=deployment_id)
    name = str(getattr(args, "name", "") or "").strip()
    if not name:
        raise CloudSlotError("pass --deployment-id or --name to address a slot")
    project_id = _first_env("PROJECT_ID") if getattr(args, "project_id", None) is None else args.project_id
    matches = [
        row
        for row in client.cloud_deployments.list(project_id=project_id)
        if str(row.get("name") or "") == name and str(row.get("state") or "") != "retired"
    ]
    if not matches:
        raise CloudSlotError(f"no live cloud slot named {name!r}")
    if len(matches) > 1:
        ids = ", ".join(str(row.get("deployment_id")) for row in matches)
        raise CloudSlotError(f"name {name!r} is ambiguous across deployments: {ids}; use --deployment-id")
    return client.cloud_deployments.get(deployment_id=str(matches[0]["deployment_id"]))


def _build_source(args: argparse.Namespace) -> dict | None:
    commit = str(getattr(args, "commit", "") or "").strip()
    if not commit:
        return None
    evidence = str(getattr(args, "evidence_sha", "") or "").strip() or commit
    instance_id = str(getattr(args, "instance_id", "") or "").strip() or str(args.name)
    return {
        "kind": "project_git",
        "source_commit_sha": commit,
        "evidence_commit_sha": evidence,
        "instance_id": instance_id,
    }


def _wait_running(client: ManagedResearchClient, deployment_id: str, args: argparse.Namespace) -> dict:
    return client.cloud_deployments.wait_until_running(
        deployment_id=deployment_id,
        timeout_seconds=float(args.timeout),
        poll_seconds=float(args.poll),
    )


def _health_probe(service_url: str, health_path: str) -> dict:
    url = urllib.parse.urljoin(service_url.rstrip("/") + "/", health_path.lstrip("/"))
    request = urllib.request.Request(url, method="GET")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            status = int(response.status)
            return {"ok": 200 <= status < 300, "url": url, "status": status,
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
    except Exception as exc:  # noqa: BLE001 - report any probe failure verbatim
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - started, 3)}


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


# ---- verbs ------------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> int:
    client = _client(args)
    project_id = _require(args.project_id or _first_env("PROJECT_ID"), "PROJECT_ID")
    name = str(args.name)

    existing = [
        row
        for row in client.cloud_deployments.list(project_id=project_id)
        if str(row.get("name") or "") == name and str(row.get("state") or "") != "retired"
    ]
    if existing and not args.reuse:
        raise CloudSlotError(
            f"a live slot named {name!r} already exists ({existing[0].get('deployment_id')}); "
            "pass --reuse to adopt it or choose another --name"
        )
    if existing:
        deployment = client.cloud_deployments.get(deployment_id=str(existing[0]["deployment_id"]))
        action = "reused"
    else:
        source = _build_source(args)
        metadata = {"tool": "cloud-slot", "verb": "up"}
        if source is not None:
            metadata["source_commit_sha"] = source["source_commit_sha"]
        deployment = client.cloud_deployments.create(
            project_id=project_id,
            name=name,
            topology_id=args.topology_id,
            topology_version=args.topology_version,
            host_kind=args.host_kind,
            metadata=metadata,
            source=source,
        )
        action = "created"

    deployment_id = str(deployment.get("deployment_id") or "")
    if not deployment_id:
        raise CloudSlotError(f"{action} deployment has no deployment_id: {deployment}")

    result = {"action": action, "deployment": _summary(deployment)}
    if args.wait:
        running = _wait_running(client, deployment_id, args)
        result["deployment"] = _summary(running)
        service_url = str(running.get("service_url") or "").strip()
        if service_url:
            result["health"] = _health_probe(service_url, args.health_path)
    _emit(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    deployment_id = str(deployment["deployment_id"])
    queued = client.cloud_deployments.deploy(deployment_id=deployment_id, reason=args.reason)
    result = {"action": "redeploy", "deployment": _summary(queued)}
    if args.wait:
        running = _wait_running(client, deployment_id, args)
        result["deployment"] = _summary(running)
        service_url = str(running.get("service_url") or "").strip()
        if service_url:
            result["health"] = _health_probe(service_url, args.health_path)
    _emit(result)
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    deployment_id = str(deployment["deployment_id"])
    vm_name = str(deployment.get("vm_name") or "").strip()
    confirm = str(args.confirm_vm_name or "").strip()
    if args.delete_vm and confirm != vm_name:
        raise CloudSlotError(
            "--delete-vm destroys the VM: --confirm-vm-name must exactly match "
            f"the deployment vm_name {vm_name!r} (got {confirm!r})"
        )
    retired = client.cloud_deployments.retire(
        deployment_id=deployment_id,
        reason=args.reason,
        delete_vm=bool(args.delete_vm),
        confirm_vm_name=confirm if args.delete_vm else None,
    )
    _emit({"action": "retire", "delete_vm": bool(args.delete_vm), "deployment": _summary(retired)})
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    client = _client(args)
    project_id = args.project_id if args.project_id is not None else _first_env("PROJECT_ID")
    rows = client.cloud_deployments.list(project_id=project_id, limit=args.limit)
    _emit({"count": len(rows), "deployments": [_summary(row) for row in rows]})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    _emit(deployment)
    return 0


# ---- parser -----------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloud-slot", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-base", default=None, help="backend API base (default: $SYNTH_BACKEND_BASE)")
    parser.add_argument("--api-key", default=None, help="authorization (default: $SYNTH_API_KEY)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_wait_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--wait", dest="wait", action="store_true", default=True,
                       help="poll until running + health (default)")
        p.add_argument("--no-wait", dest="wait", action="store_false",
                       help="return immediately after the control-plane call")
        p.add_argument("--timeout", type=float, default=1800.0, help="max seconds to poll for running")
        p.add_argument("--poll", type=float, default=10.0, help="poll interval seconds")
        p.add_argument("--health-path", default="/health", help="service health path probed once running")

    p_up = sub.add_parser("up", help="create (or --reuse) a cloud slot and wait until it serves")
    p_up.add_argument("--name", required=True)
    p_up.add_argument("--project-id", default=None)
    p_up.add_argument("--topology-id", default="synth-dev")
    p_up.add_argument("--topology-version", default=None)
    p_up.add_argument("--host-kind", default="exe_dev")
    p_up.add_argument("--commit", default=None, help="pin a remote-branch commit sha (project git source)")
    p_up.add_argument("--evidence-sha", default=None, help="evidence commit sha (default: --commit)")
    p_up.add_argument("--instance-id", default=None, help="stable source instance id (default: --name)")
    p_up.add_argument("--reuse", action="store_true", help="adopt an existing live slot of the same name")
    add_wait_flags(p_up)
    p_up.set_defaults(func=cmd_up)

    p_run = sub.add_parser("run", help="redeploy a slot (push the branch's code) and reprove health")
    p_run.add_argument("--name", default=None)
    p_run.add_argument("--deployment-id", default=None)
    p_run.add_argument("--project-id", default=None)
    p_run.add_argument("--reason", default="cloud-slot run")
    add_wait_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_down = sub.add_parser("down", help="retire a slot (keeps the VM unless --delete-vm)")
    p_down.add_argument("--name", default=None)
    p_down.add_argument("--deployment-id", default=None)
    p_down.add_argument("--project-id", default=None)
    p_down.add_argument("--reason", default="cloud-slot down")
    p_down.add_argument("--delete-vm", action="store_true", help="destroy the VM (requires --confirm-vm-name)")
    p_down.add_argument("--confirm-vm-name", default=None, help="exact vm_name echo, required with --delete-vm")
    p_down.set_defaults(func=cmd_down)

    p_ls = sub.add_parser("ls", help="list cloud slots (what the sltop CLOUD scope renders)")
    p_ls.add_argument("--project-id", default=None)
    p_ls.add_argument("--limit", type=int, default=100)
    p_ls.set_defaults(func=cmd_ls)

    p_status = sub.add_parser("status", help="full JSON for one slot")
    p_status.add_argument("--name", default=None)
    p_status.add_argument("--deployment-id", default=None)
    p_status.add_argument("--project-id", default=None)
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CloudSlotError as exc:
        print(f"cloud-slot: {exc}", file=sys.stderr)
        return 2
    except SmrApiError as exc:
        # Surface the backend's failure *class*, not a bare status code.
        parts = [f"cloud-slot: backend error: {exc}"]
        if exc.status_code is not None:
            parts.append(f"status={exc.status_code}")
        print(" | ".join(parts), file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(f"cloud-slot: not running in budget: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # wait_until_running raises RuntimeError with the failure_reason on
        # failed/retired states — that is a typed cause, not a crash.
        print(f"cloud-slot: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
