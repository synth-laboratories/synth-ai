#!/usr/bin/env python3
"""cloud-slot — bring exe.dev cloud slots up, run code on them, take them down.

A thin, typed operator CLI over ``ManagedResearchClient.cloud_deployments`` for
the up / run / down loop against remote branches:

    uv run scripts/cloud_slot.py up   --cloud-slot slot1-cloud --commit <sha>
    uv run scripts/cloud_slot.py run  --cloud-slot slot1-cloud    # redeploy + reprove
    uv run scripts/cloud_slot.py exec --cloud-slot slot1-cloud --fencing-token <n> -- git status
    uv run scripts/cloud_slot.py services --cloud-slot slot1-cloud
    uv run scripts/cloud_slot.py logs --cloud-slot slot1-cloud --service-id backend-api
    uv run scripts/cloud_slot.py artifacts --cloud-slot slot1-cloud
    uv run scripts/cloud_slot.py artifact --cloud-slot slot1-cloud \
        --root-id reportbench-readme-smoke --path run.json --output run.json
    uv run scripts/cloud_slot.py ls                              # what sltop CLOUD shows
    uv run scripts/cloud_slot.py down --cloud-slot slot1-cloud    # retire (keep VM)
    uv run scripts/cloud_slot.py down --cloud-slot slot1-cloud --delete-vm --confirm-vm-name <vm>

Config (same env the sltop CLOUD scope reads, so what you bring up shows up there):
    SYNTH_BACKEND_URL (or SYNTH_BACKEND_BASE)  — backend API base
    SYNTH_API_KEY                              — authorization
    PROJECT_ID                                 — default project

There is one correct path per verb; failures fail closed with the backend's typed
failure class rather than a bare status code, and never blind-retry.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

from synth_ai.managed_research import CLOUD_SLOT_IDENTITIES, ManagedResearchClient
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
        "cloud_slot": payload.get("cloud_slot"),
        "name": payload.get("name"),
        "state": payload.get("state"),
        "vm_name": payload.get("vm_name"),
        "service_url": payload.get("service_url"),
        "failure_reason": payload.get("failure_reason"),
    }


def _resolve_deployment(client: ManagedResearchClient, args: argparse.Namespace) -> dict:
    """Address a slot by deployment id, canonical cloud slot, or unique name."""
    deployment_id = str(getattr(args, "deployment_id", "") or "").strip()
    if deployment_id:
        return client.cloud_deployments.get(deployment_id=deployment_id)
    cloud_slot = str(getattr(args, "cloud_slot", "") or "").strip()
    name = str(getattr(args, "name", "") or "").strip()
    if not cloud_slot and not name:
        raise CloudSlotError("pass --deployment-id, --cloud-slot, or --name to address a slot")
    project_id = _first_env("PROJECT_ID") if getattr(args, "project_id", None) is None else args.project_id
    matches = [
        row
        for row in client.cloud_deployments.list(project_id=project_id)
        if str(row.get("state") or "") != "retired"
        and (
            (cloud_slot and str(row.get("cloud_slot") or "") == cloud_slot)
            or (not cloud_slot and str(row.get("name") or "") == name)
        )
    ]
    if not matches:
        selector = f"cloud slot {cloud_slot!r}" if cloud_slot else f"name {name!r}"
        raise CloudSlotError(f"no live deployment for {selector}")
    if len(matches) > 1:
        ids = ", ".join(str(row.get("deployment_id")) for row in matches)
        raise CloudSlotError(f"selector is ambiguous across deployments: {ids}; use --deployment-id")
    return client.cloud_deployments.get(deployment_id=str(matches[0]["deployment_id"]))


def _build_source(args: argparse.Namespace, *, name: str) -> dict | None:
    commit = str(getattr(args, "commit", "") or "").strip()
    if not commit:
        return None
    evidence = str(getattr(args, "evidence_sha", "") or "").strip() or commit
    instance_id = str(getattr(args, "instance_id", "") or "").strip() or name
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


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _emit(payload: object) -> None:
    print(json.dumps(_jsonable(payload), default=str, indent=2, sort_keys=True))


# ---- verbs ------------------------------------------------------------------

def cmd_up(args: argparse.Namespace) -> int:
    client = _client(args)
    project_id = _require(args.project_id or _first_env("PROJECT_ID"), "PROJECT_ID")
    cloud_slot = str(args.cloud_slot)
    name = str(args.name or cloud_slot)

    existing = [
        row
        for row in client.cloud_deployments.list(project_id=project_id)
        if str(row.get("cloud_slot") or "") == cloud_slot
        and str(row.get("state") or "") != "retired"
    ]
    if existing and not args.reuse:
        raise CloudSlotError(
            f"{cloud_slot} already has a live deployment ({existing[0].get('deployment_id')}); "
            "pass --reuse to adopt it or retire that deployment first"
        )
    if existing:
        deployment = client.cloud_deployments.get(deployment_id=str(existing[0]["deployment_id"]))
        action = "reused"
    else:
        source = _build_source(args, name=name)
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
            cloud_slot=cloud_slot,
        )
        action = "created"

    deployment_id = str(deployment.get("deployment_id") or "")
    if not deployment_id:
        raise CloudSlotError(f"{action} deployment has no deployment_id: {deployment}")

    result = {"action": action, "deployment": _summary(deployment)}
    exit_code = 0
    if args.wait:
        running = _wait_running(client, deployment_id, args)
        result["deployment"] = _summary(running)
        service_url = str(running.get("service_url") or "").strip()
        if service_url:
            result["health"] = _health_probe(service_url, args.health_path)
            if result["health"].get("ok") is not True:
                exit_code = 1
        else:
            result["health"] = {"ok": False, "error": "running deployment has no service_url"}
            exit_code = 1
    _emit(result)
    return exit_code


def cmd_run(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    deployment_id = str(deployment["deployment_id"])
    queued = client.cloud_deployments.deploy(
        deployment_id=deployment_id,
        reason=args.reason,
        fencing_token=args.fencing_token,
    )
    result = {"action": "redeploy", "deployment": _summary(queued)}
    exit_code = 0
    if args.wait:
        running = _wait_running(client, deployment_id, args)
        result["deployment"] = _summary(running)
        service_url = str(running.get("service_url") or "").strip()
        if service_url:
            result["health"] = _health_probe(service_url, args.health_path)
            if result["health"].get("ok") is not True:
                exit_code = 1
        else:
            result["health"] = {"ok": False, "error": "running deployment has no service_url"}
            exit_code = 1
    _emit(result)
    return exit_code


def cmd_exec(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise CloudSlotError("exec requires command argv after '--'")
    result = client.cloud_deployments.exec(
        deployment_id=str(deployment["deployment_id"]),
        argv=argv,
        fencing_token=args.fencing_token,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
        max_output_bytes=args.max_output_bytes,
    )
    _emit({"action": "exec", "deployment": _summary(deployment), "result": result})
    return int(result["exit_code"])


def cmd_services(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    services = client.cloud_deployments.services(
        deployment_id=str(deployment["deployment_id"]),
    )
    _emit({"deployment": _summary(deployment), "service_discovery": services})
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    workspace = client.cloud_deployments.workspace(
        deployment_id=str(deployment["deployment_id"]),
    )
    _emit({"deployment": _summary(deployment), "workspace": workspace})
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    receipt = client.cloud_deployments.materialize_workspace(
        deployment_id=str(deployment["deployment_id"]),
        repository=args.repository,
        branch=args.branch,
        source_commit_sha=args.source_commit_sha,
        fencing_token=args.fencing_token,
    )
    _emit({"action": "materialize", "deployment": _summary(deployment), "receipt": receipt})
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    logs = client.cloud_deployments.logs(
        deployment_id=str(deployment["deployment_id"]),
        service_id=args.service_id,
        tail=args.tail,
    )
    _emit({"deployment": _summary(deployment), "logs": logs})
    return int(logs["exit_code"])


def cmd_artifacts(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    artifacts = client.cloud_deployments.artifacts(
        deployment_id=str(deployment["deployment_id"]),
        root_id=args.root_id,
        relative_prefix=args.relative_prefix,
        after=args.after,
        limit=args.limit,
    )
    _emit({"deployment": _summary(deployment), "artifact_inventory": artifacts})
    return 0


def cmd_artifact(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    deployment_id = str(deployment["deployment_id"])
    target = Path(args.output).expanduser()
    if os.path.lexists(target) and not args.force:
        raise CloudSlotError(f"output already exists: {target}; pass --force to replace it")
    if not target.parent.is_dir():
        raise CloudSlotError(f"output parent does not exist: {target.parent}")

    temporary = target.with_name(f".{target.name}.cloud-slot-{os.getpid()}.tmp")
    if os.path.lexists(temporary):
        raise CloudSlotError(f"temporary output already exists: {temporary}")
    expected_size: int | None = None
    expected_sha256: str | None = None
    expected_modified_at_ns: int | None = None
    offset = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as output:
            while True:
                chunk = client.cloud_deployments.artifact_content(
                    deployment_id=deployment_id,
                    root_id=args.root_id,
                    relative_path=args.path,
                    offset=offset,
                    max_bytes=args.chunk_bytes,
                    include_sha256=offset == 0,
                )
                if chunk["offset"] != offset:
                    raise CloudSlotError(
                        f"artifact chunk offset changed: expected {offset}, got {chunk['offset']}"
                    )
                if expected_size is None:
                    expected_size = int(chunk["size_bytes"])
                    expected_sha256 = chunk["sha256"]
                    expected_modified_at_ns = int(chunk["modified_at_ns"])
                    if not expected_sha256:
                        raise CloudSlotError("artifact response omitted the requested SHA-256")
                elif (
                    int(chunk["size_bytes"]) != expected_size
                    or int(chunk["modified_at_ns"]) != expected_modified_at_ns
                ):
                    raise CloudSlotError("artifact identity changed during retrieval")
                if chunk["root_id"] != args.root_id or chunk["relative_path"] != args.path:
                    raise CloudSlotError("artifact response identity did not match the request")
                try:
                    content = base64.b64decode(chunk["content_base64"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise CloudSlotError("artifact response contained invalid base64") from exc
                if len(content) != int(chunk["bytes_returned"]):
                    raise CloudSlotError("artifact response byte count did not match its content")
                if not content and not chunk["eof"]:
                    raise CloudSlotError("artifact retrieval made no progress before EOF")
                output.write(content)
                digest.update(content)
                offset += len(content)
                if chunk["eof"]:
                    break
            output.flush()
            os.fsync(output.fileno())
        if expected_size is None or offset != expected_size:
            raise CloudSlotError(
                f"artifact size mismatch: expected {expected_size}, retrieved {offset}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise CloudSlotError("artifact SHA-256 mismatch")
        if args.force:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()

    _emit(
        {
            "action": "artifact_retrieved",
            "deployment": _summary(deployment),
            "root_id": args.root_id,
            "relative_path": args.path,
            "output": str(target),
            "size_bytes": offset,
            "sha256": actual_sha256,
        }
    )
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    deployment_id = str(deployment["deployment_id"])
    vm_name = str(deployment.get("vm_name") or "").strip()
    confirm = str(args.confirm_vm_name or "").strip()
    if args.delete_vm and (not vm_name or not confirm or confirm != vm_name):
        raise CloudSlotError(
            "--delete-vm destroys the VM: --confirm-vm-name must exactly match "
            f"the nonempty deployment vm_name {vm_name!r} (got {confirm!r})"
        )
    retired = client.cloud_deployments.retire(
        deployment_id=deployment_id,
        reason=args.reason,
        delete_vm=bool(args.delete_vm),
        confirm_vm_name=confirm if args.delete_vm else None,
        fencing_token=args.fencing_token,
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


def cmd_health(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    service_url = str(deployment.get("service_url") or "").strip()
    if not service_url:
        raise CloudSlotError("deployment has no service_url")
    health = _health_probe(service_url, args.health_path)
    _emit({"deployment": _summary(deployment), "health": health})
    return 0 if health.get("ok") is True else 1


def cmd_observe(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    observed = client.cloud_deployments.observe(
        deployment_id=str(deployment["deployment_id"]),
        fencing_token=args.fencing_token,
    )
    _emit({"action": "observe", "deployment": observed})
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    claim = client.cloud_deployments.acquire_claim(
        deployment_id=str(deployment["deployment_id"]),
        holder=args.holder,
        purpose=args.purpose,
        ttl_seconds=args.ttl_seconds,
    )
    _emit({"action": "claim", "deployment": _summary(deployment), "claim": claim})
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    heartbeat = client.cloud_deployments.heartbeat_claim(
        deployment_id=str(deployment["deployment_id"]),
        claim_id=args.claim_id,
    )
    _emit({"action": "heartbeat", "deployment": _summary(deployment), "heartbeat": heartbeat})
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    claim = client.cloud_deployments.release_claim(
        deployment_id=str(deployment["deployment_id"]),
        claim_id=args.claim_id,
    )
    _emit({"action": "release", "deployment": _summary(deployment), "claim": claim})
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    client = _client(args)
    deployment = _resolve_deployment(client, args)
    claims = client.cloud_deployments.get_claims(
        deployment_id=str(deployment["deployment_id"]),
    )
    _emit({"deployment": _summary(deployment), "claims": claims})
    return 0


# ---- parser -----------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloud-slot", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend-base", default=None, help="backend API base (default: $SYNTH_BACKEND_URL)")
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

    def add_address_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cloud-slot", choices=CLOUD_SLOT_IDENTITIES, default=None)
        p.add_argument("--name", default=None)
        p.add_argument("--deployment-id", default=None)
        p.add_argument("--project-id", default=None)

    p_up = sub.add_parser("up", help="create (or --reuse) a cloud slot and wait until it serves")
    p_up.add_argument("--cloud-slot", choices=CLOUD_SLOT_IDENTITIES, required=True)
    p_up.add_argument("--name", default=None, help="deployment name (default: --cloud-slot)")
    p_up.add_argument("--project-id", default=None)
    p_up.add_argument("--topology-id", default="synth-dev")
    p_up.add_argument("--topology-version", default=None)
    p_up.add_argument("--host-kind", default="exe_dev")
    p_up.add_argument("--commit", default=None, help="pin a remote-branch commit sha (project git source)")
    p_up.add_argument("--evidence-sha", default=None, help="evidence commit sha (default: --commit)")
    p_up.add_argument("--instance-id", default=None, help="stable source instance id (default: --name)")
    p_up.add_argument("--reuse", action="store_true", help="adopt the existing live deployment for this slot")
    add_wait_flags(p_up)
    p_up.set_defaults(func=cmd_up)

    p_run = sub.add_parser("run", help="redeploy a slot (push the branch's code) and reprove health")
    add_address_flags(p_run)
    p_run.add_argument("--reason", default="cloud-slot run")
    p_run.add_argument("--fencing-token", type=int, default=None)
    add_wait_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_exec = sub.add_parser(
        "exec",
        help="execute command argv in the declared workspace (requires active claim fencing)",
    )
    add_address_flags(p_exec)
    p_exec.add_argument("--fencing-token", type=int, required=True)
    p_exec.add_argument("--cwd", default=None, help="path relative to the declared workspace root")
    p_exec.add_argument("--timeout-seconds", type=int, default=300)
    p_exec.add_argument("--max-output-bytes", type=int, default=65_536)
    p_exec.add_argument("argv", nargs=argparse.REMAINDER, help="command argv after '--'")
    p_exec.set_defaults(func=cmd_exec)

    p_services = sub.add_parser(
        "services",
        help="discover topology-declared services and endpoints",
    )
    add_address_flags(p_services)
    p_services.set_defaults(func=cmd_services)

    p_workspace = sub.add_parser("workspace", help="show declared and live workspace source proof")
    add_address_flags(p_workspace)
    p_workspace.set_defaults(func=cmd_workspace)

    p_materialize = sub.add_parser(
        "materialize",
        help="materialize an exact declared repository source (requires active claim fencing)",
    )
    add_address_flags(p_materialize)
    p_materialize.add_argument("--repository", required=True)
    p_materialize.add_argument("--branch", required=True)
    p_materialize.add_argument("--source-commit-sha", required=True)
    p_materialize.add_argument("--fencing-token", type=int, required=True)
    p_materialize.set_defaults(func=cmd_materialize)

    p_logs = sub.add_parser("logs", help="read bounded logs for a declared service")
    add_address_flags(p_logs)
    p_logs.add_argument("--service-id", required=True)
    p_logs.add_argument("--tail", type=int, default=200)
    p_logs.set_defaults(func=cmd_logs)

    p_artifacts = sub.add_parser(
        "artifacts",
        help="list declared artifact roots or one root's paginated files",
    )
    add_address_flags(p_artifacts)
    p_artifacts.add_argument("--root-id", default=None)
    p_artifacts.add_argument("--relative-prefix", default=None)
    p_artifacts.add_argument("--after", default=None, help="next_after cursor; requires --root-id")
    p_artifacts.add_argument("--limit", type=int, default=100)
    p_artifacts.set_defaults(func=cmd_artifacts)

    p_artifact = sub.add_parser(
        "artifact",
        help="retrieve a complete declared artifact through verified bounded chunks",
    )
    add_address_flags(p_artifact)
    p_artifact.add_argument("--root-id", required=True)
    p_artifact.add_argument("--path", required=True, help="file path relative to the declared root")
    p_artifact.add_argument("--output", required=True, help="local destination file")
    p_artifact.add_argument("--chunk-bytes", type=int, default=131_072)
    p_artifact.add_argument("--force", action="store_true", help="replace an existing destination")
    p_artifact.set_defaults(func=cmd_artifact)

    for command in ("down", "retire"):
        p_down = sub.add_parser(command, help="retire a slot (keeps the VM unless --delete-vm)")
        add_address_flags(p_down)
        p_down.add_argument("--reason", default=f"cloud-slot {command}")
        p_down.add_argument("--delete-vm", action="store_true", help="destroy the VM (requires --confirm-vm-name)")
        p_down.add_argument("--confirm-vm-name", default=None, help="exact vm_name echo, required with --delete-vm")
        p_down.add_argument("--fencing-token", type=int, default=None)
        p_down.set_defaults(func=cmd_down)

    p_ls = sub.add_parser("ls", help="list cloud slots (what the sltop CLOUD scope renders)")
    p_ls.add_argument("--project-id", default=None)
    p_ls.add_argument("--limit", type=int, default=100)
    p_ls.set_defaults(func=cmd_ls)

    p_status = sub.add_parser("status", help="full JSON for one slot")
    add_address_flags(p_status)
    p_status.set_defaults(func=cmd_status)

    p_health = sub.add_parser("health", help="probe the declared service URL for one slot")
    add_address_flags(p_health)
    p_health.add_argument("--health-path", default="/health")
    p_health.set_defaults(func=cmd_health)

    p_observe = sub.add_parser("observe", help="refresh substrate health and lifecycle state")
    add_address_flags(p_observe)
    p_observe.add_argument(
        "--fencing-token",
        type=int,
        default=None,
        help="active claim fencing token (required when a claim is active)",
    )
    p_observe.set_defaults(func=cmd_observe)

    p_claim = sub.add_parser("claim", help="acquire a TTL claim and fencing token")
    add_address_flags(p_claim)
    p_claim.add_argument("--holder", required=True)
    p_claim.add_argument("--purpose", required=True)
    p_claim.add_argument(
        "--ttl-seconds",
        type=int,
        default=900,
        help="claim lifetime (default: 900; covers default exec and materialization budgets)",
    )
    p_claim.set_defaults(func=cmd_claim)

    p_heartbeat = sub.add_parser("heartbeat", help="renew a cloud-slot claim")
    add_address_flags(p_heartbeat)
    p_heartbeat.add_argument("--claim-id", required=True)
    p_heartbeat.set_defaults(func=cmd_heartbeat)

    p_release = sub.add_parser("release", help="release a cloud-slot claim idempotently")
    add_address_flags(p_release)
    p_release.add_argument("--claim-id", required=True)
    p_release.set_defaults(func=cmd_release)

    p_claims = sub.add_parser("claims", help="show active claim and fencing truth")
    add_address_flags(p_claims)
    p_claims.set_defaults(func=cmd_claims)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (CloudSlotError, ValueError) as exc:
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
