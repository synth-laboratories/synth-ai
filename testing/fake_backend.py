from __future__ import annotations

import base64
import itertools
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from nacl.public import PrivateKey

TEST_API_KEY = "test-key"


def _build_state(public_key_b64: str) -> dict[str, Any]:
    return {
        "counters": {
            "container": itertools.count(1),
            "tunnel": itertools.count(1),
            "managed_lease": itertools.count(1),
            "synth_lease": itertools.count(1),
            "pool": itertools.count(1),
            "task": itertools.count(1),
            "pool_rollout": itertools.count(1),
            "global_rollout": itertools.count(1),
        },
        "containers": {},
        "managed_tunnels": {},
        "managed_leases": {},
        "synth_leases": {},
        "pools": {},
        "tasks": {},
        "pool_rollouts": {},
        "global_rollouts": {},
        "env_key_uploads": [],
        "credentials": [],
        "backend_public_key_b64": public_key_b64,
    }


def build_fake_backend() -> FastAPI:
    app = FastAPI()
    private_key = PrivateKey.generate()
    public_key_b64 = base64.b64encode(bytes(private_key.public_key)).decode("utf-8")
    app.state.data = _build_state(public_key_b64)

    @app.middleware("http")
    async def require_auth(request: Request, call_next):  # type: ignore[override]
        if request.url.path.startswith("/__test__/"):
            return await call_next(request)
        if request.headers.get("authorization") != f"Bearer {TEST_API_KEY}":
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    @app.get("/__test__/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/__test__/state")
    async def test_state() -> dict[str, Any]:
        state = app.state.data
        return {
            "env_key_upload_count": len(state["env_key_uploads"]),
            "env_key_uploads": list(state["env_key_uploads"]),
            "credential_count": len(state["credentials"]),
        }

    @app.post("/__test__/reset")
    async def reset_state() -> dict[str, Any]:
        app.state.data = _build_state(app.state.data["backend_public_key_b64"])
        return {"ok": True}

    @app.post("/__test__/seed-env-key")
    async def seed_env_key(payload: dict[str, Any]) -> dict[str, Any]:
        plaintext = str(payload.get("plaintext") or "").strip()
        if not plaintext:
            raise HTTPException(status_code=400, detail="plaintext is required")
        record = {
            "name": "ENVIRONMENT_API_KEY",
            "plaintext": plaintext,
            "created_at": _now_iso(),
        }
        app.state.data["credentials"].append(record)
        return {"ok": True, "credential_count": len(app.state.data["credentials"])}

    @app.get("/api/v1/env-keys/verify")
    async def verify_env_keys() -> dict[str, Any]:
        if not app.state.data["env_key_uploads"]:
            raise HTTPException(status_code=404, detail="ENVIRONMENT_API_KEY not configured")
        return {"ok": True}

    @app.get("/api/v1/crypto/public-key")
    async def crypto_public_key() -> dict[str, Any]:
        return {"public_key": app.state.data["backend_public_key_b64"]}

    @app.post("/api/v1/env-keys")
    async def upload_env_key(payload: dict[str, Any]) -> dict[str, Any]:
        ciphertext_b64 = str(payload.get("ciphertext_b64") or "").strip()
        if not ciphertext_b64:
            raise HTTPException(status_code=400, detail="ciphertext_b64 is required")
        record = {
            "name": str(payload.get("name") or ""),
            "ciphertext_b64": ciphertext_b64,
            "created_at": _now_iso(),
        }
        app.state.data["env_key_uploads"].append(record)
        return {"ok": True, "credential_id": f"cred-{len(app.state.data['env_key_uploads'])}"}

    @app.get("/v1/credentials")
    @app.get("/api/v1/credentials")
    async def list_credentials() -> dict[str, Any]:
        return {"credentials": list(app.state.data["credentials"])}

    @app.post("/v1/containers")
    async def create_container(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        container_id = f"container-{next(state['counters']['container'])}"
        record = {
            "id": container_id,
            "name": payload["name"],
            "task_type": payload["task_type"],
            "status": "ready",
            "definition": payload.get("definition", {}),
            "environment_config": payload.get("environment_config"),
            "internal_url": payload.get("internal_url") or f"http://{container_id}.internal",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        state["containers"][container_id] = record
        return record

    @app.get("/v1/containers")
    async def list_containers() -> list[dict[str, Any]]:
        return list(app.state.data["containers"].values())

    @app.get("/v1/containers/{container_id}")
    async def get_container(container_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["containers"], container_id, "container")
        return record

    @app.delete("/v1/containers/{container_id}", status_code=204)
    async def delete_container(container_id: str) -> Response:
        _lookup(app.state.data["containers"], container_id, "container")
        del app.state.data["containers"][container_id]
        return Response(status_code=204)

    @app.get("/v1/tunnels/health")
    async def tunnels_health() -> dict[str, Any]:
        return {"status": "ok", "providers": ["ngrok", "cloudflared", "synthtunnel"]}

    @app.get("/v1/tunnels/")
    async def list_tunnels(
        status_filter: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        tunnels = list(app.state.data["managed_tunnels"].values())
        if not include_deleted:
            tunnels = [item for item in tunnels if item["status"] != "deleted"]
        if status_filter:
            tunnels = [item for item in tunnels if item["status"] == status_filter]
        return tunnels

    @app.post("/v1/tunnels/")
    async def create_tunnel(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        tunnel_id = f"tunnel-{next(state['counters']['tunnel'])}"
        record = {
            "id": tunnel_id,
            "provider": "ngrok",
            "status": "active",
            "subdomain": payload["subdomain"],
            "local_host": payload.get("local_host", "127.0.0.1"),
            "local_port": payload["local_port"],
            "public_url": f"https://{payload['subdomain']}.ngrok.app",
        }
        state["managed_tunnels"][tunnel_id] = record
        return record

    @app.delete("/v1/tunnels/{tunnel_id}")
    async def delete_tunnel(tunnel_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["managed_tunnels"], tunnel_id, "tunnel")
        record["status"] = "deleted"
        return record

    @app.post("/v1/tunnels/rotate")
    async def rotate_tunnel(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "rotated",
            "local_host": payload.get("local_host", "127.0.0.1"),
            "local_port": payload.get("local_port", 8000),
            "reason": payload.get("reason"),
            "public_url": "https://rotated.ngrok.app",
        }

    @app.post("/v1/tunnels/lease")
    async def create_managed_lease(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        lease_id = f"lease-{next(state['counters']['managed_lease'])}"
        record = {
            "lease_id": lease_id,
            "status": "active",
            "client_instance_id": payload["client_instance_id"],
            "local_host": payload["local_host"],
            "local_port": payload["local_port"],
            "provider_preference": payload["provider_preference"],
            "requested_ttl_seconds": payload["requested_ttl_seconds"],
            "reuse_connector": payload["reuse_connector"],
            "app_name": payload.get("app_name"),
            "idempotency_key": payload.get("idempotency_key"),
            "public_url": f"https://{lease_id}.ngrok.app",
        }
        state["managed_leases"][lease_id] = record
        return record

    @app.get("/v1/tunnels/lease")
    async def list_managed_leases(
        client_instance_id: str | None = None,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        leases = list(app.state.data["managed_leases"].values())
        if client_instance_id:
            leases = [item for item in leases if item["client_instance_id"] == client_instance_id]
        if not include_expired:
            leases = [item for item in leases if item["status"] != "deleted"]
        return leases

    @app.post("/v1/tunnels/lease/{lease_id}/heartbeat")
    async def heartbeat_lease(lease_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["managed_leases"], lease_id, "managed lease")
        record["status"] = "heartbeat_ok"
        record["heartbeat"] = payload
        return record

    @app.post("/v1/tunnels/lease/{lease_id}/release")
    async def release_lease(lease_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["managed_leases"], lease_id, "managed lease")
        record["status"] = "released"
        return record

    @app.post("/v1/tunnels/lease/{lease_id}/refresh")
    async def refresh_lease(lease_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["managed_leases"], lease_id, "managed lease")
        record["status"] = "refreshed"
        record["requested_ttl_seconds"] = payload["requested_ttl_seconds"]
        return record

    @app.delete("/v1/tunnels/lease/{lease_id}")
    async def delete_lease(lease_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["managed_leases"], lease_id, "managed lease")
        record["status"] = "deleted"
        return record

    @app.post("/api/v1/synthtunnel/leases", status_code=201)
    async def create_synth_lease(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        lease_id = f"synth-lease-{next(state['counters']['synth_lease'])}"
        route_token = f"rt-{lease_id}"
        record = {
            "lease_id": lease_id,
            "status": "PENDING",
            "route_token": route_token,
            "client_instance_id": payload["client_instance_id"],
            "public_base_url": "https://st.usesynth.ai",
            "public_url": f"https://st.usesynth.ai/s/{route_token}",
            "agent_connect": {
                "transport": "ws",
                "url": "wss://st.usesynth.ai/agent",
                "agent_token": f"agent-{lease_id}",
            },
            "worker_token": f"worker-{lease_id}",
            "expires_at": _future_iso(hours=1),
            "limits": {"max_inflight": 128},
            "heartbeat": {"required": False, "recommended_interval_seconds": 25},
            "local_target": payload["local_target"],
            "metadata": payload.get("metadata", {}),
            "capabilities": payload.get("capabilities", {}),
        }
        state["synth_leases"][lease_id] = record
        return record

    @app.get("/api/v1/synthtunnel/leases/{lease_id}")
    async def get_synth_lease(lease_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["synth_leases"], lease_id, "synth lease")
        return {
            "lease_id": lease_id,
            "status": record["status"],
            "last_seen_at": _now_iso(),
            "connected": False,
            "inflight": 0,
            "expires_at": record["expires_at"],
        }

    @app.delete("/api/v1/synthtunnel/leases/{lease_id}")
    async def delete_synth_lease(lease_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["synth_leases"], lease_id, "synth lease")
        record["status"] = "EXPIRED"
        return {
            "lease_id": lease_id,
            "status": record["status"],
            "last_seen_at": _now_iso(),
            "connected": False,
            "inflight": 0,
            "expires_at": _now_iso(),
        }

    @app.post("/api/v1/synthtunnel/leases/{lease_id}/token:refresh")
    async def refresh_synth_token(lease_id: str) -> dict[str, Any]:
        _lookup(app.state.data["synth_leases"], lease_id, "synth lease")
        return {"lease_id": lease_id, "worker_token": f"worker-{lease_id}-refreshed"}

    @app.post("/v1/pools")
    async def create_pool(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        pool_id = str(
            payload.get("pool_id") or payload.get("id") or f"pool-{next(state['counters']['pool'])}"
        )
        if pool_id in state["pools"]:
            raise HTTPException(status_code=409, detail="pool already exists")
        record = {
            "id": pool_id,
            "name": payload.get("name", pool_id),
            "target": payload.get("target", "harbor"),
            "status": "ready",
            "config": payload,
        }
        state["pools"][pool_id] = record
        state["tasks"][pool_id] = {}
        state["pool_rollouts"][pool_id] = {}
        return record

    @app.get("/v1/pools")
    async def list_pools(
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        items = list(app.state.data["pools"].values())
        if state:
            items = [item for item in items if item["status"] == state]
        return {"items": items[:limit], "cursor": cursor}

    @app.get("/v1/pools/{pool_id}")
    async def get_pool(pool_id: str) -> dict[str, Any]:
        return _lookup(app.state.data["pools"], pool_id, "pool")

    @app.put("/v1/pools/{pool_id}")
    async def replace_pool(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["pools"], pool_id, "pool")
        record.update(payload)
        return record

    @app.patch("/v1/pools/{pool_id}")
    async def update_pool(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["pools"], pool_id, "pool")
        record.update(payload)
        return record

    @app.delete("/v1/pools/{pool_id}")
    async def delete_pool(pool_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["pools"], pool_id, "pool")
        record["status"] = "deleted"
        return record

    @app.get("/v1/pools/{pool_id}/urls")
    async def get_pool_urls(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"public_url": f"https://{pool_id}.harbor.synth.ai"}

    @app.get("/v1/pools/{pool_id}/metrics")
    async def get_pool_metrics(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "active_rollouts": 1, "success_rate": 1.0}

    @app.get("/v1/pools/{pool_id}/tasks")
    async def list_tasks(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"items": list(app.state.data["tasks"][pool_id].values())}

    @app.post("/v1/pools/{pool_id}/tasks")
    async def create_task(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        _lookup(state["pools"], pool_id, "pool")
        task_id = payload.get("id") or f"task-{next(state['counters']['task'])}"
        record = {"id": task_id, "pool_id": pool_id, **payload}
        state["tasks"][pool_id][task_id] = record
        return record

    @app.put("/v1/pools/{pool_id}/tasks/{task_id}")
    async def update_task(pool_id: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        record.clear()
        record.update({"id": task_id, "pool_id": pool_id, **payload})
        return record

    @app.patch("/v1/pools/{pool_id}/tasks/{task_id}")
    async def patch_task(pool_id: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        record.update(payload)
        return record

    @app.delete("/v1/pools/{pool_id}/tasks/{task_id}")
    async def delete_task(pool_id: str, task_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        record["status"] = "deleted"
        return record

    @app.get("/v1/pools/{pool_id}/container/health")
    async def pool_container_health(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "status": "healthy"}

    @app.get("/v1/pools/{pool_id}/container/info")
    async def pool_container_info(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "kind": "harbor_code"}

    @app.get("/v1/pools/{pool_id}/container/metadata")
    async def pool_container_metadata(pool_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "metadata": {"owner": "testing"}}

    @app.post("/v1/pools/{pool_id}/container/rollout")
    async def execute_pool_rollout(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "accepted": True, "request": payload}

    @app.post("/v1/pools/{pool_id}/container/prompt-learning/evaluate")
    async def evaluate_pool(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        return {"pool_id": pool_id, "status": "accepted", "request": payload}

    @app.get("/v1/pools/{pool_id}/tasks/{task_id}/container/health")
    async def task_container_health(pool_id: str, task_id: str) -> dict[str, Any]:
        _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        return {"pool_id": pool_id, "task_id": task_id, "status": "healthy"}

    @app.get("/v1/pools/{pool_id}/tasks/{task_id}/container/info")
    async def task_container_info(pool_id: str, task_id: str) -> dict[str, Any]:
        _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        return {"pool_id": pool_id, "task_id": task_id, "kind": "harbor_code"}

    @app.get("/v1/pools/{pool_id}/tasks/{task_id}/container/metadata")
    async def task_container_metadata(pool_id: str, task_id: str) -> dict[str, Any]:
        _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        return {"pool_id": pool_id, "task_id": task_id, "metadata": {"owner": "testing"}}

    @app.post("/v1/pools/{pool_id}/tasks/{task_id}/container/rollout")
    async def execute_task_rollout(
        pool_id: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        return {"pool_id": pool_id, "task_id": task_id, "accepted": True, "request": payload}

    @app.post("/v1/pools/{pool_id}/tasks/{task_id}/container/prompt-learning/evaluate")
    async def evaluate_task(
        pool_id: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        _lookup(app.state.data["tasks"][pool_id], task_id, "task")
        return {"pool_id": pool_id, "task_id": task_id, "status": "accepted", "request": payload}

    @app.post("/v1/pools/{pool_id}/rollouts")
    async def create_pool_rollout(pool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        _lookup(state["pools"], pool_id, "pool")
        rollout_id = f"pool-rollout-{next(state['counters']['pool_rollout'])}"
        record = {
            "id": rollout_id,
            "pool_id": pool_id,
            "state": "running",
            "status": "running",
            "request": payload,
            "_get_count": 0,
        }
        state["pool_rollouts"][pool_id][rollout_id] = record
        return record

    @app.get("/v1/pools/{pool_id}/rollouts")
    async def list_pool_rollouts(
        pool_id: str,
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        _lookup(app.state.data["pools"], pool_id, "pool")
        items = list(app.state.data["pool_rollouts"][pool_id].values())
        if state:
            items = [item for item in items if item["state"] == state]
        return {"items": items[:limit], "cursor": cursor}

    @app.get("/v1/pools/{pool_id}/rollouts/{rollout_id}")
    async def get_pool_rollout(pool_id: str, rollout_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        record["_get_count"] = int(record.get("_get_count", 0)) + 1
        if record["status"] == "running":
            record["status"] = "completed"
            record["state"] = "completed"
        return record

    @app.post("/v1/pools/{pool_id}/rollouts/{rollout_id}/cancel")
    async def cancel_pool_rollout(pool_id: str, rollout_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        record["state"] = "cancelled"
        record["status"] = "cancelled"
        return record

    @app.get("/v1/pools/{pool_id}/rollouts/{rollout_id}/artifacts")
    async def pool_rollout_artifacts(pool_id: str, rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        return {"items": [{"artifact_id": f"artifact-{rollout_id}", "kind": "report"}]}

    @app.get("/v1/pools/{pool_id}/rollouts/{rollout_id}/usage")
    async def pool_rollout_usage(pool_id: str, rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        return {"tokens": 42, "compute_seconds": 3}

    @app.get("/v1/pools/{pool_id}/rollouts/{rollout_id}/summary")
    async def pool_rollout_summary(pool_id: str, rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        return {"status": "completed", "score": 1.0}

    @app.get("/v1/pools/{pool_id}/rollouts/{rollout_id}/events")
    async def pool_rollout_events(pool_id: str, rollout_id: str) -> StreamingResponse:
        _lookup(app.state.data["pool_rollouts"][pool_id], rollout_id, "pool rollout")
        return StreamingResponse(
            _event_stream(
                [
                    {"event": "rollout.started", "rollout_id": rollout_id},
                    {"event": "rollout.completed", "rollout_id": rollout_id},
                ]
            ),
            media_type="text/event-stream",
        )

    @app.post("/v1/rollouts")
    async def create_global_rollout(payload: dict[str, Any]) -> dict[str, Any]:
        state = app.state.data
        rollout_id = f"rollout-{next(state['counters']['global_rollout'])}"
        record = {"id": rollout_id, "state": "running", "request": payload}
        state["global_rollouts"][rollout_id] = record
        return record

    @app.get("/v1/rollouts")
    async def list_global_rollouts(
        state: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        items = list(app.state.data["global_rollouts"].values())
        if state:
            items = [item for item in items if item["state"] == state]
        return {"items": items[:limit], "cursor": cursor}

    @app.get("/v1/rollouts/{rollout_id}")
    async def get_global_rollout(rollout_id: str) -> dict[str, Any]:
        return _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")

    @app.post("/v1/rollouts/{rollout_id}/cancel")
    async def cancel_global_rollout(rollout_id: str) -> dict[str, Any]:
        record = _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")
        record["state"] = "cancelled"
        return record

    @app.get("/v1/rollouts/{rollout_id}/artifacts")
    async def global_rollout_artifacts(rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")
        return {"items": [{"artifact_id": f"artifact-{rollout_id}", "kind": "report"}]}

    @app.get("/v1/rollouts/{rollout_id}/usage")
    async def global_rollout_usage(rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")
        return {"tokens": 24, "compute_seconds": 2}

    @app.get("/v1/rollouts/{rollout_id}/summary")
    async def global_rollout_summary(rollout_id: str) -> dict[str, Any]:
        _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")
        return {"status": "completed", "score": 1.0}

    @app.get("/v1/rollouts/{rollout_id}/events")
    async def global_rollout_events(rollout_id: str) -> StreamingResponse:
        _lookup(app.state.data["global_rollouts"], rollout_id, "global rollout")
        return StreamingResponse(
            _event_stream(
                [
                    {"event": "rollout.started", "rollout_id": rollout_id},
                    {"event": "rollout.completed", "rollout_id": rollout_id},
                ]
            ),
            media_type="text/event-stream",
        )

    @app.get("/v1/queue/status")
    async def queue_status() -> dict[str, Any]:
        return {"queued": 0, "running": 1}

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {"targets": ["harbor", "openenv"], "tunnels": ["ngrok", "synthtunnel"]}

    return app


def _lookup(items: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    try:
        return items[key]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"{label} not found") from exc


def _event_stream(events: list[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _future_iso(*, hours: int) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
