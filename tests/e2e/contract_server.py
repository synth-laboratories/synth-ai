"""Deterministic local HTTP + SSE + Streamable-HTTP /mcp contract server for 4-surface E2E tests."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import socket
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
import yaml
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route

from synth_ai.aclient import AsyncSynthClient
from synth_ai.mcp.server import build_server

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

ROOT = Path(__file__).resolve().parents[2]
VALID_API_KEY = "sk_e2e_local_valid_key"
VALID_MCP_BEARER = "mcp_e2e_oauth_bearer_token"
# Reserved run IDs triggering deterministic refusal / lifecycle behaviors.
RUN_ID_NOT_FOUND = "run_ZZZZZZZZZZZZZZZZZZZZZZ"
RUN_ID_FAILED = "run_FFFFFFFFFFFFFFFFFFFFFF"
RUN_ID_PAUSED = "run_PPPPPPPPPPPPPPPPPPPPPP"
RUN_ID_TIMEOUT = "run_TTTTTTTTTTTTTTTTTTTTTT"
EVT_ID_1 = "evt_1111111111111111111111"
EVT_ID_2 = "evt_2222222222222222222222"


def _make_id(prefix: str) -> str:
    tail = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")
    return f"{prefix}_{tail}"


def _load_examples() -> dict[str, dict[str, object]]:
    contract = yaml.safe_load((ROOT / "contracts/synth-v1.openapi.yaml").read_text())
    paths = contract["paths"]

    def example(path: str, method: str, code: str) -> dict[str, object]:
        app_json = paths[path][method]["responses"][code]["content"]["application/json"]
        if "example" in app_json:
            return copy.deepcopy(app_json["example"])
        first = next(iter(app_json["examples"].values()))
        return copy.deepcopy(first["value"])

    return {
        "upload": example("/v1/uploads", "post", "201"),
        "file": example("/v1/files", "post", "201"),
        "run": example("/v1/runs", "post", "202"),
        "limits": example("/v1/runs/{run_id}/limits", "get", "200"),
        "pause": example("/v1/runs/{run_id}/pause", "post", "202"),
        "resume": example("/v1/runs/{run_id}/resume", "post", "202"),
        "message": example("/v1/runs/{run_id}/messages", "post", "202"),
        "report": example("/v1/runs/{run_id}/report", "get", "200"),
        "result": example("/v1/runs/{run_id}/result", "get", "200"),
    }


def _problem(
    code: str,
    status: int,
    *,
    detail: str | None = None,
    retryable: bool | None = None,
    stage: str = "contract_e2e",
) -> dict[str, object]:
    body: dict[str, object] = {
        "type": f"https://docs.usesynth.ai/errors/{code}",
        "title": code,
        "status": status,
        "detail": detail or f"Deterministic refusal: {code}",
        "instance": f"urn:synth-e2e:{uuid.uuid4().hex[:12]}",
        "code": code,
        "service": "synth-e2e-mock",
        "stage": stage,
        "outcome": "not_accepted",
    }
    if retryable is not None:
        body["retryable"] = retryable
    return body


@dataclass
class _UploadRecord:
    upload_id: str
    file_id: str
    revision_id: str
    name: str
    media_type: str
    size_bytes: int
    digest_sha256: str
    content: bytes | None = None


@dataclass
class _OutputRecord:
    output_id: str
    revision_id: str
    filename: str
    media_type: str
    content: bytes
    digest_sha256: str


@dataclass
class _RunRecord:
    run_id: str
    message: str
    topology: str
    version: int
    state: str = "running"
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    limits_revision: int = 1
    budget_amount: str = "2.50"
    wall_time_seconds: int = 600
    maximum_tokens: int = 100_000
    pause_retention_seconds: int = 3600
    sse_connections: int = 0
    sse_last_event_ids: list[str | None] = field(default_factory=list)
    messages: list[dict[str, object]] = field(default_factory=list)
    outputs: list[_OutputRecord] = field(default_factory=list)
    control_history: list[str] = field(default_factory=list)


class ContractState:
    """In-memory deterministic state backing the local HTTP + SSE + /mcp E2E server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.examples = _load_examples()
        self.uploads: dict[str, _UploadRecord] = {}
        self.files: dict[str, _UploadRecord] = {}
        self.runs: dict[str, _RunRecord] = {}
        self.mcp_auth_checks: list[tuple[str | None, int]] = []
        self._seed_special_runs()

    def _seed_special_runs(self) -> None:
        failed = _RunRecord(
            run_id=RUN_ID_FAILED,
            message="deterministic failed run",
            topology="swarms_v2",
            version=1,
            state="failed",
            execution_closed=True,
            accounting_final=True,
            cleanup_complete=True,
        )
        paused = _RunRecord(
            run_id=RUN_ID_PAUSED,
            message="deterministic paused run",
            topology="swarms_v2",
            version=1,
            state="paused",
            execution_closed=False,
            accounting_final=False,
            cleanup_complete=False,
        )
        timeout_run = _RunRecord(
            run_id=RUN_ID_TIMEOUT,
            message="deterministic slow run for timeout",
            topology="swarms_v2",
            version=1,
            state="running",
            execution_closed=False,
            accounting_final=False,
            cleanup_complete=False,
        )
        for item in (failed, paused, timeout_run):
            self._attach_outputs(item)
            self.runs[item.run_id] = item

    def _attach_outputs(self, run: _RunRecord) -> None:
        result_md = f"# Result for {run.run_id}\n\nTopology: {run.topology}@{run.version}\nGoal: {run.message}\n".encode()
        notes_txt = f"Supplemental artifact for {run.run_id}\n".encode()
        run.outputs = [
            _OutputRecord(
                output_id=_make_id("out"),
                revision_id=_make_id("rev"),
                filename="result.md",
                media_type="text/markdown",
                content=result_md,
                digest_sha256=hashlib.sha256(result_md).hexdigest(),
            ),
            _OutputRecord(
                output_id=_make_id("out"),
                revision_id=_make_id("rev"),
                filename="notes.txt",
                media_type="text/plain",
                content=notes_txt,
                digest_sha256=hashlib.sha256(notes_txt).hexdigest(),
            ),
        ]

    def snapshot_json(self, run: _RunRecord) -> dict[str, object]:
        snap = copy.deepcopy(self.examples["run"])
        snap["run_id"] = run.run_id
        snap["state"] = run.state
        snap["result_available"] = run.execution_closed
        snap["execution_closed"] = run.execution_closed
        snap["accounting_final"] = run.accounting_final
        snap["cleanup_complete"] = run.cleanup_complete
        snap["limits_revision"] = run.limits_revision
        snap["optimizer"] = {
            "topology": run.topology,
            "version": run.version,
            "params": {},
            "reason": "explicit_pin",
            "definition_digest": "0" * 64,
        }
        snap["charge"] = {
            "amount": {"amount": "0.12", "currency": "USD"},
            "final": run.accounting_final,
        }
        if run.state == "paused":
            snap["pause"] = {
                "cause": "customer",
                "operation_id": "ctl_e2e_pause",
                "limit": None,
                "requested_at_ms": 1794300000000,
                "drained_at_ms": 1794300001000,
                "mode": "warm",
            }
        return snap

    def limits_json(self, run: _RunRecord) -> dict[str, object]:
        doc = copy.deepcopy(self.examples["limits"])
        doc["run_id"] = run.run_id
        doc["revision"] = run.limits_revision
        doc["status"] = run.state
        doc["funding"] = {"held": run.budget_amount, "settled": "0.00", "currency": "USD"}
        doc["primary"] = [
            {
                "dimension": "spend_micros",
                "unit": "micros",
                "effective_cap": int(float(run.budget_amount) * 1_000_000),
                "used": 100_000,
                "remaining": max(0, int(float(run.budget_amount) * 1_000_000) - 100_000),
            },
            {
                "dimension": "wall_time_seconds",
                "unit": "seconds",
                "effective_cap": run.wall_time_seconds,
                "used": 10,
                "remaining": max(0, run.wall_time_seconds - 10),
            },
        ]
        return doc


def _build_app(state: ContractState) -> Starlette:
    def check_auth(request: Request) -> JSONResponse | None:
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        if token != VALID_API_KEY:
            return JSONResponse(_problem("api_key_invalid", 401), status_code=401)
        return None

    async def create_upload(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        body = await request.json()
        upload_id = _make_id("upl")
        file_id = _make_id("fil")
        revision_id = _make_id("rev")
        rec = _UploadRecord(
            upload_id=upload_id,
            file_id=file_id,
            revision_id=revision_id,
            name=str(body["name"]),
            media_type=str(body["media_type"]),
            size_bytes=int(body["size_bytes"]),
            digest_sha256=str(body["digest_sha256"]),
        )
        state.uploads[upload_id] = rec
        payload = copy.deepcopy(state.examples["upload"])
        payload["upload_id"] = upload_id
        payload["state"] = "pending"
        payload["file"] = {
            "file_id": file_id,
            "revision_id": revision_id,
            "name": rec.name,
            "media_type": rec.media_type,
            "size_bytes": rec.size_bytes,
            "digest_sha256": rec.digest_sha256,
            "created_at": "2026-11-24T00:00:00Z",
        }
        payload["put"] = {
            "url": f"{state.base_url}/_blob/{upload_id}",
            "headers": {"Content-Type": rec.media_type},
            "expires_at": "2026-11-25T00:00:00Z",
        }
        return JSONResponse(payload, status_code=201)

    async def put_blob(request: Request) -> Response:
        upload_id = request.path_params["upload_id"]
        rec = state.uploads.get(upload_id)
        if rec is None:
            return JSONResponse(_problem("upload_not_found", 404), status_code=404)
        rec.content = await request.body()
        return Response(status_code=200)

    async def finalize_upload(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        upload_id = request.path_params["upload_id"]
        rec = state.uploads.get(upload_id)
        if rec is None:
            return JSONResponse(_problem("upload_not_found", 404), status_code=404)
        if rec.content is None:
            return JSONResponse(_problem("upload_bytes_missing", 409), status_code=409)
        state.files[rec.file_id] = rec
        payload = copy.deepcopy(state.examples["upload"])
        payload["upload_id"] = upload_id
        payload["state"] = "committed"
        payload["file"] = {
            "file_id": rec.file_id,
            "revision_id": rec.revision_id,
            "name": rec.name,
            "media_type": rec.media_type,
            "size_bytes": len(rec.content),
            "digest_sha256": hashlib.sha256(rec.content).hexdigest(),
            "created_at": "2026-11-24T00:00:00Z",
        }
        return JSONResponse(payload, status_code=200)

    async def create_file_inline(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        body = await request.json()
        content = str(body["content"]).encode("utf-8")
        file_id = _make_id("fil")
        revision_id = _make_id("rev")
        rec = _UploadRecord(
            upload_id=_make_id("upl"),
            file_id=file_id,
            revision_id=revision_id,
            name=str(body["name"]),
            media_type=str(body["media_type"]),
            size_bytes=len(content),
            digest_sha256=hashlib.sha256(content).hexdigest(),
            content=content,
        )
        state.files[file_id] = rec
        return JSONResponse(
            {
                "file_id": file_id,
                "revision_id": revision_id,
                "name": rec.name,
                "media_type": rec.media_type,
                "size_bytes": rec.size_bytes,
                "digest_sha256": rec.digest_sha256,
                "created_at": "2026-11-24T00:00:00Z",
            },
            status_code=201,
        )

    async def create_run(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        body = await request.json()
        message = str(body.get("message", ""))
        # Deterministic refusal triggers keyed on explicit marker prefixes.
        refusal_map: dict[str, tuple[int, str, dict[str, str]]] = {
            "REFUSE:402": (402, "spend_entitlement_exceeded", {}),
            "REFUSE:403": (403, "provider_preset_not_allowed", {}),
            "REFUSE:422": (422, "launch_budget_below_actor_floor", {}),
            "REFUSE:429": (429, "rate_limit_exceeded", {"Retry-After": "2"}),
            "REFUSE:503": (503, "overloaded", {}),
        }
        for prefix, (status, code, headers) in refusal_map.items():
            if message.startswith(prefix):
                return JSONResponse(
                    _problem(code, status, retryable=False),
                    status_code=status,
                    headers=headers,
                )
        optimizer = body.get("optimizer")
        topology = "swarms_v2"
        version = 1
        if isinstance(optimizer, dict):
            topology = str(optimizer.get("topology", "swarms_v2"))
            version = int(optimizer.get("version", 1))
        limits = body.get("limits", {})
        charge = limits.get("maximum_charge", {}) if isinstance(limits, dict) else {}
        run_id = _make_id("run")
        run = _RunRecord(
            run_id=run_id,
            message=message,
            topology=topology,
            version=version,
            budget_amount=str(charge.get("amount", "2.50")),
            wall_time_seconds=int(limits.get("wall_time_seconds") or 600),
        )
        state._attach_outputs(run)
        state.runs[run_id] = run
        return JSONResponse(state.snapshot_json(run), status_code=202)

    async def get_run(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        return JSONResponse(state.snapshot_json(run), status_code=200)

    async def get_limits(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        return JSONResponse(state.limits_json(run), status_code=200)

    async def update_limits(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        body = await request.json()
        expected = int(body["expected_revision"])
        if expected != run.limits_revision:
            prob = _problem("limits_revision_conflict", 409)
            prob["current_revision"] = run.limits_revision
            return JSONResponse(prob, status_code=409)
        changes = body.get("limits", {})
        if isinstance(changes, dict):
            if "maximum_charge" in changes:
                run.budget_amount = str(changes["maximum_charge"]["amount"])
            if "wall_time_seconds" in changes:
                run.wall_time_seconds = int(changes["wall_time_seconds"])
            if "maximum_tokens" in changes:
                run.maximum_tokens = int(changes["maximum_tokens"])
            if "pause_retention_seconds" in changes:
                run.pause_retention_seconds = int(changes["pause_retention_seconds"])
        run.limits_revision += 1
        return JSONResponse(state.limits_json(run), status_code=200)

    async def pause_run(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        run.state = "paused"
        run.control_history.append("pause")
        receipt = copy.deepcopy(state.examples["pause"])
        receipt["resource_id"] = run.run_id
        return JSONResponse(receipt, status_code=202)

    async def resume_run(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        run.state = "running"
        run.control_history.append("resume")
        receipt = copy.deepcopy(state.examples["resume"])
        receipt["resource_id"] = run.run_id
        return JSONResponse(receipt, status_code=202)

    async def cancel_run(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        run.state = "cancelled"
        run.execution_closed = True
        run.accounting_final = True
        run.cleanup_complete = True
        run.control_history.append("cancel")
        return JSONResponse(state.snapshot_json(run), status_code=202)

    async def send_message(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        body = await request.json()
        msg = copy.deepcopy(state.examples["message"])
        msg["message_id"] = _make_id("msg")
        msg["run_id"] = run.run_id
        msg["kind"] = str(body["kind"])
        msg["sequence"] = len(run.messages) + 1
        msg["body"] = {"kind": "available", "text": str(body["text"])}
        run.messages.append(msg)
        return JSONResponse(msg, status_code=202)

    async def list_messages(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        return JSONResponse(
            {
                "items": run.messages,
                "next_cursor": None,
                "has_more": False,
                "availability": {"kind": "available"},
            },
            status_code=200,
        )

    async def get_report(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        report = copy.deepcopy(state.examples["report"])
        report["run_id"] = run.run_id
        return JSONResponse(report, status_code=200)

    async def get_result(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        primary = run.outputs[0]
        result = copy.deepcopy(state.examples["result"])
        result["run_id"] = run.run_id
        result["status"] = "completed" if run.state != "failed" else "failed"
        result["state"] = "completed" if run.state != "failed" else "failed"
        result["run"] = state.snapshot_json(run)
        result["answer"] = primary.content.decode("utf-8")
        result["result_file"] = {
            "run_id": run.run_id,
            "output_id": primary.output_id,
            "state": "available",
            "filename": primary.filename,
            "reference": {
                "revision_id": primary.revision_id,
                "digest_sha256": primary.digest_sha256,
                "byte_size": str(len(primary.content)),
                "media_type": "text/markdown",
            },
        }
        return JSONResponse(result, status_code=200)

    async def list_outputs(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        cursor = request.query_params.get("cursor")
        # Force deterministic 2-page pagination across the 2 outputs.
        start = 0 if cursor is None else int(cursor)
        item = run.outputs[start]
        next_cursor = str(start + 1) if start + 1 < len(run.outputs) else None
        serialized = {
            "run_id": run.run_id,
            "output_id": item.output_id,
            "state": "available",
            "filename": item.filename,
            "reference": {
                "revision_id": item.revision_id,
                "digest_sha256": item.digest_sha256,
                "byte_size": str(len(item.content)),
                "media_type": item.media_type,
            },
        }
        payload: dict[str, object] = {
            "items": [serialized],
            "has_more": next_cursor is not None,
        }
        if next_cursor is not None:
            payload["next_cursor"] = next_cursor
        return JSONResponse(payload, status_code=200)

    async def get_output_content(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        output_id = request.path_params["output_id"]
        revision_id = request.query_params.get("revision_id")
        output = next((item for item in run.outputs if item.output_id == output_id), None)
        if output is None:
            return JSONResponse(_problem("resource_not_found", 404), status_code=404)
        if revision_id is not None and revision_id != output.revision_id:
            return JSONResponse(_problem("invalid_request", 422), status_code=422)
        return Response(
            content=output.content,
            status_code=200,
            media_type=output.media_type,
            headers={
                "X-Synth-Revision-Id": output.revision_id,
                "X-Synth-Digest-Sha256": output.digest_sha256,
            },
        )

    async def list_events(request: Request) -> Response:
        if denied := check_auth(request):
            return denied
        run = state.runs.get(request.path_params["run_id"])
        if run is None:
            return JSONResponse(_problem("run_not_found", 404), status_code=404)
        if "text/event-stream" not in request.headers.get("accept", ""):
            return JSONResponse(
                {"items": [], "has_more": False, "resume_cursor": "2"},
                status_code=200,
            )

        last_event_id = request.headers.get("last-event-id")
        run.sse_connections += 1
        run.sse_last_event_ids.append(last_event_id)

        def make_event(seq: int, event_id: str, kind: str) -> str:
            payload = {
                "event_id": event_id,
                "run_id": run.run_id,
                "sequence": seq,
                "kind": kind,
                "original_kind": kind,
                "usage": None,
                "observed_at": "2026-11-24T00:00:00Z",
                "approval_id": None,
            }
            return f"id: {seq}\ndata: {json.dumps(payload)}\n\n"

        async def stream() -> AsyncGenerator[bytes, None]:
            if run.run_id == RUN_ID_TIMEOUT:
                # Hold stream open briefly so short client timeouts raise wait_timeout (exit 2).
                yield b": heartbeat\n\n"
                await asyncio.sleep(2.0)
                return
            if run.run_id == RUN_ID_PAUSED:
                milestone = {
                    "milestone": "execution_closed",
                    "run_id": run.run_id,
                    "state": "paused",
                    "execution_closed": False,
                    "accounting_final": False,
                    "cleanup_complete": False,
                    "result_available": False,
                    "failure": None,
                    "final": False,
                }
                yield f"id: 1.1\nevent: milestone\ndata: {json.dumps(milestone)}\n\n".encode()
                await asyncio.sleep(0.2)
                return
            if run.run_id == RUN_ID_FAILED:
                milestone = {
                    "milestone": "cleanup_complete",
                    "run_id": run.run_id,
                    "state": "failed",
                    "execution_closed": True,
                    "accounting_final": True,
                    "cleanup_complete": True,
                    "result_available": True,
                    "failure": None,
                    "final": True,
                }
                yield f"id: 1.3\nevent: milestone\ndata: {json.dumps(milestone)}\n\n".encode()
                return
            if last_event_id is None:
                # First stream connection: emit event 1, then retryable owner handoff error frame.
                yield make_event(1, EVT_ID_1, "run.started").encode()
                handoff = _problem("event_owner_unavailable", 503, retryable=True)
                handoff["last_event_id"] = "1"
                yield f"event: error\ndata: {json.dumps(handoff)}\n\n".encode()
                return
            # Reconnect with Last-Event-ID=1: emit event 2 and settle the run completely.
            yield make_event(2, EVT_ID_2, "run.completed").encode()
            run.state = "completed"
            run.execution_closed = True
            run.accounting_final = True
            run.cleanup_complete = True
            milestone = {
                "milestone": "cleanup_complete",
                "run_id": run.run_id,
                "state": "completed",
                "execution_closed": True,
                "accounting_final": True,
                "cleanup_complete": True,
                "result_available": True,
                "failure": None,
                "final": True,
            }
            yield f"id: 2.3\nevent: milestone\ndata: {json.dumps(milestone)}\n\n".encode()

        return StreamingResponse(stream(), media_type="text/event-stream")

    mcp_session_manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
        nonlocal mcp_session_manager
        async with AsyncSynthClient(VALID_API_KEY, base_url=state.base_url) as client:
            mcp_server = build_server(client, upload_descriptor=None)
            mcp_session_manager = StreamableHTTPSessionManager(
                app=mcp_server,
                stateless=True,
                json_response=True,
            )
            async with mcp_session_manager.run():
                yield

    async def hosted_mcp_asgi(scope: Scope, receive: Receive, send: Send) -> None:
        headers = {
            k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])
        }
        auth_header = headers.get("authorization")
        token = (
            auth_header.removeprefix("Bearer ").strip()
            if auth_header and auth_header.startswith("Bearer ")
            else None
        )
        if token != VALID_MCP_BEARER:
            state.mcp_auth_checks.append((token, 401))
            response = JSONResponse(_problem("unauthenticated", 401), status_code=401)
            await response(scope, receive, send)
            return
        state.mcp_auth_checks.append((token, 200))
        assert mcp_session_manager is not None
        await mcp_session_manager.handle_request(scope, receive, send)

    routes = [
        Route("/v1/uploads", create_upload, methods=["POST"]),
        Route("/_blob/{upload_id}", put_blob, methods=["PUT"]),
        Route("/v1/uploads/{upload_id}/finalize", finalize_upload, methods=["POST"]),
        Route("/v1/files", create_file_inline, methods=["POST"]),
        Route("/v1/runs", create_run, methods=["POST"]),
        Route("/v1/runs/{run_id}", get_run, methods=["GET"]),
        Route("/v1/runs/{run_id}/limits", get_limits, methods=["GET"]),
        Route("/v1/runs/{run_id}/limits", update_limits, methods=["POST"]),
        Route("/v1/runs/{run_id}/pause", pause_run, methods=["POST"]),
        Route("/v1/runs/{run_id}/resume", resume_run, methods=["POST"]),
        Route("/v1/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/v1/runs/{run_id}/messages", send_message, methods=["POST"]),
        Route("/v1/runs/{run_id}/messages", list_messages, methods=["GET"]),
        Route("/v1/runs/{run_id}/report", get_report, methods=["GET"]),
        Route("/v1/runs/{run_id}/result", get_result, methods=["GET"]),
        Route("/v1/runs/{run_id}/outputs", list_outputs, methods=["GET"]),
        Route("/v1/runs/{run_id}/outputs/{output_id}/content", get_output_content, methods=["GET"]),
        Route("/v1/runs/{run_id}/events", list_events, methods=["GET"]),
        Mount("/mcp", app=hosted_mcp_asgi),
    ]
    return Starlette(routes=routes, lifespan=lifespan)


class LocalContractServer:
    """Thread-hosted loopback HTTP + SSE + Streamable-HTTP /mcp server."""

    def __init__(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = int(probe.getsockname()[1])
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.mcp_url = f"{self.base_url}/mcp"
        self.state = ContractState(self.base_url)
        self._app = _build_app(self.state)
        self._config = uvicorn.Config(
            self._app, host="127.0.0.1", port=self.port, log_level="error"
        )
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, name="e2e-contract", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() >= deadline:
                raise RuntimeError("LocalContractServer failed to start within 10s")
            time.sleep(0.02)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)
