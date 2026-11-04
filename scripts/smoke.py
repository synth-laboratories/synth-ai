"""Bounded live qualification; explicit --execute, $0.30 aggregate reservation caps.

Reads SYNTH_API_KEY / SYNTH_API_BASE_URL. Optional SYNTH_CLERK_SESSION_TOKEN
qualifies positive session-only key CRUD; otherwise verifies their scoped-key refusal.
Checkout creates an unpaid session. Receipt contains identifiers/status only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from synth_ai import AsyncSynthClient
from synth_ai._generated.operations import OPERATIONS
from synth_ai.errors import SwarmsAPIError


def require_successful_run(snapshot):
    if snapshot.state != "completed":
        raise RuntimeError("first run did not complete successfully: " + snapshot.state)


async def qualify_inference(client, project_id, receipt, save):
    if receipt.get("inference", {}).get("replay_verified"):
        return
    receipt.setdefault("inference_key", "public-sdk-inference:" + str(uuid4()))
    save()
    options = dict(
        project_id=project_id, budget="0.10", ttl=60, idempotency_key=receipt["inference_key"]
    )
    session = await client.inference.session(**options)
    replay = await client.inference.session(**options)
    if session != replay:
        raise RuntimeError("inference idempotent replay changed the session")
    if session.model != "gpt-6-luna" or session.max_output_tokens <= 0:
        raise RuntimeError("inference catalogue admission differs")
    receipt["inference"] = {
        "replay_verified": True,
        "expires_at": session.expires_at,
        "model": session.model,
        "max_output_tokens": session.max_output_tokens,
        "public_revoke_served": False,
        "expiry_cleanup": "pending owner proof",
        "cleanup_complete": False,
    }
    receipt["operations"]["createInferenceSession"] = {"status": "passed"}
    save()


def apply_inference_cleanup_proof(receipt, path):
    proof = json.loads(path.read_text())
    if (
        proof.get("schema") != "synth.inference.cleanup-proof.v1"
        or proof.get("idempotency_key") != receipt["inference_key"]
    ):
        raise ValueError("inference cleanup proof identity differs")
    required = ("gateway_revoked", "core_closed", "hold_released", "charge_final")
    if not all(proof.get(name) is True for name in required):
        raise ValueError("inference cleanup proof is incomplete")
    expiry = datetime.fromisoformat(receipt["inference"]["expires_at"].replace("Z", "+00:00"))
    observed = datetime.fromisoformat(proof["observed_at"].replace("Z", "+00:00"))
    if (
        expiry.tzinfo is None
        or observed.tzinfo is None
        or observed < expiry
        or observed > datetime.now(timezone.utc) + timedelta(seconds=60)
    ):
        raise ValueError("inference cleanup proof time is outside the verified expiry window")
    receipt["inference"].update(
        cleanup_complete=True,
        expiry_cleanup="verified",
        cleanup_proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cleanup_proof=str(path.resolve()),
    )


async def run(output: Path, resume: bool = False, cleanup_proof: Path | None = None) -> None:
    receipt = {
        "schema": "synth.public-sdk.smoke.v2",
        "base_url": os.environ["SYNTH_API_BASE_URL"],
        "aggregate_maximum_usd": "0.30",
        "operations": {},
        "runs": [],
        "complete": False,
    }
    if resume:
        receipt = json.loads(output.read_text())
        if receipt.get("schema") != "synth.public-sdk.smoke.v2":
            raise ValueError(
                "receipt predates the inference reservation; retain it and use a new v2 receipt"
            )
        if receipt["base_url"] != os.environ["SYNTH_API_BASE_URL"]:
            raise ValueError("cannot resume against a different API")
        if len(receipt["runs"]) > 2:
            raise ValueError("receipt exceeds the two-run budget")
    elif output.exists():
        raise ValueError("receipt already exists; use --resume to preserve the run budget")
    receipt["failures"] = {}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(receipt, indent=2) + "\n")

    async def call(client, operation, **kwargs):
        try:
            result = await client._transport.invoke(operation, **kwargs)
        except SwarmsAPIError as error:
            receipt["operations"][operation] = {"status": error.status, "code": error.code}
            save()
            raise
        receipt["operations"][operation] = {"status": "passed"}
        save()
        return result

    def key():
        return "public-sdk-smoke:" + str(uuid4())

    async def capture(response: httpx.Response):
        request = response.request
        operation = next(
            (
                name
                for name, spec in OPERATIONS.items()
                if spec["method"] == request.method
                and re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", spec["path"]), request.url.path)
            ),
            None,
        )
        if operation is None:
            return
        entry = {
            "operation": operation,
            "status": response.status_code,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "headers_present": {
                name: name in response.headers
                for name in (
                    "x-request-id",
                    "traceparent",
                    "ratelimit-limit",
                    "ratelimit-remaining",
                    "ratelimit-reset",
                    "retry-after",
                    "etag",
                    "strict-transport-security",
                    "idempotent-replayed",
                )
            },
        }
        if response.status_code >= 400 and "application/problem+json" in response.headers.get(
            "content-type", ""
        ):
            await response.aread()
            entry["problem_code"] = response.json().get("code")
        receipt.setdefault("responses", []).append(entry)
        save()

    async with (
        httpx.AsyncClient(event_hooks={"response": [capture]}) as http,
        AsyncSynthClient(http_client=http) as client,
    ):
        session = os.getenv("SYNTH_CLERK_SESSION_TOKEN")
        if session and not receipt.get("key_crud_positive"):
            async with httpx.AsyncClient(
                headers={"X-Clerk-Session-Token": session}, event_hooks={"response": [capture]}
            ) as http:
                async with AsyncSynthClient(http_client=http) as admin:
                    minted = await call(
                        admin,
                        "createApiKey",
                        body={
                            "name": "public-sdk-smoke",
                            "purpose": "automation",
                            "scopes": ["account:read"],
                            "expires_at": (
                                datetime.now(timezone.utc) + timedelta(hours=1)
                            ).isoformat(),
                        },
                        idempotency_key=key(),
                    )
                    kid = minted.key_id
                    try:
                        await call(admin, "listApiKeys", query={"limit": 1})
                        await call(
                            admin,
                            "updateApiKey",
                            path={"key_id": kid},
                            body={
                                "name": "public-sdk-smoke-updated",
                                "scopes": minted.scopes,
                                "expected_version": minted.authorization_version,
                            },
                        )
                        rotated = await call(
                            admin, "rotateApiKey", path={"key_id": kid}, idempotency_key=key()
                        )
                        kid = rotated.key_id
                    finally:
                        await call(admin, "revokeApiKey", path={"key_id": kid})
            receipt["key_crud_positive"] = True
            save()
        elif not receipt.get("key_crud_positive"):
            for op in [
                "createApiKey",
                "listApiKeys",
                "updateApiKey",
                "rotateApiKey",
                "revokeApiKey",
            ]:
                kwargs = {
                    "path": {
                        name: "key_AAAAAAAAAAAAAAAAAAAAAQ" for name in OPERATIONS[op]["path_params"]
                    }
                }
                if op == "createApiKey":
                    kwargs["body"] = {"name": "public-sdk-smoke"}
                if op == "updateApiKey":
                    kwargs["body"] = {
                        "name": "public-sdk-smoke-updated",
                        "scopes": ["account:read"],
                        "expected_version": 1,
                    }
                try:
                    await call(client, op, **kwargs)
                except SwarmsAPIError as error:
                    if error.status != 403 or error.code != "session_required":
                        raise
                else:
                    raise RuntimeError("session boundary accepted API key: " + op)
        await call(client, "getAccount")
        await call(client, "getBalance")
        await call(client, "listPlans")
        await call(client, "listBillingHistory", query={"limit": 1})
        await call(client, "listInvoices", query={"limit": 1})
        await call(client, "listFamilies")
        if receipt["runs"]:
            existing = await client.swarms.get(receipt["runs"][0]["run_id"])
            project = await call(client, "getProject", path={"project_id": existing.project_id})
        else:
            receipt.setdefault("project_key", key())
            save()
            project = await call(
                client,
                "createProject",
                body={"name": "public-sdk-smoke"},
                idempotency_key=receipt["project_key"],
            )
        await call(client, "getProject", path={"project_id": project.project_id})
        await call(client, "listProjects", query={"limit": 1})

        await qualify_inference(client, project.project_id, receipt, save)

        async def launch(message, index):
            if len(receipt["runs"]) > index:
                return receipt["runs"][index]["run_id"]
            if index not in (0, 1):
                raise ValueError("two-run budget exceeded")
            receipt.setdefault("launch_keys", {}).setdefault(str(index), key())
            save()
            result = await call(
                client,
                "createRun",
                body={
                    "project_id": project.project_id,
                    "message": message,
                    "program": {"id": "swarms.general", "version": "1"},
                    "limits": {
                        "maximum_charge": {"amount": "0.10", "currency": "USD"},
                        "wall_time_seconds": 120,
                        "maximum_tasks": 1,
                        "maximum_parallel_tasks": 1,
                    },
                },
                idempotency_key=receipt["launch_keys"][str(index)],
            )
            receipt["runs"].append({"run_id": result.run_id})
            save()
            return result.run_id

        async def closed(run_id):
            end = time.monotonic() + 240
            while time.monotonic() < end:
                snapshot = await call(client, "getRun", path={"run_id": run_id})
                for item in receipt["runs"]:
                    if item["run_id"] == run_id:
                        item.update(
                            state=snapshot.state,
                            execution_closed=snapshot.execution_closed,
                            accounting_final=snapshot.accounting_final,
                            cleanup_complete=snapshot.cleanup_complete,
                            observed_at=datetime.now(timezone.utc).isoformat(),
                        )
                save()
                if (
                    snapshot.execution_closed
                    and snapshot.accounting_final
                    and snapshot.cleanup_complete
                ):
                    for item in receipt["runs"]:
                        if item["run_id"] == run_id:
                            item.update(
                                state=snapshot.state,
                                execution_closed=True,
                                accounting_final=True,
                                cleanup_complete=snapshot.cleanup_complete,
                            )
                    save()
                    return snapshot
                await asyncio.sleep(2)
            raise RuntimeError("run closure deadline exceeded: " + run_id)

        first = await launch(
            "Create a file hello.txt containing exactly hello and publish it as the task output.", 0
        )
        try:
            require_successful_run(await closed(first))
            await call(client, "listRuns", query={"project_id": project.project_id, "limit": 1})
            await call(client, "listRunEvents", path={"run_id": first}, query={"limit": 100})
            async with asyncio.timeout(15):
                async for _ in client.swarms.follow(first):
                    pass
            outputs = await call(client, "listRunOutputs", path={"run_id": first})
            if not outputs.items:
                raise RuntimeError("first run produced no output; content gate is unverified")
            out = await call(
                client,
                "getRunOutput",
                path={"run_id": first, "output_id": outputs.items[0].output_id},
            )
            ref = out.reference
            if not ref:
                raise RuntimeError("output reference missing")
            content = await call(
                client,
                "getRunOutputContent",
                path={"run_id": first, "output_id": out.output_id},
                query={"revision_id": ref["revision_id"], "digest_sha256": ref["digest_sha256"]},
            )
            assert hashlib.sha256(content).hexdigest() == ref["digest_sha256"]
        except (RuntimeError, SwarmsAPIError, TimeoutError) as error:
            receipt["failures"]["first_run"] = type(error).__name__
            save()
        finally:
            snapshot = await client.swarms.get(first)
            if not snapshot.execution_closed:
                await client.swarms.cancel(first, idempotency_key=key())
                try:
                    await closed(first)
                except RuntimeError:
                    receipt["failures"]["first_run_closure"] = "deadline_exceeded"
                    save()
        second = await launch("Write a short explanation of addition.", 1)
        try:
            await call(client, "cancelRun", path={"run_id": second}, idempotency_key=key())
        finally:
            try:
                await closed(second)
            except RuntimeError:
                receipt["failures"]["second_run_closure"] = "deadline_exceeded"
                save()
        receipt.setdefault("checkout_key", key())
        save()
        await call(
            client,
            "createCheckout",
            body={"pack_id": "credits.10"},
            idempotency_key=receipt["checkout_key"],
        )
        receipt["api_complete"] = (
            not receipt["failures"]
            and set(receipt["operations"]) == set(OPERATIONS)
            and all(item.get("cleanup_complete") for item in receipt["runs"])
        )
        if cleanup_proof is not None:
            apply_inference_cleanup_proof(receipt, cleanup_proof)
        receipt["complete"] = receipt["api_complete"] and receipt["inference"].get(
            "cleanup_complete", False
        )
        save()
        if not receipt["complete"]:
            raise RuntimeError(
                "qualification incomplete; resume the retained receipt after resolving failures"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse the receipt's two runs and mutation keys"
    )
    parser.add_argument(
        "--inference-cleanup-proof",
        type=Path,
        help="Retained owner verification of gateway revoke, core closure, hold release and final charge",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args.output, args.resume, args.inference_cleanup_proof))
    except SwarmsAPIError as error:
        raise SystemExit(
            f"smoke refused: status={error.status} code={error.code} outcome={error.outcome}"
        ) from None
