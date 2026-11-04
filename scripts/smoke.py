"""Bounded live qualification; explicit --execute, $0.20 aggregate run caps.

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
import time
from pathlib import Path
from uuid import uuid4

import httpx

from synth_ai import AsyncSynthClient
from synth_ai._generated.operations import OPERATIONS
from synth_ai.errors import SwarmsAPIError


async def run(output: Path) -> None:
    receipt = {
        "schema": "synth.public-sdk.smoke.v1",
        "base_url": os.environ["SYNTH_API_BASE_URL"],
        "aggregate_maximum_usd": "0.20",
        "operations": {},
        "runs": [],
        "complete": False,
    }
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

    async with AsyncSynthClient() as client:
        await call(client, "getAccount")
        await call(client, "getBalance")
        await call(client, "listPlans")
        await call(client, "listBillingHistory", query={"limit": 1})
        await call(client, "listInvoices", query={"limit": 1})
        await call(client, "listFamilies")
        project = await call(
            client, "createProject", body={"name": "public-sdk-smoke"}, idempotency_key=key()
        )
        await call(client, "getProject", path={"project_id": project.project_id})
        await call(client, "listProjects", query={"limit": 1})

        async def launch(message):
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
                idempotency_key=key(),
            )
            receipt["runs"].append({"run_id": result.run_id})
            save()
            return result.run_id

        async def closed(run_id):
            end = time.monotonic() + 240
            while time.monotonic() < end:
                snapshot = await call(client, "getRun", path={"run_id": run_id})
                if snapshot.execution_closed and snapshot.accounting_final:
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
            "Create a file hello.txt containing exactly hello and publish it as the task output."
        )
        try:
            await closed(first)
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
        finally:
            snapshot = await client.swarms.get(first)
            if not snapshot.execution_closed:
                await client.swarms.cancel(first, idempotency_key=key())
                await closed(first)
        second = await launch("Write a short explanation of addition.")
        try:
            await call(client, "cancelRun", path={"run_id": second}, idempotency_key=key())
        finally:
            await closed(second)
        await call(client, "createCheckout", body={"pack_id": "credits.10"}, idempotency_key=key())
        session = os.getenv("SYNTH_CLERK_SESSION_TOKEN")
        if session:
            async with httpx.AsyncClient(headers={"X-Clerk-Session-Token": session}) as http:
                async with AsyncSynthClient(http_client=http) as admin:
                    minted = await call(
                        admin,
                        "createApiKey",
                        body={"name": "public-sdk-smoke"},
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
                        await call(
                            admin, "rotateApiKey", path={"key_id": kid}, idempotency_key=key()
                        )
                    finally:
                        await call(admin, "revokeApiKey", path={"key_id": kid})
        else:
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
        assert set(receipt["operations"]) == set(OPERATIONS)
        receipt["complete"] = True
        receipt["key_crud_positive"] = bool(session)
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.output))
    except SwarmsAPIError as error:
        raise SystemExit(
            f"smoke refused: status={error.status} code={error.code} outcome={error.outcome}"
        ) from None
