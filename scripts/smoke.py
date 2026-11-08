"""Bounded live qualification; explicit --execute, $0.20 aggregate run caps.

Reads SYNTH_API_KEY / SYNTH_API_BASE_URL. Optional SYNTH_CLERK_SESSION_TOKEN
qualifies positive session-only key CRUD; otherwise verifies their scoped-key refusal.
Checkout creates an unpaid session. Receipt contains identifiers/status only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, TypeVar, overload
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from synth_ai import AsyncSynthClient
from synth_ai._generated.models import (
    CheckoutSession,
    MintedApiKey,
    OutputPage,
    Problem,
    RunOutput,
    RunSnapshot,
)
from synth_ai._generated.operations import OPERATIONS
from synth_ai.errors import SwarmsAPIError


class RunStateView(Protocol):
    @property
    def state(self) -> str: ...


class OperationObservation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    status: int | Literal["passed"]
    code: str | None = None


class ResponseObservation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    operation: str
    status: int
    observed_at: str
    headers_present: dict[str, bool]
    problem_code: str | None = None


class ObservedRun(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)
    run_id: str
    state: str | None = None
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    observed_at: str | None = None

    def observe(self, snapshot: RunSnapshot) -> None:
        self.state = snapshot.state
        self.execution_closed = snapshot.execution_closed
        self.accounting_final = snapshot.accounting_final
        self.cleanup_complete = snapshot.cleanup_complete
        self.observed_at = datetime.now(UTC).isoformat()


class SmokeReceipt(BaseModel):
    """Mutable observation state; the persisted JSON is validated before resuming."""

    model_config = ConfigDict(
        strict=True, extra="forbid", populate_by_name=True, validate_assignment=True
    )
    schema_: Literal["synth.public-sdk.smoke.v3"] = Field(
        default="synth.public-sdk.smoke.v3", alias="schema"
    )
    base_url: str
    aggregate_maximum_usd: Literal["0.20"] = "0.20"
    operations: dict[str, OperationObservation] = Field(default_factory=dict)
    runs: list[ObservedRun] = Field(default_factory=list, max_length=2)
    complete: bool = False
    failures: dict[str, str] = Field(default_factory=dict)
    responses: list[ResponseObservation] = Field(default_factory=list)
    key_crud_positive: bool = False
    launch_keys: dict[str, str] = Field(default_factory=dict)
    checkout_key: str | None = None
    api_complete: bool = False


T = TypeVar("T", bound=BaseModel)


def require_successful_run(snapshot: RunStateView) -> None:
    if snapshot.state != "completed":
        raise RuntimeError("first run did not complete successfully: " + snapshot.state)


async def run(output: Path, resume: bool = False) -> None:
    receipt = SmokeReceipt(base_url=os.environ["SYNTH_API_BASE_URL"])
    if resume:
        receipt = SmokeReceipt.model_validate_json(output.read_text())
        if receipt.base_url != os.environ["SYNTH_API_BASE_URL"]:
            raise ValueError("cannot resume against a different API")
        if len(receipt.runs) > 2:
            raise ValueError("receipt exceeds the two-run budget")
    elif output.exists():
        raise ValueError("receipt already exists; use --resume to preserve the run budget")
    receipt.failures = {}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(
            receipt.model_dump_json(indent=2, by_alias=True, exclude_none=True) + "\n"
        )

    @overload
    async def call(
        client: AsyncSynthClient,
        operation: str,
        model: type[T],
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> T: ...

    @overload
    async def call(
        client: AsyncSynthClient,
        operation: str,
        model: None = None,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> BaseModel | bytes | None: ...

    async def call(
        client: AsyncSynthClient,
        operation: str,
        model: type[T] | None = None,
        *,
        path: Mapping[str, str] | None = None,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> T | BaseModel | bytes | None:
        try:
            result = await client._transport.invoke(
                operation, path=path, query=query, body=body, idempotency_key=idempotency_key
            )
        except SwarmsAPIError as error:
            receipt.operations[operation] = OperationObservation(
                status=error.status, code=error.code
            )
            save()
            raise
        receipt.operations[operation] = OperationObservation(status="passed")
        save()
        if model is not None and not isinstance(result, model):
            raise TypeError("operation returned the wrong model: " + operation)
        return result

    def key() -> str:
        return "public-sdk-smoke:" + str(uuid4())

    async def capture(response: httpx.Response) -> None:
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
        entry = ResponseObservation(
            operation=operation,
            status=response.status_code,
            observed_at=datetime.now(UTC).isoformat(),
            headers_present={
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
        )
        if response.status_code >= 400 and "application/problem+json" in response.headers.get(
            "content-type", ""
        ):
            await response.aread()
            entry.problem_code = Problem.model_validate_json(response.content).code
        receipt.responses.append(entry)
        save()

    async with (
        httpx.AsyncClient(event_hooks={"response": [capture]}) as http,
        AsyncSynthClient(http_client=http) as client,
    ):
        session = os.getenv("SYNTH_CLERK_SESSION_TOKEN")
        if session is not None and session != "" and not receipt.key_crud_positive:
            async with httpx.AsyncClient(
                headers={"X-Clerk-Session-Token": session}, event_hooks={"response": [capture]}
            ) as http:
                async with AsyncSynthClient(http_client=http) as admin:
                    minted = await call(
                        admin,
                        "createApiKey",
                        MintedApiKey,
                        body={
                            "name": "public-sdk-smoke",
                            "purpose": "automation",
                            "scopes": ["account:read"],
                            "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
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
                                "scopes": [scope for scope in minted.scopes],
                                "expected_version": minted.authorization_version,
                            },
                        )
                        rotated = await call(
                            admin,
                            "rotateApiKey",
                            MintedApiKey,
                            path={"key_id": kid},
                            idempotency_key=key(),
                        )
                        kid = rotated.key_id
                    finally:
                        await call(admin, "revokeApiKey", path={"key_id": kid})
            receipt.key_crud_positive = True
            save()
        elif not receipt.key_crud_positive:
            for op in [
                "createApiKey",
                "listApiKeys",
                "updateApiKey",
                "rotateApiKey",
                "revokeApiKey",
            ]:
                path = {
                    name: "key_AAAAAAAAAAAAAAAAAAAAAQ" for name in OPERATIONS[op]["path_params"]
                }
                body: dict[str, JsonValue] | None = None
                if op == "createApiKey":
                    body = {"name": "public-sdk-smoke"}
                if op == "updateApiKey":
                    body = {
                        "name": "public-sdk-smoke-updated",
                        "scopes": ["account:read"],
                        "expected_version": 1,
                    }
                try:
                    await call(client, op, path=path, body=body)
                except SwarmsAPIError as error:
                    if error.status != 403 or error.code != "session_required":
                        raise
                else:
                    raise RuntimeError("session boundary accepted API key: " + op)
        await call(client, "getAccount")
        await call(client, "getBalance")
        await call(client, "listPlans")
        await call(client, "listBillingHistory", query={"limit": 1})
        await call(client, "listStatements", query={"limit": 1})
        await call(client, "listLimits")
        await call(client, "getUsageSummary")
        await call(client, "listUsageRecords", query={"limit": 1})
        await call(client, "listFamilies")

        async def launch(message: str, index: int) -> str:
            if len(receipt.runs) > index:
                return receipt.runs[index].run_id
            if index not in (0, 1):
                raise ValueError("two-run budget exceeded")
            receipt.launch_keys.setdefault(str(index), key())
            save()
            result = await call(
                client,
                "createRun",
                RunSnapshot,
                body={
                    "message": message,
                    "profile": "swarms-v2",
                    "limits": {
                        "maximum_charge": {"amount": "0.10", "currency": "USD"},
                        "wall_time_seconds": 120,
                    },
                },
                idempotency_key=receipt.launch_keys[str(index)],
            )
            receipt.runs.append(ObservedRun(run_id=result.run_id))
            save()
            return result.run_id

        async def closed(run_id: str) -> RunSnapshot:
            end = time.monotonic() + 240
            while time.monotonic() < end:
                snapshot = await call(client, "getRun", RunSnapshot, path={"run_id": run_id})
                for item in receipt.runs:
                    if item.run_id == run_id:
                        item.observe(snapshot)
                save()
                if snapshot.finished:
                    return snapshot
                await asyncio.sleep(2)
            raise RuntimeError("run closure deadline exceeded: " + run_id)

        first = await launch(
            "Create a file hello.txt containing exactly hello and publish it as the task output.", 0
        )
        try:
            require_successful_run(await closed(first))
            await call(client, "listRuns", query={"limit": 1})
            await call(client, "listRunEvents", path={"run_id": first}, query={"limit": 100})
            async with asyncio.timeout(15):
                async for _ in client.swarms.follow(first):
                    pass
            outputs = await call(client, "listRunOutputs", OutputPage, path={"run_id": first})
            if not outputs.items:
                raise RuntimeError("first run produced no output; content gate is unverified")
            out = await call(
                client,
                "getRunOutput",
                RunOutput,
                path={"run_id": first, "output_id": outputs.items[0].output_id},
            )
            ref = out.reference
            if ref is None:
                raise RuntimeError("output reference missing")
            content = await call(
                client,
                "getRunOutputContent",
                path={"run_id": first, "output_id": out.output_id},
                query={"revision_id": ref["revision_id"], "digest_sha256": ref["digest_sha256"]},
            )
            if not isinstance(content, bytes):
                raise TypeError("output content must be bytes")
            if hashlib.sha256(content).hexdigest() != ref["digest_sha256"]:
                raise ValueError("download digest differs from immutable output reference")
        except (RuntimeError, SwarmsAPIError, TimeoutError) as error:
            receipt.failures["first_run"] = type(error).__name__
            save()
        finally:
            snapshot = await client.swarms.get(first)
            if not snapshot.execution_closed:
                await client.swarms.cancel(first, idempotency_key=key())
                try:
                    await closed(first)
                except RuntimeError:
                    receipt.failures["first_run_closure"] = "deadline_exceeded"
                    save()
        second = await launch("Write a short explanation of addition.", 1)
        try:
            await call(client, "cancelRun", path={"run_id": second}, idempotency_key=key())
        finally:
            try:
                await closed(second)
            except RuntimeError:
                receipt.failures["second_run_closure"] = "deadline_exceeded"
                save()
        if receipt.checkout_key is None:
            receipt.checkout_key = key()
        save()
        checkout = await call(
            client,
            "createCheckout",
            CheckoutSession,
            body={"pack_id": "credits.10"},
            idempotency_key=receipt.checkout_key,
        )
        await call(client, "getCheckout", path={"id": checkout.checkout_id})
        receipt.api_complete = (
            not receipt.failures
            and set(receipt.operations) == set(OPERATIONS)
            and all(item.cleanup_complete for item in receipt.runs)
        )
        receipt.complete = receipt.api_complete
        save()
        if not receipt.complete:
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
    args = parser.parse_args()
    try:
        asyncio.run(run(args.output, args.resume))
    except SwarmsAPIError as error:
        raise SystemExit(
            f"smoke refused: status={error.status} code={error.code} outcome={error.outcome}"
        ) from None
