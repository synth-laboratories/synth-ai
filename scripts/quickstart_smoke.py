"""Opt-in SDK, CLI and stdio-MCP quickstarts; at most three $4 run caps.

This exercises a single uploaded task on each surface. It is not a hosted OAuth,
capacity, load, provider-invoice or release acceptance test. A retained receipt
must be resumed with the same target and identities after an uncertain response.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import signal
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

import synth_ai
from synth_ai import AsyncSynthClient
from synth_ai.models import InputFile, RunInputs

SURFACES = ("sdk", "cli", "mcp")
MAX_RUN_USD = "4.00"
WALL_SECONDS = 600


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)
    surface: Literal["sdk", "cli", "mcp"]
    upload_key: str
    launch_key: str
    cancel_key: str
    run_id: str | None = None
    status: Literal["not_started", "launch_pending", "running", "passed", "failed"] = "not_started"
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    charge_usd: str | None = None
    charge_final: bool = False
    answer_verified: bool = False
    output_id: str | None = None
    output_sha256: str | None = None
    output_file: str | None = None
    cancel_requested: bool = False
    observed_at: str | None = None
    failure_kind: str | None = None
    failure_code: str | None = None


class Receipt(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, validate_assignment=True, populate_by_name=True
    )
    schema_: Literal["synth.quickstart-smoke.v1"] = Field(
        default="synth.quickstart-smoke.v1", alias="schema"
    )
    base_url: str
    fixture_org_id: str
    cohort_sha256: str
    sdk_version: str
    sdk_tree_sha256: str
    runner_sha256: str
    nonce: str
    aggregate_maximum_usd: Literal["12.00"] = "12.00"
    arms: list[Arm] = Field(min_length=3, max_length=3)
    provider_invoice_reconciled: Literal[False] = False
    uploaded_fixtures_retained: Literal[True] = True

    def validate_intent(self, target: str, fixture_org_id: str, cohort_sha256: str) -> None:
        if (
            self.base_url != target
            or self.fixture_org_id != fixture_org_id
            or not fixture_org_id
            or self.cohort_sha256 != cohort_sha256
            or re.fullmatch(r"[0-9a-f]{64}", cohort_sha256) is None
            or tuple(a.surface for a in self.arms) != SURFACES
        ):
            raise ValueError("receipt target or surface mismatch")
        if (
            self.sdk_version != synth_ai.__version__
            or self.sdk_tree_sha256 != sdk_tree_digest()
            or self.runner_sha256 != runner_digest()
        ):
            raise ValueError("receipt SDK or runner identity differs")
        if re.fullmatch(r"[0-9a-f]{32}", self.nonce) is None:
            raise ValueError("receipt nonce is invalid")
        for arm in self.arms:
            if (
                arm.upload_key != f"quickstart:{self.nonce}:{arm.surface}:upload"
                or arm.launch_key != f"quickstart:{self.nonce}:{arm.surface}:launch"
                or arm.cancel_key != f"quickstart:{self.nonce}:{arm.surface}:cancel"
            ):
                raise ValueError("receipt mutation keys are invalid")
            if arm.run_id is not None:
                valid_run_id(arm.run_id)
            if arm.cancel_requested and arm.run_id is None:
                raise ValueError("cancel intent lacks run identity")
            if arm.status in {"running", "passed", "failed"} and arm.run_id is None:
                raise ValueError("receipt run identity is missing")
            if arm.status == "passed" and not (
                arm.execution_closed
                and arm.accounting_final
                and arm.cleanup_complete
                and arm.charge_final
                and arm.answer_verified
                and arm.output_id is not None
                and arm.output_sha256 is not None
                and arm.output_file is not None
                and arm.charge_usd is not None
            ):
                raise ValueError("passed arm lacks finality evidence")
            if arm.charge_usd is not None:
                parse_charge(arm.charge_usd)
                if arm.status == "passed":
                    check_charge(arm.charge_usd)
        ids = [a.run_id for a in self.arms if a.run_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("receipt has duplicate run identities")

    @property
    def journey_passed(self) -> bool:
        return all(
            a.status == "passed"
            and a.execution_closed
            and a.accounting_final
            and a.cleanup_complete
            and a.charge_final
            and a.answer_verified
            and a.output_id is not None
            and a.output_sha256 is not None
            and a.output_file is not None
            for a in self.arms
        )

    def save(self, path: Path) -> None:
        data = self.model_dump(mode="json", by_alias=True)
        data["journey_passed"] = self.journey_passed
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            os.chmod(tmp, 0o600)
            json.dump(data, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> Receipt:
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value.pop("journey_passed", None)
        return cls.model_validate(value)


def validate_evidence(receipt: Receipt, receipt_path: Path) -> None:
    for arm in receipt.arms:
        if arm.status != "passed":
            continue
        if arm.output_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", arm.output_sha256) is None:
            raise ValueError("passed arm has invalid output digest")
        expected = receipt_path.with_name(
            f"{receipt_path.stem}.{arm.surface}.{arm.output_sha256}.output"
        )
        if arm.output_file != str(expected) or expected.is_symlink() or not expected.is_file():
            raise ValueError("passed arm output evidence is missing")
        content = expected.read_bytes()
        if (
            len(content) > 1_048_576
            or hashlib.sha256(content).hexdigest() != arm.output_sha256
            or receipt.nonce.encode() not in content
        ):
            raise ValueError("passed arm output evidence differs")


def sdk_tree_digest() -> str:
    if synth_ai.__file__ is None:
        raise ValueError("SDK source location unavailable")
    root = Path(synth_ai.__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    if not paths:
        raise ValueError("SDK source is empty")
    for path in paths:
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def runner_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def new_receipt(target: str, fixture_org_id: str, cohort_sha256: str) -> Receipt:
    nonce = uuid4().hex
    return Receipt(
        base_url=target,
        fixture_org_id=fixture_org_id,
        cohort_sha256=cohort_sha256,
        sdk_version=synth_ai.__version__,
        sdk_tree_sha256=sdk_tree_digest(),
        runner_sha256=runner_digest(),
        nonce=nonce,
        arms=[
            Arm(
                surface=surface,
                upload_key=f"quickstart:{nonce}:{surface}:upload",
                launch_key=f"quickstart:{nonce}:{surface}:launch",
                cancel_key=f"quickstart:{nonce}:{surface}:cancel",
            )
            for surface in SURFACES
        ],
    )


def target_url() -> str:
    target = os.environ.get("SYNTH_API_BASE_URL", "").rstrip("/")
    parsed = urlsplit(target)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("SYNTH_API_BASE_URL must be an origin")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("non-local API targets require HTTPS")
    if not bool(os.environ.get("SYNTH_API_KEY")):
        raise ValueError("SYNTH_API_KEY is required")
    return target


class SurfaceFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("surface operation refused")
        self.code = code


def safe_code(value: JsonValue) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[a-z0-9_]{1,80}", value):
        return value
    return None


def child_environment() -> dict[str, str]:
    # Only the target credential and transport settings belong in client processes.
    names = (
        "SYNTH_API_KEY",
        "SYNTH_API_BASE_URL",
        "PATH",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SSL_CERT_FILE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


async def cli(*args: str) -> dict[str, JsonValue]:
    process = await asyncio.create_subprocess_exec(
        str(Path(sys.executable).with_name("synth")),
        *args,
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_environment(),
    )
    try:
        stdout, _stderr = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        raise
    if process.returncode != 0:
        try:
            body = TypeAdapter(dict[str, JsonValue]).validate_json(_stderr)
            code = safe_code(body.get("code"))
        except ValueError:
            code = None
        raise SurfaceFailure(code or "cli_operation_failed")
    return TypeAdapter(dict[str, JsonValue]).validate_python(json.loads(stdout))


async def mcp_call(
    session: ClientSession, name: str, arguments: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    result = await session.call_tool(name, arguments)
    if result.isError:
        try:
            if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
                raise ValueError("invalid error shape")
            body = TypeAdapter(dict[str, JsonValue]).validate_json(result.content[0].text)
            code = safe_code(body.get("code"))
        except ValueError:
            code = None
        raise SurfaceFailure(code or "mcp_operation_failed")
    if len(result.content) != 1 or not isinstance(result.content[0], TextContent):
        raise RuntimeError("MCP response shape invalid")
    return TypeAdapter(dict[str, JsonValue]).validate_python(json.loads(result.content[0].text))


def valid_run_id(value: JsonValue) -> str:
    if not isinstance(value, str) or re.fullmatch(r"run_[A-Za-z0-9_-]{22}", value) is None:
        raise ValueError("run identity is invalid")
    return value


async def launch(
    client: AsyncSynthClient, session: ClientSession, arm: Arm, task: Path, goal: str
) -> str:
    if arm.surface == "cli":
        value = await cli(
            "run",
            goal,
            "--budget",
            MAX_RUN_USD,
            "--wall",
            str(WALL_SECONDS),
            "--file",
            str(task),
            "--idempotency-key",
            arm.launch_key,
        )
        return valid_run_id(value.get("run_id"))
    if arm.surface == "sdk":
        file = await client.swarms.upload_file(str(task), idempotency_key=arm.upload_key)
        inputs = RunInputs(
            files=[
                InputFile(
                    file_id=file.file_id,
                    revision_id=file.revision_id,
                    digest_sha256=file.digest_sha256,
                )
            ]
        )
        run = await client.swarms.launch(
            goal,
            budget=MAX_RUN_USD,
            wall_time_seconds=WALL_SECONDS,
            inputs=inputs,
            idempotency_key=arm.launch_key,
        )
        return run.run_id
    file = await mcp_call(
        session,
        "swarms_upload_file",
        {
            "body": {"name": task.name, "media_type": "text/markdown", "content": task.read_text()},
            "idempotency_key": arm.upload_key,
        },
    )
    file_ref = TypeAdapter(dict[str, str]).validate_python(
        {k: file[k] for k in ("file_id", "revision_id", "digest_sha256")}
    )
    body = TypeAdapter(dict[str, JsonValue]).validate_python(
        {
            "message": goal,
            "limits": {
                "maximum_charge": {"amount": MAX_RUN_USD, "currency": "USD"},
                "wall_time_seconds": WALL_SECONDS,
            },
            "inputs": {"files": [file_ref]},
        }
    )
    value = await mcp_call(
        session, "swarms_run_launch", {"body": body, "idempotency_key": arm.launch_key}
    )
    return valid_run_id(value.get("run_id"))


def parse_charge(dollars: str) -> Decimal:
    try:
        value = Decimal(dollars)
    except InvalidOperation as error:
        raise ValueError("run charge is malformed") from error
    if not value.is_finite() or value < 0:
        raise ValueError("run charge is malformed")
    return value


def check_charge(dollars: str) -> None:
    if parse_charge(dollars) > Decimal(MAX_RUN_USD):
        raise ValueError("run charge is outside its cap")


async def verify_surface(
    client: AsyncSynthClient, session: ClientSession, arm: Arm, nonce: str, receipt_path: Path
) -> tuple[str, bool, str, str, str]:
    if arm.run_id is None:
        raise ValueError("run identity missing")
    if arm.surface == "sdk":
        result = (await client.swarms.result(arm.run_id)).model_dump(mode="json")
    elif arm.surface == "cli":
        status = await cli("status", arm.run_id)
        if valid_run_id(status.get("run_id")) != arm.run_id:
            raise ValueError("CLI status identifies a different run")
        result = await cli("result", arm.run_id)
    else:
        status = await mcp_call(session, "swarms_run_get", {"run_id": arm.run_id})
        if valid_run_id(status.get("run_id")) != arm.run_id:
            raise ValueError("MCP status identifies a different run")
        result = await mcp_call(session, "swarms_run_result", {"run_id": arm.run_id})
    result_run = result.get("run")
    if not isinstance(result_run, dict) or valid_run_id(result_run.get("run_id")) != arm.run_id:
        raise ValueError("result identifies a different run")
    charge = result.get("charge")
    if not isinstance(charge, dict):
        raise ValueError("run charge is unavailable")
    amount = charge.get("amount")
    if not isinstance(amount, dict) or amount.get("currency") != "USD":
        raise ValueError("run charge is unavailable")
    answer = result.get("answer")
    if not isinstance(answer, str) or nonce not in answer:
        raise ValueError("answer did not contain the task nonce")
    outputs = await client.swarms.outputs(arm.run_id)
    evidence: tuple[str, str, str] | None = None
    for output in outputs.items:
        reference = output.reference
        if reference is None or "revision_id" not in reference or "digest_sha256" not in reference:
            continue
        digest = reference["digest_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("output digest is invalid")
        destination = receipt_path.with_name(f"{receipt_path.stem}.{arm.surface}.{digest}.output")
        if destination.is_symlink():
            raise ValueError("output evidence path is a symlink")
        if destination.exists():
            if (
                destination.stat().st_size > 1_048_576
                or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("retained output evidence differs")
        else:
            await client.swarms.download_output(
                arm.run_id,
                output.output_id,
                str(destination),
                revision_id=reference["revision_id"],
                digest_sha256=digest,
                max_bytes=1_048_576,
            )
        content = destination.read_bytes()
        if len(content) > 1_048_576 or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("downloaded output evidence differs")
        if nonce.encode() in content:
            evidence = (output.output_id, digest, str(destination))
            break
    if evidence is None:
        raise ValueError("no digest-verified output contains the task nonce")
    dollars = amount.get("amount")
    if not isinstance(dollars, str):
        raise ValueError("run charge is malformed")
    return dollars, charge.get("final") is True, *evidence


async def cancel_and_reconcile(
    client: AsyncSynthClient, receipt: Receipt, arm: Arm, path: Path
) -> None:
    if arm.run_id is None:
        raise ValueError("cannot cancel an unknown run identity")
    arm.cancel_requested = True
    receipt.save(path)
    run = await client.swarms.get(arm.run_id)
    if run.run_id != arm.run_id:
        raise ValueError("cancel status identifies a different run")
    if not run.execution_closed:
        await client.swarms.cancel(arm.run_id, idempotency_key=arm.cancel_key)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        run = await client.swarms.get(arm.run_id)
        if run.run_id != arm.run_id:
            raise ValueError("cancel status identifies a different run")
        arm.observed_at = datetime.now(UTC).isoformat()
        arm.execution_closed = run.execution_closed
        arm.accounting_final = run.accounting_final
        arm.cleanup_complete = run.cleanup_complete
        if run.charge is not None:
            arm.charge_usd = run.charge.amount.amount
            arm.charge_final = run.charge.final
        receipt.save(path)
        if run.finished:
            arm.status = "failed"
            receipt.save(path)
            return
        await asyncio.sleep(30)
    raise RuntimeError("cancellation finality is pending; resume the retained receipt")


async def execute(receipt: Receipt, path: Path) -> None:
    from mcp.client.stdio import stdio_client

    task = path.with_name(path.stem + "-task.md")
    content = f"# Task\nReply with exactly this token: {receipt.nonce}\n"
    if task.exists() and task.read_text() != content:
        raise ValueError("retained task differs from receipt")
    if not task.exists():
        task.write_text(content)
    goal = "Follow the uploaded task and publish the requested answer."
    server = StdioServerParameters(
        command=str(Path(sys.executable).with_name("synth-ai-mcp")), env=child_environment()
    )
    with open(os.devnull, "w") as errors:
        async with (
            AsyncSynthClient() as client,
            stdio_client(server, errlog=errors) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            account = await client.account.get()
            if account.org.id != receipt.fixture_org_id:
                raise ValueError("credential is not scoped to the retained fixture organization")
            for arm in receipt.arms:
                if arm.status == "passed":
                    continue
                if arm.status == "failed":
                    raise RuntimeError("failed run retained; do not create a replacement")
                if arm.cancel_requested:
                    await cancel_and_reconcile(client, receipt, arm, path)
                    raise RuntimeError("cancelled run retained; do not create a replacement")
                try:
                    if arm.run_id is None:
                        arm.status = "launch_pending"
                        receipt.save(path)
                        arm.run_id = await launch(client, session, arm, task, goal)
                        arm.status = "running"
                        receipt.save(path)
                        receipt.validate_intent(
                            receipt.base_url, receipt.fixture_org_id, receipt.cohort_sha256
                        )
                    deadline = time.monotonic() + WALL_SECONDS + 120
                    while time.monotonic() < deadline:
                        run = await client.swarms.get(arm.run_id)
                        if run.run_id != arm.run_id:
                            raise ValueError("status identifies a different run")
                        arm.observed_at = datetime.now(UTC).isoformat()
                        print(
                            f"{arm.observed_at} {arm.surface}: run={arm.run_id} state={run.state} execution_closed={run.execution_closed} accounting_final={run.accounting_final} cleanup_complete={run.cleanup_complete}",
                            flush=True,
                        )
                        arm.execution_closed = run.execution_closed
                        arm.accounting_final = run.accounting_final
                        arm.cleanup_complete = run.cleanup_complete
                        if run.charge is not None:
                            arm.charge_usd = run.charge.amount.amount
                            arm.charge_final = run.charge.final
                        receipt.save(path)
                        if arm.charge_usd is not None:
                            check_charge(arm.charge_usd)
                        if run.finished:
                            if run.state != "completed":
                                arm.status = "failed"
                                receipt.save(path)
                                raise RuntimeError("run did not complete; receipt retained")
                            break
                        await asyncio.sleep(30)
                    else:
                        raise RuntimeError("run has not reached finality")
                    (
                        arm.charge_usd,
                        arm.charge_final,
                        arm.output_id,
                        arm.output_sha256,
                        arm.output_file,
                    ) = await verify_surface(client, session, arm, receipt.nonce, path)
                    receipt.save(path)
                    check_charge(arm.charge_usd)
                    arm.answer_verified = True
                    if not arm.charge_final:
                        receipt.save(path)
                        raise RuntimeError("charge is not final; resume the same receipt")
                    arm.status = "passed"
                    arm.failure_kind = None
                    arm.failure_code = None
                    receipt.save(path)
                    print(
                        f"{arm.surface}: passed run={arm.run_id} charge={arm.charge_usd}",
                        flush=True,
                    )
                except (Exception, asyncio.CancelledError) as error:
                    arm.failure_kind = type(error).__name__
                    arm.failure_code = error.code if isinstance(error, SurfaceFailure) else None
                    receipt.save(path)
                    duplicate_identity = any(
                        other is not arm and other.run_id == arm.run_id for other in receipt.arms
                    )
                    if (
                        arm.run_id is not None
                        and not arm.execution_closed
                        and not duplicate_identity
                    ):
                        try:
                            await cancel_and_reconcile(client, receipt, arm, path)
                        except Exception:
                            # The retained cancel intent must be reconciled before another admission.
                            receipt.save(path)
                    raise


async def cancel_outstanding(receipt: Receipt, path: Path) -> None:
    async with AsyncSynthClient() as client:
        account = await client.account.get()
        if account.org.id != receipt.fixture_org_id:
            raise ValueError("credential is not scoped to the retained fixture organization")
        for arm in receipt.arms:
            if arm.run_id is not None and not (
                arm.execution_closed and arm.accounting_final and arm.cleanup_complete
            ):
                await cancel_and_reconcile(client, receipt, arm, path)
        if any(arm.status == "launch_pending" and arm.run_id is None for arm in receipt.arms):
            raise RuntimeError("unknown launch must be reconciled using the retained key")


async def run_with_signals(receipt: Receipt, path: Path, cancel_only: bool) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("acceptance task is unavailable")
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        if cancel_only:
            await cancel_outstanding(receipt, path)
        else:
            validate_evidence(receipt, path)
            await execute(receipt, path)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cancel-outstanding",
        action="store_true",
        help="reconcile known runs without admitting new ones",
    )
    parser.add_argument("--fixture-org", required=True, help="dedicated test organization ID")
    parser.add_argument(
        "--cohort-file", type=Path, required=True, help="frozen content-addressed cohort record"
    )
    args = parser.parse_args()
    if args.cancel_outstanding and not args.resume:
        raise ValueError("cancellation requires an existing receipt")
    args.output = args.output.resolve()
    target = target_url()
    cohort_bytes = args.cohort_file.read_bytes()
    if not cohort_bytes:
        raise ValueError("cohort record is empty")
    cohort_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output.with_name(args.output.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            receipt = Receipt.load(args.output)
        else:
            if args.output.exists():
                raise ValueError("receipt exists; resume the existing intent")
            receipt = new_receipt(target, args.fixture_org, cohort_sha256)
            receipt.save(args.output)
        receipt.validate_intent(target, args.fixture_org, cohort_sha256)
        print("quickstart maximum: $12.00; expected actual: unknown until runs settle", flush=True)
        try:
            asyncio.run(run_with_signals(receipt, args.output, args.cancel_outstanding))
        except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
            # Do not print response bodies, credentials, URLs or task content.
            print(
                f"quickstart incomplete: {type(error).__name__}; retain and resume receipt",
                file=sys.stderr,
            )
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
