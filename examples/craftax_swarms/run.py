"""Launch one Craftax swarm through the public SDK; see README.md for acceptance scope."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from synth_ai import AsyncSynthClient, CompletionMilestone, __version__
from synth_ai.errors import SwarmsAPIError, SynthError
from synth_ai.models import LaunchRequest, Limits, OwnerResourceRequest, RunReport, RunSnapshot

MAXIMUM_BUDGET_USD = Decimal("50")
MAXIMUM_TASK_BYTES = 32768


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)
    schema_version: Literal["synth.craftax-sdk-run.v1"] = "synth.craftax-sdk-run.v1"
    base_url: str
    sdk_version: str
    request_sha256: str
    cohort_sha256: str
    maximum_usd: str
    idempotency_key: str
    created_at: str
    run_id: str | None = None
    state: str | None = None
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    problems_observed: bool = False
    problem_codes: list[str] = Field(default_factory=list)
    final_charge_usd: str | None = None
    failed_model_calls: int | None = None
    executing_ms: int | None = None
    evaluation_ids: list[str] = Field(default_factory=list)
    lifecycle_clean: bool = False
    # Run closure does not attest the benchmark pairing or installed service identities.
    benchmark_pair_verified: Literal[False] = False
    installed_cohort_verified: Literal[False] = False
    error_type: str | None = None
    error_code: str | None = None
    observed_at: str | None = None


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if (
        (parsed.scheme != "https" and not local)
        or parsed.hostname in {None, ""}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("base_url_requires_https_or_loopback_without_credentials_or_path")
    return value.rstrip("/")


def request(
    task: str, budget: str, wall_seconds: int, evaluation_micros: int, template: str | None = None
) -> LaunchRequest:
    try:
        amount = Decimal(budget)
    except InvalidOperation as error:
        raise ValueError("run_budget_must_be_decimal_usd") from error
    if not amount.is_finite() or not Decimal("0") < amount <= MAXIMUM_BUDGET_USD:
        raise ValueError("run_budget_must_be_positive_and_at_most_50_usd")
    if amount * 1_000_000 != (amount * 1_000_000).to_integral_value():
        raise ValueError("run_budget_requires_whole_micros")
    if not 0 < evaluation_micros <= amount * 1_000_000:
        raise ValueError("evaluation_cap_must_fit_run_budget")
    if not 1 <= wall_seconds <= 3600:
        raise ValueError("wall_seconds_must_be_between_1_and_3600")
    if not task.strip() or len(task.encode("utf-8")) > MAXIMUM_TASK_BYTES:
        raise ValueError("task_must_be_nonempty_and_at_most_32768_bytes")
    return LaunchRequest(
        message=task,
        profile="swarms-v2",
        limits=Limits.model_validate(
            {
                "maximum_charge": {"amount": budget, "currency": "USD"},
                "wall_time_seconds": wall_seconds,
                "parallel_workers": 3,
            }
        ),
        resources=[
            OwnerResourceRequest.model_validate(
                {
                    "plugin": "pool",
                    "roles": ["planner", "worker", "reviewer"],
                    "limits": {"spend_micros": evaluation_micros, "rollouts": 240},
                    "selection": {},
                    **({"evaluation_templates": [template]} if template is not None else {}),
                }
            )
        ],
    )


def error_code(error: BaseException) -> str:
    if isinstance(error, SwarmsAPIError):
        return error.problem.code
    if type(error) is ValueError:
        # Our local ValueErrors carry stable codes, never request or response bodies.
        return str(error)
    return type(error).__name__


def save(path: Path, receipt: Receipt) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(receipt.model_dump_json(indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare(
    path: Path, *, base_url: str, launch: LaunchRequest, cohort_sha256: str, resume: bool
) -> Receipt:
    digest = hashlib.sha256(launch.model_dump_json(exclude_unset=True).encode()).hexdigest()
    if resume:
        receipt = Receipt.model_validate_json(path.read_text())
        if (
            receipt.base_url,
            receipt.request_sha256,
            receipt.cohort_sha256,
            receipt.sdk_version,
        ) != (base_url, digest, cohort_sha256, __version__):
            raise ValueError("resume_identity_changed")
        if (
            receipt.run_id is None
            and (datetime.now(UTC) - datetime.fromisoformat(receipt.created_at)).total_seconds()
            >= 23 * 3600
        ):
            raise ValueError("unknown_admission_requires_reconciliation_before_idempotency_expiry")
        return receipt
    if path.exists():
        raise ValueError("receipt_exists_use_resume")
    assert launch.limits is not None
    receipt = Receipt(
        base_url=base_url,
        sdk_version=__version__,
        request_sha256=digest,
        cohort_sha256=cohort_sha256,
        maximum_usd=launch.limits.maximum_charge.amount,
        idempotency_key="craftax-sdk:" + str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
    )
    save(path, receipt)
    return receipt


def observe(receipt: Receipt, snapshot: RunSnapshot, report: RunReport) -> None:
    if snapshot.run_id != receipt.run_id or report.run_id != receipt.run_id:
        raise ValueError("observation_run_identity_mismatch")
    receipt.state = snapshot.state
    receipt.execution_closed = snapshot.execution_closed
    receipt.accounting_final = snapshot.accounting_final
    receipt.cleanup_complete = snapshot.cleanup_complete
    receipt.problems_observed = snapshot.problems_observed
    receipt.problem_codes = [] if snapshot.problems is None else [p.code for p in snapshot.problems]
    receipt.final_charge_usd = (
        snapshot.charge.amount.amount
        if snapshot.charge is not None and snapshot.charge.final
        else None
    )
    receipt.failed_model_calls = (
        sum(row.failed_calls for row in report.by_model) + report.unattributed.failed_calls
    )
    receipt.executing_ms = report.timings.executing_ms
    receipt.evaluation_ids = (
        [] if report.evaluations is None else [row.evaluation_id for row in report.evaluations]
    )
    receipt.lifecycle_clean = (
        snapshot.state == "completed"
        and snapshot.finished
        and snapshot.problems_observed
        and snapshot.problems == []
        and receipt.final_charge_usd is not None
        and Decimal(receipt.final_charge_usd) <= Decimal(receipt.maximum_usd)
        and receipt.failed_model_calls == 0
    )
    receipt.observed_at = datetime.now(UTC).isoformat()


async def execute(
    client: AsyncSynthClient, *, path: Path, receipt: Receipt, launch: LaunchRequest, timeout: float
) -> None:
    receipt.error_type = None
    receipt.error_code = None
    save(path, receipt)
    try:
        if receipt.run_id is None:
            admitted = await client.swarms.launch_request(
                launch, idempotency_key=receipt.idempotency_key
            )
            receipt.run_id = admitted.run_id
            save(path, receipt)
        snapshot = await client.swarms.wait(
            receipt.run_id, timeout=timeout, until=CompletionMilestone.FINISHED
        )
        report = await client.swarms.report(receipt.run_id)
        observe(receipt, snapshot, report)
    except (SynthError, OSError, ValueError) as error:
        receipt.error_type = type(error).__name__
        receipt.error_code = error_code(error)
        save(path, receipt)
        raise
    save(path, receipt)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=endpoint)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--template-file", required=True, type=Path)
    parser.add_argument("--cohort-file", required=True, type=Path)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--evaluation-micros", required=True, type=int)
    parser.add_argument("--wall-seconds", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    task = args.task_file.read_text()
    template = None
    if args.template_file is not None:
        from .prepare import Prepared

        prepared = Prepared.model_validate_json(args.template_file.read_text())
        if (
            prepared.base_url != args.base_url
            or prepared.task_sha256 != hashlib.sha256(task.encode()).hexdigest()
        ):
            raise ValueError("prepared_template_environment_or_task_mismatch")
        template = prepared.evaluator_revision_id
    launch = request(task, args.budget, args.wall_seconds, args.evaluation_micros, template)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("observation_timeout_must_be_positive")
    cohort_digest = hashlib.sha256(args.cohort_file.read_bytes()).hexdigest()
    if not args.execute:
        print(json.dumps({"validated": True, "maximum_usd": args.budget, "network_calls": 0}))
        return 0
    secret = os.environ[args.api_key_env]
    if not secret:
        raise ValueError("api_key_environment_variable_empty")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.with_suffix(args.receipt.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = prepare(
            args.receipt,
            base_url=args.base_url,
            launch=launch,
            cohort_sha256=cohort_digest,
            resume=args.resume,
        )
        async with AsyncSynthClient(secret, base_url=args.base_url, retries=0) as client:
            await execute(
                client, path=args.receipt, receipt=receipt, launch=launch, timeout=args.timeout
            )
        print(receipt.model_dump_json(indent=2))
    return 0 if receipt.lifecycle_clean else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (SynthError, OSError, ValueError, KeyError) as error:
        # Arbitrary error bodies can contain provider data; retain only the class.
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "code": error_code(error),
                    "resume_required": True,
                }
            )
        )
        raise SystemExit(2) from None
