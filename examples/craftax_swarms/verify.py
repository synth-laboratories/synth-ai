"""Verify paired Craftax metadata through public report/rollout-summary reads only."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from synth_ai import AsyncSynthClient
from synth_ai.errors import SynthError
from synth_ai.models import EvaluationEvaluator, EvaluationRolloutSummary, RunEvaluation, RunReport

from .prepare import Prepared, checked_source
from .run import Receipt, endpoint, error_code, observe

EXPECTED_INSTANCES = frozenset(f"episode-{index}" for index in range(100))
MAXIMUM_SUMMARY_PAGES = 100


class Aggregate(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    n: Literal[100]
    aggregation: Literal["mean"]
    score: float


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    run_id: str
    baseline_evaluation_id: str
    candidate_evaluation_id: str
    baseline_policy_revision_id: str
    candidate_policy_revision_id: str
    evaluator_revision_id: str
    dataset_revision_id: str
    baseline_attestation_digest: str
    candidate_attestation_digest: str
    baseline_mean: float
    candidate_mean: float
    improvement: float
    instances: Literal[100] = 100
    policy_errors: Literal[0] = 0
    policy_timeouts: Literal[0] = 0
    pair_metadata_verified: Literal[True] = True
    # The authenticated API exposes trusted signed-evidence references. This
    # checker does not fetch signer keys or claim local signature verification.
    signatures_verified_locally: Literal[False] = False


class Instance(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    instance_id: str
    outcome: Literal["scored"]
    score: float | str


def baseline_revision() -> str:
    baseline, _, _ = checked_source()
    payload = {
        "schema_version": "synth.policy.v1",
        "name": "baseline",
        "files": {"policy.py": baseline},
        "command": ["python", "-I", "-B", "-u", "/opt/policy_worker.py", "/candidate/policy.py"],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def candidates(report: RunReport, prepared: Prepared) -> tuple[RunEvaluation, list[RunEvaluation]]:
    if report.evaluations is None:
        raise ValueError("run_evaluation_metadata_unavailable")
    baseline_id = baseline_revision()
    confirmed = [
        row
        for row in report.evaluations
        if row.status == "completed"
        and row.dataset_revision_id == prepared.dataset_revision_id
        and row.aggregate is not None
        and row.aggregate.get("n") == 100
    ]
    baseline = [row for row in confirmed if row.policy_revision_id == baseline_id]
    if len(baseline) != 1:
        raise ValueError("exactly_one_source_bound_baseline_confirmation_required")
    choices = [
        row
        for row in confirmed
        if row.policy_revision_id != baseline_id
        and row.evaluator_revision_id == baseline[0].evaluator_revision_id
    ]
    if not choices:
        raise ValueError("matching_candidate_confirmation_missing")
    for row in [baseline[0], *choices]:
        Aggregate.model_validate(row.aggregate)
        if row.attestation_digest is None:
            raise ValueError("evaluation_attestation_reference_missing")
    return baseline[0], choices


def summary_instances(summary: EvaluationRolloutSummary, row: RunEvaluation) -> set[str]:
    if (
        summary.evaluation_id != row.evaluation_id
        or summary.status != "completed"
        or summary.attestation_digest != row.attestation_digest
    ):
        raise ValueError("evaluation_summary_identity_mismatch")
    counts = summary.outcome_counts
    if counts.total != 100 or counts.policy_error != 0 or counts.policy_timeout != 0:
        raise ValueError("evaluation_confirmation_has_missing_or_failed_instances")
    if summary.columns != ["instance_id", "outcome", "score", "metrics"]:
        raise ValueError("evaluation_summary_columns_changed")
    instances = [
        Instance.model_validate(dict(zip(summary.columns, values, strict=True)))
        for values in summary.instances
    ]
    names = {instance.instance_id for instance in instances}
    if len(names) != len(instances) or not names <= EXPECTED_INSTANCES:
        raise ValueError("evaluation_summary_instances_invalid")
    return names


async def confirm(client: AsyncSynthClient, row: RunEvaluation) -> None:
    observed: set[str] = set()
    seen_cursors: set[str] = set()
    cursor = None
    for _ in range(MAXIMUM_SUMMARY_PAGES):
        summary = await client.evaluations.rollout_summary(row.evaluation_id, cursor=cursor)
        instances = summary_instances(summary, row)
        if observed & instances:
            raise ValueError("evaluation_summary_duplicate_page_instances")
        observed |= instances
        if summary.next_cursor is None:
            if summary.truncated or observed != EXPECTED_INSTANCES:
                raise ValueError("evaluation_summary_incomplete")
            return
        if summary.next_cursor in seen_cursors:
            raise ValueError("evaluation_summary_cursor_cycle")
        cursor = summary.next_cursor
        seen_cursors.add(cursor)
    raise ValueError("evaluation_summary_page_limit_exceeded")


async def verify_template(client: AsyncSynthClient, prepared: Prepared, evaluator_id: str) -> None:
    template = await client.evaluations.get_input("evaluator", prepared.evaluator_revision_id)
    owned = await client.evaluations.get_input("evaluator", evaluator_id)
    if not isinstance(template.payload, EvaluationEvaluator) or not isinstance(
        owned.payload, EvaluationEvaluator
    ):
        raise ValueError("evaluation_registry_payload_kind_mismatch")
    if template.revision_id != prepared.evaluator_revision_id or owned.revision_id != evaluator_id:
        raise ValueError("evaluation_registry_revision_mismatch")
    if template.payload.pool_id != prepared.template_pool_revision_id:
        raise ValueError("template_pool_binding_mismatch")
    if owned.payload.pool_id == prepared.template_pool_revision_id:
        raise ValueError("evaluation_reused_template_pool")
    if template.payload.model_dump(exclude={"name", "pool_id"}) != owned.payload.model_dump(
        exclude={"name", "pool_id"}
    ):
        raise ValueError("evaluation_differs_from_granted_template")


async def verify(client: AsyncSynthClient, receipt: Receipt, prepared: Prepared) -> Pair:
    if receipt.run_id is None or not receipt.lifecycle_clean:
        raise ValueError("clean_run_receipt_required")
    if receipt.base_url != prepared.base_url:
        raise ValueError("template_environment_mismatch")
    report = await client.swarms.report(receipt.run_id)
    if report.run_id != receipt.run_id:
        raise ValueError("report_run_identity_mismatch")
    snapshot = await client.swarms.get(receipt.run_id)
    observe(receipt, snapshot, report)
    if not receipt.lifecycle_clean:
        raise ValueError("live_run_no_longer_clean")
    baseline, choices = candidates(report, prepared)
    await verify_template(client, prepared, baseline.evaluator_revision_id)
    await confirm(client, baseline)
    for row in choices:
        await confirm(client, row)
    candidate = max(choices, key=lambda row: Aggregate.model_validate(row.aggregate).score)
    baseline_score = Aggregate.model_validate(baseline.aggregate).score
    candidate_score = Aggregate.model_validate(candidate.aggregate).score
    assert baseline.attestation_digest is not None and candidate.attestation_digest is not None
    return Pair(
        run_id=receipt.run_id,
        baseline_evaluation_id=baseline.evaluation_id,
        candidate_evaluation_id=candidate.evaluation_id,
        baseline_policy_revision_id=baseline.policy_revision_id,
        candidate_policy_revision_id=candidate.policy_revision_id,
        evaluator_revision_id=baseline.evaluator_revision_id,
        dataset_revision_id=baseline.dataset_revision_id,
        baseline_attestation_digest=baseline.attestation_digest,
        candidate_attestation_digest=candidate.attestation_digest,
        baseline_mean=baseline_score,
        candidate_mean=candidate_score,
        improvement=candidate_score - baseline_score,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=endpoint)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--template-file", required=True, type=Path)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = Receipt.model_validate_json(args.receipt.read_text())
    prepared = Prepared.model_validate_json(args.template_file.read_text())
    if receipt.base_url != args.base_url:
        raise ValueError("receipt_environment_mismatch")
    async with AsyncSynthClient(
        os.environ[args.api_key_env], base_url=args.base_url, retries=0
    ) as client:
        pair = await verify(client, receipt, prepared)
    args.output.write_text(pair.model_dump_json(indent=2) + "\n")
    print(pair.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (SynthError, OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "code": error_code(error),
                    "pair_metadata_verified": False,
                }
            )
        )
        raise SystemExit(2) from None
