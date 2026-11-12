"""Register and evaluate a customer task using only the public SDK and API key."""

import argparse
import json
import os
from pathlib import Path

from pydantic import JsonValue

from synth_ai import SynthClient
from synth_ai.models import (
    EvaluationDataset,
    EvaluationEvaluationRequest,
    EvaluationEvaluator,
    EvaluationLimits,
    EvaluationPolicy,
    EvaluationPool,
    EvaluationPoolProvider,
    EvaluationTask,
    EvaluationTaskInstance,
)

ROOT = Path(__file__).resolve().parent


def run(
    client: SynthClient,
    *,
    candidate_image: str,
    evaluator_image: str,
    provider: EvaluationPoolProvider,
    region: str,
    maximum_charge_micros: int,
    operation: str,
) -> dict[str, JsonValue]:
    api = client.swarms
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"values": {"type": "array", "maxItems": 64, "items": {"type": "integer"}}},
        "required": ["values"],
        "additionalProperties": False,
    }
    task = api.register_evaluation_input(
        "task",
        EvaluationTask(
            name="customer-sorted-integers",
            description=(ROOT / "instruction.md").read_text(),
            candidate_contract={
                "transport": "json_lines",
                "input": {"values": [2, 1, 2]},
                "output": {"values": [1, 2]},
            },
            input_schema=schema,
        ),
    )
    pool = api.register_evaluation_input(
        "pool",
        EvaluationPool(
            name="customer-integer-pool", provider=provider, concurrency=3, region=region
        ),
    )
    evaluator = api.register_evaluation_input(
        "evaluator",
        EvaluationEvaluator(
            name="customer-integer-verifier",
            image=evaluator_image,
            candidate_image=candidate_image,
            command=["python", "-u", "/tests/bridge.py"],
            pool_id=pool.revision_id,
            task_id="customer-sorted-integers",
            task_digest=task.revision_id,
            input_schema=schema,
            region=region,
            limits=EvaluationLimits(
                timeout_seconds=30,
                cpu_cores=1,
                memory_mb=256,
                storage_mb=512 * 1024 if provider == "modal" else 512,
            ),
        ),
    )
    cases: list[list[JsonValue]] = [[5, -2, 5, 0, 9, -2], [], [7, 7, 7]]
    dataset = api.register_evaluation_input(
        "dataset",
        EvaluationDataset(
            name="customer-integer-cases",
            instances=[
                EvaluationTaskInstance(id=f"case-{i}", seed=i, inputs={"values": values})
                for i, values in enumerate(cases)
            ],
        ),
    )
    policy = api.register_evaluation_input(
        "policy",
        EvaluationPolicy(
            name="customer-sort-policy",
            files={"sort_stream.py": (ROOT / "candidate.py").read_text()},
            command=["python", "-I", "-u", "/candidate/sort_stream.py"],
        ),
    )
    request = EvaluationEvaluationRequest(
        policy_revision_id=policy.revision_id,
        evaluator_revision_id=evaluator.revision_id,
        dataset_revision_id=dataset.revision_id,
        max_concurrency=3,
    )
    quote = api.preflight_evaluation(request, maximum_charge_micros=maximum_charge_micros)
    if quote.maximum_charge_micros > maximum_charge_micros:
        raise RuntimeError("quoted charge exceeds the requested cap")
    evaluation = api.evaluate(
        request, maximum_charge_micros=maximum_charge_micros, idempotency_key=operation
    )
    result = api.wait_evaluation(evaluation.evaluation_id, timeout=180, poll_interval=2)
    document = result.model_dump(mode="json", exclude_none=True)
    if result.status != "completed" or result.attestation is None:
        raise RuntimeError(json.dumps(document))
    evidence = document["attestation"]["evidence"]
    if (
        evidence["evaluation_id"] != evaluation.evaluation_id
        or evidence["request"]["evaluator_revision_id"] != evaluator.revision_id
        or evidence["aggregate"]["score"] != 1.0
    ):
        raise RuntimeError("evaluation evidence differs from the submitted task or expected score")
    return {
        "quote": quote.model_dump(mode="json"),
        "revisions": {
            kind: rev.revision_id
            for kind, rev in (
                ("task", task),
                ("pool", pool),
                ("evaluator", evaluator),
                ("dataset", dataset),
                ("policy", policy),
            )
        },
        "evaluation": document,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--candidate-image", required=True)
    p.add_argument("--evaluator-image", required=True)
    p.add_argument("--provider", choices=("docker", "modal"), required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--maximum-charge-micros", type=int, required=True)
    p.add_argument("--operation", required=True)
    args = p.parse_args()
    with SynthClient(os.environ["SYNTH_API_KEY"], base_url=args.url) as client:
        result = run(
            client,
            candidate_image=args.candidate_image,
            evaluator_image=args.evaluator_image,
            provider=EvaluationPoolProvider(args.provider),
            region=args.region,
            maximum_charge_micros=args.maximum_charge_micros,
            operation=args.operation,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
