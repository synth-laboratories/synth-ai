"""Register reusable Craftax input templates, never a run-owned execution pool."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from synth_ai import AsyncSynthClient
from synth_ai.errors import SynthError
from synth_ai.models import (
    EvaluationDataset,
    EvaluationEvaluator,
    EvaluationEvaluatorResultFormat,
    EvaluationInput,
    EvaluationLimits,
    EvaluationPool,
    EvaluationPoolProvider,
    EvaluationTask,
    EvaluationTaskInstance,
)

from .run import endpoint, error_code

HERE = Path(__file__).resolve().parent
TASK = "craftax-classic-enginebench-policy-v1"
EVALUATOR_IMAGE = "ghcr.io/synth-laboratories/synth-evaluation-benchmark-verifier@sha256:bdbd01e0d4445ae4ad1131b78a4430341d541eabe41da367a2eab3807d780814"
CANDIDATE_IMAGE = "ghcr.io/synth-laboratories/synth-evaluation-benchmark-candidate@sha256:5a4ba6a750c672333fad1954b4b64d1e32c9e414213d5e1ec500c36de593f7b7"


class Prepared(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["synth.craftax-template.v1"] = "synth.craftax-template.v1"
    base_url: str
    evaluator_revision_id: str
    dataset_revision_id: str
    template_pool_revision_id: str
    baseline_sha256: str
    field_sha256: str
    task_sha256: str
    provider: str
    region: str
    execution_pool_created: Literal[False] = False
    compute_started: Literal[False] = False


def checked_source() -> tuple[str, EvaluationDataset, str]:
    sources = json.loads((HERE / "sources.json").read_text())
    baseline = (HERE / "baseline.txt").read_bytes()
    raw_field = (HERE / "field.json").read_bytes()
    if hashlib.sha256(baseline).hexdigest() != sources["baseline"]["sha256"]:
        raise ValueError("baseline_source_digest_mismatch")
    field_digest = hashlib.sha256(raw_field).hexdigest()
    if field_digest != sources["field"]["sha256"]:
        raise ValueError("field_source_digest_mismatch")
    field = json.loads(raw_field)
    if field["max_steps"] != 250 or len(field["seeds"]) != 256:
        raise ValueError("craftax_public_field_shape_changed")
    # First 100 public-development instances, not held-out benchmark evidence.
    dataset = EvaluationDataset(
        name="craftax-public-development-v3-confirm-100",
        instances=[
            EvaluationTaskInstance(id=f"episode-{index}", seed=seed, inputs={"max_steps": 250})
            for index, seed in enumerate(field["seeds"][:100])
        ],
    )
    return baseline.decode(), dataset, field_digest


def task_text(baseline: str, evaluator: str, dataset: str) -> str:
    digest = hashlib.sha256(baseline.encode()).hexdigest()
    return f"""Run CyberneticsBench v2 Craftax Classic, task {TASK}.
Create this swarm's own project through forge_project_create. Never reuse a project.
The planner must call evaluation_pool_register with name craftax, evaluator_revision_id
{evaluator}, and concurrency 16 to create a fresh run-owned evaluation pool and evaluator.
Use the returned evaluator revision for every evaluation. Do not create or deploy legacy pools.
The grant's template is reusable configuration, not an execution pool.

Use dataset_revision_id {dataset}. It is the first 100 public-development-v3
instances episode-0 through episode-99, 250 steps each. This is public development,
not sealed or held-out evidence. SCREEN uses episode-0 through episode-7.
CONFIRM uses all 100 instances. Never change the task, evaluator, field or step limit.

Register the exact baseline below as name baseline, sole file policy.py, command
["python", "-I", "-B", "-u", "/opt/policy_worker.py", "/candidate/policy.py"].
Expected file SHA256: {digest}. Verify that identity before evaluating.
Implement reset(context) and act(observation). act returns {{"actions": [name]}};
observations expose local views and the full action catalog, not the seed.
Baseline policy.py:
```python
{baseline}```

Develop up to three distinct candidate policies. SCREEN each on the eight instances,
maximum_charge_micros 290000, max_concurrency 8. Reserve enough budget for two CONFIRMs.
Then evaluate baseline and best valid candidate on the same 100 instances,
maximum_charge_micros 3600000 per evaluation, max_concurrency 16.
Register each policy immutably before evaluation. Poll existing evaluations; never
relaunch an accepted evaluation because observation took time. No policy errors may
be counted as successful episodes. If a policy fails, repair during SCREEN.

Use the evaluation metadata and signed attestations, not an invented score.
Publish both exact policy files and the baseline/candidate evaluation ids, revisions,
100-episode means, policy_error metrics, and signed evidence digests. Report a failure
honestly if there is no valid candidate. Do not claim an improvement if there is none.
The baseline is the repository's deterministic noop control, not the stronger policy
used by previous private staging experiments, so their numerical scores are not comparable.
After independent task review, close the objective with its independent verdict and
conclude. Leave run-owned pool cleanup to the platform and wait for final settlement.
"""


async def register(
    client: AsyncSynthClient, *, base_url: str, provider: EvaluationPoolProvider, region: str
) -> tuple[Prepared, str]:
    baseline, dataset, field_digest = checked_source()
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"max_steps": {"type": "integer", "minimum": 1, "maximum": 1000}},
        "required": ["max_steps"],
        "additionalProperties": False,
    }

    async def put(
        kind: Literal["task", "pool", "evaluator", "dataset"], payload: EvaluationInput
    ) -> str:
        digest = hashlib.sha256(payload.model_dump_json(exclude_unset=True).encode()).hexdigest()
        revision = await client.evaluations.register(
            kind, payload, idempotency_key="craftax-template:" + kind + ":" + digest
        )
        return revision.revision_id

    task = await put(
        "task",
        EvaluationTask(
            name=TASK,
            description="Implement reset(context) and act(observation); return one legal Craftax Classic action. Maximize provisional achievement score.",
            candidate_contract={
                "entrypoints": ["reset(context)", "act(observation)"],
                "transport": "json-lines",
                "runtime": "python3.12",
                "sdk_modules": ["craftax_sdk"],
            },
            input_schema=schema,
        ),
    )
    pool = await put(
        "pool",
        EvaluationPool(
            name="craftax-template-configuration", provider=provider, region=region, concurrency=16
        ),
    )
    evaluator = await put(
        "evaluator",
        EvaluationEvaluator(
            name=TASK,
            task_id=TASK,
            task_digest=task,
            pool_id=pool,
            image=EVALUATOR_IMAGE,
            candidate_image=CANDIDATE_IMAGE,
            command=["python", "-u", "-m", "cybernetics_evaluators.episode", TASK],
            input_schema=schema,
            region=region,
            result_format=EvaluationEvaluatorResultFormat.SYNTH_EVALUATION_RESULT_V2,
            limits=EvaluationLimits(
                timeout_seconds=120,
                cpu_cores=1,
                memory_mb=512,
                storage_mb=524288 if provider == EvaluationPoolProvider.MODAL else 512,
            ),
        ),
    )
    dataset_revision = await put("dataset", dataset)
    text = task_text(baseline, evaluator, dataset_revision)
    return Prepared(
        base_url=base_url,
        evaluator_revision_id=evaluator,
        dataset_revision_id=dataset_revision,
        template_pool_revision_id=pool,
        baseline_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
        field_sha256=field_digest,
        task_sha256=hashlib.sha256(text.encode()).hexdigest(),
        provider=provider.value,
        region=region,
    ), text


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=endpoint)
    parser.add_argument("--provider", required=True, choices=["docker", "modal"])
    parser.add_argument("--region", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    checked_source()
    if not args.execute:
        print(json.dumps({"validated": True, "network_calls": 0, "compute_started": False}))
        return
    if args.output.exists():
        raise ValueError("output_exists_choose_new_path_or_reuse_prepared_template")
    async with AsyncSynthClient(
        os.environ[args.api_key_env], base_url=args.base_url, retries=0
    ) as client:
        prepared, text = await register(
            client,
            base_url=args.base_url,
            provider=EvaluationPoolProvider(args.provider),
            region=args.region,
        )
    args.output.mkdir(parents=True)
    (args.output / "template.json").write_text(prepared.model_dump_json(indent=2) + "\n")
    (args.output / "task.txt").write_text(text)
    print(prepared.model_dump_json(indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (SynthError, OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "code": error_code(error),
                    "compute_started": False,
                }
            )
        )
        raise SystemExit(2) from None
