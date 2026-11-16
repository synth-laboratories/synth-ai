import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from examples.craftax_swarms import run
from synth_ai import AsyncSynthClient
from synth_ai.errors import TransportFailureError
from synth_ai.models import RunReport, RunSnapshot


def example(contract, path):
    return copy.deepcopy(
        contract["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["example"]
    )


def prepared(tmp_path):
    path = tmp_path / "receipt.json"
    launch = run.request(
        "Use agent MCP tools to create your own Craftax project and pool.", "14.00", 2400, 8_650_000
    )
    receipt = run.prepare(
        path, base_url="http://127.0.0.1:8000", launch=launch, cohort_sha256="a" * 64, resume=False
    )
    return path, launch, receipt


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "https://example.com/v1",
        "https://example.com?token=secret",
    ],
)
def test_credentials_and_nonlocal_plaintext_are_refused(url):
    with pytest.raises(ValueError):
        run.endpoint(url)


@pytest.mark.parametrize(
    "budget,cap", [("51", 1), ("0", 1), ("NaN", 1), ("1.0000001", 1), ("1", 1_000_001)]
)
def test_invalid_charge_bounds_refuse_before_network(budget, cap):
    with pytest.raises(ValueError):
        run.request("task", budget, 600, cap)


def test_launch_has_no_existing_pool_or_project(tmp_path):
    _, launch, _ = prepared(tmp_path)
    wire = launch.model_dump(mode="json", exclude_unset=True)
    assert wire["resources"][0]["selection"] == {}
    assert "project_id" not in wire
    assert "research_task" not in wire


@pytest.mark.parametrize("change", ["request", "endpoint", "cohort"])
def test_resume_refuses_identity_changes(tmp_path, change):
    path, launch, _ = prepared(tmp_path)
    if change == "request":
        launch = run.request("different", "14.00", 2400, 8_650_000)
    with pytest.raises(ValueError, match="resume_identity_changed"):
        run.prepare(
            path,
            base_url="https://example.com" if change == "endpoint" else "http://127.0.0.1:8000",
            launch=launch,
            cohort_sha256=("b" if change == "cohort" else "a") * 64,
            resume=True,
        )


def test_unknown_admission_expiry_never_launches_again(tmp_path):
    path, launch, receipt = prepared(tmp_path)
    receipt.created_at = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    run.save(path, receipt)
    with pytest.raises(ValueError, match="requires_reconciliation"):
        run.prepare(
            path,
            base_url=receipt.base_url,
            launch=launch,
            cohort_sha256=receipt.cohort_sha256,
            resume=True,
        )


async def test_only_public_sdk_launch_and_resume_retains_identity(tmp_path):
    path, launch, receipt = prepared(tmp_path)
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout("body must never be retained", request=request)

    async with AsyncSynthClient(
        "fixture",
        base_url=receipt.base_url,
        retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        for _ in range(2):
            with pytest.raises(TransportFailureError):
                await run.execute(client, path=path, receipt=receipt, launch=launch, timeout=1)
            receipt = run.prepare(
                path,
                base_url=receipt.base_url,
                launch=launch,
                cohort_sha256=receipt.cohort_sha256,
                resume=True,
            )
    assert len(requests) == 2
    assert all(r.url.path == "/v1/runs" and r.method == "POST" for r in requests)
    assert requests[0].headers["idempotency-key"] == requests[1].headers["idempotency-key"]
    assert requests[0].content == requests[1].content
    assert "body must never" not in path.read_text()
    assert "Use agent MCP" not in path.read_text()


@pytest.mark.parametrize(
    "missing",
    ["execution_closed", "accounting_final", "cleanup_complete", "problems_observed", "charge"],
)
def test_completion_without_finality_is_not_clean(contract, tmp_path, missing):
    _, _, receipt = prepared(tmp_path)
    snapshot = example(contract, "/v1/runs/{run_id}")
    snapshot.update(
        state="completed",
        execution_closed=True,
        accounting_final=True,
        cleanup_complete=True,
        problems_observed=True,
        problems=[],
        charge={"amount": {"amount": "1.00", "currency": "USD"}, "final": True},
    )
    snapshot[missing] = None if missing == "charge" else False
    report = example(contract, "/v1/runs/{run_id}/report")
    report["run_id"] = snapshot["run_id"]
    receipt.run_id = snapshot["run_id"]
    run.observe(receipt, RunSnapshot.model_validate(snapshot), RunReport.model_validate(report))
    assert not receipt.lifecycle_clean
    assert not receipt.benchmark_pair_verified
    assert not receipt.installed_cohort_verified
    # Typed report summaries must not retain arbitrary upstream problem bodies.
    assert "message" not in json.loads(receipt.model_dump_json())


async def test_prepare_uses_only_immutable_public_registration(tmp_path):
    import hashlib

    from examples.craftax_swarms import prepare
    from synth_ai.models import EvaluationPoolProvider

    requests = []

    def handle(request):
        requests.append(request)
        payload = json.loads(request.content)
        kind = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(
            201,
            json={
                "kind": kind,
                "revision_id": "sha256:" + hashlib.sha256(request.content).hexdigest(),
                "name": payload["name"],
                "payload": payload,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://fixture.test", http_client=http
        ) as client:
            prepared, task = await prepare.register(
                client,
                base_url="https://fixture.test",
                provider=EvaluationPoolProvider.MODAL,
                region="us",
            )
    assert [r.url.path for r in requests] == [
        "/v1/evaluation-inputs/" + k for k in ("task", "pool", "evaluator", "dataset")
    ]
    assert all(r.method == "POST" for r in requests)
    assert not prepared.compute_started and not prepared.execution_pool_created
    assert "evaluation_pool_register" in task
    assert "forge_project_create" in task
    assert "candidate" in task
    launch = run.request(task, "14", 2400, 8_650_000, prepared.evaluator_revision_id)
    assert launch.resources[0].selection.pool_ids is None
    assert launch.resources[0].evaluation_templates == [prepared.evaluator_revision_id]
    assert hashlib.sha256(task.encode()).hexdigest() == prepared.task_sha256


def test_public_field_and_baseline_are_pinned():
    from examples.craftax_swarms import prepare

    baseline, dataset, digest = prepare.checked_source()
    assert len(dataset.instances) == 100
    assert [row.id for row in dataset.instances] == [f"episode-{i}" for i in range(100)]
    assert all(row.inputs == {"max_steps": 250} for row in dataset.instances)
    assert '"noop"' in baseline
    assert len(digest) == 64


def pair_fixture(contract):
    from examples.craftax_swarms import prepare, verify
    from synth_ai.models import RunReport

    report = example(contract, "/v1/runs/{run_id}/report")
    baseline = {
        "evaluation_id": "baseline-evaluation",
        "status": "completed",
        "policy_revision_id": verify.baseline_revision(),
        "evaluator_revision_id": "sha256:" + "c" * 64,
        "dataset_revision_id": "sha256:" + "d" * 64,
        "aggregate": {"n": 100, "aggregation": "mean", "score": 0.0},
        "attestation_digest": "sha256:" + "a" * 64,
    }
    candidate = {
        **baseline,
        "evaluation_id": "candidate-evaluation",
        "policy_revision_id": "sha256:" + "e" * 64,
        "aggregate": {"n": 100, "aggregation": "mean", "score": 0.1},
        "attestation_digest": "sha256:" + "b" * 64,
    }
    report["evaluations"] = [baseline, candidate]
    template = prepare.Prepared(
        base_url="https://fixture.test",
        evaluator_revision_id="sha256:" + "f" * 64,
        dataset_revision_id=baseline["dataset_revision_id"],
        template_pool_revision_id="sha256:" + "1" * 64,
        baseline_sha256="2" * 64,
        field_sha256="3" * 64,
        task_sha256="4" * 64,
        provider="modal",
        region="us",
    )
    return RunReport.model_validate(report), template


def summary_fixture(evaluation):
    return {
        "evaluation_id": evaluation.evaluation_id,
        "status": "completed",
        "view": "summary",
        "attestation_digest": evaluation.attestation_digest,
        "outcome_counts": {
            "total": 100,
            "by_status": {"completed": 100},
            "by_error_code": {},
            "policy_error": 0,
            "policy_timeout": 0,
            "policy_timeout_reported": True,
            "infrastructure_requeued": 0,
            "infrastructure_requeues_total": 0,
        },
        "columns": ["instance_id", "outcome", "score", "metrics"],
        "metric_keys": [],
        "constant_metrics": {},
        "instances": [[f"episode-{i}", "scored", 0.0, []] for i in range(100)],
        "next_cursor": None,
        "truncated": False,
    }


@pytest.mark.parametrize(
    "change", ["dataset", "baseline_policy", "evaluator", "attestation", "count"]
)
def test_pair_refuses_unbound_or_unattested_comparison(contract, change):
    from examples.craftax_swarms import verify

    report, template = pair_fixture(contract)
    data = report.model_dump(mode="json")
    row = data["evaluations"][0 if change == "baseline_policy" else 1]
    if change == "dataset":
        row["dataset_revision_id"] = "sha256:" + "0" * 64
    elif change == "baseline_policy":
        row["policy_revision_id"] = "sha256:" + "0" * 64
    elif change == "evaluator":
        row["evaluator_revision_id"] = "sha256:" + "0" * 64
    elif change == "attestation":
        row["attestation_digest"] = None
    else:
        row["aggregate"]["n"] = 8
    with pytest.raises(ValueError):
        verify.candidates(RunReport.model_validate(data), template)


@pytest.mark.parametrize(
    "change",
    [
        "policy_error",
        "policy_timeout",
        "unknown_instance",
        "duplicate_instance",
        "unscored",
        "digest",
    ],
)
def test_confirmation_refuses_incomplete_or_erroneous_metadata(contract, change):
    from examples.craftax_swarms import verify
    from synth_ai.models import EvaluationRolloutSummary

    report, template = pair_fixture(contract)
    baseline, _ = verify.candidates(report, template)
    summary = summary_fixture(baseline)
    if change in {"policy_error", "policy_timeout"}:
        summary["outcome_counts"][change] = 1
    elif change == "unknown_instance":
        summary["instances"][0][0] = "episode-100"
    elif change == "duplicate_instance":
        summary["instances"][0][0] = "episode-1"
    elif change == "unscored":
        summary["instances"][0][1] = "failed"
    else:
        summary["attestation_digest"] = "wrong"
    with pytest.raises(ValueError):
        verify.summary_instances(EvaluationRolloutSummary.model_validate(summary), baseline)


async def test_metadata_verifier_reads_only_summary_and_returns_exact_instance_set(contract):
    from examples.craftax_swarms import verify

    report, template = pair_fixture(contract)
    baseline, choices = verify.candidates(report, template)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=summary_fixture(baseline))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://fixture.test", http_client=http
        ) as client:
            await verify.confirm(client, baseline)
    assert len(choices) == 1
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.params["view"] == "summary"
    assert requests[0].url.path.endswith("/rollouts")


@pytest.mark.parametrize("change", ["same_pool", "different_image", "matching_clone"])
async def test_template_binding_requires_fresh_pool_and_exact_evaluator(contract, change):
    from examples.craftax_swarms import prepare, verify

    report, template = pair_fixture(contract)
    baseline, _ = verify.candidates(report, template)
    payload = {
        "name": "template",
        "image": prepare.EVALUATOR_IMAGE,
        "candidate_image": prepare.CANDIDATE_IMAGE,
        "command": ["python", "-m", "cybernetics_evaluators.episode", prepare.TASK],
        "pool_id": template.template_pool_revision_id,
        "task_id": prepare.TASK,
        "task_digest": "sha256:" + "2" * 64,
        "input_schema": {},
        "limits": {"timeout_seconds": 120, "cpu_cores": 1, "memory_mb": 512, "storage_mb": 512},
        "region": "us",
    }
    owned = {**payload, "name": "run-owned", "pool_id": "sha256:" + "3" * 64}
    if change == "same_pool":
        owned["pool_id"] = payload["pool_id"]
    if change == "different_image":
        owned["image"] = "example/verifier@sha256:" + "0" * 64

    def handle(request):
        revision = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(
            200,
            json={
                "kind": "evaluator",
                "revision_id": revision,
                "payload": payload if revision == template.evaluator_revision_id else owned,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url=template.base_url, http_client=http
        ) as client:
            if change == "matching_clone":
                await verify.verify_template(client, template, baseline.evaluator_revision_id)
            else:
                with pytest.raises(ValueError):
                    await verify.verify_template(client, template, baseline.evaluator_revision_id)
