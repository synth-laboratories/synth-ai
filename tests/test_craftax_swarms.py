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
