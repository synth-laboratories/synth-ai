import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts import smoke


@pytest.mark.parametrize("state", ["failed", "cancelled", "partial", "running"])
def test_output_does_not_substitute_for_successful_execution(state):
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        smoke.require_successful_run(SimpleNamespace(state=state, result_available=True))


def test_completed_run_is_required():
    smoke.require_successful_run(SimpleNamespace(state="completed"))


@pytest.mark.parametrize("case", ["existing", "different_api", "too_many_runs", "legacy_schema"])
async def test_smoke_budget_preflight(tmp_path, monkeypatch, case):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "synth.public-sdk.smoke.v1"
                if case == "legacy_schema"
                else "synth.public-sdk.swarms-smoke.v4",
                "base_url": "https://other.example.test"
                if case == "different_api"
                else "https://api.example.test",
                "runs": [
                    {"run_id": f"run-{i}"} for i in range(3 if case == "too_many_runs" else 0)
                ],
            }
        )
    )
    before = receipt.read_bytes()

    def network_forbidden(*args, **kwargs):
        pytest.fail("invalid receipts must be rejected before constructing a client")

    monkeypatch.setattr(smoke.httpx, "AsyncClient", network_forbidden)
    with pytest.raises(ValueError):
        await smoke.run(receipt, resume=case != "existing")
    assert receipt.read_bytes() == before


@pytest.fixture
def smoke_api(contract, monkeypatch, tmp_path):
    """Exercise the real SDK and response hooks through an in-memory HTTP boundary."""
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "absent-auth.json"))
    requests = []
    run_ids = ["run_AAAAAAAAAAAAAAAAAAAAAA", "run_BBBBBBBBBBBBBBBBBBBBBB"]
    output_id = "out_AAAAAAAAAAAAAAAAAAAAAA"
    content = b"hello"
    reference = {
        "revision_id": "revision-fixture",
        "digest_sha256": hashlib.sha256(content).hexdigest(),
    }
    output = {
        "run_id": run_ids[0],
        "output_id": output_id,
        "state": "available",
        "reference": reference,
        "filename": "hello.txt",
    }
    snapshot = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    state = {"content": content, "launch_count": 0, "cancel_keys": [], "pending_flag": None}

    def handler(request):
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/runs":
            index = state["launch_count"]
            assert index < 2, "aggregate two-run cap exceeded"
            state["launch_count"] += 1
            return httpx.Response(202, json={**snapshot, "run_id": run_ids[index]})
        for index, run_id in enumerate(run_ids):
            run_path = f"/v1/runs/{run_id}"
            closed = {
                **snapshot,
                "run_id": run_id,
                "state": "completed" if index == 0 else "cancelled",
                "execution_closed": True,
                "accounting_final": state["pending_flag"] != "accounting_final",
                "cleanup_complete": state["pending_flag"] != "cleanup_complete",
                "result_available": index == 0,
            }
            if path == run_path and request.method == "GET":
                return httpx.Response(200, json=closed)
            if path == run_path + "/cancel" and request.method == "POST":
                state["cancel_keys"].append(request.headers["Idempotency-Key"])
                return httpx.Response(202, json=closed)
        first = f"/v1/runs/{run_ids[0]}"
        if path == first + "/events":
            return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
        if path == first + "/outputs":
            return httpx.Response(200, json={"has_more": False, "items": [output]})
        if path == first + f"/outputs/{output_id}":
            return httpx.Response(200, json=output)
        if path == first + f"/outputs/{output_id}/content":
            assert request.url.params["digest_sha256"] == reference["digest_sha256"]
            assert request.url.params["revision_id"] == reference["revision_id"]
            return httpx.Response(200, content=state["content"])
        pytest.fail(f"unexpected operation: {request.method} {path}")

    original_client = httpx.AsyncClient

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(smoke.httpx, "AsyncClient", client)
    return state, requests, run_ids


async def test_smoke_completes_two_run_journey_and_resumes_without_new_spend(smoke_api, tmp_path):
    state, requests, run_ids = smoke_api
    output = tmp_path / "smoke.json"
    await smoke.run(output)
    receipt = smoke.SmokeReceipt.from_json(output.read_text())
    assert receipt.smoke_complete
    assert json.loads(output.read_text())["smoke_complete"] is True
    assert [run.run_id for run in receipt.runs] == run_ids
    assert {entry.operation for entry in receipt.responses} < set(smoke.OPERATIONS)
    launches = [request for request in requests if request.url.path == "/v1/runs"]
    assert len(launches) == 2
    for index, request in enumerate(launches):
        payload = json.loads(request.content)
        assert payload["limits"] == {
            "maximum_charge": {"amount": "4.00", "currency": "USD"},
            "wall_time_seconds": 600,
        }
        assert payload["profile"] == "swarms-v2"
        assert request.headers["Idempotency-Key"] == receipt.launch_keys[str(index)]
    assert len(receipt.cancel_keys) == 1
    assert state["cancel_keys"] == list(receipt.cancel_keys.values())
    await smoke.run(output, resume=True)
    resumed = smoke.SmokeReceipt.from_json(output.read_text())
    assert resumed.smoke_complete
    assert resumed.launch_keys == receipt.launch_keys
    assert resumed.cancel_keys == receipt.cancel_keys
    assert state["launch_count"] == 2
    assert state["cancel_keys"] == list(receipt.cancel_keys.values()) * 2


async def test_smoke_digest_failure_still_closes_both_runs(smoke_api, tmp_path):
    state, _, _ = smoke_api
    state["content"] = b"wrong content"
    output = tmp_path / "smoke.json"
    with pytest.raises(RuntimeError, match="smoke incomplete"):
        await smoke.run(output)
    receipt = smoke.SmokeReceipt.from_json(output.read_text())
    assert receipt.failures == {"first_run": "DataIntegrityError"}
    assert not receipt.output_verified
    assert not receipt.smoke_complete
    assert all(run.finished for run in receipt.runs)
    assert receipt.cancellation_requested


@pytest.mark.parametrize("flag", ["accounting_final", "cleanup_complete"])
async def test_smoke_requires_accounting_and_cleanup(smoke_api, tmp_path, monkeypatch, flag):
    state, _, _ = smoke_api
    state["pending_flag"] = flag
    clock = iter(range(0, 2000, 100))
    monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    async def skip_sleep(_seconds):
        pass

    monkeypatch.setattr(smoke.asyncio, "sleep", skip_sleep)
    with pytest.raises(RuntimeError, match="smoke incomplete"):
        await smoke.run(tmp_path / "smoke.json")
    receipt = smoke.SmokeReceipt.from_json((tmp_path / "smoke.json").read_text())
    assert not receipt.smoke_complete
    assert not any(run.finished for run in receipt.runs)


def test_smoke_summary_cannot_override_missing_evidence():
    source = json.dumps(
        {
            "schema": "synth.public-sdk.swarms-smoke.v4",
            "base_url": "https://api.example.test",
            "smoke_complete": True,
        }
    )
    assert not smoke.SmokeReceipt.from_json(source).smoke_complete


async def test_v3_receipt_is_preserved_and_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    output = tmp_path / "old.json"
    source = (
        '{"schema":"synth.public-sdk.smoke.v3","base_url":"https://api.example.test","runs":[]}'
    )
    output.write_text(source)

    def network_forbidden(*args, **kwargs):
        pytest.fail("legacy receipt must be refused before networking")

    monkeypatch.setattr(smoke.httpx, "AsyncClient", network_forbidden)
    with pytest.raises(ValueError):
        await smoke.run(output, resume=True)
    assert output.read_text() == source


async def test_smoke_reuses_launch_key_saved_before_acknowledgement(smoke_api, tmp_path):
    state, requests, _ = smoke_api
    output = tmp_path / "smoke.json"
    output.write_text(
        smoke.SmokeReceipt(
            base_url="https://api.example.test", launch_keys={"0": "retained-before-launch"}
        ).to_json()
    )
    await smoke.run(output, resume=True)
    launches = [request for request in requests if request.url.path == "/v1/runs"]
    assert launches[0].headers["Idempotency-Key"] == "retained-before-launch"
    assert state["launch_count"] == 2
    assert smoke.SmokeReceipt.from_json(output.read_text()).smoke_complete


def test_v4_receipt_preserves_original_budget_and_deadline():
    receipt = smoke.SmokeReceipt.from_json(
        json.dumps(
            {"schema": "synth.public-sdk.swarms-smoke.v4", "base_url": "https://api.example.test"}
        )
    )
    assert receipt.aggregate_maximum_usd == "0.20"
    assert receipt.run_limits == ("0.10", 120)
    assert smoke.SmokeReceipt.from_json(receipt.to_json()).run_limits == ("0.10", 120)


def test_receipt_refuses_mismatched_schema_and_budget():
    with pytest.raises(ValueError, match="receipt budget does not match schema"):
        smoke.SmokeReceipt.from_json(
            json.dumps(
                {
                    "schema": "synth.public-sdk.swarms-smoke.v4",
                    "base_url": "https://api.example.test",
                    "aggregate_maximum_usd": "8.00",
                }
            )
        )


async def test_v4_resume_keeps_original_launch_intent(smoke_api, tmp_path):
    state, requests, _ = smoke_api
    output = tmp_path / "v4.json"
    output.write_text(
        json.dumps(
            {
                "schema": "synth.public-sdk.swarms-smoke.v4",
                "base_url": "https://api.example.test",
                "launch_keys": {"0": "retained-v4-key"},
            }
        )
    )
    await smoke.run(output, resume=True)
    launches = [request for request in requests if request.url.path == "/v1/runs"]
    assert state["launch_count"] == 2
    assert launches[0].headers["Idempotency-Key"] == "retained-v4-key"
    for request in launches:
        limits = json.loads(request.content)["limits"]
        assert limits["maximum_charge"]["amount"] == "0.10"
        assert limits["wall_time_seconds"] == 120
    assert smoke.SmokeReceipt.from_json(output.read_text()).aggregate_maximum_usd == "0.20"
