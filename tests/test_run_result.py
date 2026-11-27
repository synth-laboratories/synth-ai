"""`swarms.result()` is one `getRunResult` read: the answer, its source and the charge."""

import json
from pathlib import Path

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ConflictError, ResponseContractError, ResultUnavailableError
from synth_ai.models import (
    OptimizerPin,
    OptimizerPinTopology,
    ResolvedOptimizer,
    RunCharge,
    RunResult,
)


def snapshot(contract, **changes):
    value = dict(
        contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    value.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
    value.update(changes)
    return value


def output(run_id, suffix, media_type):
    return {
        "run_id": run_id,
        "output_id": "out_" + suffix * 22,
        "state": "available",
        "filename": suffix + ".out",
        "reference": {
            "run_id": run_id,
            "output_id": "out_" + suffix * 22,
            "revision_id": "rev_" + suffix,
            "digest_sha256": "0" * 64,
            "byte_size": "18",
            "media_type": media_type,
        },
    }


def unavailable_tree(run_id):
    return {
        "state": "unavailable",
        "run_id": run_id,
        "code": "run_records_projection_unavailable",
        "stage": "run_result",
        "cause": "Canonical research record projection is unavailable for this cohort.",
    }


def record(kind):
    """Schema control only; not a retained record or run receipt."""
    return {
        "source": "research",
        "kind": kind,
        "record_id": "contract-control-" + kind,
        "revision": "1",
        "digest_sha256": "a" * 64,
    }


TERMINAL = {"completed", "partial", "failed", "cancelled"}


def body(
    state,
    *,
    answer=None,
    answer_output=None,
    objective_answer=None,
    attempt_tree=None,
    result_file=None,
):
    return {
        "run_id": state["run_id"],
        "status": state["state"] if state["state"] in TERMINAL else "completed",
        "accounting": None,
        "run": state,
        "state": state["state"],
        "answer": answer,
        "answer_output": answer_output,
        "objective_answer": objective_answer,
        "charge": state.get("charge"),
        "attempt_tree": attempt_tree or unavailable_tree(state["run_id"]),
        "result_file": result_file,
    }


def problem(code, status=409):
    return {
        "type": "about:blank",
        "title": "refused",
        "detail": code,
        "status": status,
        "instance": "urn:test",
        "code": code,
        "service": "core",
        "stage": "getRunResult",
        "outcome": "not_accepted",
    }


def handler(state, result, calls):
    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/result"):
            if isinstance(result, tuple):
                return httpx.Response(result[0], json=result[1])
            return httpx.Response(200, json=result)
        return httpx.Response(200, json=state)

    return handle


async def read(state, result, calls, run=None):
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler(state, result, calls))) as http,
        AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client,
    ):
        return await client.swarms.result(run or state["run_id"])


async def test_result_is_one_read_of_core_choice(contract):
    state = snapshot(contract)
    source = output(state["run_id"], "C", "text/plain; charset=utf-8")
    calls = []
    result = await read(
        state, body(state, answer="The answer is 42.\n", answer_output=source), calls
    )
    assert isinstance(result, RunResult)
    assert result.run.run_id == state["run_id"] and result.state == state["state"]
    assert result.answer == "The answer is 42.\n"
    assert (
        result.answer_output is not None and result.answer_output.output_id == source["output_id"]
    )
    assert result.objective_answer is None
    assert calls == ["/v1/runs/" + state["run_id"] + "/result"]


def test_sync_result_accepts_a_snapshot_and_reports_no_answer(contract):
    state = snapshot(contract)
    calls = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler(state, body(state), calls)))
    with SynthClient("fixture", base_url="https://example.test", http_client=http) as client:
        finished = client.swarms.get(state["run_id"])
        calls.clear()
        result = client.swarms.result(finished)
    assert result.answer is None and result.answer_output is None
    assert calls == ["/v1/runs/" + state["run_id"] + "/result"]


async def test_result_reads_an_objective_answer(contract):
    state = snapshot(contract)
    objective = {
        "schema_version": "synth.objective-answer.v1",
        "objective_id": "codename",
        "definition_revision": 1,
        "answer": {"text": "nonce-6b98", "by": "planner", "revision": 1, "sources": []},
    }
    result = await read(
        state,
        body(
            state,
            answer="nonce-6b98",
            answer_output=output(state["run_id"], "J", "application/json"),
            objective_answer=objective,
        ),
        [],
    )
    assert result.answer == "nonce-6b98"
    assert result.objective_answer is not None
    assert result.objective_answer.objective_id == "codename"
    assert result.objective_answer.answer.text == "nonce-6b98"


async def test_unfinished_run_raises_result_unavailable_with_a_fresh_snapshot(contract):
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    calls = []
    with pytest.raises(ResultUnavailableError) as caught:
        await read(state, (409, problem("result_unavailable")), calls)
    assert caught.value.code == "result_unavailable"
    assert caught.value.snapshot is not None and not caught.value.snapshot.finished
    assert isinstance(caught.value.__cause__, ConflictError)
    assert calls == ["/v1/runs/" + state["run_id"] + "/result", "/v1/runs/" + state["run_id"]]


@pytest.mark.parametrize("code", ["result_answer_too_large", "result_answer_not_text"])
async def test_other_result_refusals_propagate(contract, code):
    state = snapshot(contract)
    with pytest.raises(ConflictError) as caught:
        await read(state, (409, problem(code)), [])
    assert caught.value.code == code


async def test_result_carries_the_run_charge(contract):
    state = snapshot(
        contract, charge={"amount": {"amount": "0.02365822", "currency": "USD"}, "final": True}
    )
    result = await read(state, body(state), [])
    assert isinstance(result.charge, RunCharge)
    assert result.charge.amount.amount == "0.02365822" and result.charge.final is True
    uncharged = snapshot(contract)
    uncharged.pop("charge", None)
    assert (await read(uncharged, body(uncharged), [])).charge is None


async def test_result_carries_the_resolved_optimizer_and_attempt_tree(contract):
    optimizer = {
        "topology": "swarms_v2",
        "version": 1,
        "params": {},
        "reason": "environment_default",
        "definition_digest": "b" * 64,
    }
    state = snapshot(contract, optimizer=optimizer)
    tree = {
        "state": "available",
        "run_id": state["run_id"],
        "protocol": record("protocol"),
        "experiment": record("experiment"),
        "research_log": record("research_log"),
        "report": record("report"),
        "attempts": [],
        "notes": [],
    }
    result_md = dict(output(state["run_id"], "R", "text/markdown"), filename="result.md")
    result = await read(state, body(state, attempt_tree=tree, result_file=result_md), [])
    assert result.result_file is not None
    assert result.result_file.filename == "result.md"
    assert result.result_file.reference.media_type == "text/markdown"
    assert result.result_file.reference.digest_sha256 == "0" * 64
    assert result.run.optimizer is not None
    assert result.run.optimizer.topology == "swarms_v2"
    assert result.attempt_tree.state == "available"
    assert result.attempt_tree.report.kind == "report"
    unavailable = await read(state, body(state), [])
    assert unavailable.result_file is None
    assert unavailable.attempt_tree.state == "unavailable"
    assert unavailable.attempt_tree.code == "run_records_projection_unavailable"


async def test_result_without_an_attempt_tree_violates_the_contract(contract):
    state = snapshot(contract)
    missing = body(state)
    missing.pop("attempt_tree")
    with pytest.raises(ResponseContractError):
        await read(state, missing, [])


async def test_mcp_result_tool_carries_result_md(contract):
    from synth_ai.mcp.dispatch import dispatch

    state = snapshot(contract)
    result_md = dict(output(state["run_id"], "R", "text/markdown"), filename="result.md")
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler(state, body(state, result_file=result_md), []))
        ) as http,
        AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client,
    ):
        model = await dispatch(client, "getRunResult", {"run_id": state["run_id"]})
    assert model is not None
    document = model.model_dump(mode="json")
    assert document["result_file"]["filename"] == "result.md"
    assert document["result_file"]["output_id"] == result_md["output_id"]


async def test_result_file_is_required_and_names_result_md(contract):
    state = snapshot(contract)
    missing = body(state)
    missing.pop("result_file")
    with pytest.raises(ResponseContractError):
        await read(state, missing, [])
    misnamed = body(state, result_file=output(state["run_id"], "R", "text/markdown"))
    with pytest.raises(ResponseContractError):
        await read(state, misnamed, [])
    wrong_type = dict(output(state["run_id"], "R", "text/plain"), filename="result.md")
    with pytest.raises(ResponseContractError):
        await read(state, body(state, result_file=wrong_type), [])


@pytest.mark.parametrize("topology", ["forgetting", "mira", "scalemas", "future_topology"])
def test_optimizer_models_accept_forward_compatible_topologies(topology):
    assert {t.value for t in OptimizerPinTopology} == {"swarms_v2", "fork_flush"}
    pin = OptimizerPin.model_validate({"topology": topology, "version": 1})
    assert pin.topology == topology
    resolved = ResolvedOptimizer.model_validate(
        {
            "topology": topology,
            "version": 1,
            "params": {},
            "reason": "explicit_pin",
            "definition_digest": "b" * 64,
        }
    )
    assert resolved.topology == topology


def test_public_run_profiles_exclude_research_topologies():
    from synth_ai._generated.policy import PROFILE_REQUIRES_RESEARCH_TASK

    assert set(PROFILE_REQUIRES_RESEARCH_TASK) == {"swarms-v2", "fork-flush"}


# Contract 1.37.0 (SYN-5637): RunResult carries run_id, its terminal status and the final
# ledger-backed accounting. The fixture is Core's published /result example at 1.37.0.
ENVELOPE = json.loads(
    (Path(__file__).parent / "fixtures" / "core_run_result_1_37_0.json").read_text()
)


def envelope(**changes):
    value = json.loads(json.dumps(ENVELOPE))
    value.update(changes)
    return value


async def test_result_envelope_round_trips_run_id_status_and_accounting():
    fixture = envelope()
    result = await read(fixture["run"], fixture, [])
    assert result.run_id == fixture["run_id"] and result.status == "completed"
    assert isinstance(result.accounting, RunCharge)
    assert result.accounting.final is True and result.accounting.amount.amount == "0.10"
    assert result.model_dump(mode="json", exclude_unset=True) == fixture


async def test_result_envelope_accepts_null_accounting_and_refuses_nonfinal():
    fixture = envelope(accounting=None, status="cancelled")
    result = await read(fixture["run"], fixture, [])
    assert result.accounting is None and result.status == "cancelled"
    nonfinal = envelope(
        accounting={"amount": {"amount": "0.10", "currency": "USD"}, "final": False}
    )
    with pytest.raises(ResponseContractError):
        await read(nonfinal["run"], nonfinal, [])
    for missing in ("run_id", "status", "accounting"):
        absent = envelope()
        absent.pop(missing)
        with pytest.raises(ResponseContractError):
            await read(absent["run"], absent, [])
    with pytest.raises(ResponseContractError):
        await read(ENVELOPE["run"], envelope(status="running"), [])


async def test_mcp_result_tool_carries_the_result_envelope():
    from synth_ai.mcp.dispatch import dispatch

    fixture = envelope()
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler(fixture["run"], fixture, []))
        ) as http,
        AsyncSynthClient("fixture", base_url="https://example.test", http_client=http) as client,
    ):
        model = await dispatch(client, "getRunResult", {"run_id": fixture["run_id"]})
    assert model is not None
    document = model.model_dump(mode="json", exclude_unset=True)
    assert document == fixture
