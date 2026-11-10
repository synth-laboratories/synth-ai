"""`swarms.result()` is one `getRunResult` read: the answer, its source and the charge."""

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ConflictError, ResultUnavailableError
from synth_ai.models import RunCharge, RunResult


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
    return dict(
        run_id=run_id,
        output_id="out_" + suffix * 22,
        state="available",
        filename=suffix + ".out",
        reference=dict(
            run_id=run_id,
            output_id="out_" + suffix * 22,
            revision_id="rev_" + suffix,
            digest_sha256="0" * 64,
            byte_size="18",
            media_type=media_type,
        ),
    )


def body(state, *, answer=None, answer_output=None, objective_answer=None):
    return dict(
        run=state,
        state=state["state"],
        answer=answer,
        answer_output=answer_output,
        objective_answer=objective_answer,
        charge=state.get("charge"),
    )


def problem(code, status=409):
    return dict(
        type="about:blank",
        title="refused",
        detail=code,
        status=status,
        instance="urn:test",
        code=code,
        service="core",
        stage="getRunResult",
        outcome="not_accepted",
    )


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
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(state, result, calls))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
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
    objective = dict(
        schema_version="synth.objective-answer.v1",
        objective_id="codename",
        definition_revision=1,
        answer=dict(text="nonce-6b98", by="planner", revision=1, sources=[]),
    )
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
