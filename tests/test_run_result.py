"""`swarms.result()` reads a finished run's outputs and final text answer."""

import hashlib

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import DataIntegrityError, ResultUnavailableError
from synth_ai.models import RunResult


def snapshot(contract, **changes):
    value = dict(
        contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    value.update(execution_closed=True, accounting_final=True, cleanup_complete=True)
    value.update(changes)
    return value


def output(run_id, suffix, media_type, content, *, state="available"):
    return dict(
        run_id=run_id,
        output_id="out_" + suffix * 22,
        state=state,
        filename=suffix + ".out",
        reference=dict(
            run_id=run_id,
            output_id="out_" + suffix * 22,
            revision_id="out_" + suffix * 22,
            digest_sha256=hashlib.sha256(content).hexdigest(),
            byte_size=str(len(content)),
            media_type=media_type,
        ),
    )


def handler(state, outputs, contents, calls):
    def handle(request):
        calls.append(request.url.path)
        path = request.url.path
        if path.endswith("/outputs"):
            cursor = request.url.params.get("cursor")
            # Two pages prove the record holds every output, in order.
            if cursor is None and len(outputs) > 1:
                return httpx.Response(
                    200, json=dict(has_more=True, next_cursor="page-2", items=outputs[:1])
                )
            items = outputs[1:] if cursor == "page-2" else outputs
            return httpx.Response(200, json=dict(has_more=False, items=items))
        if path.endswith("/content"):
            output_id = path.split("/")[-2]
            assert request.url.params["revision_id"] == output_id
            return httpx.Response(200, content=contents[output_id])
        return httpx.Response(200, json=state)

    return handle


async def test_result_returns_outputs_and_last_text_answer(contract):
    state = snapshot(contract)
    answer_text = b"The answer is 42.\n"
    items = [
        output(state["run_id"], "A", "text/markdown", b"draft"),
        output(state["run_id"], "B", "application/gzip", b"\x1f\x8b"),
        output(state["run_id"], "C", "text/plain; charset=utf-8", answer_text),
    ]
    contents = {
        item["output_id"]: c for item, c in zip(items, [b"draft", b"", answer_text], strict=True)
    }
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(state, items, contents, calls))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            finished = await client.swarms.get(state["run_id"])
            calls.clear()
            result = await client.swarms.result(finished)
    assert isinstance(result, RunResult)
    assert result.run == finished
    assert [item.output_id for item in result.outputs] == [item["output_id"] for item in items]
    assert result.answer is not None and result.answer.output_id == items[2]["output_id"]
    assert result.content == "The answer is 42.\n"
    # A finished snapshot is not reread; only the answer's bytes are fetched.
    assert not any(path.endswith(state["run_id"]) for path in calls)
    assert sum(path.endswith("/content") for path in calls) == 1


def test_sync_result_without_text_output_leaves_content_absent(contract):
    state = snapshot(contract)
    items = [output(state["run_id"], "B", "application/gzip", b"\x1f\x8b")]
    calls = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler(state, items, {}, calls)))
    with SynthClient("fixture", base_url="https://example.test", http_client=http) as client:
        result = client.swarms.result(state["run_id"])
    assert result.run.run_id == state["run_id"]
    assert [item.output_id for item in result.outputs] == [items[0]["output_id"]]
    assert result.answer is None and result.content is None
    assert not any(path.endswith("/content") for path in calls)


async def test_result_refuses_unfinished_run_after_a_fresh_read(contract):
    state = snapshot(
        contract, execution_closed=True, accounting_final=False, cleanup_complete=False
    )
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(state, [], {}, calls))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            stale = await client.swarms.get(state["run_id"])
            calls.clear()
            with pytest.raises(ResultUnavailableError) as caught:
                await client.swarms.result(stale)
    assert caught.value.snapshot is not None and not caught.value.snapshot.finished
    assert caught.value.code == "result_unavailable"
    assert calls == ["/v1/runs/" + state["run_id"]]


@pytest.mark.parametrize("bound", ["outputs", "bytes"])
async def test_result_bounds_refuse_instead_of_truncating(contract, bound):
    state = snapshot(contract)
    big = b"x" * 64
    items = [
        output(state["run_id"], "A", "text/plain", big),
        output(state["run_id"], "C", "text/plain", big),
    ]
    contents = {item["output_id"]: big for item in items}
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler(state, items, contents, calls))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            with pytest.raises(DataIntegrityError, match="max_" + bound):
                if bound == "outputs":
                    await client.swarms.result(state["run_id"], max_outputs=1)
                else:
                    await client.swarms.result(state["run_id"], max_bytes=63)
    assert not any(path.endswith("/content") for path in calls)
