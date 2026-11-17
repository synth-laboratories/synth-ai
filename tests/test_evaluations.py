import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.models import EvaluationEvaluationRequest, EvaluationResponse, EvaluationTaskInstance

REVISION = "sha256:" + "a" * 64


def example(contract, path, method, status):
    return contract["paths"][path][method]["responses"][status]["content"]["application/json"][
        "example"
    ]


def seeds(count):
    return [{"id": f"episode-{i}", "seed": i, "inputs": {"level": i % 4}} for i in range(count)]


async def test_dataset_registration_sends_the_contract_body_and_returns_the_revision():
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append((request.method, request.url.path, request.headers, body))
        return httpx.Response(
            201,
            json={
                "kind": "dataset",
                "revision_id": REVISION,
                "name": body["name"],
                "payload": body,
            },
        )

    instances = [*seeds(255), EvaluationTaskInstance(id="episode-255", seed=2**63 - 1, inputs={})]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            revision = await api.evaluations.register_dataset(
                "craftax-256", instances, idempotency_key="dataset-1"
            )
    assert revision.revision_id == REVISION
    method, path, headers, body = calls[0]
    assert (method, path) == ("POST", "/v1/evaluation-inputs/dataset")
    assert headers["Idempotency-Key"] == "dataset-1"
    assert body == {
        "name": "craftax-256",
        "instances": [*seeds(255), {"id": "episode-255", "seed": 2**63 - 1, "inputs": {}}],
    }


@pytest.mark.parametrize(
    "instances",
    [
        [],
        seeds(1025),
        [*seeds(2), {"id": "episode-0", "seed": 9, "inputs": {}}],
        [{"id": "episode-0", "seed": -1, "inputs": {}}],
        [{"id": "episode-0", "seed": 2**63, "inputs": {}}],
        [{"id": "episode-0", "seed": True, "inputs": {}}],
        [{"id": "-bad", "seed": 0, "inputs": {}}],
        [{"id": "episode-0", "seed": 0}],
        [{"id": "episode-0", "seed": 0, "inputs": {}, "extra": 1}],
    ],
)
async def test_invalid_dataset_is_refused_before_any_request(instances):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("request sent"))
    ) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(ValueError):
                await api.evaluations.register_dataset("craftax", instances)


def test_get_input_reads_the_contract_revision(contract):
    body = example(contract, "/v1/evaluation-inputs/{kind}/{revision_id}", "get", "200")
    paths = []

    def handle(request):
        paths.append(request.url.raw_path.decode())
        return httpx.Response(200, json=body)

    with SynthClient(
        "fixture",
        base_url="https://example.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        revision = client.evaluations.get_input("policy", body["revision_id"])
    assert paths == ["/v1/evaluation-inputs/policy/" + body["revision_id"].replace(":", "%3A")]
    assert revision.model_dump(mode="json", exclude_unset=True) == body


async def test_preflight_create_get_and_wait_round_trip_contract_examples(contract):
    created = example(contract, "/v1/evaluations", "post", "202")
    quote = example(contract, "/v1/evaluations/preflight", "post", "200")
    finished = dict(created, status="completed")
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path, request.content))
        if request.url.path == "/v1/evaluations/preflight":
            return httpx.Response(200, json=quote)
        if request.method == "POST":
            return httpx.Response(202, json=created)
        return httpx.Response(200, json=finished if len(calls) > 3 else created)

    request = EvaluationEvaluationRequest.model_validate(created["request"])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            quoted = await api.evaluations.preflight(request, maximum_charge_micros=3000)
            evaluation = await api.evaluations.create(request, maximum_charge_micros=3000)
            fetched = await api.evaluations.get(evaluation.evaluation_id)
            waited = await api.evaluations.wait(
                evaluation.evaluation_id, timeout=5, poll_interval=0.01
            )
    assert quoted.maximum_charge_micros == quote["maximum_charge_micros"]
    assert evaluation.model_dump(mode="json", exclude_unset=True) == created
    assert isinstance(fetched, EvaluationResponse) and fetched == evaluation
    assert waited.status == "completed"
    sent = {"request": created["request"], "maximum_charge_micros": 3000}
    assert [(m, p) for m, p, _ in calls[:3]] == [
        ("POST", "/v1/evaluations/preflight"),
        ("POST", "/v1/evaluations"),
        ("GET", "/v1/evaluations/" + created["evaluation_id"]),
    ]
    assert json.loads(calls[0][2]) == sent and json.loads(calls[1][2]) == sent


def test_evaluations_are_not_a_swarms_surface():
    client = AsyncSynthClient("fixture", base_url="https://example.test")
    assert not any("evaluat" in name for name in dir(client.swarms))


def report_page():
    return {"schema_version":"synth.evaluation-rollout-page.v1","evaluation_id":"eval_"+"a"*40,
            "rollouts":[{"ordinal":0,"instance_id":"episode-0","rollout_id":None,"status":"completed",
                "result":{"schema_version":"synth.evaluation-result.v2","score":0.5,"metrics":{},
                    "report":{"schema_version":"synth.episode-report.v1","environment":"fixture","outcome":"died","ended_step":121,"termination_reason":"thirst"}},
                "report_status":"available","error_code":None,"execution_digest":None,"result_digest":None}],
            "next_cursor":"next-page","attestation_digest":None}


async def test_rollout_pages_preserve_typed_report_and_opaque_cursor():
    paths=[]
    def handle(request):
        paths.append(request)
        return httpx.Response(200,json=report_page())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient("fixture",base_url="https://example.test",http_client=http) as api:
            page=await api.evaluations.rollouts("eval_"+"a"*40,cursor="opaque-cursor",instance_ids=["episode-0","episode-1"])
    assert page.next_cursor=="next-page" and page.rollouts[0].result.report.termination_reason=="thirst"
    assert paths[0].method=="GET" and paths[0].url.params.get_list("instance_ids")==["episode-0","episode-1"]
    assert paths[0].url.params["cursor"]=="opaque-cursor"


def test_sync_rollout_pages_use_the_same_public_operation():
    with SynthClient("fixture",base_url="https://example.test",transport=httpx.MockTransport(lambda request:httpx.Response(200,json=report_page()))) as api:
        page=api.evaluations.rollouts("eval_"+"a"*40)
    assert page.rollouts[0].result.report.ended_step==121
