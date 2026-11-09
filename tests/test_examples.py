import copy
import hashlib
import json

import httpx
from test_routing_lifecycle import event

from examples import sdk_workflows as recipes
from synth_ai import AsyncSynthClient, StreamCheckpoint


async def test_documented_workflows_execute_against_wire_fixtures(contract, tmp_path):
    def example(path, method="get", status="200"):
        return copy.deepcopy(
            contract["paths"][path][method]["responses"][status]["content"]["application/json"][
                "example"
            ]
        )

    snapshot = example("/v1/runs/{run_id}")
    snapshot.update(
        result_available=True, execution_closed=True, accounting_final=True, cleanup_complete=True
    )
    calls = []
    content = b"verified content"
    digest = hashlib.sha256(content).hexdigest()
    file = example("/v1/files", "post", "201")
    file.update(size_bytes=len(content), digest_sha256=digest)
    uploaded = False

    def handle(request):
        nonlocal uploaded
        calls.append(request)
        path = request.url.path
        if request.headers.get("accept") == "text/event-stream":
            item = event(snapshot["run_id"], "A")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="id: 1\ndata: " + json.dumps(item) + "\n\n",
            )
        if path == "/v1/profiles":
            return httpx.Response(
                200,
                json={
                    "default_profile": "swarms-v2",
                    "profiles": [
                        {
                            "id": "swarms-v2",
                            "revision": "1",
                            "name": "General",
                            "description": "General",
                            "requires_research_task": False,
                            "default": True,
                            "available": True,
                        }
                    ],
                },
            )
        if path.endswith("/content"):
            return httpx.Response(200, content=content)
        if "/outputs/" in path:
            output = example("/v1/runs/{run_id}/outputs/{output_id}")
            output["reference"] = {"revision_id": "revision", "digest_sha256": digest}
            return httpx.Response(200, json=output)
        if path == "/v1/usage/summary":
            return httpx.Response(200, json=example(path))
        if path == "/v1/uploads" or path.startswith("/v1/uploads/"):
            return httpx.Response(
                201 if path == "/v1/uploads" else 200,
                json={
                    "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                    "state": "committed" if uploaded else "pending",
                    "file": file,
                    **(
                        {}
                        if uploaded
                        else {
                            "put": {
                                "url": "https://storage.example.test/blob",
                                "headers": {},
                                "expires_at": "2099-01-01T00:00:00Z",
                            }
                        }
                    ),
                },
            )
        if path == "/blob":
            uploaded = True
            return httpx.Response(503)
        return httpx.Response(202 if request.method == "POST" else 200, json=snapshot)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            await recipes.required_budget(client, "task")
            await recipes.select_profile(client, "task")
            await recipes.constrained_routing(client, "task")
            await recipes.wait_result_then_finished(client, snapshot["run_id"])
            applied, saved = [], []

            async def apply(item):
                applied.append(item.event_id)

            await recipes.restartable_stream(
                client,
                snapshot["run_id"],
                StreamCheckpoint(),
                apply,
                lambda state: saved.append(state.to_dict()),
            )
            assert applied and saved[-1]["last_event_id"] == "1"
            destination = tmp_path / "output"
            await recipes.verified_artifact(
                client, snapshot["run_id"], "out_AAAAAAAAAAAAAAAAAAAAAA", str(destination)
            )
            assert destination.read_bytes() == content
            path = tmp_path / "task.txt"
            path.write_bytes(content)
            assert (
                await recipes.recover_upload(client, str(path), upload_key="retained")
            ).digest_sha256 == digest
            await recipes.usage_readback(
                client, since="2026-11-01T00:00:00Z", until="2026-11-02T00:00:00Z"
            )
    launches = [json.loads(request.content) for request in calls if request.url.path == "/v1/runs"]
    assert all(body["limits"]["maximum_charge"]["amount"] == "0.10" for body in launches)
    assert launches[-1]["routing"]["providers"]["only"] == ["openai"]


async def test_explicit_timeout_recipe_is_the_only_cancellation_path(contract):
    import asyncio

    snapshot = copy.deepcopy(
        contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["example"]
    )
    cancelled = False
    calls = []

    async def handle(request):
        nonlocal cancelled
        calls.append(request)
        if request.url.path.endswith("/cancel"):
            assert request.headers["idempotency-key"] == "cancel-once"
            cancelled = True
        if not cancelled:
            await asyncio.sleep(1)
        if request.headers["accept"] == "text/event-stream":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
        return httpx.Response(
            200,
            json={
                **snapshot,
                "execution_closed": True,
                "accounting_final": True,
                "cleanup_complete": True,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            assert (
                await recipes.cancel_after_timeout(
                    client, snapshot["run_id"], timeout=0.01, cancel_key="cancel-once"
                )
            ).finished
    assert sum(request.method == "POST" for request in calls) == 1
