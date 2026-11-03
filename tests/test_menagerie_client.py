"""SYN-5183: Menagerie export/archive/restore/lineage/serving-identity client.

Offline exchanges against the backend wire shape (``app/api/v1/menagerie.py``):
no capability negotiation, typed ``backend.menagerie`` refusals.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts.menagerie import MENAGERIE_OPERATIONS
from synth_ai.errors import (
    ConflictError,
    ContractError,
    NotFoundError,
    OutcomeUncertainError,
    PermissionDeniedError,
    UnsupportedOperationError,
)
from synth_ai.options import ClientOptions
from synth_ai.sdk.menagerie import DeploymentWrite, LineageWrite

CHECKPOINT = "mdlckpt_" + "a" * 64
ARTIFACT = "mdlart_" + "b" * 64
EDGE = {
    "schema_version": "synth.menagerie.v1",
    "source": {"kind": "checkpoint", "id": CHECKPOINT},
    "relation": "export",
    "target": {"kind": "artifact", "id": ARTIFACT},
    "replayed": False,
    "created_at": "2026-10-29T00:00:00+00:00",
}


def deployment(deployment_id="tinker-r1", status="active"):
    return {
        "schema_version": "synth.menagerie.v1",
        "deployment_id": deployment_id,
        "model": {"kind": "checkpoint", "id": CHECKPOINT},
        "provider": "tinker",
        "route_id": "r1",
        "serving_revision": None,
        "verification_strength": "provider_reference",
        "status": status,
        "replayed": False,
        "created_at": "2026-10-29T00:00:00+00:00",
    }


def refusal(status, code, stage="lifecycle"):
    return httpx.Response(
        status,
        json={
            "detail": {
                "error_code": code,
                "service": "backend.menagerie",
                "stage": stage,
                "cause": f"fixture {code}",
            }
        },
    )


class Recorder:
    def __init__(self, routes):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.routes[(request.method, request.url.path)]


def sync_client(handler, **options):
    network = httpx.Client(transport=httpx.MockTransport(handler))
    return SynthClient(
        api_key="fixture",
        base_url="https://backend.test",
        options=ClientOptions(http_client=network, **options),
    )


def test_lifecycle_lineage_and_serving_round_trip_without_negotiation():
    prefix = "/api/v1/menagerie"
    recorder = Recorder(
        {
            ("GET", prefix + "/export"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.export.v1",
                    "org_id": "org",
                    "sections": {"lineage_edges": [{"relation": "export"}]},
                    "manifest_digest": "sha256:" + "c" * 64,
                },
            ),
            ("POST", prefix + "/lineage"): httpx.Response(200, json=EDGE),
            ("GET", f"{prefix}/models/{CHECKPOINT}/lineage"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.v1",
                    "model": EDGE["source"],
                    "derived_from": [EDGE],
                    "derivatives": [],
                },
            ),
            ("PUT", prefix + "/deployments/tinker-r1"): httpx.Response(200, json=deployment()),
            ("DELETE", prefix + "/deployments/tinker-r1"): httpx.Response(
                200, json=deployment(status="retired")
            ),
            ("DELETE", f"{prefix}/models/{CHECKPOINT}"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.v1",
                    "model": EDGE["source"],
                    "status": "archived",
                    "replayed": False,
                    "retained_aliases": 2,
                    "retained_eval_links": 1,
                },
            ),
            ("POST", f"{prefix}/models/{CHECKPOINT}/restore"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.v1",
                    "model": EDGE["source"],
                    "status": "active",
                    "replayed": True,
                },
            ),
            ("DELETE", prefix + "/catalog/records/tinker-lora"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.v1",
                    "record_id": "tinker-lora",
                    "head_revision": 1,
                    "archived": True,
                    "replayed": False,
                    "revoked_grants": 1,
                },
            ),
            ("POST", prefix + "/catalog/records/tinker-lora/restore"): httpx.Response(
                200,
                json={
                    "schema_version": "synth.menagerie.v1",
                    "record_id": "tinker-lora",
                    "head_revision": 1,
                    "archived": False,
                    "replayed": False,
                },
            ),
        }
    )
    with sync_client(recorder) as client:
        menagerie = client.menagerie
        exported = menagerie.export()
        assert exported.sections["lineage_edges"] == ({"relation": "export"},)
        edge = menagerie.record_lineage(
            LineageWrite(source="legacy-model-id", relation="export", target=ARTIFACT)
        )
        assert (edge.source.id, edge.target.kind) == (CHECKPOINT, "artifact")
        assert [e.relation for e in menagerie.lineage(CHECKPOINT).derived_from] == ["export"]
        recorded = menagerie.record_deployment(
            "tinker-r1", DeploymentWrite(model=CHECKPOINT, provider="tinker", route_id="r1")
        )
        assert recorded.verification_strength == "provider_reference"
        assert menagerie.retire_deployment("tinker-r1").status == "retired"
        assert menagerie.archive_model(CHECKPOINT).retained_aliases == 2
        assert menagerie.restore_model(CHECKPOINT).replayed is True
        assert menagerie.archive_record("tinker-lora").revoked_grants == 1
        assert menagerie.restore_record("tinker-lora").archived is False
    paths = [(r.method, r.url.path) for r in recorder.requests]
    assert not any(path.endswith("/capabilities") for _, path in paths)
    assert len(paths) == 9
    lineage_request = recorder.requests[1]
    assert json.loads(lineage_request.content) == {
        "source": "legacy-model-id",
        "relation": "export",
        "target": ARTIFACT,
    }
    assert lineage_request.headers["Authorization"] == "Bearer fixture"
    assert "X-Synth-Contract" not in lineage_request.headers
    assert "Idempotency-Key" in lineage_request.headers
    assert json.loads(recorder.requests[3].content)["verification_strength"] == (
        "provider_reference"
    )


@pytest.mark.parametrize(
    ("status", "code", "error"),
    [
        (409, "model_has_dependents", ConflictError),
        (409, "lineage_cycle", ConflictError),
        (409, "restore_conflict", ConflictError),
        (404, "model_not_found", NotFoundError),
        (403, "serving_strength_unearned", PermissionDeniedError),
    ],
)
def test_typed_refusals_carry_code_stage_and_cause(status, code, error):
    with (
        sync_client(lambda request: refusal(status, code)) as client,
        pytest.raises(error) as raised,
    ):
        client.menagerie.archive_model(CHECKPOINT)
    assert raised.value.code == code
    assert raised.value.status == status
    assert raised.value.stage == "lifecycle"
    assert raised.value.cause == f"fixture {code}"


def test_uncertain_write_and_contract_violations():
    uncertain = sync_client(lambda request: httpx.Response(502, text="bad gateway"))
    with uncertain as client, pytest.raises(OutcomeUncertainError):
        client.menagerie.restore_model(CHECKPOINT)
    other = deployment(deployment_id="someone-else")
    with sync_client(lambda request: httpx.Response(200, json=other)) as client:
        # A write whose receipt names another subject is never accepted.
        with pytest.raises(OutcomeUncertainError) as raised:
            client.menagerie.retire_deployment("tinker-r1")
        assert isinstance(raised.value.__cause__, ContractError)
        with pytest.raises(ValueError):
            client.menagerie.retire_deployment("tinker:r1")
        with pytest.raises(ValueError):
            client.menagerie.archive_model("../export")
    with (
        sync_client(
            lambda request: httpx.Response(200, json={"schema_version": "synth.menagerie.v1"})
        ) as client,
        pytest.raises(ContractError),
    ):
        client.menagerie.lineage(CHECKPOINT)


def test_release_pinned_clients_refuse_menagerie_routes():
    with (
        sync_client(
            lambda request: httpx.Response(500), release_manifest_sha256="a" * 64
        ) as client,
        pytest.raises(UnsupportedOperationError) as raised,
    ):
        client.menagerie.export()
    assert raised.value.code == "menagerie_release_not_qualified"
    assert len({operation.id for operation in MENAGERIE_OPERATIONS}) == 10
    assert all(operation.contract == "menagerie.v1" for operation in MENAGERIE_OPERATIONS)


def test_async_client_restores_record():
    asyncio.run(_async_restore())


async def _async_restore():
    body = {
        "schema_version": "synth.menagerie.v1",
        "record_id": "tinker-lora",
        "head_revision": 2,
        "archived": False,
        "replayed": True,
    }
    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as network,
        AsyncSynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(async_http_client=network),
        ) as client,
    ):
        restored = await client.menagerie.restore_record("tinker-lora")
    assert (restored.head_revision, restored.replayed) == (2, True)
