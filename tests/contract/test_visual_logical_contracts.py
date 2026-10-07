"""SYN-4005: current logical Visual responses work in sync/async operations.

Response fixtures validate against the committed generated backend contract.
Transport requests are in-process and content/preview stay byte operations.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import jsonschema
import pytest
from synth_ai.sdk.research.contracts.visuals import LogicalVisual, VisualPatch
from synth_ai.sdk.research.visuals import AsyncVisualsAPI, VisualsAPI

WIRE = {
    "visual_id": "visual",
    "org_id": "org",
    "project_id": "project",
    "title": "Visual",
    "lifecycle": "active",
    "current_revision": None,
    "resolved_revision": None,
    "resolved_materialization_id": None,
    "revisions": [],
    "releases": [],
    "derivation": None,
    "operation_receipts": [],
    "inaccessible_sources": False,
    "hosted_artifact_id": None,
    "created_at": "2026-10-06T00:00:00Z",
    "updated_at": "2026-10-06T00:00:00Z",
}


def test_logical_fixture_is_real_backend_response__SYN4005():
    specification = json.loads(
        (Path(__file__).with_name("fixtures") / "backend_full_openapi.generated.json").read_text()
    )
    schema = specification["components"]["schemas"]["SmrVisualLogicalResponse"]
    jsonschema.Draft202012Validator(
        {
            "$defs": specification["components"]["schemas"],
            **json.loads(json.dumps(schema).replace("#/components/schemas/", "#/$defs/")),
        }
    ).validate(WIRE)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_logical_retrieve_update_and_byte_reads__SYN4005(mode):
    if mode == "sync":
        transport = SimpleNamespace(
            execute=Mock(return_value=WIRE), request_bytes=Mock(return_value=b"content")
        )
        client = VisualsAPI(transport)
        assert isinstance(client.retrieve("visual"), LogicalVisual)
        assert client.update("visual", VisualPatch(title="Renamed")).visual_id == "visual"
        assert client.retrieve_content("visual") == b"content"
        assert client.retrieve_preview("visual") == b"content"
    else:
        transport = SimpleNamespace(
            execute=AsyncMock(return_value=WIRE), request_bytes=AsyncMock(return_value=b"content")
        )
        client = AsyncVisualsAPI(transport)

        async def run():
            assert isinstance(await client.retrieve("visual"), LogicalVisual)
            assert (
                await client.update("visual", VisualPatch(title="Renamed"))
            ).visual_id == "visual"
            assert await client.retrieve_content("visual") == b"content"
            assert await client.retrieve_preview("visual") == b"content"

        asyncio.run(run())
    request = transport.execute.call_args_list[1].args[0]
    assert request.path == "/smr/visuals/visual"
    assert request.body == {"title": "Renamed"}


def test_logical_visual_rejects_fabricated_lifecycle_and_boolean__SYN4005():
    with pytest.raises(ValueError):
        LogicalVisual.from_wire({**WIRE, "lifecycle": "draft"})
    with pytest.raises(ValueError):
        LogicalVisual.from_wire({**WIRE, "inaccessible_sources": []})
