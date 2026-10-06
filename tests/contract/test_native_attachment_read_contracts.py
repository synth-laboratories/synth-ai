"""SYN-3999 typed native evidence survives SDK sync/async and real MCP handler."""

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from synth_ai.sdk.research.scientific_records import ScientificRecordsAPI, AsyncScientificRecordsAPI
from synth_ai.mcp.research.tools.scientific_records import build_scientific_record_tools
from synth_ai.sdk.research.errors import ResearchApiError

PAGE = json.loads(
    (Path(__file__).with_name("fixtures") / "native_attachment_page.json").read_text()
)
PROJECT = PAGE["attachments"][0]["attachment"]["scope"]["project_id"]


class Transport:
    def __init__(self):
        self.calls = []

    def request_json(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return PAGE


def test_sync_async_and_mcp_read_null_record_without_erasing_delivery_receipt__SYN3999():
    transport = Transport()
    api = ScientificRecordsAPI(transport)
    page = api.native_attachments(PROJECT, "forge-only-experiment")
    native = page.attachments[0]
    assert native.attachment.payload.value is None
    assert native.attachment.payload.outcome == "null"
    assert native.attachment.producer.authority == "mloky"
    assert native.status == "attached" and native.receipt.reference.kind == "experiment"

    class AsyncTransport(Transport):
        async def request_json(self, *args, **kwargs):
            return super().request_json(*args, **kwargs)

    assert (
        asyncio.run(
            AsyncScientificRecordsAPI(AsyncTransport()).native_attachments(
                PROJECT, "forge-only-experiment"
            )
        )
        == page
    )

    @contextmanager
    def clients(_):
        yield SimpleNamespace(records=api)

    selected = next(
        tool
        for tool in build_scientific_record_tools(clients)
        if tool.name == "research_get_native_attachments"
    )
    wire = selected.handler({"project_id": PROJECT, "record_id": "forge-only-experiment"})
    assert json.loads(json.dumps(wire)) == page.model_dump(mode="json")


def test_native_page_scope_substitution_and_boolean_limits_refuse__SYN3999():
    api = ScientificRecordsAPI(Transport())
    with pytest.raises(ResearchApiError):
        api.native_attachments("foreign", "forge-only-experiment")
    with pytest.raises(ResearchApiError):
        api.native_attachments(PROJECT, "foreign-experiment")
    with pytest.raises(ValueError):
        api.native_attachments(PROJECT, "forge-only-experiment", limit=True)
