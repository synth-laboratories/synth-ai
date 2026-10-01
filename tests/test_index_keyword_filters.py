"""Keyword constraints and transport parity without provider or network calls."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from synth_ai.cli import cli
from synth_ai.mcp.research.tools.index import IndexPrivateSearchRequest, IndexSearchRequest
from synth_ai.sdk.index import PublicIndexClient, SearchFilters, SearchSpec
from test_index_cli_search_route import _paid_result, _run
from test_index_cli_search_route import seen as seen
from test_index_public_search import _delivered, _mount, _public_search_block, _routes, _tool


@pytest.mark.parametrize(
    "filters",
    [
        {"tags_none": ["obsolete", "obsolete"]},
        {"tags_none": [f"tag-{number}" for number in range(11)]},
        {"tags_none": ["not a valid identifier"]},
        {"tags_all": ["cybernetics"], "tags_none": ["cybernetics"]},
        {"tags_any": ["cybernetics"], "tags_none": ["cybernetics"]},
    ],
)
def test_invalid_keyword_filters_are_rejected(filters: dict) -> None:
    with pytest.raises(ValidationError):
        SearchFilters.model_validate(filters)


def test_empty_exclusions_preserve_existing_intent_bytes() -> None:
    intent = SearchSpec(query="q", filters=SearchFilters(tags_any=("optimization",)))
    assert json.loads(intent.model_dump_json())["filters"] == {
        "kinds": [],
        "research_areas": [],
        "workflow_stages": [],
        "tags_any": ["optimization"],
        "tags_all": [],
    }


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_keyword_schema_remains_typed(mode: str) -> None:
    assert SearchFilters.model_json_schema(mode=mode)["properties"]["tags_none"]["maxItems"] == 10


def test_partial_any_exclusion_remains_valid_and_serializes() -> None:
    filters = SearchFilters(tags_any=("optimization", "obsolete"), tags_none=("obsolete",))
    assert filters.model_dump(mode="json")["tags_none"] == ["obsolete"]


@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_anonymous_sdk_preserves_filter_intent(mode: str) -> None:
    with PublicIndexClient(base_url="https://api.example.test") as client:
        requests = _mount(client, lambda request: _delivered())
        client.public_search(
            "q",
            mode=mode,
            filters=SearchFilters(tags_all=("optimization",), tags_none=("obsolete",)),
        )
    assert len(requests) == 1
    assert "authorization" not in requests[0].headers
    body = json.loads(requests[0].content)
    assert body["mode"] == mode
    assert body["filters"]["tags_all"] == ["optimization"]
    assert body["filters"]["tags_none"] == ["obsolete"]


@pytest.mark.parametrize("route", ["public", "keyed"])
def test_cli_filter_options_reach_selected_route(seen: list[httpx.Request], route: str) -> None:
    options = [f"--{route}"]
    if route == "keyed":
        options += ["--api-key", "sk-test"]
    result = _run(
        [
            "q",
            *options,
            "--tag-any",
            "optimization",
            "--tag-any",
            "reasoning",
            "--tag-all",
            "benchmark",
            "--tag-none",
            "obsolete",
        ]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(seen[-1].content)
    assert body["filters"]["tags_any"] == ["optimization", "reasoning"]
    assert body["filters"]["tags_all"] == ["benchmark"]
    assert body["filters"]["tags_none"] == ["obsolete"]


@pytest.mark.parametrize("route", ["public", "keyed"])
def test_cli_contradiction_is_rejected_before_transport(
    seen: list[httpx.Request],
    route: str,
) -> None:
    options = [f"--{route}"]
    if route == "keyed":
        options += ["--api-key", "sk-test"]
    result = _run(["q", *options, "--tag-all", "cybernetics", "--tag-none", "cybernetics"])
    assert result.exit_code != 0
    assert "must not overlap" in result.stdout + result.stderr
    assert not seen


def test_durable_cli_preserves_private_scope_and_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SearchSpec] = []

    def create(spec: SearchSpec, *, idempotency_key: str):
        captured.append(spec)
        assert idempotency_key == "saved-request"
        return SimpleNamespace(snapshot=SimpleNamespace(model_dump=lambda **kwargs: {}))

    @contextmanager
    def client(**kwargs):
        yield SimpleNamespace(index=SimpleNamespace(searches=SimpleNamespace(create=create)))

    monkeypatch.setattr("synth_ai.SynthClient", client)
    result = CliRunner().invoke(
        cli,
        [
            "index",
            "searches",
            "create",
            "q",
            "--api-key",
            "sk-test",
            "--idempotency-key",
            "saved-request",
            "--private-collection",
            "testing-collection",
            "--tag-all",
            "cybernetics",
            "--tag-none",
            "obsolete",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0].scope.visibility == "private"
    assert captured[0].scope.collection_ids == ("testing-collection",)
    assert captured[0].filters.tags_all == ("cybernetics",)
    assert captured[0].filters.tags_none == ("obsolete",)


def test_mcp_anonymous_filters_reach_backend_and_schema() -> None:
    requests: list[httpx.Request] = []

    def search(request: httpx.Request):
        requests.append(request)
        return _delivered()

    tool = _tool(_routes(capabilities=_public_search_block(), search=search))
    tool.handler({"query": "q", "filters": {"tags_none": ["obsolete"]}})
    assert json.loads(requests[0].content)["filters"]["tags_none"] == ["obsolete"]
    assert "authorization" not in requests[0].headers
    assert tool.input_schema["$defs"]["SearchFilters"]["properties"]["tags_none"]["maxItems"] == 10


def test_mcp_nested_search_preserves_keyword_contract() -> None:
    request = IndexPrivateSearchRequest.model_validate(
        {
            "idempotency_key": "saved-request",
            "search": {
                "query": "q",
                "scope": {"visibility": "private", "collection_ids": ["testing-collection"]},
                "filters": {"tags_all": ["cybernetics"], "tags_none": ["obsolete"]},
            },
        }
    )
    assert request.search.filters.tags_none == ("obsolete",)
    assert request.search.scope.visibility == "private"
    with pytest.raises(ValidationError):
        IndexSearchRequest.model_validate(
            {
                "query": "q",
                "filters": {"tags_all": ["cybernetics"], "tags_none": ["cybernetics"]},
            }
        )


@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_keyed_sdk_private_filter_transport(mode: str) -> None:
    from synth_ai.client import SynthClient
    from test_index_deep_create_retry import _mount as mount_keyed

    spec = SearchSpec(
        query="Compare testing evidence",
        mode=mode,
        scope={"visibility": "private", "collection_ids": ["testing-collection"]},
        filters=SearchFilters(tags_all=("cybernetics",), tags_none=("obsolete",)),
        billing={"allow_wallet": True, "max_charge_cents": 25},
    )
    if mode == "fast":
        result = _paid_result()
        result["usage"]["billing_scope"] = "private"
        response = httpx.Response(200, json=result)
    else:
        response = httpx.Response(
            200,
            json={
                "search_id": "search-private-1",
                "spec": spec.model_dump(mode="json"),
                "state": "queued",
                "requested_mode": "deep",
                "created_at": "2026-10-01T00:00:00Z",
                "updated_at": "2026-10-01T00:00:00Z",
            },
        )
    with SynthClient(api_key="sk-test", base_url="https://api.example.test") as client:
        requests = mount_keyed(client, [response])
        if mode == "fast":
            client.index.search(spec, idempotency_key="private-intent-1")
        else:
            client.index.searches.create(spec, idempotency_key="private-intent-1")
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["idempotency-key"] == "private-intent-1"
    assert request.url.path == (
        "/api/v1/index/search" if mode == "fast" else "/api/v1/index/searches"
    )
    assert json.loads(request.content) == spec.model_dump(mode="json", exclude_none=mode == "deep")


@pytest.mark.parametrize("resource", ["contribution", "revision", "assessment", "asset"])
@pytest.mark.parametrize("search_id", [None, "60bba243-5ac1-4ff3-a5a8-c52e90648e5b"])
def test_exact_reads_forward_optional_receipt_without_changing_default_wire(
    resource: str,
    search_id: str | None,
) -> None:
    from synth_ai.sdk.index.client import ContributionsAPI
    from synth_ai.sdk.index.contracts import ContributionReference

    calls = []
    api = ContributionsAPI(lambda call: calls.append(call), asynchronous=False)
    reference = ContributionReference(contribution_id="contribution-1", revision_id="revision-1")
    if resource == "contribution":
        api.retrieve(reference.contribution_id, search_id=search_id)
    elif resource == "revision":
        api.revisions.retrieve(reference, search_id=search_id)
    elif resource == "assessment":
        api.assessments.list(reference, search_id=search_id)
    else:
        api.assets.retrieve(reference, "asset-1", search_id=search_id)
    assert len(calls) == 1
    method, path, kwargs = calls[0].request()
    assert method == "GET"
    assert path.startswith("/api/v1/index/contributions/contribution-1")
    if search_id is None:
        assert "params" not in kwargs
    else:
        assert kwargs["params"] == {"search_id": search_id}


def test_invalid_read_receipt_is_rejected_before_transport() -> None:
    from synth_ai.sdk.index.client import ContributionsAPI

    calls = []
    api = ContributionsAPI(lambda call: calls.append(call), asynchronous=False)
    with pytest.raises(ValueError, match="search_id must"):
        api.retrieve("contribution-1", search_id="not a safe identifier")
    assert calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_asset_transport_carries_receipt_for_sync_and_async(asynchronous: bool) -> None:
    import asyncio

    from synth_ai.client import AsyncSynthClient, SynthClient
    from synth_ai.sdk.index.contracts import ContributionReference
    from test_index_deep_create_retry import _mount as mount_keyed

    reference = ContributionReference(contribution_id="contribution-1", revision_id="revision-1")
    receipt = "60bba243-5ac1-4ff3-a5a8-c52e90648e5b"
    if asynchronous:

        async def run():
            async with AsyncSynthClient(
                api_key="sk-test", base_url="https://api.example.test"
            ) as client:
                requests = mount_keyed(client, [httpx.Response(200, content=b"evidence")])
                content = await client.index.contributions.assets.retrieve(
                    reference, "asset-1", search_id=receipt
                )
            return content, requests

        content, requests = asyncio.run(run())
    else:
        with SynthClient(api_key="sk-test", base_url="https://api.example.test") as client:
            requests = mount_keyed(client, [httpx.Response(200, content=b"evidence")])
            content = client.index.contributions.assets.retrieve(
                reference, "asset-1", search_id=receipt
            )
    assert content == b"evidence"
    assert requests[0].url.params["search_id"] == receipt
    assert requests[0].headers["authorization"] == "Bearer sk-test"


def test_cli_revision_status_forwards_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    def retrieve(reference, **kwargs):
        captured.append((reference, kwargs))
        return {}

    @contextmanager
    def client(**kwargs):
        yield SimpleNamespace(
            index=SimpleNamespace(
                contributions=SimpleNamespace(
                    revisions=SimpleNamespace(retrieve=retrieve),
                )
            )
        )

    monkeypatch.setattr("synth_ai.SynthClient", client)
    receipt = "60bba243-5ac1-4ff3-a5a8-c52e90648e5b"
    result = CliRunner().invoke(
        cli,
        [
            "index",
            "contribution",
            "status",
            "contribution-1",
            "revision-1",
            "--api-key",
            "sk-test",
            "--backend-url",
            "https://api.example.test",
            "--search-id",
            receipt,
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0][1] == {"search_id": receipt}


def test_mcp_exact_reads_forward_receipt_and_reject_anonymous_receipt() -> None:
    from synth_ai.mcp.research.tools.index import build_index_tools
    from synth_ai.sdk.index.client import IndexAPI

    captured = []

    def retrieve(*args, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(model_dump=lambda **kwargs: {"ok": True})

    client = object.__new__(IndexAPI)
    client.contributions = SimpleNamespace(
        retrieve=retrieve,
        revisions=SimpleNamespace(retrieve=retrieve),
    )

    @contextmanager
    def factory():
        yield client

    tools = {tool.name: tool for tool in build_index_tools(factory)}
    receipt = "60bba243-5ac1-4ff3-a5a8-c52e90648e5b"
    assert tools["index_get_contribution"].handler(
        {
            "contribution_id": "contribution-1",
            "search_id": receipt,
        }
    ) == {"ok": True}
    assert tools["index_contribution_status"].handler(
        {
            "reference": {"contribution_id": "contribution-1", "revision_id": "revision-1"},
            "search_id": receipt,
        }
    ) == {"ok": True}
    assert [kwargs for _, kwargs in captured] == [{"search_id": receipt}] * 2
    assert "search_id" in tools["index_get_contribution"].input_schema["properties"]

    @contextmanager
    def anonymous():
        yield SimpleNamespace(contributions=SimpleNamespace(retrieve=retrieve))

    tools = {tool.name: tool for tool in build_index_tools(anonymous)}
    with pytest.raises(ValueError, match="requires an API key"):
        tools["index_get_contribution"].handler(
            {
                "contribution_id": "contribution-1",
                "search_id": receipt,
            }
        )
    assert len(captured) == 2
