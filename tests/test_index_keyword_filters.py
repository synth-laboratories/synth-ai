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

from test_index_cli_search_route import _run, seen
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
