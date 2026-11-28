"""Cursor paging and filters on notebook artifact, experiment and research-log lists (SYN-6066).

Each surface (async SDK, sync SDK, CLI and MCP) must forward `limit`, `cursor` and the
list filters as wire query parameters and return the page's `next_cursor`/`has_more`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient, cli
from synth_ai.mcp.server import call_tool

BASE_URL = "https://paging.example.test"
NEXT = "cur_page_2"


def _recording_backend(seen: list[tuple[str, dict[str, str]]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"items": [], "next_cursor": NEXT, "has_more": True})

    return httpx.MockTransport(handler)


ARTIFACT_QUERY = {
    "run_id": "run_A",
    "kind": "svg",
    "origin": "human",
    "q": "50%_off",
    "limit": "5",
    "cursor": "cur_1",
}
EXPERIMENT_QUERY = {"run_id": "run_A", "q": "sweep", "limit": "7", "cursor": "cur_1"}
LOG_QUERY = {
    "query": "pack",
    "q": "pack",
    "category": "observation",
    "run_id": "run_A",
    "author_kind": "human",
    "limit": "9",
    "cursor": "cur_1",
}


async def test_async_sdk_forwards_paging_and_filters() -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    async with (
        httpx.AsyncClient(transport=_recording_backend(seen)) as http,
        AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client,
    ):
        artifacts = await client.artifacts.list_artifacts(
            "nb_1",
            run_id="run_A",
            kind="svg",
            origin="human",
            q="50%_off",
            limit=5,
            cursor="cur_1",
        )
        experiments = await client.experiments.list_experiments(
            "nb_1", run_id="run_A", q="sweep", limit=7, cursor="cur_1"
        )
        entries = await client.research_logs.search_entries(
            "nb_1",
            query="pack",
            q="pack",
            category="observation",
            run_id="run_A",
            author_kind="human",
            limit=9,
            cursor="cur_1",
        )
        # Omitted arguments stay off the wire so the server default page applies.
        await client.artifacts.list_artifacts("nb_1")
    for page in (artifacts, experiments, entries):
        assert page.has_more is True
        assert page.next_cursor == NEXT
    assert seen == [
        ("/v1/notebooks/nb_1/artifacts", ARTIFACT_QUERY),
        ("/v1/notebooks/nb_1/experiments", EXPERIMENT_QUERY),
        ("/v1/notebooks/nb_1/research-log", LOG_QUERY),
        ("/v1/notebooks/nb_1/artifacts", {}),
    ]


def test_sync_sdk_forwards_paging() -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    http = httpx.AsyncClient(transport=_recording_backend(seen))
    with SynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client:
        page = client.experiments.list_experiments("nb_1", limit=3, cursor="cur_1")
    assert (page.next_cursor, page.has_more) == (NEXT, True)
    assert seen == [("/v1/notebooks/nb_1/experiments", {"limit": "3", "cursor": "cur_1"})]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "artifacts",
                "list",
                "--notebook-id",
                "nb_1",
                "--run-id",
                "run_A",
                "--kind",
                "svg",
                "--origin",
                "human",
                "--q",
                "50%_off",
                "--limit",
                "5",
                "--cursor",
                "cur_1",
            ],
            ("/v1/notebooks/nb_1/artifacts", ARTIFACT_QUERY),
        ),
        (
            [
                "experiments",
                "list",
                "--notebook-id",
                "nb_1",
                "--run-id",
                "run_A",
                "--q",
                "sweep",
                "--limit",
                "7",
                "--cursor",
                "cur_1",
            ],
            ("/v1/notebooks/nb_1/experiments", EXPERIMENT_QUERY),
        ),
        (
            [
                "research-log",
                "search",
                "--notebook-id",
                "nb_1",
                "--query",
                "pack",
                "--q",
                "pack",
                "--category",
                "observation",
                "--run-id",
                "run_A",
                "--author-kind",
                "human",
                "--limit",
                "9",
                "--cursor",
                "cur_1",
            ],
            ("/v1/notebooks/nb_1/research-log", LOG_QUERY),
        ),
    ],
)
def test_cli_list_flags_forward_paging_and_filters(
    argv: list[str],
    expected: tuple[str, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setenv("SYNTH_API_KEY", "sk_org_a")
    code = cli.main(
        ["--base-url", BASE_URL, *argv],
        http_client=httpx.AsyncClient(transport=_recording_backend(seen)),
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["next_cursor"], payload["has_more"]) == (NEXT, True)
    assert seen == [expected]


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        (
            "swarms_artifacts_list",
            {
                "notebook_id": "nb_1",
                "run_id": "run_A",
                "kind": "svg",
                "origin": "human",
                "q": "50%_off",
                "limit": 5,
                "cursor": "cur_1",
            },
            ("/v1/notebooks/nb_1/artifacts", ARTIFACT_QUERY),
        ),
        (
            "swarms_experiments_list",
            {"notebook_id": "nb_1", "run_id": "run_A", "q": "sweep", "limit": 7, "cursor": "cur_1"},
            ("/v1/notebooks/nb_1/experiments", EXPERIMENT_QUERY),
        ),
        (
            "swarms_research_log_search",
            {
                "notebook_id": "nb_1",
                "query": "pack",
                "q": "pack",
                "category": "observation",
                "run_id": "run_A",
                "author_kind": "human",
                "limit": 9,
                "cursor": "cur_1",
            },
            ("/v1/notebooks/nb_1/research-log", LOG_QUERY),
        ),
    ],
)
async def test_mcp_list_tools_forward_paging_and_filters(
    tool: str,
    arguments: dict[str, str | int],
    expected: tuple[str, dict[str, str]],
) -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    async with (
        httpx.AsyncClient(transport=_recording_backend(seen)) as http,
        AsyncSynthClient("sk_org_a", base_url=BASE_URL, http_client=http) as client,
    ):
        payload = json.loads(await call_tool(client, tool, dict(arguments)))
    assert (payload["next_cursor"], payload["has_more"]) == (NEXT, True)
    assert seen == [expected]
