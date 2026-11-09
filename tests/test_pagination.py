import asyncio
import copy
from contextlib import aclosing

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import PaginationError

CASES = [
    ("swarms", "iter_runs", "/v1/runs", ()),
    ("swarms", "iter_files", "/v1/files", ()),
    ("swarms", "iter_outputs", "/v1/runs/{run_id}/outputs", ("run_AAAAAAAAAAAAAAAAAAAAAA",)),
    ("swarms", "iter_tasks", "/v1/runs/{run_id}/tasks", ("run_AAAAAAAAAAAAAAAAAAAAAA",)),
    ("swarms", "iter_events", "/v1/runs/{run_id}/events", ("run_AAAAAAAAAAAAAAAAAAAAAA",)),
    ("account.api_keys", "iter_keys", "/v1/api-keys", ()),
    ("account", "iter_usage_records", "/v1/usage/records", ()),
    ("billing", "iter_history", "/v1/billing/history", ()),
    ("billing", "iter_statements", "/v1/billing/statements", ()),
]


def method(api, resource, name):
    for component in resource.split("."):
        api = getattr(api, component)
    return getattr(api, name)


@pytest.mark.parametrize("resource,name,path,args", CASES)
@pytest.mark.parametrize("sync", [False, True])
async def test_lazy_page_iterators_preserve_filters_and_types(
    contract, resource, name, path, args, sync
):
    example = contract["paths"][path]["get"]["responses"]["200"]["content"]["application/json"][
        "example"
    ]
    calls = []

    def handle(request):
        calls.append(request)
        page = copy.deepcopy(example)
        page.update(has_more=len(calls) == 1, next_cursor="next" if len(calls) == 1 else None)
        return httpx.Response(200, json=page)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        if sync:
            with SynthClient("fixture", base_url="https://example.test", http_client=http) as api:
                iterator = method(api, resource, name)(*args, limit=1)
                assert not calls
                result = list(iterator)
        else:
            async with AsyncSynthClient(
                "fixture", base_url="https://example.test", http_client=http
            ) as api:
                iterator = method(api, resource, name)(*args, limit=1)
                assert not calls
                result = [item async for item in iterator]
    assert len(calls) == 2
    assert all(request.url.params["limit"] == "1" for request in calls)
    assert calls[1].url.params["cursor"] == "next"
    assert len(result) == len(example["items"]) * 2


@pytest.mark.parametrize("cursors", [["A", "A"], ["A", "B", "A"], [""], [None], ["x" * 1025]])
async def test_bad_or_cyclic_cursor_refused_before_repeated_fetch(contract, cursors):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, json=dict(items=[], has_more=True, next_cursor=cursors[len(calls) - 1])
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises((PaginationError, ValueError)):
                _ = [item async for item in api.swarms.iter_files()]
    assert len(calls) == len(cursors)


async def test_stopping_iterator_does_not_prefetch(contract):
    example = contract["paths"]["/v1/runs"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]
    example = {
        **example,
        "items": [
            contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        ],
    }
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={**example, "has_more": True, "next_cursor": "next"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            async with aclosing(api.swarms.iter_runs()) as items:
                await anext(items)
    assert len(calls) == 1


async def test_cancellation_releases_pending_page_read():
    ready, cancelled = asyncio.Event(), asyncio.Event()

    async def handle(request):
        ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            iterator = api.swarms.iter_files()
            task = asyncio.create_task(anext(iterator))
            await ready.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await iterator.aclose()
    assert cancelled.is_set()
