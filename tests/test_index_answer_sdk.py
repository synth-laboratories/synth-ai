"""Typed sync/async and MCP witnesses for grounded Index answers."""

import asyncio

import pytest
from pydantic import ValidationError
from synth_ai.mcp.research.tools.index import build_index_tools
from synth_ai.sdk.index.answer import AnswerResult, AnswerStatus
from synth_ai.sdk.index.client import _IndexRoot


def _payload():
    quote = "ColBERT uses token-level MaxSim."
    return {
        "answer_id": "answer-1",
        "search_id": "search-1",
        "request_id": "answer_request_1",
        "query": "How does ColBERT rerank?",
        "mode": "fast",
        "status": "answered",
        "answer": "It uses token-level MaxSim.",
        "claims": [{"text": "It uses token-level MaxSim.", "citation_ids": ["c1"]}],
        "citations": [
            {
                "citation_id": "c1",
                "reference": {
                    "contribution_id": "contribution-1",
                    "revision_id": "revision-1",
                },
                "asset_id": "report",
                "chunk_id": "chunk-1",
                "content_sha256": "a" * 64,
                "start_byte": 4,
                "end_byte": 4 + len(quote.encode()),
                "quote": quote,
            }
        ],
        "retrieval_versions": {
            "corpus_generation": "corpus.v1",
            "ranker_version": "ranker.v1",
            "parser_version": "parser.v1",
            "taxonomy_version": "taxonomy.v1",
            "reranker_version": "colbert.v1",
        },
        "admission_policy_version": "synth.index.answer-admission.v1",
        "synthesis_policy_version": "synth.index.cited-synthesis.v1",
        "model_version": "deepseek-ai/DeepSeek-V4.1-Flash",
        "usage": {"inference_cost_usd_micros": 10},
    }


def test_sync_answer_uses_explicit_operation_and_idempotency() -> None:
    calls = []

    def run(call):
        calls.append(call.request())
        return call.parse(_payload())

    result = _IndexRoot(run, asynchronous=False).answer(
        query="How does ColBERT rerank?",
        idempotency_key="answer-1",
    )
    assert result.status is AnswerStatus.ANSWERED
    assert calls == [
        (
            "POST",
            "/api/v1/index/answer",
            {
                "operation_id": "index.answer",
                "json_body": {
                    "query": "How does ColBERT rerank?",
                    "mode": "fast",
                    "scope": {"visibility": "public", "collection_ids": []},
                    "filters": {
                        "kinds": [],
                        "research_areas": [],
                        "workflow_stages": [],
                        "tags_any": [],
                        "tags_all": [],
                    },
                    "content": {
                        "max_results": 5,
                        "max_excerpts_per_result": 2,
                    },
                    "max_answer_tokens": 1024,
                },
                "headers": {"Idempotency-Key": "answer-1"},
            },
        )
    ]


def test_async_answer_returns_the_same_typed_contract() -> None:
    async def run(call):
        return call.parse(_payload())

    async def execute():
        return await _IndexRoot(run, asynchronous=True).answer(
            query="How does ColBERT rerank?",
            idempotency_key="answer-2",
        )

    result = asyncio.run(execute())
    assert isinstance(result, AnswerResult)
    assert result.citations[0].citation_id == "c1"


def test_answer_contract_rejects_fabricated_citation() -> None:
    payload = _payload()
    payload["claims"][0]["citation_ids"] = ["c2"]
    with pytest.raises(ValidationError):
        AnswerResult.model_validate(payload)


def test_unauthenticated_mcp_does_not_advertise_paid_answer_tool() -> None:
    tools = build_index_tools(lambda: None, include_answer=False)
    assert "index_answer" not in {tool.name for tool in tools}
