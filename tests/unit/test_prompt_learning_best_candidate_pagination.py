from __future__ import annotations

import asyncio
from typing import Any, Dict

from synth_ai.sdk.optimization.internal.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
)
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.optimization.models import PolicyCandidate


def _build_job() -> PromptLearningJob:
    config = PromptLearningJobConfig(
        config_dict={"prompt_learning": {"algorithm": "gepa"}},
        backend_url="https://backend.test",
        api_key="api-key",
        container_api_key="container-key",
    )
    return PromptLearningJob(config, job_id="pl_pagination", skip_health_check=True)


def test_list_candidates_async_forwards_pagination_parameters(monkeypatch) -> None:
    job = _build_job()
    captured: Dict[str, Any] = {}

    async def fake_list_candidates(self, job_id: str, **kwargs: Any) -> Dict[str, Any]:
        captured["self"] = self
        captured["job_id"] = job_id
        captured["kwargs"] = kwargs
        return {"items": [{"candidate_id": "cand_3"}], "next_cursor": None}

    monkeypatch.setattr(PromptLearningClient, "list_candidates", fake_list_candidates)

    result = asyncio.run(
        job.list_candidates_async(
            status="evaluated",
            limit=2,
            cursor="cursor-2",
            sort="objective:desc,created_at:desc",
            include="prompt_text",
        )
    )
    assert result == {"items": [{"candidate_id": "cand_3"}], "next_cursor": None}
    assert captured["job_id"] == "pl_pagination"
    assert captured["kwargs"] == {
        "algorithm": None,
        "mode": None,
        "status": "evaluated",
        "limit": 2,
        "cursor": "cursor-2",
        "sort": "objective:desc,created_at:desc",
        "include": "prompt_text",
    }


def test_list_candidates_sync_wrapper(monkeypatch) -> None:
    job = _build_job()

    async def fake_list_candidates_async(**kwargs: Any) -> Dict[str, Any]:
        assert kwargs["limit"] == 3
        assert kwargs["cursor"] == "cursor-3"
        return {"items": [{"candidate_id": "cand_5"}], "next_cursor": None}

    monkeypatch.setattr(job, "list_candidates_async", fake_list_candidates_async)

    result = job.list_candidates(limit=3, cursor="cursor-3")
    assert result == {"items": [{"candidate_id": "cand_5"}], "next_cursor": None}


def test_list_candidates_typed_async_returns_typed_page(monkeypatch) -> None:
    job = _build_job()

    async def fake_list_candidates_typed(self, job_id: str, **kwargs: Any):
        assert job_id == "pl_pagination"
        assert kwargs["include"] == "artifact_payload"
        from synth_ai.sdk.optimization.models import PolicyCandidatePage

        return PolicyCandidatePage.from_dict(
            {
                "items": [
                    {
                        "candidate_id": "cand_typed_1",
                        "artifact_kind": "program_code",
                        "artifact_payload": {"candidate_code": "def solve(x): return x"},
                        "artifact_preview": "def solve(x): return x",
                    }
                ],
                "next_cursor": None,
                "job_id": job_id,
            }
        )

    monkeypatch.setattr(
        PromptLearningClient,
        "list_candidates_typed",
        fake_list_candidates_typed,
    )

    page = asyncio.run(job.list_candidates_typed_async(include="artifact_payload"))
    assert page.job_id == "pl_pagination"
    assert len(page.items) == 1
    assert isinstance(page.items[0], PolicyCandidate)
    assert page.items[0].artifact_kind == "program_code"


def test_get_candidate_typed_sync_wrapper(monkeypatch) -> None:
    job = _build_job()

    async def fake_get_candidate_typed(self, job_id: str, candidate_id: str) -> PolicyCandidate:
        assert job_id == "pl_pagination"
        assert candidate_id == "cand_typed_2"
        return PolicyCandidate.from_dict(
            {
                "candidate_id": "cand_typed_2",
                "artifact_kind": "dsl_config",
                "artifact_payload": {"alpha": 1.1},
                "status": "evaluated",
            }
        )

    monkeypatch.setattr(
        PromptLearningClient,
        "get_candidate_typed",
        fake_get_candidate_typed,
    )

    candidate = job.get_candidate_typed("cand_typed_2")
    assert isinstance(candidate, PolicyCandidate)
    assert candidate.candidate_id == "cand_typed_2"
    assert candidate.artifact_kind == "dsl_config"
