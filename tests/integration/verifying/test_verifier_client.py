"""Integration tests for Verifier Client.

These tests validate verifier graph completions:
1. Client initialization
2. Scoring with trace data
3. Error handling
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ["SYNTH_SILENCE_EXPERIMENTAL"] = "1"

pytestmark = [pytest.mark.integration]


class TestVerifierClientInit:
    """Test VerifierClient initialization."""

    def test_client_creates_with_valid_params(self) -> None:
        from synth_ai.sdk.graphs import VerifierClient

        client = VerifierClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        assert client is not None

    def test_client_accepts_custom_timeout(self) -> None:
        from synth_ai.sdk.graphs import VerifierClient

        client = VerifierClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
            timeout=120.0,
        )

        assert client is not None


class TestVerifierClientEvaluate:
    """Test VerifierClient evaluation functionality."""

    @pytest.fixture
    def client(self):
        from synth_ai.sdk.graphs import VerifierClient

        return VerifierClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self, client, monkeypatch) -> None:
        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {
                    "output": {
                        "status": "ok",
                        "outcome_review": {"total": 0.85, "criteria": {"accuracy": {"score": 0.85}}},
                        "event_reviews": [],
                    }
                }

        monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        trace = {
            "session_id": "test-session",
            "event_history": [{"type": "lm_call", "metadata": {"turn": 1}}],
        }

        result = await client.evaluate(
            session_trace=trace,
            rubric={"event": [], "outcome": [{"id": "accuracy", "weight": 1.0, "description": "Correctness"}]},
            policy_name="test_policy",
            task_app_id="math",
            options={"outcome": True},
            job_id="zero_shot_verifier_single",
        )

        assert result["output"]["status"] == "ok"
        assert result["output"]["outcome_review"]["total"] == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_captures_request_params(self, client, monkeypatch) -> None:
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {"output": {"status": "ok", "outcome_review": {}, "event_reviews": []}}

        monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        trace = {"session_id": "test", "event_history": []}

        await client.evaluate(
            session_trace=trace,
            rubric={"event": [], "outcome": []},
            policy_name="my_policy",
            task_app_id="heartdisease",
            options={"outcome": True, "provider": "groq"},
            job_id="zero_shot_verifier_single",
        )

        assert captured_payload["input"]["policy_name"] == "my_policy"
        assert captured_payload["input"]["task_app"]["id"] == "heartdisease"
        assert captured_payload["input"]["options"]["provider"] == "groq"


class TestVerifierClientErrorHandling:
    """Test VerifierClient error handling."""

    @pytest.fixture
    def client(self):
        from synth_ai.sdk.graphs import VerifierClient

        return VerifierClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_handles_validation_error(self, client, monkeypatch) -> None:
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=400,
                    url="/api/graphs/completions",
                    message="validation_error",
                    body_snippet="Invalid rubric",
                )

        monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(ValueError, match="graph_completions_validation_error"):
            await client.evaluate(
                session_trace={},
                rubric={"event": [], "outcome": []},
                policy_name="test",
                task_app_id="test",
                options={},
                job_id="zero_shot_verifier_single",
            )

    @pytest.mark.asyncio
    async def test_handles_auth_error(self, client, monkeypatch) -> None:
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=401,
                    url="/api/graphs/completions",
                    message="unauthorized",
                    body_snippet="Invalid API key",
                )

        monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(PermissionError, match="graph_completions_auth_error"):
            await client.evaluate(
                session_trace={},
                rubric={"event": [], "outcome": []},
                policy_name="test",
                task_app_id="test",
                options={},
                job_id="zero_shot_verifier_single",
            )

    @pytest.mark.asyncio
    async def test_handles_not_found_error(self, client, monkeypatch) -> None:
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=404,
                    url="/api/graphs/completions",
                    message="not_found",
                    body_snippet="Graph not found",
                )

        monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(FileNotFoundError, match="graph_completions_not_found"):
            await client.evaluate(
                session_trace={},
                rubric={"event": [], "outcome": []},
                policy_name="test",
                task_app_id="nonexistent",
                options={},
                job_id="zero_shot_verifier_single",
            )


class TestGraphTarget:
    """Test GraphTarget TypedDict."""

    def test_graph_target_accepts_valid_fields(self) -> None:
        from synth_ai.sdk.graphs import GraphTarget

        target: GraphTarget = {
            "kind": "zero_shot",
            "verifier_shape": "zero_shot_verifier_single",
        }

        assert target["kind"] == "zero_shot"
        assert target["verifier_shape"] == "zero_shot_verifier_single"
