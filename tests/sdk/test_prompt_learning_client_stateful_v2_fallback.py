from __future__ import annotations

import asyncio

import pytest

from synth_ai.core.errors import HTTPError
from synth_ai.sdk.optimization.internal.learning import prompt_learning_client as plc_module


class _RouteUnavailableHttpClient:
    post_calls: list[str] = []
    get_calls: list[str] = []

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _RouteUnavailableHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post_json(self, path: str, *, json):  # noqa: ANN001
        del json
        _RouteUnavailableHttpClient.post_calls.append(path)
        if path == "/v2/candidates/submit":
            raise HTTPError(status=405, url="mock://submit-v2", message="method_not_allowed")
        raise AssertionError(f"unexpected POST path: {path}")

    async def get(self, path: str, *, params=None):  # noqa: ANN001
        del params
        _RouteUnavailableHttpClient.get_calls.append(path)
        if path in {
            "/v2/offline/jobs/pl_123/state/baseline-info",
            "/v2/offline/jobs/pl_123/state-envelope",
        }:
            raise HTTPError(status=404, url=f"mock://{path}", message="not_found")
        raise AssertionError(f"unexpected GET path: {path}")


def _client() -> plc_module.PromptLearningClient:
    return plc_module.PromptLearningClient(
        base_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def test_prompt_learning_client_submit_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RouteUnavailableHttpClient.post_calls = []
    _RouteUnavailableHttpClient.get_calls = []
    monkeypatch.setattr(plc_module, "RustCoreHttpClient", _RouteUnavailableHttpClient)

    with pytest.raises(HTTPError) as exc_info:
        asyncio.run(
            _client().submit_candidates(
                job_id="pl_123",
                algorithm_kind="gepa",
                candidates=[
                    {
                        "candidate_type": "gepa_prompt_candidate",
                        "stage_prompts": [{"stage_id": "root", "instruction_text": "x"}],
                    }
                ],
            )
        )
    assert exc_info.value.status == 405
    assert _RouteUnavailableHttpClient.post_calls == ["/v2/candidates/submit"]


def test_prompt_learning_client_state_reads_have_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RouteUnavailableHttpClient.post_calls = []
    _RouteUnavailableHttpClient.get_calls = []
    monkeypatch.setattr(plc_module, "RustCoreHttpClient", _RouteUnavailableHttpClient)

    with pytest.raises(HTTPError) as baseline_exc:
        asyncio.run(_client().get_state_baseline_info("pl_123"))
    assert baseline_exc.value.status == 404
    assert _RouteUnavailableHttpClient.get_calls == ["/v2/offline/jobs/pl_123/state/baseline-info"]

    _RouteUnavailableHttpClient.get_calls = []
    with pytest.raises(HTTPError) as envelope_exc:
        asyncio.run(_client().get_state_envelope("pl_123"))
    assert envelope_exc.value.status == 404
    assert _RouteUnavailableHttpClient.get_calls == ["/v2/offline/jobs/pl_123/state-envelope"]
