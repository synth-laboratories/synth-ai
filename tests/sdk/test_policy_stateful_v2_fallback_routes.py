from __future__ import annotations

import asyncio

import pytest

from synth_ai.core.errors import HTTPError
from synth_ai.sdk.optimization.policy import v1 as policy_v1


class _RouteUnavailableHttpClient:
    calls: list[str] = []

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _RouteUnavailableHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post_json(self, path: str, *, json):  # noqa: ANN001
        del json
        _RouteUnavailableHttpClient.calls.append(path)
        if path == "/v2/candidates/submit":
            raise HTTPError(status=405, url="mock://submit-v2", message="method_not_allowed")
        raise AssertionError(f"unexpected path: {path}")

    async def get(self, path: str, *, params=None):  # noqa: ANN001
        del params
        _RouteUnavailableHttpClient.calls.append(path)
        if path in {
            "/v2/offline/jobs/pl_123/state/baseline-info",
            "/v2/offline/jobs/pl_123/state-envelope",
        }:
            raise HTTPError(status=404, url=f"mock://{path}", message="not_found")
        raise AssertionError(f"unexpected path: {path}")


class _HardFailureHttpClient:
    calls: list[str] = []

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _HardFailureHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post_json(self, path: str, *, json):  # noqa: ANN001
        del json
        _HardFailureHttpClient.calls.append(path)
        raise HTTPError(status=401, url="mock://submit", message="unauthorized")


def _job() -> policy_v1.PolicyOptimizationOfflineJob:
    return policy_v1.PolicyOptimizationOfflineJob(
        job_id="pl_123",
        backend_url="http://localhost:8080",
        api_key="test-key",
        api_version="v2",
    )


def test_submit_candidates_does_not_fallback_on_route_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RouteUnavailableHttpClient.calls = []
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _RouteUnavailableHttpClient)
    with pytest.raises(HTTPError) as exc_info:
        asyncio.run(
            _job().submit_candidates_async(
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
    assert _RouteUnavailableHttpClient.calls == ["/v2/candidates/submit"]


def test_state_reads_do_not_fallback_on_route_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RouteUnavailableHttpClient.calls = []
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _RouteUnavailableHttpClient)

    with pytest.raises(HTTPError) as baseline_exc:
        asyncio.run(_job().get_state_baseline_info_async())
    assert baseline_exc.value.status == 404
    assert _RouteUnavailableHttpClient.calls == ["/v2/offline/jobs/pl_123/state/baseline-info"]

    _RouteUnavailableHttpClient.calls = []
    with pytest.raises(HTTPError) as envelope_exc:
        asyncio.run(_job().get_state_envelope_async())
    assert envelope_exc.value.status == 404
    assert _RouteUnavailableHttpClient.calls == ["/v2/offline/jobs/pl_123/state-envelope"]


def test_submit_candidates_does_not_fallback_on_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _HardFailureHttpClient.calls = []
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _HardFailureHttpClient)

    with pytest.raises(HTTPError) as exc_info:
        asyncio.run(
            _job().submit_candidates_async(
                algorithm_kind="gepa",
                candidates=[],
            )
        )
    assert exc_info.value.status == 401
    assert _HardFailureHttpClient.calls == ["/v2/candidates/submit"]
