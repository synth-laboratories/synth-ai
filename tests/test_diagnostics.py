import dataclasses
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient
from synth_ai._transport import RetryPolicy
from synth_ai.diagnostics import DiagnosticPhase
from synth_ai.errors import ClientClosedError, local_problem


async def test_diagnostics_are_safe_and_hook_failures_cannot_change_outcome(contract):
    events, calls = [], []
    snapshot = contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["example"]

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                503,
                json=local_problem("busy", "admission")
                .model_copy(update={"retryable": True})
                .model_dump(mode="json"),
            )
        return httpx.Response(202, json=snapshot)

    def hook(event):
        events.append(event)
        raise RuntimeError("sink failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "secret-key",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(backoff_seconds=0, jitter_seconds=0),
            diagnostic_hook=hook,
        ) as api:
            await api.swarms.launch("private prompt", budget="1", idempotency_key="private-key")
    assert [event.phase for event in events] == [DiagnosticPhase.RETRY, DiagnosticPhase.SUCCEEDED]
    assert events[-1].attempts == 2 and events[-1].elapsed_seconds >= 0
    encoded = json.dumps([dataclasses.asdict(event) for event in events])
    assert all(secret not in encoded for secret in ["secret-key", "private prompt", "private-key"])
    assert len(events[0].idempotency_correlation) == 64


async def test_closed_sdk_does_not_close_injected_client_or_allow_new_requests():
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: pytest.fail("network used"))
    )
    api = AsyncSynthClient("key", base_url="https://example.test", http_client=http)
    await api.close()
    await api.close()
    assert not http.is_closed
    with pytest.raises(ClientClosedError):
        await api.swarms.get("run")
    await http.aclose()


@pytest.mark.parametrize("operation", ["get", "follow", "download"])
async def test_injected_redirect_policy_cannot_forward_credentials(operation, tmp_path):
    from synth_ai.errors import MalformedErrorResponseError

    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.host == "example.test"
        return httpx.Response(307, headers={"Location": "https://unrelated.example.test/private"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), follow_redirects=True
    ) as http:
        async with AsyncSynthClient(
            "secret", base_url="https://example.test", http_client=http
        ) as api:
            with pytest.raises(MalformedErrorResponseError) as caught:
                if operation == "get":
                    await api.swarms.get("run")
                elif operation == "follow":
                    await anext(api.swarms.follow("run"))
                else:
                    await api.swarms.download_output(
                        "run",
                        "out",
                        str(tmp_path / "download"),
                        revision_id="rev",
                        digest_sha256="0" * 64,
                    )
    assert caught.value.problem.upstream_status == 307 and len(calls) == 1


async def test_deadline_suppresses_retry_before_another_request():
    from synth_ai.diagnostics import RetrySuppressionReason
    from synth_ai.errors import SynthError

    calls, events = [], []

    def handle(request):
        calls.append(request)
        problem = local_problem("busy", "admission").model_copy(update={"retryable": True})
        return httpx.Response(
            503, headers={"Retry-After": "20"}, json=problem.model_dump(mode="json")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "key",
            base_url="https://example.test",
            http_client=http,
            retry_policy=RetryPolicy(deadline_seconds=10, jitter_seconds=0),
            diagnostic_hook=events.append,
        ) as api:
            with pytest.raises(SynthError):
                await api.swarms.get("run")
    assert len(calls) == 1
    assert [event.phase for event in events] == [
        DiagnosticPhase.RETRY_SUPPRESSED,
        DiagnosticPhase.FAILED,
    ]
    suppressed = events[0].suppressed_retry
    assert events[0].attempts == 1 and suppressed.reason is RetrySuppressionReason.DEADLINE
    assert suppressed.delay_seconds == 20 and 0 < suppressed.remaining_seconds <= 10
