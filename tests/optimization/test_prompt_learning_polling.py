from __future__ import annotations

import json

from synth_ai.sdk.optimization.internal import prompt_learning_polling


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


def test_poll_prompt_learning_until_complete(monkeypatch) -> None:
    payloads = [
        {"status": "running", "best_score": 0.1},
        {"status": "succeeded", "best_score": 0.5},
    ]

    def _fake_http_get(*_args, **_kwargs):
        return _DummyResponse(payloads.pop(0))

    monkeypatch.setattr(prompt_learning_polling, "http_get", _fake_http_get)

    result = prompt_learning_polling.poll_prompt_learning_until_complete(
        backend_url="http://example.com",
        api_key="key",
        job_id="pl_test",
        timeout=1.0,
        interval=0.0,
        progress=False,
        on_status=None,
        request_timeout=0.1,
    )

    assert result.succeeded
    assert result.best_score == 0.5
