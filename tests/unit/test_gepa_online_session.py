from __future__ import annotations

from typing import Any, List

import pytest

from synth_ai.sdk.optimization.policy import gepa_online_session as module
from synth_ai.sdk.optimization.policy.gepa_online_session import GepaOnlineSession


@pytest.mark.asyncio
async def test_get_prompt_urls_async_raises_on_non_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRustClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "DummyRustClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args
            return None

        async def get(self, path: str, params: Any = None) -> List[str]:
            del path, params
            return ["bad-shape"]

    monkeypatch.setattr(module, "RustCoreHttpClient", DummyRustClient)
    session = GepaOnlineSession(
        session_id="session-bad-prompt",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    with pytest.raises(ValueError, match="GEPA prompt endpoint"):
        await session.get_prompt_urls_async()
