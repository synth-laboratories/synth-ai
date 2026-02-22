from __future__ import annotations

import logging

from synth_ai.sdk.container._impl.server import ContainerConfig, create_container, log_info


def _make_cfg(*, cors_origins=None) -> ContainerConfig:  # type: ignore[no-untyped-def]
    async def rollout(request, fastapi_request):  # pragma: no cover - not exercised
        del request, fastapi_request
        return {"reward_info": {}}

    return ContainerConfig(
        app_id="unit-cors-log",
        name="Unit Cors Log",
        description="unit",
        provide_taskset_description=lambda: {"splits": ["train"], "sizes": {"train": 1}},
        provide_task_instances=lambda seeds: [],
        rollout=rollout,
        cors_origins=cors_origins,
    )


def test_log_info_emits_info_record(caplog) -> None:  # type: ignore[no-untyped-def]
    with caplog.at_level(logging.INFO):
        log_info("container-start", ctx={"app_id": "demo"})
    assert any("container-start" in rec.message for rec in caplog.records)
    assert any("app_id" in rec.message for rec in caplog.records)


def test_create_container_cors_empty_origins_disables_credentials() -> None:
    app = create_container(_make_cfg(cors_origins=[]))
    cors = next((m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"), None)
    assert cors is not None
    assert cors.kwargs["allow_origins"] == ["*"]
    assert cors.kwargs["allow_credentials"] is False


def test_create_container_cors_explicit_origins_keeps_credentials_enabled() -> None:
    app = create_container(_make_cfg(cors_origins=["https://example.com"]))
    cors = next((m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"), None)
    assert cors is not None
    assert cors.kwargs["allow_origins"] == ["https://example.com"]
    assert cors.kwargs["allow_credentials"] is True
