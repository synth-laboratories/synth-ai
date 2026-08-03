from __future__ import annotations

import pytest
from synth_ai.sdk.research.contracts.factory_operations import (
    FactoryWakeDueRequest,
)
from synth_ai.sdk.research.session.factories import FactoriesAPI


def test_factory_wake_contract_round_trips_exact_effort_scope() -> None:
    request = FactoryWakeDueRequest.from_contract_wire(
        {
            "launch_request": None,
            "effort_ids": ["effort-1"],
            "limit": 1,
            "allow_overlap": False,
            "continue_on_error": True,
        }
    )

    assert request.effort_ids == ("effort-1",)
    assert request.to_contract_wire()["effort_ids"] == ["effort-1"]
    with pytest.raises(ValueError, match="unique non-empty"):
        FactoryWakeDueRequest(effort_ids=("effort-1", "effort-1"))


def test_factory_wake_effort_scope_is_append_only_for_sdk_callers() -> None:
    legacy = FactoryWakeDueRequest(None, 3, True, True, False, "preview-1", "token-1")

    assert legacy.limit == 3
    assert legacy.allow_overlap is True
    assert legacy.dry_run is True
    assert legacy.confirmed_preview_id == "preview-1"
    assert legacy.effort_ids == ()


def test_factories_api_sends_effort_scope_on_preview() -> None:
    observed: dict[str, object] = {}

    class Client:
        def wake_due_factory_efforts(
            self,
            factory_id: str,
            request: FactoryWakeDueRequest,
        ) -> dict[str, object]:
            observed["factory_id"] = factory_id
            observed["request"] = request.to_wire()
            return {
                "factory_id": factory_id,
                "evaluated_at": None,
                "dry_run": True,
                "confirmation_required": False,
                "ready": 0,
                "launched": 0,
                "skipped": 0,
                "failed": 0,
                "efforts": [],
                "request_contract": request.to_contract_wire(),
            }

    result = FactoriesAPI(Client()).wake_due(  # type: ignore[arg-type]
        "factory-1",
        effort_ids=("effort-1",),
        limit=1,
        dry_run=True,
    )

    assert result.dry_run is True
    assert observed == {
        "factory_id": "factory-1",
        "request": {
            "effort_ids": ["effort-1"],
            "limit": 1,
            "allow_overlap": False,
            "dry_run": True,
            "continue_on_error": True,
        },
    }
