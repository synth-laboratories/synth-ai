"""Response extensions never loosen request constructors or known wire values."""

import json
from copy import deepcopy
from datetime import datetime
from typing import Literal

import pytest
from pydantic import AwareDatetime, TypeAdapter, ValidationError
from synth_ai.sdk.index.types.search import (
    KeyedSearchRequest,
    SearchEvent,
    SearchResponse,
    SearchSnapshot,
)
from synth_ai.sdk.swarms.types.problems import ProblemCode
from synth_ai.sdk.swarms.types.runs import RunSnapshot, RunState, SwarmEvent, TaskState
from synth_ai.types.base import Model, Page, ResponseModel
from synth_ai.types.money import Money
from synth_ai.types.unknown import UnknownValue
from tests.test_index_retention import LEGACY_ZDR, METADATA, options, result_body, snapshot_body
from tests.test_run_control_observation import snapshot

pytestmark = pytest.mark.unit


def test_index_and_swarms_extensions_are_ignored_at_every_model_depth():
    body = deepcopy(result_body(LEGACY_ZDR))
    body["new_field"] = {"future": True}
    body["metadata"] = {**METADATA, "future_timing": 1}
    body["retention"] = {**LEGACY_ZDR, "future_policy": "new"}
    result = SearchResponse.model_validate_json(json.dumps(body))
    assert result.response == "answer" and result.retention.mode == "zdr"
    swarms = snapshot("running").model_dump(mode="json")
    swarms["future"] = True
    swarms["metadata"]["future"] = True
    assert RunSnapshot.model_validate_json(json.dumps(swarms)).state == "running"


def test_request_constructor_and_known_values_remain_strict_after_response_parse():
    with pytest.raises(ValidationError):
        KeyedSearchRequest(query="q", options=options(), future=True)
    with pytest.raises(ValidationError):
        Money(coefficient=1, scale=0, currency="USD", future=True)
    with pytest.raises(ValidationError):
        Money(coefficient="1", scale=0, currency="USD")
    body = deepcopy(result_body(LEGACY_ZDR))
    body["usage"]["cost"] = {
        "state": "measured",
        "amount": {
            "coefficient": "1",
            "scale": 0,
            "currency": "USD",
        },
        "source": "fixture",
        "rate_version": "v1",
    }
    with pytest.raises(ValidationError):
        SearchResponse.model_validate_json(json.dumps(body))
    # A failed response must also reset its validation scope.
    with pytest.raises(ValidationError):
        KeyedSearchRequest(query="q", options=options(), future=True)


def test_shared_money_keeps_its_class_equality_and_currency_semantics():
    class Observation(ResponseModel):
        amount: Money

    body = {
        "metadata": METADATA,
        "amount": {
            "coefficient": 1,
            "scale": 0,
            "currency": "USD",
            "future": "ignored",
        },
    }
    result = Observation.model_validate_json(json.dumps(body))
    assert type(result.amount) is Money
    assert result.amount == Money(coefficient=1, scale=0, currency="USD")
    body["amount"]["currency"] = "unknown_future_currency"
    with pytest.raises(ValidationError):
        Observation.model_validate_json(json.dumps(body))


def test_response_identifier_accepts_unicode_but_request_identifier_does_not():
    body = snapshot("running").model_dump(mode="json")
    body["run_id"] = "run with spaces/日本語"
    body["metadata"]["request_id"] = "request / 雪"
    result = RunSnapshot.model_validate_json(json.dumps(body))
    assert result.run_id == body["run_id"] and result.metadata.request_id == "request / 雪"
    from synth_ai.options import ReadOptions

    with pytest.raises(ValidationError):
        ReadOptions(request_id=body["run_id"])


def test_response_page_accepts_future_larger_pages_under_wire_byte_limit():
    body = {"metadata": METADATA, "items": list(range(1001)), "next_cursor": None, "complete": True}
    assert len(Page[int].model_validate_json(json.dumps(body)).items) == 1001


def test_shared_unions_mappings_and_strict_json_tuples_dates_keep_semantics():
    class Left(Model):
        kind: Literal["left"]
        left: int

    class Right(Model):
        kind: Literal["right"]
        right: int

    class Observation(ResponseModel):
        variant: Left | Right
        by_name: dict[str, Left]
        at: AwareDatetime
        names: tuple[str, ...]

    body = {
        "metadata": METADATA,
        "variant": {"kind": "right", "right": 2, "future": 3},
        "by_name": {"data key stays": {"kind": "left", "left": 1, "future": True}},
        "at": "2026-10-19T12:00:00Z",
        "names": ["a", "b"],
    }
    result = Observation.model_validate_json(json.dumps(body))
    assert type(result.variant) is Right and result.variant.right == 2
    assert result.by_name["data key stays"].left == 1
    assert result.names == ("a", "b") and isinstance(result.at, datetime)
    assert "future" in body["variant"]  # Caller data is never edited.
    with pytest.raises(ValidationError):
        Right(kind="right", right=2, future=3)


@pytest.mark.parametrize("annotation", [RunState, TaskState, ProblemCode])
def test_future_discriminants_keep_the_actual_bounded_wire_value(annotation):
    adapter = TypeAdapter(annotation)
    value = adapter.validate_json('"future_waiting_state"')
    assert isinstance(value, UnknownValue) and value == "future_waiting_state"
    assert adapter.dump_json(value) == b'"future_waiting_state"'
    with pytest.raises(ValidationError):
        adapter.validate_json("123")


def test_known_and_unknown_run_search_states_do_not_fabricate_completion():
    body = snapshot("running").model_dump(mode="json")
    assert not isinstance(RunSnapshot.model_validate_json(json.dumps(body)).state, UnknownValue)
    body["state"] = "future_waiting_state"
    value = RunSnapshot.model_validate_json(json.dumps(body))
    assert isinstance(value.state, UnknownValue) and not value.execution_closed
    search = SearchSnapshot.model_validate_json(
        json.dumps(snapshot_body(state="future_waiting_state"))
    )
    assert isinstance(search.state, UnknownValue) and not search.execution_closed


@pytest.mark.parametrize("model", [SwarmEvent, SearchEvent])
def test_future_event_kind_is_observable_with_its_original_value(model):
    field = "run_id" if model is SwarmEvent else "search_id"
    value = model.model_validate_json(
        json.dumps(
            {
                field: "resource-1",
                "event_id": "event-1",
                "sequence": 1,
                "kind": "future_observation",
                "future_detail": {"owner": "new"},
            }
        )
    )
    assert isinstance(value.kind, UnknownValue) and value.kind == "future_observation"


def test_response_schema_relaxes_shared_views_without_changing_request_schema():
    from synth_ai.options import ReadOptions
    from synth_ai.types.response_values import response_schema

    response = response_schema(RunSnapshot)
    assert response["additionalProperties"] is True
    assert "pattern" not in response["properties"]["run_id"]
    assert response["properties"]["run_id"]["maxLength"] == 512
    request = response_schema(ReadOptions)
    assert request["additionalProperties"] is False
    assert request["properties"]["request_id"]["anyOf"][0]["pattern"]


def test_reference_extensions_preserve_class_equality_and_native_digest_binding():
    from synth_ai.sdk.index.references import ExactReference

    class Observation(ResponseModel):
        reference: ExactReference

    exact = ExactReference(
        authority="forge",
        kind="revision",
        record_id="record",
        revision="r1",
        digest_sha256="a" * 64,
    )
    body = {"metadata": METADATA, "reference": {**exact.model_dump(mode="json"), "future": True}}
    value = Observation.model_validate_json(json.dumps(body))
    assert type(value.reference) is ExactReference and value.reference == exact
    with pytest.raises(ValidationError):
        ExactReference(**body["reference"])
    for record in ("intern-exec:bad", "other:bad"):
        body["reference"].update(authority="mloky", kind="mloky_execution", record_id=record)
        with pytest.raises(ValidationError):
            Observation.model_validate_json(json.dumps(body))


def test_strict_progress_tolerates_metadata_extensions_without_forwarding_unknown_data():
    from synth_ai.sdk.index.types.search import StrictSearchProgress

    body = {
        "search_id": "search-1",
        "event_id": "search-1.1",
        "sequence": 1,
        "kind": "progress",
        "stage": "working",
        "search_calls": 1,
        "read_calls": 0,
        "input_tokens": 1,
        "output_tokens": 1,
        "future_metadata": {"new_counter": 2},
    }
    result = StrictSearchProgress.model_validate_json(json.dumps(body))
    assert "future_metadata" not in result.model_dump(mode="json")
    body["query"] = "private query"
    with pytest.raises(ValidationError):
        StrictSearchProgress.model_validate_json(json.dumps(body))
