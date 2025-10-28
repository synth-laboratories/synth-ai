from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum

import pytest

from synth_ai.tracing_v3.serialization import (
    dumps_http_json,
    normalize_for_json,
    serialize_trace_for_http,
)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Inner:
    when: datetime
    data: bytes


@dataclass
class Outer:
    id: str
    inner: Inner
    tags: set[str]
    score: float
    amount: Decimal
    day: date
    choice: Color


def test_normalize_simple_primitives():
    assert normalize_for_json({"a": 1, "b": True, "c": None}) == {"a": 1, "b": True, "c": None}


def test_normalize_dataclass_and_bytes_and_datetime():
    now = datetime.now(UTC).replace(microsecond=123000)
    raw = Outer(
        id="x",
        inner=Inner(when=now, data=b"hello"),
        tags={"t1", "t2"},
        score=1.5,
        amount=Decimal("3.14"),
        day=(now + timedelta(days=1)).date(),
        choice=Color.BLUE,
    )
    norm = normalize_for_json(raw)
    assert norm["id"] == "x"
    assert norm["inner"]["when"].startswith(str(now.date()))
    assert norm["inner"]["data"] == base64.b64encode(b"hello").decode("ascii")
    assert sorted(norm["tags"]) == ["t1", "t2"]
    assert isinstance(norm["amount"], float)
    assert norm["choice"] == "blue"


def test_normalize_nan_and_inf():
    payload = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf")}
    norm = normalize_for_json(payload)
    assert norm == {"nan": None, "inf": None, "ninf": None}
    out = dumps_http_json(payload)
    # JSON must parse and contain nulls
    assert json.loads(out) == {"nan": None, "inf": None, "ninf": None}


def test_serialize_trace_for_http_accepts_dataclass_and_dict():
    now = datetime.now(UTC)
    inner = Inner(when=now, data=b"abc")
    s1 = serialize_trace_for_http(inner)
    obj1 = json.loads(s1)
    assert obj1["data"] == base64.b64encode(b"abc").decode("ascii")

    d = {"when": now, "data": b"xyz"}
    s2 = serialize_trace_for_http(d)
    obj2 = json.loads(s2)
    assert obj2["data"] == base64.b64encode(b"xyz").decode("ascii")


def test_no_allow_nan_round_trip_and_compact():
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "é"}}
    s = dumps_http_json(payload)
    assert s == '{"a":1,"b":[1,2,3],"c":{"d":"é"}}'
    assert json.loads(s) == payload


