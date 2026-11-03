"""SYN-4551 rolling compat: old/new SDK x old/new backend for `synth.problems`.

`fixtures/problems_rolling_compat.json` is the customer wire the backend renders
per negotiated version (byte-equal to backend
tests/customer_api/fixtures/problems_rolling_compat.json, where the backend
asserts it renders exactly these bodies). `fixtures/sdk_v1_problems.py` is the
frozen pre-v2 SDK decoder.

* old backend -> new SDK: old backends only emit v1 origins; v1 is a subset.
* new backend -> new SDK: this SDK asks for v2 and decodes `docker` as known
  and future origins as UnknownOrigin.
* new backend -> old SDK: old SDKs never send the accept header, so they get
  the v1 down-map, which the frozen decoder accepts; the v2 bodies it would
  reject are never sent to it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms import Problem, UnknownOrigin

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"
COMPAT = json.loads((FIXTURES / "problems_rolling_compat.json").read_text())
OWNER = json.loads((FIXTURES / "orchestra_problems_owner_wire.json").read_text())


def _old_sdk():
    spec = importlib.util.spec_from_file_location(
        "sdk_v1_problems", FIXTURES / "sdk_v1_problems.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD_SDK = _old_sdk()


def decode(model, body):
    return model.model_validate_json(json.dumps(body))


@pytest.mark.parametrize("case", ["docker", "future_origin"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_new_sdk_decodes_every_new_backend_body(case, version) -> None:
    for body in COMPAT[case][version]:
        decode(Problem, body)


def test_new_sdk_types_v2_origins() -> None:
    (docker,) = (decode(Problem, b) for b in COMPAT["docker"]["v2"])
    assert docker.origin == "docker" and not isinstance(docker.origin, UnknownOrigin)
    (future,) = (decode(Problem, b) for b in COMPAT["future_origin"]["v2"])
    assert isinstance(future.origin, UnknownOrigin) and future.origin == "kubernetes"
    assert isinstance(future.reason_chain[0].service, UnknownOrigin)


@pytest.mark.parametrize("case", ["docker", "future_origin"])
def test_old_sdk_decodes_the_v1_down_map_and_would_reject_v2(case) -> None:
    for body in COMPAT[case]["v1"]:
        decode(OLD_SDK.Problem, body)
    with pytest.raises(ValidationError):
        for body in COMPAT[case]["v2"]:
            decode(OLD_SDK.Problem, body)


def test_v1_down_map_keeps_the_provider_in_the_cause() -> None:
    (docker,) = COMPAT["docker"]["v1"]
    assert docker["origin"] == "host"
    assert docker["reason_chain"][0]["cause"]["sandbox_provider"] == "docker"
    (future,) = COMPAT["future_origin"]["v1"]
    assert future["origin"] == "unknown"
    assert future["reason_chain"][0]["cause"]["origin_tried"] == "kubernetes"


def test_new_sdk_decodes_old_backend_v1_bodies() -> None:
    from datetime import UTC, datetime

    for run, body in OWNER.items():
        if run == "docker-sandbox-unreachable":
            continue
        for item in body["problems"]:
            wire = dict(item)
            wire["since"] = datetime.fromtimestamp(wire.pop("since_unix_ms") / 1000, UTC)
            wire["last_seen"] = datetime.fromtimestamp(wire.pop("last_seen_unix_ms") / 1000, UTC)
            decode(Problem, json.loads(json.dumps(wire, default=lambda v: v.isoformat())))


def test_every_request_asks_for_problems_v2() -> None:
    from synth_ai.contracts.swarms import GET
    from synth_ai.core.http.request import build_request
    from synth_ai.sdk.swarms.types.problems import (
        ACCEPT_PROBLEMS_HEADER,
        PROBLEMS_SCHEMA_V2,
    )

    with httpx.Client() as client:
        request = build_request(
            client,
            "https://api.example.test",
            "sk_test",
            GET,
            "/swarms/runs/run-1",
            b"",
            None,
            None,
            None,
            5.0,
            1.0,
        )
    assert request.headers[ACCEPT_PROBLEMS_HEADER] == PROBLEMS_SCHEMA_V2
