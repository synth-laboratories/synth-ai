"""Exercise the real public SDK against a stateful HTTP authority.

These component tests deliberately return stale/no-op/cross-run views. They
prove the acceptance assertion rejects them; they are not cached-slot proof.
"""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from scripts.acceptance.live_limits import LimitTargets, LiveLimitsError, raise_live_limits
from synth_ai import SynthClient
from synth_ai.errors import SwarmsAPIError

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"
TARGETS = LimitTargets("2.00", 2000, 120, 120)


class Authority:
    def __init__(self, contract, defect=None):
        self.view = deepcopy(
            contract["paths"]["/v1/runs/{run_id}/limits"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        )
        self.view.update(run_id=RUN, status="running", revision=1)
        self.view["primary"] = [
            {"dimension": "spend_micros", "effective_cap": 1_000_000},
            {
                "dimension": "tokens",
                "effective_cap": None,
                "requested_cap": 1000,
                "state": "metered",
            },
            {"dimension": "execution_ms", "effective_cap": 60_000},
            {"dimension": "pause_retention_ms", "effective_cap": 60_000},
        ]
        self.defect = defect
        self.requests = []

    def handle(self, request):
        assert request.url.path == f"/v1/runs/{RUN}/limits"
        self.requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["expected_revision"] == self.view["revision"]
            self.view["revision"] += 1
            field, value = next(iter(body["limits"].items()))
            positions = {
                "maximum_charge": 0,
                "maximum_tokens": 1,
                "wall_time_seconds": 2,
                "pause_retention_seconds": 3,
            }
            index = positions[field]
            if self.defect != "noop":
                cap = 2_000_000 if index == 0 else value if index == 1 else value * 1000
                self.view["primary"][index]["requested_cap" if index == 1 else "effective_cap"] = (
                    cap
                )
            if self.defect == "other_cap":
                self.view["primary"][3]["effective_cap"] = 1
            if self.defect == "terminal":
                self.view["status"] = "completed"
            if self.defect == "cross_run":
                self.view["run_id"] = "run_BBBBBBBBBBBBBBBBBBBBBB"
            if self.defect == "unknown":
                raise httpx.ReadTimeout("response lost", request=request)
            return httpx.Response(200, json=self.view)
        result = deepcopy(self.view)
        if self.defect == "stale" and result["revision"] > 1:
            result["revision"] -= 1
        return httpx.Response(200, json=result)


def execute(authority, path, targets=TARGETS):
    http = httpx.AsyncClient(transport=httpx.MockTransport(authority.handle))
    try:
        with SynthClient(
            "fixture", base_url="https://example.test", http_client=http, retries=0
        ) as client:
            return raise_live_limits(
                client,
                run_id=RUN,
                targets=targets,
                mutation_prefix="owned-run",
                receipt_directory=path,
            )

    finally:
        asyncio.run(http.aclose())


def test_all_four_fields_round_trip_with_next_views(contract, tmp_path):
    authority = Authority(contract)
    path = tmp_path / "limits"
    views = execute(authority, path)
    assert [view.revision for view in views] == [2, 3, 4, 5]
    assert len(authority.requests) == 9
    posts = [request for request in authority.requests if request.method == "POST"]
    assert len({request.headers["Idempotency-Key"] for request in posts}) == 4
    assert [json.loads(request.content)["expected_revision"] for request in posts] == [1, 2, 3, 4]
    assert views[1].primary[1].effective_cap is None
    assert views[1].primary[1].requested_cap == 2000
    assert json.loads((path / "verdict.json").read_text())["verdict"] == "Pass"
    assert (path / "0-intent.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        execute(authority, path)
    assert len(authority.requests) == 9


@pytest.mark.parametrize(
    "defect,code",
    [
        ("noop", "limit_update_not_observed"),
        ("other_cap", "limit_update_not_observed"),
        ("stale", "limit_revision_mismatch"),
        ("terminal", "run_not_running"),
        ("cross_run", "limit_revision_mismatch"),
    ],
)
def test_false_success_is_rejected_and_evidence_retained(contract, tmp_path, defect, code):
    authority = Authority(contract, defect)
    path = tmp_path / "limits"
    with pytest.raises(LiveLimitsError) as raised:
        execute(authority, path)
    assert raised.value.code == code
    assert (path / "0-intent.json").exists()
    assert (path / "0-view.json").exists()
    assert not (path / "verdict.json").exists()
    assert len(authority.requests) == 3


def test_all_targets_validated_before_first_mutation(contract, tmp_path):
    authority = Authority(contract)
    with pytest.raises(LiveLimitsError, match="limit_target_not_increased"):
        execute(authority, tmp_path / "limits", LimitTargets("2", 2000, 120, 30))
    assert len(authority.requests) == 1


def test_missing_dimension_refuses_before_mutation(contract, tmp_path):
    authority = Authority(contract)
    authority.view["primary"].pop()
    with pytest.raises(LiveLimitsError, match="limit_dimension_invalid"):
        execute(authority, tmp_path / "limits")
    assert len(authority.requests) == 1


def test_uncertain_response_keeps_original_intent_and_refuses_rerun(contract, tmp_path):
    authority = Authority(contract, "unknown")
    path = tmp_path / "limits"
    with pytest.raises(SwarmsAPIError):
        execute(authority, path)
    intent = json.loads((path / "0-intent.json").read_text())
    assert intent["idempotency_key"] == "owned-run:spend_micros"
    assert intent["request"]["expected_revision"] == 1
    assert not (path / "verdict.json").exists()
    with pytest.raises(FileExistsError):
        execute(authority, path)
    assert len(authority.requests) == 2


AT_MAXIMUM = 86_400_000  # pause_retention_seconds le=86400 in the public contract.


def test_retention_at_contract_maximum_is_held_and_asserted_unchanged(contract, tmp_path):
    authority = Authority(contract)
    authority.view["primary"][3]["effective_cap"] = AT_MAXIMUM
    path = tmp_path / "limits"
    views = execute(authority, path, LimitTargets("2.00", 2000, 120, None))
    assert [view.revision for view in views] == [2, 3, 4]
    posts = [json.loads(r.content) for r in authority.requests if r.method == "POST"]
    assert [next(iter(body["limits"])) for body in posts] == [
        "maximum_charge",
        "maximum_tokens",
        "wall_time_seconds",
    ]
    assert all(view.primary[3].effective_cap == AT_MAXIMUM for view in views)
    verdict = json.loads((path / "verdict.json").read_text())
    assert verdict["held_at_contract_maximum"] == ["pause_retention_ms"]


def test_held_dimension_below_contract_maximum_refuses_before_mutation(contract, tmp_path):
    authority = Authority(contract)  # retention 60 s: a raise is possible, so it is owed.
    with pytest.raises(LiveLimitsError, match="limit_target_not_increased"):
        execute(authority, tmp_path / "limits", LimitTargets("2.00", 2000, 120, None))
    assert len(authority.requests) == 1


def test_held_dimension_that_moves_is_not_observed(contract, tmp_path):
    authority = Authority(contract, "other_cap")  # every POST rewrites retention to 1.
    authority.view["primary"][3]["effective_cap"] = AT_MAXIMUM
    with pytest.raises(LiveLimitsError) as raised:
        execute(authority, tmp_path / "limits", LimitTargets("2.00", 2000, 120, None))
    assert raised.value.code == "limit_update_not_observed"
