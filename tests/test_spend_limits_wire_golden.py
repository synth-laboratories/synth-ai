"""The SDK's `spend` wire output is pinned to a golden file the backend also tests.

`tests/fixtures/spend_limits_wire_v1.json` is copied byte-for-byte into the
backend (`tests/fixtures/spend_limits_wire_v1.json`), whose test asserts the
run-create route accepts each case or rejects it with the recorded code. Drift
on either side fails that side's test. Regenerate with
`python tests/test_spend_limits_wire_golden.py` and copy the file across.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from synth_ai.sdk.research.contracts.swarms import (
    ProviderBinding,
    ResourceLimit,
    ResourceProvider,
    RunPolicy,
    RunPolicyLimits,
)
from synth_ai.sdk.research.public import (
    LimitAction,
    Resource,
    ResourceCap,
    SpendLimit,
    SwarmSpec,
)

GOLDEN = Path(__file__).parent / "fixtures" / "spend_limits_wire_v1.json"
_WIRE_FIELDS = ("spend", "limit", "run_policy", "providers", "timebox_seconds")
_OPENROUTER_8_USD = (
    ProviderBinding(ResourceProvider.OPENROUTER, limit=ResourceLimit(max_spend_usd=8)),
)


@dataclass(frozen=True)
class _Case:
    name: str
    spend: SpendLimit
    expect: str
    legacy: dict[str, Any] = field(default_factory=dict)
    # False: the SDK refuses this combination client-side; the golden body is
    # built raw so the backend's own refusal is tested too.
    sdk_builds: bool = True

    def request(self) -> dict[str, Any]:
        if self.sdk_builds:
            wire = SwarmSpec(objective="golden", spend=self.spend, **self.legacy).to_wire()
        else:
            wire = SwarmSpec(objective="golden", **self.legacy).to_wire()
            wire["spend"] = self.spend.to_wire()
        return {key: wire[key] for key in _WIRE_FIELDS if key in wire}


_CASES = (
    _Case("total_only", SpendLimit(max_usd=25), "ok"),
    _Case("empty", SpendLimit(), "ok"),
    _Case(
        "per_resource_detail",
        SpendLimit(
            max_usd=25,
            warn_at=0.9,
            resources=(
                ResourceCap(Resource.INFERENCE, usd=8, model="openai/gpt-5.6-luna"),
                ResourceCap(
                    Resource.INFERENCE,
                    tokens=2_000_000,
                    actor_type="worker",
                    on_exhaustion=LimitAction.STOP,
                ),
                ResourceCap(Resource.ALL, usd=15, provider="modal"),
                ResourceCap(Resource.GPU, usd=10, provider="modal", sku="H100"),
                ResourceCap(Resource.MISC, usd=5, provider="exa"),
                ResourceCap(Resource.WALLCLOCK, seconds=600, on_exhaustion=LimitAction.STOP),
            ),
        ),
        "ok",
    ),
    _Case(
        "legacy_agrees",
        SpendLimit(max_usd=25),
        "ok",
        legacy={"limit": ResourceLimit(max_spend_usd=25)},
    ),
    _Case(
        "legacy_run_policy_minimum_agrees",
        SpendLimit(max_usd=25),
        "ok",
        legacy={
            "limit": ResourceLimit(max_spend_usd=30),
            "run_policy": RunPolicy(limits=RunPolicyLimits(total_cost_cents=2500)),
        },
    ),
    _Case(
        "legacy_gpu_hours_never_conflicts",
        SpendLimit(max_usd=25),
        "ok",
        legacy={"limit": ResourceLimit(max_gpu_hours=2)},
    ),
    _Case(
        "wall_clock_hours",
        SpendLimit(
            resources=(
                ResourceCap(Resource.GPU, hours=4),
                ResourceCap(Resource.GPU, hours=1.5, provider="modal", sku="H100"),
                ResourceCap(Resource.SANDBOX, hours=6),
                ResourceCap(Resource.VM, hours=1),
            )
        ),
        "ok",
    ),
    _Case(
        "browser_hours_not_metered",
        SpendLimit(resources=(ResourceCap(Resource.BROWSER, hours=1),)),
        "spend_metric_not_metered",
    ),
    _Case(
        "training_tokens_not_metered",
        SpendLimit(resources=(ResourceCap(Resource.TRAINING, train_tokens=1000),)),
        "spend_metric_not_metered",
    ),
    _Case(
        "legacy_disagrees",
        SpendLimit(max_usd=30),
        "spend_conflicts_with_legacy_limit",
        legacy={"limit": ResourceLimit(max_spend_usd=25)},
        sdk_builds=False,
    ),
    _Case(
        "legacy_provider_cap_omitted",
        SpendLimit(max_usd=25),
        "spend_conflicts_with_legacy_limit",
        legacy={"providers": _OPENROUTER_8_USD},
        sdk_builds=False,
    ),
)


def _render() -> str:
    cases = [
        {
            "name": case.name,
            "expect": case.expect,
            "sdk_builds": case.sdk_builds,
            "request": case.request(),
        }
        for case in _CASES
    ]
    return json.dumps({"schema": "spend_limits_wire.v1", "cases": cases}, indent=2) + "\n"


def test_sdk_wire_matches_golden() -> None:
    assert GOLDEN.read_text() == _render(), (
        "SDK spend wire drifted from the golden file shared with the backend; "
        "regenerate it and update the backend copy in the same change"
    )


@pytest.mark.parametrize(
    "case", [case for case in _CASES if not case.sdk_builds], ids=lambda case: case.name
)
def test_sdk_refuses_the_cases_it_marks_unbuildable(case: _Case) -> None:
    with pytest.raises(ValueError):
        SwarmSpec(objective="golden", spend=case.spend, **case.legacy)


if __name__ == "__main__":
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(_render())
    sys.stdout.write(f"wrote {GOLDEN}\n")
