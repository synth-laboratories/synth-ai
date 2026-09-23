from __future__ import annotations

import json
from decimal import Decimal

import pytest
from synth_ai.sdk.research.contracts.canonical_usage import SmrResourceLimitSelector
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
    SpendMetric,
    SwarmSpec,
)

# --- measures and the metric each compiles to --------------------------------------------


@pytest.mark.parametrize(
    ("cap", "metric", "limit"),
    [
        (ResourceCap(Resource.ALL, usd=15), SpendMetric.SPEND_USD_CENTS, 1500),
        (ResourceCap(Resource.MISC, usd=5), SpendMetric.SPEND_USD_CENTS, 500),
        (ResourceCap(Resource.INFERENCE, usd=8), SpendMetric.SPEND_USD_CENTS, 800),
        (ResourceCap(Resource.INFERENCE, tokens=2_000_000), SpendMetric.TOKENS, 2_000_000),
        (ResourceCap(Resource.TRAINING, train_tokens=10), SpendMetric.TRAIN_TOKENS, 10),
        (ResourceCap(Resource.TRAINING, sample_tokens=10), SpendMetric.SAMPLE_TOKENS, 10),
        (ResourceCap(Resource.TRAINING, usd=20), SpendMetric.SPEND_USD_CENTS, 2000),
        (ResourceCap(Resource.GPU, hours=4), SpendMetric.GPU_HOURS, Decimal(4)),
        (ResourceCap(Resource.SANDBOX, hours=6), SpendMetric.SANDBOX_HOURS, Decimal(6)),
        (ResourceCap(Resource.BROWSER, hours=1), SpendMetric.BROWSER_HOURS, Decimal(1)),
        (ResourceCap(Resource.VM, hours=2), SpendMetric.VM_HOURS, Decimal(2)),
        (ResourceCap(Resource.GPU, usd=10), SpendMetric.SPEND_USD_CENTS, 1000),
        (ResourceCap(Resource.WALLCLOCK, seconds=600), SpendMetric.WALLCLOCK_SECONDS, 600),
    ],
)
def test_each_measure_compiles_to_the_contract_metric(
    cap: ResourceCap, metric: SpendMetric, limit: int | Decimal
) -> None:
    assert cap.metric is metric
    assert cap.limit == limit


@pytest.mark.parametrize(
    ("resource", "measure"),
    [
        (Resource.ALL, {"tokens": 1}),
        (Resource.ALL, {"hours": 1}),
        (Resource.MISC, {"tokens": 1}),
        (Resource.MISC, {"hours": 1}),
        (Resource.INFERENCE, {"hours": 1}),
        (Resource.INFERENCE, {"train_tokens": 1}),
        (Resource.TRAINING, {"tokens": 1}),
        (Resource.GPU, {"tokens": 1}),
        (Resource.GPU, {"seconds": 1}),
        (Resource.SANDBOX, {"sample_tokens": 1}),
        (Resource.WALLCLOCK, {"usd": 1}),
        (Resource.WALLCLOCK, {"hours": 1}),
    ],
)
def test_wrong_unit_for_resource_is_rejected(resource: Resource, measure: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="not a valid measure"):
        ResourceCap(resource, **measure)  # type: ignore[arg-type]


def test_exactly_one_measure_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one.*got none"):
        ResourceCap(Resource.GPU)
    with pytest.raises(ValueError, match="exactly one.*got hours, usd"):
        ResourceCap(Resource.GPU, usd=1, hours=1)


def test_wallclock_takes_no_selectors() -> None:
    for selector in ("provider", "model", "sku", "actor_type"):
        with pytest.raises(ValueError, match=f"wallclock caps take no selectors; got {selector}"):
            ResourceCap(Resource.WALLCLOCK, seconds=60, **{selector: "x"})  # type: ignore[arg-type]


def test_empty_selector_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        ResourceCap(Resource.GPU, hours=1, provider="  ")


@pytest.mark.parametrize(
    "measure",
    [
        {"usd": -1},
        {"usd": Decimal("-0.01")},
        {"hours": -0.5},
        {"tokens": -1},
        {"seconds": -1},
    ],
)
def test_negative_limits_are_rejected(measure: dict[str, object]) -> None:
    resource = {
        "usd": Resource.GPU,
        "hours": Resource.GPU,
        "tokens": Resource.INFERENCE,
        "seconds": Resource.WALLCLOCK,
    }[next(iter(measure))]
    with pytest.raises(ValueError, match="non-negative"):
        ResourceCap(resource, **measure)  # type: ignore[arg-type]


def test_zero_is_a_valid_limit() -> None:
    assert ResourceCap(Resource.INFERENCE, tokens=0).limit == 0
    assert ResourceCap(Resource.ALL, usd=0).limit == 0


@pytest.mark.parametrize(
    ("resource", "measure"),
    [
        (Resource.INFERENCE, {"tokens": 1.5}),
        (Resource.INFERENCE, {"tokens": True}),
        (Resource.TRAINING, {"train_tokens": 2.0}),
        (Resource.TRAINING, {"sample_tokens": Decimal(3)}),
        (Resource.WALLCLOCK, {"seconds": 60.0}),
    ],
)
def test_integer_metrics_require_integers(resource: Resource, measure: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ResourceCap(resource, **measure)  # type: ignore[arg-type]


def test_hours_allow_at_most_four_decimal_places() -> None:
    assert ResourceCap(Resource.GPU, hours=Decimal("1.2345")).limit == Decimal("1.2345")
    assert ResourceCap(Resource.GPU, hours=0.0001).limit == Decimal("0.0001")
    # Trailing zeros are not extra precision.
    assert ResourceCap(Resource.GPU, hours=Decimal("1.50000")).limit == Decimal("1.5")
    with pytest.raises(ValueError, match="at most 4 decimal places"):
        ResourceCap(Resource.GPU, hours=Decimal("1.23456"))
    with pytest.raises(ValueError, match="at most 4 decimal places"):
        ResourceCap(Resource.SANDBOX, hours=0.00001)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "5", True])
def test_non_numeric_amounts_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        ResourceCap(Resource.ALL, usd=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ResourceCap(Resource.GPU, hours=value)  # type: ignore[arg-type]


# --- dollars ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usd", "cents"),
    [
        (0.1, 10),
        (0.29, 29),  # 0.29 * 100 == 28.999999999999996 in binary floating point
        (19.99, 1999),
        (25, 2500),
        (Decimal("8.07"), 807),
        (Decimal("1E+2"), 10_000),
    ],
)
def test_usd_converts_exactly_to_integer_cents(usd: float | int | Decimal, cents: int) -> None:
    cap = ResourceCap(Resource.INFERENCE, usd=usd)
    assert cap.limit == cents
    assert isinstance(cap.limit, int)
    assert SpendLimit(max_usd=usd).max_usd_cents == cents


@pytest.mark.parametrize("usd", [0.001, Decimal("1.005"), 12.345])
def test_sub_cent_usd_is_rejected(usd: float | Decimal) -> None:
    with pytest.raises(ValueError, match="whole number of cents"):
        ResourceCap(Resource.INFERENCE, usd=usd)
    with pytest.raises(ValueError, match="whole number of cents"):
        SpendLimit(max_usd=usd)


# --- SpendLimit ---------------------------------------------------------------------------


def test_spend_limit_defaults() -> None:
    spend = SpendLimit(max_usd=25)
    assert spend.on_exhaustion is LimitAction.PAUSE
    assert spend.warn_at == 0.9
    assert spend.to_wire() == {
        "max_usd_cents": 2500,
        "on_exhaustion": "pause",
        "warn_at_fraction": 0.9,
        "caps": [],
    }


@pytest.mark.parametrize("warn_at", [0, -0.1, 1.01, True])
def test_warn_at_must_be_in_half_open_unit_interval(warn_at: float) -> None:
    with pytest.raises(ValueError, match=r"warn_at must be a number in \(0, 1\]"):
        SpendLimit(max_usd=1, warn_at=warn_at)


def test_warn_at_of_one_is_allowed() -> None:
    assert SpendLimit(warn_at=1).warn_at == 1.0


def test_duplicate_selector_and_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate cap"):
        SpendLimit(
            resources=(
                ResourceCap(Resource.GPU, usd=10, provider="modal"),
                ResourceCap(Resource.GPU, usd=20, provider="modal", on_exhaustion=LimitAction.STOP),
            )
        )


def test_same_selector_with_different_metrics_is_allowed() -> None:
    spend = SpendLimit(
        resources=(
            ResourceCap(Resource.GPU, usd=10, provider="modal"),
            ResourceCap(Resource.GPU, hours=2, provider="modal"),
            ResourceCap(Resource.GPU, hours=1, provider="modal", sku="H100"),
        )
    )
    assert len(spend.resources) == 3


def test_max_usd_must_not_duplicate_an_unselected_all_cap() -> None:
    with pytest.raises(ValueError, match="max_usd already sets the unselected ALL dollar cap"):
        SpendLimit(max_usd=25, resources=(ResourceCap(Resource.ALL, usd=25),))


def test_unselected_all_cap_is_equivalent_to_max_usd() -> None:
    explicit = SpendLimit(resources=(ResourceCap(Resource.ALL, usd=25),))
    shorthand = SpendLimit(max_usd=25)
    assert explicit.effective_caps() == shorthand.effective_caps()


def test_selected_all_caps_bound_a_slice_of_total_spend() -> None:
    spend = SpendLimit(
        max_usd=25,
        resources=(
            ResourceCap(Resource.ALL, usd=15, provider="modal"),
            ResourceCap(Resource.ALL, usd=5, actor_type="reviewer"),
        ),
    )
    assert [cap.to_wire() for cap in spend.resources] == [
        {
            "selector": {"resource": "all", "provider": "modal"},
            "metric": "spend_usd_cents",
            "limit": 1500,
        },
        {
            "selector": {"resource": "all", "actor_type": "reviewer"},
            "metric": "spend_usd_cents",
            "limit": 500,
        },
    ]


def test_misc_caps_are_dollars_narrowed_by_provider_and_sku() -> None:
    cap = ResourceCap(Resource.MISC, usd=5, provider="exa", sku="search")
    assert cap.to_wire() == {
        "selector": {"resource": "misc", "provider": "exa", "sku": "search"},
        "metric": "spend_usd_cents",
        "limit": 500,
    }


def test_per_resource_caps_may_exceed_max_usd() -> None:
    spend = SpendLimit(max_usd=5, resources=(ResourceCap(Resource.INFERENCE, usd=50),))
    assert spend.max_usd_cents == 500


@pytest.mark.parametrize(
    "measure",
    [
        {"resource": Resource.GPU, "hours": 1},
        {"resource": Resource.SANDBOX, "hours": 1},
        {"resource": Resource.BROWSER, "hours": 1},
        {"resource": Resource.VM, "hours": 1},
        {"resource": Resource.TRAINING, "train_tokens": 1},
        {"resource": Resource.TRAINING, "sample_tokens": 1},
    ],
)
def test_metrics_the_backend_does_not_meter_yet_are_still_built(
    measure: dict[str, object],
) -> None:
    # Phase 1 of the backend answers these with spend_metric_not_metered; the SDK stays
    # forward compatible and lets the backend decide.
    resource = measure.pop("resource")
    SpendLimit(resources=(ResourceCap(resource, **measure),))  # type: ignore[arg-type]


# --- wire ---------------------------------------------------------------------------------


def test_to_wire_matches_the_phase_one_contract_example() -> None:
    spend = SpendLimit(
        max_usd=Decimal("25.00"),
        resources=(
            ResourceCap(Resource.GPU, usd=10, provider="modal", sku="H100"),
            ResourceCap(
                Resource.INFERENCE,
                tokens=2_000_000,
                actor_type="worker",
                on_exhaustion=LimitAction.STOP,
            ),
            ResourceCap(Resource.ALL, usd=15, provider="modal"),
        ),
    )
    assert spend.to_wire() == {
        "max_usd_cents": 2500,
        "on_exhaustion": "pause",
        "warn_at_fraction": 0.9,
        "caps": [
            {
                "selector": {"resource": "gpu", "provider": "modal", "sku": "H100"},
                "metric": "spend_usd_cents",
                "limit": 1000,
            },
            {
                "selector": {"resource": "inference", "actor_type": "worker"},
                "metric": "tokens",
                "limit": 2000000,
                "on_exhaustion": "stop",
            },
            {
                "selector": {"resource": "all", "provider": "modal"},
                "metric": "spend_usd_cents",
                "limit": 1500,
            },
        ],
    }


def test_hours_serialize_as_json_numbers() -> None:
    payload = SpendLimit(
        on_exhaustion=LimitAction.STOP,
        warn_at=0.5,
        resources=(
            ResourceCap(Resource.GPU, hours=4),
            ResourceCap(Resource.SANDBOX, hours=Decimal("1.2345")),
        ),
    ).to_wire()
    assert "max_usd_cents" not in payload
    assert json.loads(json.dumps(payload)) == {
        "on_exhaustion": "stop",
        "warn_at_fraction": 0.5,
        "caps": [
            {"selector": {"resource": "gpu"}, "metric": "gpu_hours", "limit": 4},
            {"selector": {"resource": "sandbox"}, "metric": "sandbox_hours", "limit": 1.2345},
        ],
    }


# --- SwarmSpec ----------------------------------------------------------------------------


def test_swarm_spec_serializes_spend() -> None:
    spend = SpendLimit(max_usd=25, resources=(ResourceCap(Resource.WALLCLOCK, seconds=600),))
    payload = SwarmSpec(objective="measure", spend=spend).to_wire()
    assert payload["spend"] == spend.to_wire()
    assert "spend" not in SwarmSpec(objective="measure").to_wire()


def test_swarm_spec_rejects_non_spend_limit() -> None:
    with pytest.raises(ValueError, match="spend must be SpendLimit"):
        SwarmSpec(objective="measure", spend={"max_usd_cents": 1})  # type: ignore[arg-type]


def test_legacy_fields_still_work_without_spend() -> None:
    payload = SwarmSpec(
        objective="measure",
        limit=ResourceLimit(max_spend_usd=10, max_tokens=5, max_gpu_hours=1.5),
        run_policy=RunPolicy(limits=RunPolicyLimits(total_cost_cents=900)),
        providers=(ProviderBinding(ResourceProvider.OPENROUTER, ResourceLimit(max_spend_usd=3)),),
    ).to_wire()
    assert payload["limit"] == {"max_spend_usd": 10, "max_tokens": 5, "max_gpu_hours": 1.5}
    assert payload["run_policy"] == {"limits": {"total_cost_cents": 900}}
    assert payload["providers"] == [{"provider": "openrouter", "limit": {"max_spend_usd": 3}}]
    assert "spend" not in payload


def test_agreeing_legacy_fields_are_accepted() -> None:
    SwarmSpec(
        objective="measure",
        limit=ResourceLimit(max_spend_usd=25.0, max_tokens=1000, max_gpu_hours=2),
        # The smaller legacy total wins, so 2500 cents is the compiled ALL cap.
        run_policy=RunPolicy(limits=RunPolicyLimits(total_cost_cents=4000)),
        providers=(
            ProviderBinding(
                ResourceProvider.OPENROUTER, ResourceLimit(max_spend_usd=3, max_tokens=50)
            ),
        ),
        spend=SpendLimit(
            max_usd=25,
            resources=(
                ResourceCap(Resource.INFERENCE, tokens=1000),
                ResourceCap(Resource.INFERENCE, usd=3, provider="openrouter"),
                ResourceCap(Resource.INFERENCE, tokens=50, provider="openrouter"),
            ),
        ),
    )


def test_explicit_all_cap_agrees_with_legacy_total() -> None:
    SwarmSpec(
        objective="measure",
        run_policy=RunPolicy(limits=RunPolicyLimits(total_cost_cents=2500)),
        spend=SpendLimit(resources=(ResourceCap(Resource.ALL, usd=25),)),
    )


@pytest.mark.parametrize(
    ("legacy", "source"),
    [
        ({"limit": ResourceLimit(max_spend_usd=30)}, "limit.max_spend_usd"),
        (
            {"run_policy": RunPolicy(limits=RunPolicyLimits(total_cost_cents=1000))},
            "run_policy.limits.total_cost_cents",
        ),
        ({"limit": ResourceLimit(max_tokens=99)}, "limit.max_tokens"),
        (
            {
                "providers": (
                    ProviderBinding(ResourceProvider.MODAL, ResourceLimit(max_spend_usd=1)),
                )
            },
            r"providers\[0\].limit.max_spend_usd",
        ),
        (
            {"providers": (ProviderBinding(ResourceProvider.MODAL, ResourceLimit(max_tokens=7)),)},
            r"providers\[0\].limit.max_tokens",
        ),
    ],
)
def test_disagreeing_legacy_field_is_rejected(legacy: dict[str, object], source: str) -> None:
    spend = SpendLimit(max_usd=25, resources=(ResourceCap(Resource.INFERENCE, tokens=100),))
    with pytest.raises(ValueError, match=f"spend conflicts with legacy {source}"):
        SwarmSpec(objective="measure", spend=spend, **legacy)  # type: ignore[arg-type]


def test_legacy_provider_cap_conflicts_only_with_its_own_provider() -> None:
    with pytest.raises(ValueError, match="spend conflicts with legacy"):
        SwarmSpec(
            objective="measure",
            providers=(ProviderBinding(ResourceProvider.MODAL, ResourceLimit(max_spend_usd=1)),),
            spend=SpendLimit(resources=(ResourceCap(Resource.INFERENCE, usd=1, provider="xai"),)),
        )


def test_legacy_fields_outside_the_spend_contract_do_not_conflict() -> None:
    # max_gpu_hours, wall clock and concurrency are not compiled into spend caps in phase 1.
    SwarmSpec(
        objective="measure",
        timebox_seconds=60,
        limit=ResourceLimit(max_gpu_hours=3, max_wallclock_seconds=30, max_concurrent_actors=2),
        spend=SpendLimit(max_usd=5),
    )


# --- read model ---------------------------------------------------------------------------


def _selector(**extra: object) -> dict[str, object]:
    return {
        "kind": "resource",
        "capability": None,
        "provider": "modal",
        "model": None,
        "actor_type": None,
        "actor_id": None,
        "resource_id": None,
        **extra,
    }


def test_selector_reads_resource_and_sku() -> None:
    selector = SmrResourceLimitSelector.from_wire(_selector(resource="gpu", sku="H100"))
    assert selector.resource == "gpu"
    assert selector.sku == "H100"
    assert selector.spend_resource is Resource.GPU


def test_selector_tolerates_missing_resource_and_sku() -> None:
    selector = SmrResourceLimitSelector.from_wire(
        {key: value for key, value in _selector().items() if key != "provider"} | {"kind": "run"}
    )
    assert selector.resource is None
    assert selector.sku is None
    assert selector.spend_resource is None


def test_selector_unknown_resource_stays_readable() -> None:
    selector = SmrResourceLimitSelector.from_wire(_selector(resource="quantum"))
    assert selector.resource == "quantum"
    assert selector.spend_resource is None


def test_unselected_all_cap_keeps_run_kind() -> None:
    selector = SmrResourceLimitSelector.from_wire(
        _selector(kind="run", provider=None, resource="all")
    )
    assert selector.kind == "run"
    assert selector.spend_resource is Resource.ALL
