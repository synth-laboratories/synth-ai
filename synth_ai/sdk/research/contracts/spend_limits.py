"""Typed Swarm spend limits: one total dollar cap plus per-resource caps.

A ``SpendLimit`` compiles to the ``spend`` object on the run-create routes
(``POST /smr/runs:one-off`` and ``POST /smr/projects/{id}/trigger``). The backend is
the contract authority and validates everything again; the checks here fail early
with the same rules so a bad cap never leaves the client.

# See: SPEND_LIMITS_DESIGN.md, "Phase 1 wire contract"
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.sdk.research.contracts.common import require_text


class Resource(StrEnum):
    """What a cap meters. Every usage row maps to exactly one resource.

    ``ALL`` is the aggregate: it matches every priced charge, ``MISC`` included, and
    takes the same selectors as any other resource. ``MISC`` is the catch-all for
    metered charges with no dedicated resource (metered tools, third-party APIs, a
    new provider before it gets its own type).
    """

    ALL = "all"
    INFERENCE = "inference"
    TRAINING = "training"
    GPU = "gpu"
    SANDBOX = "sandbox"
    BROWSER = "browser"
    VM = "vm"
    WALLCLOCK = "wallclock"
    MISC = "misc"


class LimitAction(StrEnum):
    """What happens when a cap is exhausted. ``PAUSE`` lets someone extend the cap."""

    PAUSE = "pause"
    STOP = "stop"


class SpendMetric(StrEnum):
    """The wire metric a cap is measured in.

    Phase 1 of the backend meters only ``SPEND_USD_CENTS``, ``TOKENS`` and
    ``WALLCLOCK_SECONDS``. It rejects caps on the ``*_HOURS``, ``TRAIN_TOKENS`` and
    ``SAMPLE_TOKENS`` metrics with ``spend_metric_not_metered`` rather than accept a
    cap it cannot enforce. The SDK still builds them, so it needs no change when the
    backend starts metering them.
    """

    SPEND_USD_CENTS = "spend_usd_cents"
    TOKENS = "tokens"
    TRAIN_TOKENS = "train_tokens"
    SAMPLE_TOKENS = "sample_tokens"
    GPU_HOURS = "gpu_hours"
    SANDBOX_HOURS = "sandbox_hours"
    BROWSER_HOURS = "browser_hours"
    VM_HOURS = "vm_hours"
    WALLCLOCK_SECONDS = "wallclock_seconds"


_ALLOWED_METRICS: dict[Resource, frozenset[SpendMetric]] = {
    Resource.ALL: frozenset({SpendMetric.SPEND_USD_CENTS}),
    Resource.INFERENCE: frozenset({SpendMetric.SPEND_USD_CENTS, SpendMetric.TOKENS}),
    Resource.TRAINING: frozenset(
        {SpendMetric.SPEND_USD_CENTS, SpendMetric.TRAIN_TOKENS, SpendMetric.SAMPLE_TOKENS}
    ),
    Resource.GPU: frozenset({SpendMetric.SPEND_USD_CENTS, SpendMetric.GPU_HOURS}),
    Resource.SANDBOX: frozenset({SpendMetric.SPEND_USD_CENTS, SpendMetric.SANDBOX_HOURS}),
    Resource.BROWSER: frozenset({SpendMetric.SPEND_USD_CENTS, SpendMetric.BROWSER_HOURS}),
    Resource.VM: frozenset({SpendMetric.SPEND_USD_CENTS, SpendMetric.VM_HOURS}),
    Resource.WALLCLOCK: frozenset({SpendMetric.WALLCLOCK_SECONDS}),
    Resource.MISC: frozenset({SpendMetric.SPEND_USD_CENTS}),
}

# `hours=` means a different metric for each hour-metered resource.
_HOURS_METRIC: dict[Resource, SpendMetric] = {
    Resource.GPU: SpendMetric.GPU_HOURS,
    Resource.SANDBOX: SpendMetric.SANDBOX_HOURS,
    Resource.BROWSER: SpendMetric.BROWSER_HOURS,
    Resource.VM: SpendMetric.VM_HOURS,
}

_HOURS_DECIMAL_PLACES_MAX = 4

UsdAmount = Decimal | int | float
HoursAmount = Decimal | int | float


def _usd_to_cents(value: object, *, field_name: str) -> int:
    """Convert dollars to integer cents exactly, rejecting sub-cent precision.

    Floats go through their shortest ``repr`` so ``0.1`` means ten cents, not the
    binary approximation of it.
    """

    amount = _decimal(value, field_name=field_name)
    cents = amount * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"{field_name} must be a whole number of cents, got {value!r}")
    return int(cents)


def _decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float)):
        raise ValueError(f"{field_name} must be a Decimal, int or float")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{field_name} is not a number: {value!r}") from error
    if not amount.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if amount < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return amount


def _hours(value: object) -> Decimal:
    amount = _decimal(value, field_name="hours")
    # 1.50000 is 1.5: trailing zeros are not precision the backend has to store.
    exponent = amount.normalize().as_tuple().exponent
    if isinstance(exponent, int) and -exponent > _HOURS_DECIMAL_PLACES_MAX:
        raise ValueError(
            f"hours allows at most {_HOURS_DECIMAL_PLACES_MAX} decimal places, got {value!r}"
        )
    return amount


def _whole(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class SpendCapSelector:
    """Which usage a cap counts. Unset fields match everything."""

    resource: Resource
    provider: str | None = None
    model: str | None = None
    sku: str | None = None
    actor_type: str | None = None

    @property
    def is_unselected(self) -> bool:
        return all(
            value is None for value in (self.provider, self.model, self.sku, self.actor_type)
        )

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"resource": self.resource.value}
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("sku", self.sku),
            ("actor_type", self.actor_type),
        ):
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True, slots=True)
class ResourceCap:
    """One cap on one resource, in dollars or in that resource's native unit.

    Set exactly one measure:

    - ``usd``: any resource except ``WALLCLOCK``; sent as integer cents.
    - ``tokens``: ``INFERENCE``.
    - ``train_tokens`` / ``sample_tokens``: ``TRAINING``.
    - ``hours``: ``GPU``, ``SANDBOX``, ``BROWSER``, ``VM``; at most 4 decimal places.
    - ``seconds``: ``WALLCLOCK``.

    ``provider``, ``model``, ``sku`` (GPU type, or a free-form item for ``MISC``) and
    ``actor_type`` narrow the cap; ``WALLCLOCK`` takes none of them. ``on_exhaustion``
    overrides the ``SpendLimit`` default for this cap only.

    Phase 1 of the backend rejects ``hours``, ``train_tokens`` and ``sample_tokens``
    caps with ``spend_metric_not_metered``; see ``SpendMetric``.
    """

    resource: Resource
    _: KW_ONLY
    usd: UsdAmount | None = None
    tokens: int | None = None
    train_tokens: int | None = None
    sample_tokens: int | None = None
    hours: HoursAmount | None = None
    seconds: int | None = None
    provider: str | None = None
    model: str | None = None
    sku: str | None = None
    actor_type: str | None = None
    on_exhaustion: LimitAction | None = None
    metric: SpendMetric = field(init=False)
    limit: int | Decimal = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resource, Resource):
            raise ValueError("resource must be Resource")
        if self.on_exhaustion is not None and not isinstance(self.on_exhaustion, LimitAction):
            raise ValueError("on_exhaustion must be LimitAction")
        measures = {
            name: value
            for name, value in (
                ("usd", self.usd),
                ("tokens", self.tokens),
                ("train_tokens", self.train_tokens),
                ("sample_tokens", self.sample_tokens),
                ("hours", self.hours),
                ("seconds", self.seconds),
            )
            if value is not None
        }
        if len(measures) != 1:
            given = ", ".join(sorted(measures)) or "none"
            raise ValueError(
                f"ResourceCap({self.resource.value}) needs exactly one of usd, tokens, "
                f"train_tokens, sample_tokens, hours or seconds; got {given}"
            )
        ((measure, value),) = measures.items()
        metric, limit = self._metric_and_limit(measure, value)
        if metric not in _ALLOWED_METRICS[self.resource]:
            raise ValueError(
                f"{measure}= is not a valid measure for {self.resource.value}; allowed metrics: "
                + ", ".join(sorted(metric.value for metric in _ALLOWED_METRICS[self.resource]))
            )
        for name in ("provider", "model", "sku", "actor_type"):
            selector_value = getattr(self, name)
            if selector_value is None:
                continue
            if self.resource is Resource.WALLCLOCK:
                raise ValueError(f"wallclock caps take no selectors; got {name}")
            if not isinstance(selector_value, str):
                raise ValueError(f"{name} must be a string")
            object.__setattr__(self, name, require_text(selector_value, field_name=name))
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "limit", limit)

    def _metric_and_limit(self, measure: str, value: object) -> tuple[SpendMetric, int | Decimal]:
        if measure == "usd":
            return SpendMetric.SPEND_USD_CENTS, _usd_to_cents(value, field_name="usd")
        if measure == "hours":
            hours_metric = _HOURS_METRIC.get(self.resource)
            if hours_metric is None:
                raise ValueError(f"hours= is not a valid measure for {self.resource.value}")
            return hours_metric, _hours(value)
        whole_metric = {
            "tokens": SpendMetric.TOKENS,
            "train_tokens": SpendMetric.TRAIN_TOKENS,
            "sample_tokens": SpendMetric.SAMPLE_TOKENS,
            "seconds": SpendMetric.WALLCLOCK_SECONDS,
        }[measure]
        return whole_metric, _whole(value, field_name=measure)

    @property
    def selector(self) -> SpendCapSelector:
        return SpendCapSelector(
            resource=self.resource,
            provider=self.provider,
            model=self.model,
            sku=self.sku,
            actor_type=self.actor_type,
        )

    def to_wire(self) -> JsonObject:
        limit: int | float
        if isinstance(self.limit, Decimal):
            # Hours travel as a JSON number; whole hours stay integers on the wire.
            integral = self.limit == self.limit.to_integral_value()
            limit = int(self.limit) if integral else float(self.limit)
        else:
            limit = self.limit
        payload: JsonObject = {
            "selector": self.selector.to_wire(),
            "metric": self.metric.value,
            "limit": limit,
        }
        if self.on_exhaustion is not None:
            payload["on_exhaustion"] = self.on_exhaustion.value
        return payload


@dataclass(frozen=True, slots=True)
class SpendLimit:
    """The Swarm's spend envelope: a total dollar cap plus optional per-resource caps.

    ``max_usd`` is shorthand for ``ResourceCap(Resource.ALL, usd=...)`` with no selector;
    setting both is an error. Every cap applies and the first to exhaust acts, so
    per-resource dollar caps need not add up to ``max_usd``. Each cap inherits
    ``on_exhaustion`` unless it sets its own. ``warn_at`` is the fraction of a cap at
    which the backend warns.
    """

    max_usd: UsdAmount | None = None
    on_exhaustion: LimitAction = LimitAction.PAUSE
    warn_at: float = 0.9
    resources: tuple[ResourceCap, ...] = ()
    max_usd_cents: int | None = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.on_exhaustion, LimitAction):
            raise ValueError("on_exhaustion must be LimitAction")
        if (
            isinstance(self.warn_at, bool)
            or not isinstance(self.warn_at, (int, float))
            or not 0 < self.warn_at <= 1
        ):
            raise ValueError("warn_at must be a number in (0, 1]")
        object.__setattr__(self, "warn_at", float(self.warn_at))
        caps = tuple(self.resources)
        for cap in caps:
            if not isinstance(cap, ResourceCap):
                raise ValueError("resources must contain ResourceCap values")
        object.__setattr__(self, "resources", caps)
        max_usd_cents = (
            None if self.max_usd is None else _usd_to_cents(self.max_usd, field_name="max_usd")
        )
        object.__setattr__(self, "max_usd_cents", max_usd_cents)
        seen: set[tuple[SpendCapSelector, SpendMetric]] = set()
        if max_usd_cents is not None:
            seen.add((SpendCapSelector(Resource.ALL), SpendMetric.SPEND_USD_CENTS))
        for cap in caps:
            key = (cap.selector, cap.metric)
            if key in seen:
                if max_usd_cents is not None and key == (
                    SpendCapSelector(Resource.ALL),
                    SpendMetric.SPEND_USD_CENTS,
                ):
                    raise ValueError(
                        "max_usd already sets the unselected ALL dollar cap; "
                        "drop ResourceCap(Resource.ALL, usd=...) or max_usd"
                    )
                raise ValueError(
                    f"duplicate cap for selector {cap.selector.to_wire()} and metric "
                    f"{cap.metric.value}"
                )
            seen.add(key)

    def effective_caps(self) -> dict[tuple[SpendCapSelector, SpendMetric], int | Decimal]:
        """Every cap as the backend stores it, with ``max_usd`` compiled to its ALL cap."""

        caps: dict[tuple[SpendCapSelector, SpendMetric], int | Decimal] = {}
        if self.max_usd_cents is not None:
            caps[(SpendCapSelector(Resource.ALL), SpendMetric.SPEND_USD_CENTS)] = self.max_usd_cents
        for cap in self.resources:
            caps[(cap.selector, cap.metric)] = cap.limit
        return caps

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {}
        if self.max_usd_cents is not None:
            payload["max_usd_cents"] = self.max_usd_cents
        payload["on_exhaustion"] = self.on_exhaustion.value
        payload["warn_at_fraction"] = self.warn_at
        payload["caps"] = [cap.to_wire() for cap in self.resources]
        return payload


@dataclass(frozen=True, slots=True)
class LegacyCompiledCap:
    """A cap that a legacy launch field compiles into, named for error messages."""

    source: str
    selector: SpendCapSelector
    metric: SpendMetric
    limit: Decimal


def check_legacy_agreement(spend: SpendLimit, legacy_caps: Sequence[LegacyCompiledCap]) -> None:
    """Raise when ``spend`` and a legacy limit field would compile to different caps.

    Mirrors the backend's ``spend_conflicts_with_legacy_limit``. A legacy cap agrees only
    when ``spend`` carries the same (selector, metric) with the same limit; a legacy cap
    that ``spend`` omits is a conflict too, because silently merging or dropping it is
    exactly what ``SpendLimit`` exists to prevent.
    """

    effective = spend.effective_caps()
    for legacy in legacy_caps:
        current = effective.get((legacy.selector, legacy.metric))
        if current is None or Decimal(current) != legacy.limit:
            expected = "no such cap" if current is None else f"limit {current}"
            raise ValueError(
                f"spend conflicts with legacy {legacy.source}: it compiles to "
                f"{legacy.selector.to_wire()} {legacy.metric.value}={legacy.limit}, "
                f"but spend has {expected}; set the limit in one place"
            )


__all__ = [
    "LegacyCompiledCap",
    "LimitAction",
    "Resource",
    "ResourceCap",
    "SpendCapSelector",
    "SpendLimit",
    "SpendMetric",
    "check_legacy_agreement",
]
