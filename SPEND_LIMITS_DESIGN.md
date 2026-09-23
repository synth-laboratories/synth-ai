# Swarm spend limits: one cap, with per-resource detail

Status: proposal, 2026-09-23. Nothing is implemented yet.

This covers the SDK (`synth_ai/sdk/research/contracts/swarms.py`) and the backend
(`claude/swarm-fold-20260923`). Backend paths below are relative to the backend repo.

## Summary

Today a Swarm has five ways to limit spend:
- `limit.max_spend_usd`;
- `run_policy.limits.total_cost_cents`;
- `providers[].limit`;
- `limit.max_gpu_hours`;
- `timebox_seconds`.

They do not agree. Two are silently dropped. GPU and sandbox compute can't be capped at all.

The proposal replaces them with one typed `SpendLimit`. It has a total dollar cap, plus optional
per-resource caps that are selected by resource, provider, model or actor. Each resource cap is in
dollars or in that resource's native unit. The backend already has most of the machinery: storage,
evaluation, pause/stop effects, warnings, extensions and an itemized read model. What's missing is:
- selectors in storage;
- metering for GPU and sandbox;
- one rate card;
- a single enforcement path.

## Review: what exists today

### What the SDK accepts
- **`ResourceLimit`** (`swarms.py:371`) is a flat set of five fields: `max_spend_usd`,
  `max_tokens`, `max_wallclock_seconds`, `max_gpu_hours` and `max_concurrent_actors`.
- **`RunPolicy.limits`** (`:434`) takes `total_cost_cents`.
- **`ProviderBinding.limit`** takes a `ResourceLimit` per provider.
- **`timebox_seconds`** is a plain integer.

### What the backend does with each input

| Input | Result |
|---|---|
| `limit.max_spend_usd` / `max_tokens` / `max_wallclock_seconds` | Written as run-scoped rows in `smr_limits`, with the pause policy (`services/smr/run_start/limit_envelope.py:220-316`). |
| `limit.max_gpu_hours` | Stored and reported, but **never enforced**. Nothing meters GPU hours. The evaluator treats accounting as incomplete and returns `refuse` (`packages/smr/runtime/limits_authority.py:413-428`), and progress is hard-coded to 0 (`public_run_observability_service.py:887`). |
| `limit.max_concurrent_actors` | An admission cap, not a limit row (`services/smr/runtime/resource_queue.py`). This is fine. |
| `timebox_seconds` | Becomes `wallclock_seconds` with the stop policy. The stricter of it and `max_wallclock_seconds` wins. |
| `providers[].limit` | **Silently ignored.** The schema accepts it, it's echoed back, and nothing reads it (`services/smr/launch_axes.py:315-340`). |
| `run_policy.limits.total_cost_cents` | **Never stored.** It's a competing candidate only in the legacy spend check, where the minimum wins (`services/smr/runtime/spend_recording.py:233`). |

### Other structural problems
1. **The selector exists only in the read model.** The SDK reads `SmrResourceLimit.selector`
   (kind/provider/model/actor), but storage has no selector columns. The primary key is
   `(scope_kind, scope_id, dimension)`, and the selector is always `{kind: "run"}`.
2. **Sandbox and Modal compute cost $0.** `metered_infra_registry.py:376-422` records unpriced
   providers at zero, so a total-dollar cap doesn't actually bound compute.
3. **Rates live in five places.**
   - `PricingEngine` (`core/billing/engine.py`) handles tokens, `gpu_seconds` (rates from environment variables) and sandbox time.
   - Separate rates exist for VMs, dev-environment uptime and Modal planning.
4. **Two enforcers run side by side.** They are the legacy `spend_recording` enforcers and the canonical
   `limit_enforcement_service`. The timebox runs in both.
5. **Limits are shadowed by default.** Envelope mode defaults to `shadow`, so a missing required cap doesn't block launch.
6. **Scopes are run and objective only.** There is no project or org cap.
7. **The SDK has no client for the existing limit write routes.** They are `POST/PATCH/DELETE .../limits`.
   The SDK also parses `limit_quantity`, which the backend never sends.

## Proposed API (SDK)

```python
from synth_ai.sdk.research import SpendLimit, ResourceCap, Resource, LimitAction

spec = SwarmSpec(
    objective="...",
    spend=SpendLimit(
        max_usd=25.00,                      # total across every priced resource
        on_exhaustion=LimitAction.PAUSE,    # PAUSE (default) | STOP
        warn_at=0.9,
        resources=(
            ResourceCap(Resource.GPU, hours=4),                        # native unit
            ResourceCap(Resource.GPU, usd=10.00, provider="modal"),    # dollars, one provider
            ResourceCap(Resource.INFERENCE, usd=8.00, model="openai/gpt-5.6-luna"),
            ResourceCap(Resource.INFERENCE, tokens=2_000_000, actor_type="worker"),
            ResourceCap(Resource.SANDBOX, hours=6),
            ResourceCap(Resource.WALLCLOCK, seconds=600, on_exhaustion=LimitAction.STOP),
        ),
    ),
)
```

### Rules, validated in `__post_init__` and again in the backend

- **`Resource` is a closed enum:** `INFERENCE`, `GPU`, `SANDBOX`, `BROWSER`, `VM`, `WALLCLOCK`.
  Each has one fixed native unit:

  | Resource | Native unit |
  |---|---|
  | `INFERENCE` | tokens |
  | `GPU` | hours |
  | `SANDBOX` | hours |
  | `BROWSER` | hours |
  | `VM` | hours |
  | `WALLCLOCK` | seconds |

  Wall-clock time has no dollar cap.
- **Each `ResourceCap` sets exactly one measure:** either `usd=` or the resource's native unit. Setting
  both is an error, and so is using the wrong unit, such as `ResourceCap(Resource.GPU, tokens=...)`.
- **Selectors (`provider`, `model`, `actor_type`) are optional and narrow a cap.** Two caps with the same
  selector and measure are an error. Overlapping caps all apply, and the first to exhaust acts.
- **`max_usd` always applies to the sum.** Per-resource dollar caps don't need to add up to it, and a
  cap larger than `max_usd` is allowed but reported as redundant in the preflight.
- **Money is integer cents on the wire.** `usd=` accepts `Decimal`, `int` or `float` and is converted
  exactly, so floats never round-trip.
- **`on_exhaustion` can be set per cap.** Each cap inherits `SpendLimit.on_exhaustion` unless it sets its own.

### Wire format

The cap list maps one-to-one onto the itemized model the SDK already reads:

```json
"spend": {
  "max_usd_cents": 2500, "on_exhaustion": "pause", "warn_at_fraction": 0.9,
  "resources": [
    {"selector": {"kind": "resource", "resource": "gpu"}, "metric": "gpu_hours", "limit": 4},
    {"selector": {"kind": "resource", "resource": "gpu", "provider": "modal"}, "metric": "spend_usd_cents", "limit": 1000},
    ...
  ]
}
```

What comes back reuses `SmrResourceLimitProgressItem`. It already carries the selector, the metric, the current
and remaining amounts, the state, blockers and extension policy. The selector becomes real instead of always `run`.

### Compatibility with the old inputs

The old fields keep working, but they're compiled into `SpendLimit`, so nothing is silently dropped:

| Old input | Compiled into |
|---|---|
| `limit.max_spend_usd`, `run_policy.limits.total_cost_cents` | `max_usd`, taking the minimum. Both at once is a deprecation warning. |
| `providers[i].limit.max_spend_usd` | `ResourceCap(INFERENCE, usd=…, provider=…)` |
| `limit.max_tokens` | `ResourceCap(INFERENCE, tokens=…)` |
| `limit.max_gpu_hours` | `ResourceCap(GPU, hours=…)` |
| `timebox_seconds` / `max_wallclock_seconds` | `ResourceCap(WALLCLOCK, seconds=…, STOP)` |
| `limit.max_concurrent_actors` | unchanged; it stays an admission cap |

Passing both `spend=` and a legacy field that disagrees with it is a validation error.

## Backend changes

1. **Storage (migration).** Add `limit_id` as the primary key to `smr_limits`, plus the columns
   `selector_resource`, `selector_provider`, `selector_model` and `selector_actor_type`, with a unique index on
   `(scope_kind, scope_id, dimension, selector_*)`. Existing rows get the selector `run`. Add `spend_usd_cents`
   (integer), `gpu_hours`, `sandbox_hours`, `browser_hours` and `vm_hours` to the dimension catalog in
   `packages/smr/domain/limits.py`.
2. **One rate card.** Build `RateCard.price(resource, provider, model, quantity) -> cents`. Move the `PricingEngine`
   rates, the VM rate (`cloud_deployment_billing.py:78`) and the dev-environment rate
   (`persistence/dev_environments.py:111`) behind it. **Fail closed:** if a run has a dollar cap and uses a resource
   with no rate, launch is refused with `unpriced_resource_under_spend_cap`. It is no longer recorded at $0.
3. **Metering.** `metered_infra_registry` already records `gpu_seconds` and `sandbox_seconds`. Convert those into
   `gpu_hours` / `sandbox_hours` usage facts carrying the provider, and add both to the deployed counters
   (`limit_enforcement_service.py:56-60`). This makes GPU caps enforceable. Inference usage facts already carry the
   provider and model.
4. **Evaluation.** `evaluate_run_limit` sums usage facts that match each row's selector. The total cap sums all
   priced facts through the rate card.
5. **One enforcer.** Retire the legacy `spend_recording` enforcers, including the second timebox, and keep
   `limit_enforcement_service`.
6. **Envelope mode.** Switch the default from `shadow` to `enforce` once steps 1–5 ship.
7. **Scopes (later).** Add project and org scopes, so a team can set a monthly GPU-hour cap across Swarms.

## Phases

| Phase | Scope | Result |
|---|---|---|
| 1 | SDK `SpendLimit`, wire schema, selector columns, compile the legacy inputs, per-provider/model inference caps in dollars and tokens | The total cap and inference detail are enforced; no input is silently dropped |
| 2 | Rate card, GPU/sandbox/browser/VM metering, fail closed on unpriced resources | GPU hours and compute caps are enforced, and the total cap truly covers compute |
| 3 | One enforcer, `enforce` default, project/org scopes, SDK client for the limit CRUD and extension routes | One code path, and caps beyond a single Swarm |

## Decisions needed

1. **Disagreeing legacy spend caps.** Take the minimum with a warning (proposed), or reject?
2. **Unpriced resources under a dollar cap.** Refuse launch (proposed), or allow with a warning?
3. **Default action on exhaustion.** `pause` (the current default, which lets someone extend the cap) or `stop`?
4. **Should per-resource dollar caps be allowed to exceed `max_usd`?** Proposed: yes, reported as redundant.
