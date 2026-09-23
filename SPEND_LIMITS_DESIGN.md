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

- **`Resource` is a closed enum:** `ALL`, `INFERENCE`, `TRAINING`, `GPU`, `SANDBOX`, `BROWSER`, `VM`, `WALLCLOCK`, `MISC`.
  Each has one fixed native unit:

  | Resource | Native unit |
  |---|---|
  | `INFERENCE` | tokens |
  | `TRAINING` | tokens (train and sample) |
  | `GPU` | hours |
  | `SANDBOX` | hours |
  | `BROWSER` | hours |
  | `VM` | hours |
  | `WALLCLOCK` | seconds |
  | `MISC` | none: dollars only |
  | `ALL` | none: dollars only |

  Wall-clock time has no dollar cap.
- **`ALL` is the aggregate cap.** It sums every priced charge across every resource, `MISC` included.
  `SpendLimit.max_usd` is shorthand for `ResourceCap(Resource.ALL, usd=…)` with no selector, and they compile to the
  same row. Because `ALL` takes the same selectors as any other cap, it can bound one slice of total spend across
  every resource type:
  - `ResourceCap(Resource.ALL, usd=15, provider="modal")` caps all Modal spend: GPU, sandbox and serving.
  - `ResourceCap(Resource.ALL, usd=5, actor_type="reviewer")` caps everything reviewers cause.

  It is in dollars only, because units can't be summed across resources. It wins over nothing and nothing wins over
  it: every cap applies, and the first to exhaust acts.
- **`MISC` is the catch-all.** Any metered charge that doesn't map to a dedicated resource is recorded as `MISC`.
  That covers metered tools, third-party APIs, and a new provider before it gets its own type. So `max_usd` always
  covers it, and nothing escapes the total. `MISC` caps are in dollars only, because there is no common unit, and
  can be narrowed by `provider` and a free-form `sku`, e.g. `ResourceCap(Resource.MISC, usd=5, provider="exa")`.
  A charge still needs a price: `provider_reported` or `rate_card`. The fail-closed rule applies to `MISC` too.
  When a `MISC` provider becomes significant, it graduates to its own `Resource`, and old caps keep matching
  through the provider selector.
- **Each `ResourceCap` sets exactly one measure:** either `usd=` or the resource's native unit. Setting
  both is an error, and so is using the wrong unit, such as `ResourceCap(Resource.GPU, tokens=...)`.
- **Selectors (`provider`, `model`, `sku` for GPU type, `actor_type`) are optional and narrow a cap.** Two caps with the same
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

## Provider fit: OpenRouter, Modal, Tinker

The first draft of this proposal didn't fit all three providers. The changes below are part of the proposal now.

| Provider | What it bills | Fits the first draft? | Change |
|---|---|---|---|
| **OpenRouter** | Inference, per token, per model. The price varies with the serving stack it routes to, and hidden reasoning tokens are billed. | Yes: `ResourceCap(INFERENCE, provider="openrouter", model=…)` | Price from the cost OpenRouter reports, not our rate card. The gateway already parses `cost` / `cost_usd` (`services/smr/inference/gateway/usage.py:38`). Token caps count reasoning tokens. Reserve before each call so a cap isn't overshot by one large completion. An OpenRouter 402 (account out of credit) becomes a limit blocker naming the provider, not an actor crash. |
| **Modal** | GPU seconds priced by GPU type; CPU/memory seconds for CPU-only containers; scale-to-zero serving apps (the Laguna vLLM apps) | Partly. `GPU hours` mixes H100 and A10G, which have very different prices, and Modal is recorded at $0 today (`metered_infra_registry.py:374-416`, "unbilled until product sets a Modal container rate"). | Add a `sku` selector, e.g. `ResourceCap(GPU, hours=2, provider="modal", sku="H100")`. CPU-only Modal containers count as `SANDBOX` with `provider="modal"`. Price Modal from its GPU-type rate table, and reconcile against Modal's billing later. Until Modal has a rate, the fail-closed rule refuses dollar-capped runs that use Modal (decision 2). |
| **Tinker** | Training and sampling, per token, by base model: prefill, sample and train tokens | **No.** It is neither `INFERENCE` nor `GPU`. The backend already classifies it separately (`metered_infra_registry.py:441` → `third_party_training`; `usage_metering.py:229` → `tinker_request`). | Add `Resource.TRAINING`, native unit tokens, with metrics `train_tokens` and `sample_tokens`, e.g. `ResourceCap(TRAINING, usd=20, provider="tinker", model="Qwen3-8B")`. Price by base model from Tinker's rate table. |

### Where each price comes from

Every resource cap needs to know where its cost figure comes from. The rate card gains a `cost_source` for each provider:

| cost_source | Used for |
|---|---|
| `provider_reported` | OpenRouter. The cost is exact, per call. |
| `rate_card` | Our own infra, Daytona, and Modal by GPU type |
| `rate_card_then_reconciled` | Modal and Tinker. Estimate live, then correct from provider billing. The correction can push a run over its cap after the fact, so each progress item reports it. |

### Shared deployments

A scale-to-zero Modal app that serves many runs, such as `laguna-vllm`, can't be charged by GPU time to one run.
Options:
- (a) charge its calls as `INFERENCE provider="modal"` at an internal per-token rate;
- (b) leave it out of run caps and cap it at project or org scope (phase 3).

Option (a) is proposed.

### Tinker scope

The SDK and backend both reject Tinker for new Swarm launches
(`swarms.py` `ProviderBinding`; backend `contracts/run_policy.py:63-71`). Today's Tinker spend (gold loops, the REB
broker with its task-wide cap) runs outside Swarms. `Resource.TRAINING` is still worth defining, so that those
brokers can report and enforce against the same limit model at project scope. Whether Swarms should accept
Tinker again is a separate decision (decision 6).

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
5. **Shared Modal serving apps.** Charge per token to the calling run (proposed), or cap only at project/org scope?
6. **Tinker in Swarms.** Keep it rejected, with `TRAINING` used only by the external brokers (proposed), or re-admit it?
7. **Modal rate.** Product needs to set a Modal GPU/CPU rate table. Without it, dollar-capped Modal runs are refused.

## Phase 1 wire contract (binding for the SDK and backend implementations)

### Request

Both run-create routes, `POST /smr/runs:one-off` and `POST /smr/projects/{id}/trigger`, accept an optional `spend` object:

```json
"spend": {
  "max_usd_cents": 2500,
  "on_exhaustion": "pause",
  "warn_at_fraction": 0.9,
  "caps": [
    {"selector": {"resource": "gpu", "provider": "modal", "sku": "H100"}, "metric": "spend_usd_cents", "limit": 1000},
    {"selector": {"resource": "inference", "actor_type": "worker"}, "metric": "tokens", "limit": 2000000, "on_exhaustion": "stop"},
    {"selector": {"resource": "all", "provider": "modal"}, "metric": "spend_usd_cents", "limit": 1500}
  ]
}
```

**Top-level fields:**
- `max_usd_cents`: optional int ≥ 0. It compiles to the cap `{selector: {resource: "all"}, metric: "spend_usd_cents"}`.
- `on_exhaustion`: `"pause"` or `"stop"`. The default is `"pause"`. It is the default for every cap.
- `warn_at_fraction`: optional float in (0, 1]. The default is 0.9.

**Selector:** `resource` is required. `provider`, `model`, `sku` and `actor_type` are optional strings.
`actor_type` must be one of the backend's actor types (orchestrator, worker, reviewer, …).

**Metric, allowed per resource:**

| resource | allowed metrics |
|---|---|
| `all` | `spend_usd_cents` |
| `inference` | `spend_usd_cents`, `tokens` |
| `training` | `spend_usd_cents`, `train_tokens`, `sample_tokens` |
| `gpu` | `spend_usd_cents`, `gpu_hours` |
| `sandbox` | `spend_usd_cents`, `sandbox_hours` |
| `browser` | `spend_usd_cents`, `browser_hours` |
| `vm` | `spend_usd_cents`, `vm_hours` |
| `wallclock` | `wallclock_seconds`; no provider, model, sku or actor_type allowed |
| `misc` | `spend_usd_cents` |

**Limit values:**
- `limit` must be ≥ 0.
- `spend_usd_cents`, `tokens`, `train_tokens`, `sample_tokens` and `wallclock_seconds` are integers.
- `*_hours` metrics are decimals with at most 4 places.

**Rejections, all HTTP 422 with a stable `code`:**
- `spend_cap_duplicate`: two caps share the same (selector, metric).
- `spend_cap_metric_not_allowed`: the metric isn't allowed for the resource.
- `spend_metric_not_metered`: phase 1 does not meter `gpu_hours`, `sandbox_hours`, `browser_hours`, `vm_hours`, `train_tokens` or `sample_tokens`. It rejects caps on them rather than accept a cap it can't enforce. Phase 2 lifts this.
- `spend_conflicts_with_legacy_limit`: `spend` is given together with a legacy limit field whose compiled cap disagrees.

### How the old inputs compile

The backend compiles the legacy inputs into the same cap list when `spend` is absent:

| Legacy input | Compiled cap |
|---|---|
| `limit.max_spend_usd` and `run_policy.limits.total_cost_cents` | `all`/`spend_usd_cents`, taking the minimum. Both at once is a warning. |
| `providers[i].limit.max_spend_usd` | `inference`/`spend_usd_cents` with `provider` = that provider. This input was dropped before. |
| `providers[i].limit.max_tokens` | `inference`/`tokens` with `provider` |
| `limit.max_tokens` | `inference`/`tokens` |
| `limit.max_wallclock_seconds`, `timebox_seconds` | `wallclock`/`wallclock_seconds`, stricter value wins, `stop` for the timebox (unchanged) |
| `limit.max_gpu_hours` | Kept as today: stored, reported as `accounting_incomplete`, not enforced. Legacy callers aren't broken in phase 1, and the create response carries a warning. |

### Classifying usage into resources

Every existing spend-ledger or usage-fact row maps to exactly one resource:

| Usage | Resource |
|---|---|
| inference/model usage | `inference` |
| `tinker_request` / `third_party_training` | `training` |
| `gpu_seconds` / `third_party_gpus` | `gpu` |
| `sandbox_seconds` / Daytona container time | `sandbox` |
| browser/kernel | `browser` |
| cloud VM deployment | `vm` |
| anything else, metered tools included | `misc` |

`all` matches every row.

A cap's usage is the sum over rows matching its selector: resource, plus provider, model, sku and actor_type when each is set.

### Read model

`GET .../resource-limits` and the progress endpoints return one item per stored cap. `selector` gains `resource` and
`sku`. `selector.kind` is `"resource"` for every cap. The one exception is the unselected `all` spend cap, which keeps
`kind: "run"` for backward compatibility. `metric` is the new metric name. Extensions accept any stored cap's
selector plus metric, and no longer only `{kind: "run"}`.
