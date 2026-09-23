# Swarm spend limits: one cap, with per-resource detail

Status, 2026-09-23: **phase 1 and most of phase 2 are implemented** on unpushed branches `claude/spend-limits-20260923`.

**Backend (off origin/dev):**
- `48c497a08` phase 1
- `ee65159a8` golden wire test
- `86f63e8b2` hour metering
- `46be295ed` fail closed on unpriced usage
- `bf7ec693e` `accept_unpriced_usage` extensions
- `a34a9d80b` inference admission fix

**SDK (off origin/main):**
- `1d3bc466` types
- `936b0046`, `4406a60b` golden wire file
- `b18246b2` extensions
- `29b0907c` per-scope limit CRUD client

## Progress against the handoff (B1–B20)

**Done:**

| Item | What was done |
|---|---|
| B2 | Read-model change checked. No consumer filters on `selector.kind`. `{"kind":"run"}` still addresses every primary row, and primary `resource_limit_id`s are unchanged. Only the label changes, for unselected `tokens`/`wallclock_seconds` caps. |
| B3 | The backend branch is rewritten so every commit builds and tests on its own. Phase 1 is one commit, and the tree is identical to before. |
| B7 | `gpu_hours` is enforced: wall-clock seconds of GPU usage / 3600. GPU type (`sku`) comes from the linked spend-ledger row's `metadata.gpu_type`. A metered-infra container that carries a GPU type classifies as GPU; Modal GPU containers are recorded as `sandbox_seconds`, so this rule is what makes Modal GPU caps work. The legacy `max_gpu_hours` cap is enforced through the same path, and its "not enforced" warning is gone. |
| B8 | `sandbox_hours` and `vm_hours` are enforced from the wall-clock meters (`sandbox_seconds`, `modal_sandbox_seconds`, `dev_environment_uptime_seconds`, `cloud_deployment_vm_seconds`). A matched meter with no wall-clock basis, e.g. Daytona vCPU- or GiB-seconds, makes the cap incomplete rather than guessed. |
| B11 | Fail closed. Usage recorded without a price (`observability_only_modal`, `observability_only_unpriced_gpu`) is reported under every dollar cap that covers it. A cap whose policy sets `fail_closed_on_unpriced` pauses or stops the run with reason `unpriced_usage_under_spend_cap`. **Only caps from the `spend` API set it.** Legacy and runbook caps keep counting unpriced usage as $0, so existing Modal runs are unaffected, but their receipts now name the unpriced usage. The flag is written only when set, so legacy policy payloads are unchanged. |
| B11 follow-up | A run paused this way can be resumed. An extension with `accept_unpriced_usage=true` clears the flag in the same revisioned update, and the cap may stay the same. It is available in the SDK on every `extend_*resource_limit` method. |
| B12 | Largely already true. Token totals include hidden reasoning tokens (provider `total_tokens`, or input + output where output includes reasoning). The gateway prices from the provider-reported `cost` before falling back to contract rates (`gateway/usage.py`). |
| B15 | The not-metered rejection is lifted for `gpu_hours`, `sandbox_hours` and `vm_hours`. The golden file is updated in both repos. |
| B19 | SDK `session.scope_limits`: list, get, create, update and delete for run/objective caps. `update()` keeps the cap unless one is given (`KEEP_CAP`), because the route treats an explicit `null` as removing it. |

**Found and fixed along the way:**
- **Inference admission read selector-scoped caps as the run's caps.** A worker-only tokens cap made the whole run token-governed and could be reserved against in place of the primary row. Admission now reads primary rows only. The Postgres test fails on the old code.
- **SDK extension selectors dropped `resource`/`sku`.** A per-resource cap read from the progress model could not be extended. Now they are carried.

**D1 resolved: Modal rates** (backend `a3afaa91a`).
- **The rates file:** Modal's list prices from modal.com/pricing (fetched 2026-09-23) live in the dedicated `core/billing/resources/modal.toml`. It covers 11 GPUs, CPU and memory for functions and for sandboxes, the region and non-preemptible multipliers, and Modal's size defaults. `modal_rates.py` loads it as exact decimals.
- **What metered infra records:** each Modal window is priced at GPU + CPU + memory, with pricing policy `backend_priced_estimated_usage`. The price list's GPU name is recorded as `gpu_type`, which is the cap `sku`. The rate-card provenance and any defaulted sizes go on the record.
- **Unknown GPUs:** a window with an unknown GPU stays unpriced, so fail-closed still applies to it.
- **Billing is unchanged:** Modal is not charged to customers.
- **Behaviour change:** Modal usage now counts toward every dollar cap, legacy caps included, where it used to count as $0. A run that uses Modal can therefore reach its existing total spend cap sooner.
- **What the prices assume:** base region and preemptible (`default_multiplier = 1.0`). Raise the multiplier if our Modal usage is region-pinned or non-preemptible.

**Not done, with reasons:**

| Item | Status |
|---|---|
| Still not metered | `browser_hours` (browser usage is metered in GB-seconds, not wall time) and Tinker `train_tokens`/`sample_tokens` (Tinker reports per job with a cost, no token split). Dollar caps on BROWSER and TRAINING work. |
| B10 rate-card consolidation | Deferred. Limits need usage with a recorded cost plus a pricing policy that says when it's unpriced, and both exist. Merging `PricingEngine`, the VM rate and the dev-environment rate into one module is a billing refactor with its own risk and no limit behaviour depending on it. |
| B12 (reservation) | Per-call reservation still covers only the run's primary spend and tokens caps. Selected caps are evaluated on every spend write in the threshold band and by the 15 s ticker and sweeper, so they can overshoot by up to ~15 s of spend. Reserving against them needs per-selector counters. |
| B12 (402) | An OpenRouter 402 (account out of credit) as a limit blocker is not done. It lives in the actor/gateway error path, not in limits. |
| B13 | Reconciliation is not built. Limits sum whatever usage facts say, so correction facts from provider-usage ingest would flow through automatically. Whether Modal and Tinker ingest writes corrections is unverified. |
| B14 | Per-token pricing for shared Modal serving apps is still open. Modal container rates now exist, but a shared app's per-token rate is a gateway contract-rate route that nobody has set yet. |
| B16 | Not done on purpose. The legacy `spend_recording` enforcers are being migrated under a measured shadow-equivalence rollout (`smr.limit_wallclock_shadow_equivalence.v1`). Removing them belongs to that rollout. |
| B17 | A rollout step, not a code change. The envelope mode comes from `SMR_LIMIT_ENVELOPE_MODE` and `SMR_LIMIT_ENVELOPE_MODE_<LANE>` (default `shadow`). Before setting `enforce` for a lane, check that shadow refusals are near zero: count run-start receipts whose limit-envelope status is `shadow_refused`, per funding lane. |
| B18 | Project and org scopes need product decisions before code: the reset window (calendar month or rolling), which runs count (project-owned only?), and whether a scope cap pauses every run or refuses new launches. `scope_kind` is `run|objective` today, with a DB check. |
| B20 | `limit_quantity` is still parsed by the SDK and never sent by the backend. It is harmless (always `None`). Remove it or populate it in a follow-up. |
| B1, B4, B5 | Pushing, PRs, the migration rollout and the SDK release are waiting for the user's go-ahead. |

## Phase 1 as built

- **Proposed decisions adopted:**
  - disagreeing legacy spend caps → the minimum applies, with a warning;
  - legacy conflicts with `spend` use the strict rule: an omitted cap is a conflict;
  - the default action is `pause`;
  - Tinker stays rejected for Swarms.
- **SDK↔backend agreement:**
  - `tests/fixtures/spend_limits_wire_v1.json` is generated by the SDK and copied into the backend. Both repos test against it.
  - A manual matrix of 66 legacy×spend combinations showed 0 disagreements.
  - Backend read-model selectors parse in the SDK for every resource.
- **Migration** `20260923_add_smr_limit_selectors`:
  - verified on a scratch pgvector/pg16 through the full alembic chain;
  - down and up again round-trips;
  - the downgrade **refuses** while selector-scoped caps exist, rather than deleting them and silently lifting caps.
- **Read-model change to note for clients:** unselected `tokens` and `wallclock_seconds` caps now report `selector.kind = "resource"`, not `"run"`, per the contract. Only the unselected all-spend cap keeps `kind: "run"`.
- **Tests:**
  - backend: 317 relevant unit, integration and file-size tests pass. Three pre-existing failures fail identically on origin/dev: `test_enforced_route_refusal_precedes_capacity_checks` and two `test_other_customer_provider_defaults_pass_all_role_gates` cases.
  - SDK: 188 pass.
- **Not yet done:**
  - no push or PR;
  - no deploy;
  - the intermediate backend commits aren't individually test-gated; only the branch head is.

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
