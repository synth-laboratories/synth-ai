# Handoff: make the slot an address, not a contract field

**Date:** 2026-07-31
**Audience:** Eng owning the `synth-ai` public boundary and local multi-slot dev
**Repo:** `synth-laboratories/synth-ai` (siblings: `backend`, `testing`)
**Related:** `backend/specifications/bugs/2026-07-30_run-budget-guardrails-and-slot-authority.md`,
`testing/backend/unit/synth_ai_sdk/public_surface/test_internal_vocabulary.py`

---

## 0. The idea

Today the SDK models *which backend instance owns a run* as data: contract
fields like `slot_id`, `owner_slot_id`, `worker_pool_id`, `backend_url`. A
customer-importable module therefore has to name backend scheduling internals,
which is why `test_internal_vocabulary` has a 32-entry known-offender list.

**Proposal: let the base URL carry that identity instead.** Each slot gets its
own backend address. The client's only slot-awareness is which URL it points at:

```python
# slot 2's backend
SynthClient(base_url="http://localhost:8102")
# or
SYNTH_BACKEND_URL=http://localhost:8102
```

Everything downstream — which runs you see, which pool serves them, who owns the
limits — follows from the address you called. The backend already knows its own
slot; it does not need the client to tell it, and the client does not need a
field to find out.

If a run's owner is "the thing at the URL you called", then `owner_slot_id` is
not information the client needs. It is a field that exists only because the
address was ambiguous.

## 1. Why this is worth doing now

The immediate trigger: the limits work added `owner_slot_id` to
`SmrRunLimitLaunchEvidence` (`synth_ai/sdk/research/contracts/limit_evidence.py:109`),
which would be the **fifth** `slot_id` offender. The ratchet's own docstring says
*"Removing the last entry for a term is what finishing the job looks like."*

`slot_id` is at 4. It is the closest term to finishable. Adding a fifth moves it
the wrong way; this refactor takes it to zero.

The mechanism already exists and needs no new invention:

- `synth_ai/sdk/research/client.py:46,51` and `:116,121` — both clients accept
  `base_url` and fall back to `BACKEND_URL_BASE` via `normalize_backend_base()`.
- Env overrides already shipped: `SYNTH_BACKEND_URL`, `SYNTH_BACKEND_URL_OVERRIDE`,
  `SYNTH_API_URL`, `SYNTH_RESEARCH_BASE`, `SYNTH_REQUIRE_EXPLICIT_BACKEND`.

So the client-side change really is "point at a different URL". The work is
deleting the redundant modelling behind it.

## 2. Scope — be honest about what this does and does not fix

The 32 offenders are not one problem. Grouped by term:

| Term | Count | In scope? |
|---|---|---|
| `worker_pool` | 12 | **Candidate** — per-field judgement needed (see below) |
| `slot_id` | 4 | **Yes** — this is exactly the addressing concern |
| `railway` | 1 | **Yes** — `RAILWAY_NETWORK` in `smr_network_topology.py:10` is deploy-host addressing |
| `daytona` | 13 | **No** — different concern |
| `smr_runtime` | 2 | **No** — different concern |

**`daytona` is not this refactor.** A customer's run genuinely executes in a
sandbox, and the sandbox provider is an execution-environment fact, not an
addressing one. Changing the base URL does not make Daytona stop being where the
code runs. Those 13 need their own decision (probably: an opaque
`ExecutionEnvironment` kind that does not name the vendor). Do not let this
handoff be read as covering them.

**`worker_pool` needs per-field judgement, not a blanket sweep.** In local dev,
slot → pool is effectively 1:1, so pool identity collapses into the address. In
hosted multi-tenant operation it may be a real routing concept that survives.
Work through the 12 sites individually; some will delete, some will not. Do not
assume the count goes to zero.

Realistic target: **`slot_id` 4 → 0, `railway` 1 → 0, `worker_pool` 12 → some
smaller number.** That is 5 guaranteed removals plus whatever `worker_pool`
yields, out of 32.

## 3. The affected contracts

Modules carrying the in-scope terms:

| Module | Terms | Notes |
|---|---|---|
| `contracts/local_execution_profile.py` | `slot_id`, `worker_pool`, `daytona`, `smr_runtime` | Densest site. `LocalEvalContract` (`:239-248`) requires `slot_id`, `runtime_id`, `worker_pool_id` as **required** wire fields. |
| `contracts/run_observability.py` | `slot_id`, `worker_pool` | `:197` requires `slot_id` under `local_execution`. |
| `contracts/swarms.py` | `slot_id`, `worker_pool`, `daytona` | `:548,557,563,828` — `slot_id` is required and serialized. |
| `contracts/factory_evidence.py` | `slot_id`, `smr_runtime` | `:741`. |
| `contracts/limit_evidence.py` | `slot_id` (new) | `:109,134` `owner_slot_id`. Not yet allowlisted — see §6. |
| `contracts/smr_network_topology.py` | `railway` | `:10` `RAILWAY_NETWORK` enum member. |

Note several of these are **required** wire fields, not optional. Removing them
is a wire-contract change on the backend side too, so this is not a pure SDK
edit — see WP2.

## 4. Work packages

### WP0 — Decide the addressing scheme

Pick and write down how a slot maps to a URL, because everything else depends on
it. Options: port-per-slot (`localhost:8100+N`), hostname-per-slot
(`slot-2.local`), or path prefix. Port-per-slot is the least invasive given
`normalize_backend_base()` already handles host:port.

Deliver: the mapping, where it is configured, and how a developer discovers
their slot's URL. Confirm the backend can already be started bound to a
slot-specific port (it reads its own slot from env today).

### WP1 — Backend stamps ownership server-side

Before deleting anything from the SDK, make sure the diagnostic that motivated
`owner_slot_id` survives. The slot-authority bug doc requires run receipts to
carry owner slot and backend URL so a cross-slot incident is diagnosable.

Deliver: backend-side evidence retains owner slot and backend URL in its own
persisted records and operator-facing views. The SDK contract does not need to
expose them for that requirement to be met — operators read backend evidence,
not the customer SDK.

**This is the gating work package.** Do not remove SDK fields until the
diagnostic exists server-side, or the incident that prompted the field becomes
undiagnosable again.

### WP2 — Retire the required slot fields

For each of `local_execution_profile.py`, `run_observability.py`, `swarms.py`,
`factory_evidence.py`: remove `slot_id` from the dataclass, `from_wire`, and any
`to_wire`/serialization. Because several are **required**, coordinate the
backend stopping emission — decode should tolerate the field's absence first,
then the backend stops sending, then the field goes.

Suggested order per field: make optional → backend stops emitting → delete.
Three steps, each independently shippable.

### WP3 — `worker_pool`, case by case

Walk the 12 sites. For each, decide whether pool identity is (a) implied by the
address and deletable, or (b) a genuine hosted routing concept that stays. Record
the verdict per site; a site that stays should get a comment saying why, so the
next person does not re-litigate it.

### WP4 — `railway`

`RAILWAY_NETWORK` (`smr_network_topology.py:10`) names a deploy host in a
customer-importable enum. Replace with a vendor-neutral member describing the
topology, not the provider.

### WP5 — Shrink the ratchet, and prove it

Delete the corresponding entries from
`testing/backend/unit/synth_ai_sdk/public_surface/internal_vocabulary_allowlist.json`.

The suite enforces this for you in both directions:
`test_no_new_internal_vocabulary` fails if an offender is not listed, and
`test_allowlist_does_not_go_stale` fails if a listed entry no longer describes a
real offender. So you cannot delete a field without deleting its entry, and you
cannot delete an entry without deleting the field. Landing the two together is
the only green state.

Success for this handoff: zero `slot_id` entries, zero `railway` entries.

## 5. What does not change

- The public REST API. This is about SDK contract fields, not routes.
- Hosted/production addressing. Customers already point at one backend URL; this
  makes local multi-slot behave the same way rather than special-casing it.
- Authentication. Slot-specific URLs do not imply slot-specific credentials.

## 6. Interim decision, pending this work

`owner_slot_id` in `limit_evidence.py` is currently red against the vocabulary
guard and is **not** yet in the allowlist.

If this refactor is going to happen, do not add a fifth `slot_id` entry — that
records a boundary regression you intend to reverse within weeks. **Drop
`owner_slot_id` from the SDK contract now** and keep the value in backend-side
evidence per WP1.

Worth knowing when weighing this: `SmrRunLimitLaunchEvidence` is **not exported**
— it is absent from `synth_ai.sdk.research.public` and from `dir(synth_ai)`. It
is import-reachable but not advertised, so removing it from the SDK contract
costs no documented surface.

## 7. Open questions

1. Does anything outside the backend actually *consume* `slot_id` from these
   contracts today, or is it write-only evidence? Grep consumers before deleting;
   if a local dev tool reads it, that tool needs the URL instead.
2. Is `worker_pool_id` load-bearing for hosted routing, or also implied by the
   address? WP3 hinges on this.
3. `LocalEvalContract` also carries `runtime_id` and `launch_target`. Neither
   trips the guard, but if slot identity moves to the address, check whether they
   are still meaningful or are part of the same redundant modelling.
