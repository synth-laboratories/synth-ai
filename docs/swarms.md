# Swarms

Swarms requires organization admission. Inspect `client.account.entitlements()`;
unsupported operations raise `UnsupportedOperationError` with code
`operation_unavailable`. An SDK installation does not grant execution access.

Use `client.swarms.projects.create(name, objective=...)` for a server-authored
report project, or select an existing configured project. The README contains a
complete launch and wait example. `Limits.maximum_charge` is exact decimal money:
`Money(coefficient=250, scale=2)` means USD 2.50. Wall time, task count and parallel
task count are explicit. A local observation timeout does not stop remote work.

## Inputs and guidance

`message` describes the work. `guidance` supplies typed hints with stable IDs,
kind `question`, `outcome`, or `any`, and priority `must`, `should`, or `nice`.
The planner authors formal objectives and records each hint as covered or declined.
A reviewer verifies required objectives independently. `run.objectives()` returns
the current acknowledged objective set. `run.result()` exposes terminal answers,
verdicts and coverage alongside execution status.

```python
from pathlib import Path
from synth_ai.sdk.core import ArtifactSource
from synth_ai.sdk.swarms import Guidance, Inputs

source = ArtifactSource.from_path(Path("context.txt"), media_type="text/plain")
artifact = client.artifacts.upload(project_id, source, labels={"purpose": "input"})
inputs = Inputs(artifacts=(artifact.ref,))
guidance = (Guidance(id="comparison", kind="outcome", text="Compare both methods",
                     priority="must", evidence="Cite reproducible measurements"),)
```

The server reserves the label keys `role`, `task_id`, `logical_output_id` and any
`legacy_*` key (`artifact_reserved_labels`, HTTP 422); choose other keys.

Pass `inputs` and `guidance` to `client.swarms.create(...)` together with the
project, message and limits. Input artifacts are immutable exact revisions;
mutable aliases are resolved before launch. Artifact references are not access
grants. A stale or inaccessible revision is refused before execution.

## Resources

Resources are optional. Omission grants no additional plugin tools.

```python
from synth_ai.sdk.swarms import ResourceRequest

resources = (ResourceRequest(
    plugin="openrouter.inference", roles=("worker",),
    models=("openai/gpt-oss-20b",), limits={"spend_micros": 1_000_000, "tokens": 100_000},
),)
```

The backend checks account policy, the approved plugin version, credential
references and role limits. It freezes effective grants into the run. Runtime
messages and tool arguments cannot increase them. Index FAST uses
`plugin="index.search"` with `spend_micros` and `index_searches`; `collections`
restricts private collection access. DEEP is not a v0.23 plugin.

Workload code receives a short-lived gateway token at `/run/synth/inference_token`.
`SYNTH_INFERENCE_TOKEN_FILE`, `SYNTH_INFERENCE_BASE_URL` and
`SYNTH_INFERENCE_MODELS` identify its file, endpoint and model allowlist.
The token is revoked at teardown. Provider keys remain with the gateway.
Platform and customer-connected funding do not fall back to each other.
`run.usage().resource_usage` separates plugin, dimension, role and usage class.
Unknown or pending usage has `amount=None`; it is never reported as zero.

## Program families

A launch may select one exact program family version. `client.swarms.families()`
(MCP `swarms_families`) lists the versions this deployment declares; there is
no "latest". `swarms.general@1` runs planner-concluded code work without Forge,
Factory, Effort, wiki or a scientific project. `swarms.research@1` runs the same
host with a research binding: the Forge project and record protocol the run
publishes into.

```python
from synth_ai.sdk.swarms import Limits, ProgramChoice, ProgramFamilyRef, ResearchBinding, ResourceRequest
from synth_ai.types.money import Money

limits = Limits(maximum_charge=Money(coefficient=5, scale=0, currency="USD"),
                wall_time_seconds=3600, maximum_tasks=8, maximum_parallel_tasks=2)
forge_project_id = "63000000-0000-4000-8000-000000000001"
program = ProgramChoice(
    family=ProgramFamilyRef(id="swarms.research", version=1),
    research=ResearchBinding(forge_project_id=forge_project_id, record_protocol="forge.records@1"),
)
resources = (ResourceRequest(
    plugin="forge.research", roles=("worker",), limits={"calls": 50},
    selection={"project_ids": (forge_project_id,)},
),)
client.swarms.create(message="Reproduce the baseline", project_id=project_id,
                     limits=limits, program=program, resources=resources)
```

The research binding's Forge project must be granted by a `forge.research`
resource on the same launch, so the caller's current Forge access is checked
before acceptance. Omitting `program` keeps the project's configured program.
Preflight reports, and create refuses, the same typed codes:
`program_family_version_unknown`, `program_family_context_unbound` (research
without a binding), `program_family_context_forbidden` (a research binding or
Forge grant on the general family), `program_family_research_project_ungranted`
and `program_family_configuration_unsupported` (the project does not run a
registered program). Orchestra resolves the exact program, definition and actor
profiles before dispatch and may refuse with its own `program_family_*` code.

## Model and effort overrides

A launch may fix the model and/or reasoning effort of the roles its program
declares as overridable. Per field, precedence is the template default (the
admitted authored profile), then the launch-wide choice, then the named-role
choice. The launch-wide choice reaches declared coordinator roles only; the
evaluated system changes only through a named-role choice, and never under a
bound research protocol.

```python
from synth_ai.sdk.swarms import Limits, ModelEffortChoice, ProfileOverrides
from synth_ai.types.money import Money

limits = Limits(maximum_charge=Money(coefficient=5, scale=0, currency="USD"),
                wall_time_seconds=3600, maximum_tasks=8, maximum_parallel_tasks=2)
overrides = ProfileOverrides(
    launch=ModelEffortChoice(effort="high"),
    roles={"worker": ModelEffortChoice(canonical_model="gpt-6-luna", effort="low")},
)
client.swarms.create(message="Tune the planner", project_id=project_id,
                     limits=limits, profile_overrides=overrides)
```

`client.swarms.preflight(..., profile_overrides=overrides)` previews the same
choice; `preview.configuration.profile_overrides.resolution.actors` lists each
actor's effective `canonical_model` and `effort`, `model_source` and
`effort_source`, and `base_profile_digest`/`profile_digest`.

MCP `swarms_preflight` and `swarms_create` accept the same `profile_overrides`
object. A ready preflight echoes the exact request and its resolution: per
actor the effective model and effort, their provenance (`template_default`,
`launch` or `role`), the base and derived profile digests, whether the
evaluated system changed, and the declaration, request and resolution digests.
A derived profile differs from its base only in model and effort, so tools,
limits and per-call spend never widen. Create freezes the request; the run
snapshot's `profile_overrides` reads back the frozen request and, once the run
is prepared, the resolution the run executes with. Omitting
`profile_overrides` keeps the launch request bytes and the project's profiles
unchanged.

Preflight reports, and preparation refuses before any paid work, typed codes:
`profile_override_undeclared` (the project's program declares no override
slots), `profile_override_role_unknown`, `profile_override_role_absent`,
`profile_override_unsupported` (a model/effort pair the role does not declare),
`profile_override_protected_protocol` and `profile_override_invalid`. Nothing
is substituted.

## Status facets

`RunSnapshot.state` is unchanged. A server that supports it also returns
`status_facets`, which splits the run into four facts read from their own
authorities, so a terminal `state` never implies the rest:

| Facet | States | Authority |
| --- | --- | --- |
| `execution` | `queued`, `running`, `closing`, `closed`, `never_started`, `unknown` | Orchestra run lifecycle / program terminal; the backend's never-launched closure for a run Orchestra never started |
| `publication` | `not_required`, `pending`, `published`, `waived`, `refused`, `unknown` | Orchestra's obligations-v1 `publication` entries |
| `settlement` | `pending`, `in_doubt`, `refused`, `final`, `unknown` | Backend admission holds + Orchestra `financial.settle` receipts (the same flags as `accounting_final`) |
| `cleanup` | `pending`, `complete`, `failed`, `unknown` | Orchestra lifecycle terminalization; the never-launched closure |

Each facet has an optional `cause` with a stable `code` and the `authority` it
was derived from, for example `{"code": "program_family_context_unbound",
"authority": "orchestra.program_create"}` on a run refused at create.
`execution` has `outcome`, `started_at` and `closed_at`. `cleanup` has
`completed_at`. `publication` has the owner's open/closed/waived `counts`.
`settlement` has `customer_charge_final`, `supplier_settlement_final` and the
owner's settlement `operation_id`. These fields appear only when the server
recorded them. A fact the server did not observe is `unknown`, never a zero or
a success. An older server omits `status_facets`. A newer state decodes as an
unknown value; it is not rejected. Hosted and stdio MCP `swarms_status` return
the same object.

## Controls, results and downloads

`run.pause()`, `run.resume()` and `run.cancel()` return durable control receipts.
Acceptance and physical quiescence are distinct. Use `run.wait_for_state(...)` to
observe the intended state. Resume retains the run identity and deadline and
restores only non-conflicting paused changes onto current repository content.
A snapshot with unsupported entries or a conflicting edit is refused.

Approval-gated capabilities (any `require_approval` tool, such as `git_push`
to a remote other than the run's own Synth-hosted workspace) park until a
principal with `swarms:control` decides them. `run.approvals()` lists the parked
calls with typed `state` and `revision`; `run.decide_approval(approval_id,
decision="approve" | "refuse", expected_revision=...)` relays one decision to
Orchestra, which journals the verdict. A stale `expected_revision` is refused
with `approval_revision_stale` (re-list and decide again); replaying the same
decision for the same revision returns the original verdict. Hosted and stdio
MCP expose the same operations as `swarms_approvals` (optional
`kind`: `"tool_call"`, `"wait"` or `"gate"`, the same closed values as
`run.approvals(kind=...)`) and `swarms_approval_decide`.

The same list carries the run's open human-in-the-loop waits (`kind="wait"`,
with `wait_kind`, `options`, `answer_schema`, `answer_scope`, `answerers`,
`deadline_at` and `on_timeout`). `run.answer_wait(approval_id, decision=...,
expected_revision=...)` answers one; `wait_revision_stale`,
`wait_already_resolved` and `wait_timed_out` are typed conflicts, and replaying
the same answer returns the original receipt. MCP: `swarms_wait_answer`.
`run.events()` reports `approval.parked`, `approval.decided`, `wait.opened` and
`wait.resolved` with the `approval_id` they concern.

`run.download(destination)` publishes a new directory only after every selected
resource passes its digest and size checks. It includes `MANIFEST.json`, the
final repository tree when present, `objectives.json`, answers, and artifacts at
`artifacts/<artifact-id>/<revision>/<logical-path>`. Multiple revisions of the
same artifact remain distinct. Existing destination paths are refused.

`run.labels.set(score=0.9, grader="local-v1")` updates labels using revision
checks. Labels are mutable annotations; they do not alter the immutable run
reference, objective evidence, or produced artifact identities. List filters
can select those annotations.

## Failure recovery

Handle `OutcomeUncertainError` by resolving `error.identity` with
`client.resolve_mutation(...)`. Keep the original key, body and revision; do
not create a new operation because the reply was lost.
`LifecycleWriteRefusedError` (a subclass) means the backend refused to write
the lifecycle of an Orchestra-owned run; the owner command may still have been
dispatched, so resolve the identity and read the run the same way. Poll the retained handle
when acceptance is known. A terminal result may still have pending settlement,
cleanup or unknown usage; inspect the typed fields separately.

For an iteration seeded from a prior output, include that exact artifact in
`Inputs.artifacts` and set `Inputs.derived_from` to the same reference. The new
run records both consumption and its derivation from that output.


## Experimental research method selection

The v0.24 source API carries an explicit method release and typed overrides:

```python
from synth_ai.sdk.swarms import ForkFlushMethod, ForkFlushOverrides

method = ForkFlushMethod(overrides=ForkFlushOverrides(branches=8))
```

Pass `method=method` to the launch call. The same `method` option works on sync/async `preflight` and `create`. Omission
preserves server defaults. Fork-and-Flush accepts 2–128 branches and an optional
ordered tuple of `ForkFlushRound` allocations, expressed in integer basis points
of the aggregate charge ceiling. Defaults belong to the selected release; an
override is never silently narrowed or replaced. Full resolved parameters and
program/profile/context-policy pins are available in method-aware readbacks.

This release adds selection contracts and a locally tested Orchestra control
component. Hosted workspace/session/evaluator adapters are not yet qualified.
The backend reports `research_method_unavailable` before paid execution rather
than running another method. The SDK refuses a successful readback that omits
an explicitly selected method or changes its overrides. No paper-result or
Craftax-improvement claim follows from the local fixtures.

## Rate capacity on preflight and create

Preflight and create successes carry the account's remaining Swarms capacity as
`x-ratelimit-*` headers. The SDK exposes them as typed snapshots,
`RateLimitSnapshot(dimension, limit, remaining, reset_at, burst)`: read
`preview.rate_limits` on the `LaunchPreflight` that `preflight` returns, and
`run.rate_limits` on the `SwarmRun` that `create` returns (sync and async).

`synth-ai swarms preflight --request launch.json` prints one line per dimension,
and the MCP `swarms_preflight` and `swarms_create` results include `rate_limits`.
An empty tuple means the server reported no capacity headers.
