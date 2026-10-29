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
MCP expose the same operations as `swarms_approvals` and
`swarms_approval_decide`.

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
not create a new operation because the reply was lost. Poll the retained handle
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
