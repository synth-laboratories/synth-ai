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
artifact = client.artifacts.upload(project_id, source, labels={"role": "input"})
inputs = Inputs(artifacts=(artifact.ref,))
guidance = (Guidance(id="comparison", kind="outcome", text="Compare both methods",
                     priority="must", evidence="Cite reproducible measurements"),)
```

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
