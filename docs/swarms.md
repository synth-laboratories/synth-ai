# Swarms

`SynthClient().swarms` exposes launch, run, get, wait, cancel, list, events,
iter_events, follow, outputs, output, output_content, families and profiles. `AsyncSynthClient` exposes the same methods with `await`;
its iter_events and follow methods are async iterators.

Every launch requires `budget`, a positive decimal USD string or `Decimal`,
for example `budget="0.10"`. It must be exactly representable in millionths of a
dollar. Floats and implicit budgets are rejected. Optional `wall_time_seconds`
bounds the run; omit it to use Core's policy. Core owns task counts and concurrency.
`maximum_tasks`, `maximum_parallel_tasks`, and the old top-level `providers` and
`models` arguments have been removed in launch contract 2 and remain absent in contract 3 (SDK 0.29.0).

`launch_request(LaunchRequest(...))` is the typed low-level entry point. Request,
guidance, routing and response types are available from `synth_ai.models`.
Guidance validates UTF-8 byte limits and text rules before sending.

Each mutation uses an Idempotency-Key. Supply `idempotency_key` to preserve the
identity across process retries. Bounded 429/503 retries keep the same key and
body. An unknown outcome does not mean the effect failed; inspect the run and
retry with the same identity. Replay lasts 24 hours after completed admission.

Pages expose `items`, `has_more` and an optional `next_cursor`. Pass the cursor
verbatim. `follow(run_id, last_event_id="12")` resumes SSE after that sequence.
Output bytes require the revision and SHA256 from the output reference.
`output_content` verifies the digest before returning bytes. `download_output`
streams into a temporary file, checks the digest and optional `expected_bytes`,
and replaces the destination only after verification. Its default `max_bytes`
is 64 MiB; set an explicit higher bound for larger outputs.

`StreamCheckpoint` retains the last committed SSE sequence and deduplicates
already-delivered events within an interrupted expanded sequence. Pass it as
`follow(run_id, checkpoint=checkpoint, timeout=600)` and persist its `to_dict()`
representation. Resume with `StreamCheckpoint.from_dict(...)`. A checkpoint
cannot be used for a different run. Named SSE errors raise typed API exceptions;
EOF alone does not prove execution closed.

If a file transfer or finalization is interrupted, `UploadTransferError` exposes
`upload_id` and `idempotency_key`. Resume with
`resume_upload(upload_id, filename, idempotency_key=key)`. This renews the original
grant (requiring a credential with read and write scopes) and verifies the local bytes against its retained size and digest; it does
not create a second upload.

CLI examples (all commands: [cli.md](cli.md)):

```sh
synth run 'Compare retrieval methods' --budget 0.10
synth run 'Follow the attached task' --file ./task.md --budget 0.10 --profile swarms-v2
synth logs RUN_ID -f
synth billing
```

`synth run --file PATH` uploads the file and attaches its reference; Python callers
use `upload_file` and pass the references in `inputs`; see the README.

## Named run profiles

Call `profiles()` or `profiles(research_task="TASK")` to discover published
profiles. The result includes `default_profile` and entries with `id`, `name`,
`description`, `revision`, `available`, optional `unavailable_reason`, and
supported routing. Availability is a read-only check, not a funding reservation.
It reflects caller permissions, served recipes and task approvals. Launch
revalidates those facts and checks the requested budget and resources.

| Profile | ID | Additional input |
| --- | --- | --- |
| Swarms v2 | `swarms-v2` | None |
| Fork & Flush | `fork-flush` | `research_task` |
| Forgetting | `forgetting` | `research_task` |

Omit `profile` for Core's default. An explicit profile is never silently
replaced. The public API does not accept `program`, `method`,
`profile_overrides`, or arbitrary actor/algorithm settings. A research profile
uses the task's approved immutable parameters, evaluator, baseline, context and
actor profile. Missing, revoked or foreign approvals refuse before funding.

```python
run = client.swarms.run(
    "Improve the policy", budget="2.00",
    profile="fork-flush", research_task="published-task",
)
print(run.profile.id, run.profile.revision, run.profile.digest)
```

`run.profile` retains the named revision, compiled program definition and exact
actor profile digests. Its digest binds the resolved recipe, including approved
research settings. Updating the catalog does not relabel historical runs;
runs predating named profiles omit this field. `run.method` continues to expose
retained research details as readback, not as a launch selector.

`run.selection` reports what a research run selected. Only `selected` has an incumbent
and a downloadable `selected-snapshot.tar.gz`, a gzipped tar whose SHA-256 equals
`selection.incumbent.artifact_digest`. `candidate_files` lists the files the score
covers. Other statuses are `not_applicable` (Status Quo), `pending`, `no_selection`
and `unavailable`; a `failed` run can still read `selected`, so check both. List pages
omit `selection`, so read a single run.

## Routing and owner resources

Omit `routing` to let Core assign the providers and model. Restrict either
catalog dimension independently with exactly one of `only` and `exclude`:

```python
from synth_ai import RoutingConstraints, Only, Exclude

run = client.swarms.run(
    "Compare retrieval methods", budget="2.00",
    routing=RoutingConstraints(
        providers=Only(only=["openrouter", "azure"]),
        models=Exclude(exclude=[]),
    ),
)
print(run.routing)
```

Lists are unordered hard constraints. Empty `only`, duplicates and unknown
catalog choices are invalid; `exclude=[]` has no effect. Every intermediary and
inference host must be allowed, so an OpenRouter route through Azure needs both
names in `only`. `routing_catalog()` lists the current supported routes. Today
those are direct OpenAI and OpenRouter through Azure, serving `gpt-6-luna`.
Infeasible choices return `routing_constraints_unsatisfiable` before funding.
Readback retains the normalized request, policy/catalog revisions and the actual
model, participants and profile digest for every actor, including workers.
Core preserves existing privacy restrictions, and validates fixed research or
explicit profile choices against constraints rather than replacing them.


Routing restrictions select a supported route within the chosen profile.
Research profiles cannot rewrite their approved actor profiles. Consult the
profile's `routing` catalog for compatible routes; incompatible restrictions
refuse before funding. Internal actor presets and effort overrides are not
public launch options.

Pass owner resource intents in `resources`, for example:

```python
resources = [
    {
        "plugin": "pool",
        "roles": ["worker"],
        "limits": {"spend_micros": 100000, "rollouts": 1},
        "selection": {"pool_ids": ["my-pool"]},
    }
]
```

Constrained launches refuse pool resources with `resource_routing_unverified`
because that owner cannot yet attest its inference route. Unconstrained resource
launches retain their existing behavior. The core verifies ownership and freezes grants. `role_limits` can narrow a
resource's run limits; it cannot widen them. The pinned launch contract has no top-level `preset` or `max_output_tokens`
field.

`synth wait RUN_ID` follows a run to its finished snapshot. By default `wait` and `run` require execution
closed, accounting final and cleanup complete. Use `until="execution_closed"`
or `until="result_available"` for an earlier milestone.
Python returns a failed snapshot normally; `raise_on_failure=True` raises
`RunFailedError` with that snapshot. `WaitTimeoutError` also retains the last
snapshot and run ID. Finished runs with no result raise `ResultUnavailableError`
when waiting for a result; imported history that cannot prove the milestone is
reported explicitly. CLI exit codes are listed in [cli.md](cli.md).
The timeout bounds both streaming and final polling, and does not cancel the run.
