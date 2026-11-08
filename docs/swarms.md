# Swarms

`SynthClient().swarms` exposes launch, run, get, wait, cancel, list, events,
iter_events, follow, outputs, output, output_content and families. `AsyncSynthClient` exposes the same methods with `await`;
its iter_events and follow methods are async iterators.

Every launch requires `budget`, a positive decimal USD string or `Decimal`,
for example `budget="0.10"`. It must be exactly representable in millionths of a
dollar. Floats and implicit budgets are rejected. Optional `wall_time_seconds`
bounds the run; omit it to use Core's policy. Core owns task counts and concurrency.
`maximum_tasks`, `maximum_parallel_tasks`, and the old top-level `providers` and
`models` arguments have been removed in launch contract 2 (SDK 0.28.0).

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

CLI examples:

```sh
synth swarms launch 'Compare retrieval methods' --budget 0.10
synth swarms run --task ./task.md --budget 0.10 --program swarms.general@1
synth swarms events RUN_ID --limit 20
synth swarms attach RUN_ID --last-event-id 12
synth billing balance
synth billing checkout --json '{"credit":{"amount":"20.00","currency":"USD"}}'
```

Task directories must contain one task.md, TASK.md or task.txt file, and any other
file in the directory is refused. Task text is the launch message. To send files with
a run, upload them with `synth swarms upload` or `upload_file` and pass the references
in `inputs`; see the README.

## Execution methods

Three methods are available. Status Quo is the default program and needs no setup.

| Method | CLI | Python `method=` |
| --- | --- | --- |
| Status Quo | `--method status-quo` | omit `method` |
| Fork & Flush | `--method fork-flush --method-task TASK` | `{"release": "research.fork_flush@1", "task": "TASK"}` |
| Forgetting | `--method forgetting --method-task TASK` | `{"release": "research.forgetting@1", "task": "TASK"}` |

Fork & Flush and Forgetting run on a published task that has an approval for the method,
which pins the evaluator, baseline, context and actor profile. A task with no approval is
refused. Pass `overrides` to change a method's published settings, for example
`{"release": "research.fork_flush@1", "task": "TASK", "overrides": {"branches": 2}}`, and read
`run.method` for the release and execution pins retained with the run.

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


Choose a provider with `profile_overrides={"launch": {"preset":
"hosted.openrouter_zdr_us", "effort": "low"}}`. Role-specific choices belong
under `profile_overrides["roles"]`. Hosted presets are
`hosted.openrouter_zdr_us` (restricted OpenRouter routing) and
`hosted.openai_direct` (the default). The direct preset makes no verified
retention or residency guarantee and is not a fallback for the restricted preset. `test.codex_pool` is restricted to local slots and
staging; production returns `provider_preset_not_allowed` (403).
All three currently serve `gpt-6-luna`.

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

`synth swarms watch RUN_ID --timeout 600` emits JSON events followed by the
finished run snapshot. By default `wait`, `run` and `watch` require execution
closed, accounting final and cleanup complete. Use `until="execution_closed"`
or `until="result_available"` (CLI `--until`) for an earlier milestone.
Python returns a failed snapshot normally; `raise_on_failure=True` raises
`RunFailedError` with that snapshot. `WaitTimeoutError` also retains the last
snapshot and run ID. Finished runs with no result raise `ResultUnavailableError`
when waiting for a result; imported history that cannot prove the milestone is
reported explicitly. A failed run exits 1; completed and cancelled runs exit 0.
The timeout bounds both streaming and final polling, and does not cancel the run.
