# Swarms

`SynthClient().swarms` exposes launch, run, get, wait, cancel, list, events,
iter_events, follow, outputs, output, output_content, result, families and profiles. `AsyncSynthClient` exposes the same methods with `await`;
its iter_events and follow methods are async iterators.

Every launch requires `budget`, a positive decimal USD string or `Decimal`,
for example `budget="0.10"`. It must be exactly representable in millionths of a
dollar. Floats and implicit budgets are rejected. Optional `wall_time_seconds`
bounds the run, from 1 to 86400 (24 hours; Core refuses a larger value with 422
`invalid_launch_request`, cause `limits.wall_time_seconds`); omit it to use Core's policy.
Optional `parallel_workers`, from 1 to 32 (default 1), lets the planner run up to
that many workers at once, one per task. The budget must fund the planner, the
reviewer and every parallel worker at once; a smaller budget is refused with 422
`launch_budget_below_actor_floor` (`LaunchBudgetBelowActorFloorError`, whose detail
states the minimum), and an out-of-range value with 422 `invalid_launch_request`,
cause `limits.parallel_workers` (the SDK rejects it with a `ValidationError` before
sending). Research profiles do not accept it. Core owns task counts.
`maximum_tasks`, `maximum_parallel_tasks`, and the old top-level `providers` and
`models` arguments have been removed in launch contract 2 and remain absent in contract 3 (SDK 0.29.0).

`launch_request(LaunchRequest(...))` is the typed low-level entry point. Request,
guidance, routing and response types are available from `synth_ai.models`.
Guidance validates UTF-8 byte limits and text rules before sending.

`launch(..., continue_from=parent_run_id)` forwards an opaque parent run ID.
The CLI accepts `synth run --continue-from RUN`; MCP accepts the same field in
the `swarms_run_launch` body. Current Core source refuses continuation until
the required retained input and planner producers are available. A missing or
foreign parent returns `ContinueFromNotFoundError`, a live parent returns
`ContinueFromNotTerminalError`, and unavailable inherited inputs return
`ContinueFromInputsUnavailableError`. A refusal creates no continuation run.

`RunSnapshot.api_version` is the semantic version of the responding Swarms API.
`RunSnapshot.variant` identifies the selected variant and its semantic version
pinned when the run was admitted. Both are optional for compatibility with
older servers and runs; `None` means the version was not reported. The variant
version is separate from the profile and native program revisions.

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

`result(run)` takes a `RunSnapshot` or a run ID and returns Core's `RunResult`
(`synth_ai.models`) from one `getRunResult` read, the same record the MCP tool
`swarms_run_result` returns: `run_id`, `status` (the terminal outcome),
`accounting` (the final ledger-backed `RunCharge`, or `None` without an
accounting reader), `run` (the snapshot), `state`, `answer` (the answer
text), `answer_output` (the `RunOutput` it came from), `objective_answer` (the
parsed `ObjectiveAnswer`, when that is the source) and `charge` (`RunCharge`:
`amount` and `final`; `None` when Core omits it). Core picks the most recently
published available `text/*` output or, without one, the most recent available
`application/json` objective answer (schema `synth.objective-answer.v1`; `answer`
is its `answer.text`). Without either, `answer`, `answer_output` and
`objective_answer` are `None`. An unfinished run raises `ResultUnavailableError`
with a fresh snapshot; `result` never waits, so call `run()` or `wait()` first.
An answer over 1 MiB (`result_answer_too_large`) or not UTF-8
(`result_answer_not_text`) raises `ConflictError`; use `download_output` for
those. List every output with `outputs()` / `iter_outputs()`.

```python
run = client.swarms.run("Summarize the findings", budget="2.00")
result = client.swarms.result(run)
print(result.answer)
```

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
Omit `profile` for Core's default; an explicit profile is never silently
replaced. `run.profile` retains the named revision, compiled program definition
and exact actor profile digests; catalog updates never relabel historical runs.

## Optimizer topologies

| Public name | Machine ID | Treatment |
| --- | --- | --- |
| Swarms | `swarms_v2` | Default. The only topology served in this release. |
| Fork-and-Flush | `fork_flush` | Experimental. Opt-in only, never the default; not served yet, so a pin is refused with `optimizer_topology_unavailable`. |

`optimizer="auto"` (or omitting it) lets Core resolve the default, Swarms.
`client.swarms.families()`, `synth families`, the MCP catalog and the Console
read the same catalog. `synth_ai.topology_name(entry)` and `synth families`
print the public name with its treatment, for example `Swarms (default)` and
`Fork-and-Flush (experimental, unavailable)`. An experimental topology is never
shown as the default, whatever the catalog's `auto` flag says.

### Swarms controls

A `swarms_v2` pin accepts two optional parameters (contract 1.36.0):

| Parameter | Range | When omitted |
| --- | --- | --- |
| `max_tasks` | 1..4096, at least `limits.parallel_workers` | the larger of 4 and `parallel_workers` |
| `max_repairs_per_task` | 0..32 | 2 |

```python
from synth_ai import OptimizerParams, OptimizerPin

run = client.swarms.run(
    "Improve the policy", budget="2.00",
    optimizer=OptimizerPin(
        topology="swarms_v2", version=2,
        params=OptimizerParams(max_tasks=8, max_repairs_per_task=3),
    ),
)
print(run.optimizer.params)  # only the parameters you supplied
```

```bash
synth run "Improve the policy" --budget 2.00 \
  --optimizer swarms_v2@2 --max-tasks 8 --max-repairs-per-task 3
```

The MCP `swarms_run_launch` tool takes the same `optimizer.params` object.
Spend, time, token and concurrency limits stay authoritative; these controls
never raise a grant. Omitting both leaves the launch exactly as before. Core
refuses before funding with 422:

- `optimizer_params_unsupported`: an unknown parameter, or parameters on `auto`
  or another topology (the CLI refuses these locally);
- `optimizer_param_invalid`: a value that is not an integer;
- `optimizer_param_out_of_range`: outside the range above;
- `optimizer_param_conflict`: `max_tasks` below `limits.parallel_workers`.

The run's `optimizer` keeps its admitted identity, resolution reason, definition
digest and supplied `params`. Catalog changes cannot relabel historical runs.

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

## Directory uploads

```python
files = await client.swarms.upload_directory("./task", idempotency_key="task-upload-1")
inputs = {
    "files": [
        {
            "file_id": file.file_id,
            "revision_id": file.revision_id,
            "digest_sha256": file.digest_sha256,
        }
        for file in files
    ]
}
run = await client.swarms.launch("Solve the uploaded task", budget="4.50", inputs=inputs)
```

The synchronous client exposes the same method. File names are relative to the
chosen directory; binary bytes and dotfiles are included. Each file uses the
existing prepare, PUT and finalize protocol. An empty directory returns an empty
list. Empty directories themselves have no uploaded file representation.

Before any API request, the helper freezes the complete directory into a private
disk spool using bounded chunks. Changes to the originals after that snapshot do
not change submitted bytes. Symlink roots and entries, special files, unsupported
names, and files that change while being copied are refused. Descriptor-relative traversal
requires POSIX; unsupported platforms refuse before uploading.

Bounds come from Core's pinned `contracts/file-limits.json`: 16 MiB per file,
64 MiB total, 1,024-byte relative names and 255-byte path segments. The directory
helper also bounds traversal to 10,000 entries, counting directories as well as
files, so empty directory trees cannot make scanning unbounded. It does not apply
ignore rules. The spool is closed on success, failure or cancellation.

Use the same batch idempotency key and unchanged files to recover a partial
upload. Per-file keys are derived from that key and the relative name; changed
content retains the same key and the server can refuse a conflicting retry.
Uploads are sequential, and already committed files remain in custody if a later
file fails. `synth_ai.errors.DirectoryUploadError` retains `completed_files`, the
failed `name`, the batch `idempotency_key`, and `file_idempotency_key`; its cause
retains any upload-transfer or API refusal. Cancellation during an API transfer raises `DirectoryUploadCancelledError`,
which remains an `asyncio.CancelledError` and retains the same custody fields.
Cancellation before the snapshot completes makes no API requests. Retain the
batch key before starting.

The file-limit projection and provenance are checked by
`python scripts/gen_file_limits.py --check` in CI. This helper adds no public API
operation or hosted MCP access to local files.


For local stdio MCP, explicitly grant one directory at server startup:

```sh
synth-ai-mcp --upload-root /absolute/path/to/bundle
```

Only this invocation advertises `swarms_upload_directory`; its only argument is
`idempotency_key`. The server pins the root through descriptor-relative traversal
without following symlinks, and uploads exactly that root. No tool parameter can
select another local path. Success returns the ordered file records and batch key;
partial failure or cancellation returns an MCP error containing completed files,
the current file name and both retry keys. Hosted MCP and stdio without the grant
retain the generated public catalog with no local filesystem authority.
