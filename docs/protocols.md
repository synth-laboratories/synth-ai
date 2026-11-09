# Typing and data protocol standards

Core's pinned OpenAPI remains the authority. Update it before changing a wire
shape; regenerate models, operations, MCP schemas, and synchronous delegates.
There is no second handwritten set of request models.

Python generation keeps the transitive schema closure of implemented SDK
request/response operations, handwritten runtime imports, and public exports.
Generated files and the model registry cannot keep themselves alive. Unused
schemas remain in the pinned Core contract where other consumers may need them.
A regression check independently verifies that emitted Python has no orphan
definitions; both generator drift checks remain required.

## Python records and boundaries

Public methods have explicit typed parameters. Use `RunInputs`/`InputFile`,
`OwnerResourceRequest`/`PoolSelection`, `RoutingConstraints`, `RunGuidance`, and
`Money` instead of dictionaries describing those domain objects. The generated
schema enums are `StrEnum` classes: values serialize as their existing strings.
For example, `RunGuidanceKind.QUESTION` and `RunGuidancePriority.MUST`.
Raw JSON remains accepted by `LaunchRequest.model_validate` and the CLI JSON
boundary, where it undergoes the same validation.

Generated records are frozen: field reassignment is refused. Their nested list
and map values are ordinary Python collections, so freezing is shallow. Treat
records as values; construct and validate a new record to change one. Pydantic's
`model_construct` and `model_copy(update=...)` bypass validation and are not input
validation APIs.

Validation is strict: booleans, numeric strings and fractional numbers cannot
stand in for integer limits. Invalid input fails before sending a request. A
malformed successful response raises `ResponseContractError` and retains request
identity and mutation uncertainty. Missing optional values stay distinguishable
from wire fields explicitly set to null.

Arbitrary JSON is limited to fields the upstream contract intentionally leaves
open, such as metadata and extension payloads. Those use recursive `JsonValue`,
not `Any`, and reject non-JSON objects. Response extensions stay forward compatible.
Closed schemas reject unknown properties.

## Lifecycle and semantic events

The patterns are informed by [OpenResponses](https://www.openresponses.org/specification):
explicit records, finite known states, semantic events, and structured errors.
Synth remains its own API; this does not claim OpenResponses wire compatibility.

`CompletionMilestone.FINISHED` requires `execution_closed`, `accounting_final`,
and `cleanup_complete`. `RunSnapshot.finished` and `.reached(milestone)` share
that definition with observation. `.known_state` returns a `RunState` enum or `None`;
`.state` always retains the original server string. Unknown values never become
successful completion by default.

Event identity (`event_id`), run identity (`run_id`), sequence, and kind remain
separate. Core's `x-synth-event-kind` annotations tie a non-null objective payload
to its envelope kind. Wrong-kind payloads are refused instead of being selected
by their field shape. Null historical payloads remain readable.

`StreamCheckpoint.to_data()` returns `StreamCheckpointData`. Its JSON carries
`schema_version: 1`; restoration rejects unsupported versions and still reads the
original unversioned v1 shape. A cursor commits only at the final SSE frame of
an expanded sequence, while pending event IDs prevent duplicate delivery.

## Errors and migration

Catch `SwarmsAPIError` for server refusals and inspect its typed `problem`, status,
code, request ID and idempotency key. `ErrorKind` categorizes local transport,
contract, observation, integrity and lifecycle failures; local error classes
provide stable codes such as `wait_timeout` and `upload_incomplete`. Recovery
snapshots and checkpoints remain on the relevant exceptions.

Use attributes on `auth.status()` and stored credential records. Secrets are
excluded from record reprs. Use `model_dump(mode="json")` or `model_dump_json()`
when an actual JSON representation is needed.

## Checks

Run `uv sync --locked --extra mcp`, `uv run ty check`, `uv run ruff check synth_ai
scripts tests typing_tests`, and `uv run ruff format --check synth_ai scripts tests typing_tests`. CI also
checks both generators, regression tests, wheel metadata and an isolated wheel
installation. The wheel ships its PEP 561 marker.

[ty configuration](https://docs.astral.sh/ty/configuration/) is kept in
`pyproject.toml`; all enabled diagnostics fail the check. Ruff checks errors,
imports, bug patterns, modern syntax and Ruff-specific correctness rules. A
dedicated standards lint prevents explicit `Any` annotations and unchecked public
keyword bags from returning.


## Standards lint

Run `uv run python scripts/lint_standards.py` from the repository root. To inspect
specific files, pass their paths as arguments. The check parses source without
importing it, includes generated SDK files, and exits nonzero with
`path:line:column: code message` diagnostics. Invalid or missing inputs fail closed.
Both CI and package publication run it.

| Rule | Enforced constraint |
| --- | --- |
| SYN001 | No `Any` imports or type uses, including qualified aliases and forward annotations. |
| SYN002 | Public functions and constructors name their keyword parameters; no `**kwargs` bags. |
| SYN003 | Public boundary collections declare element types; model fields and public signatures do not use `dict[str, object]` or equivalent opaque maps. Use a named record or `JsonValue` at a real JSON boundary. |
| SYN004 | Pydantic records enable `strict=True` and `frozen=True`. Direct BaseModel/WireModel subclasses and local inheritance are checked, including explicit weakening. |
| SYN005 | No `model_construct`/legacy `construct`, copy-with-update validation bypass, or explicitly non-strict/dynamic strict validation. |

Ruff's `ANN` rules require annotations throughout production SDK code; `PGH`
rejects blanket `noqa` and type-ignore comments; `TRY002` rejects generic
`Exception` raises. Tooling scripts also require annotations; tests retain their annotation exemption.
The custom rules do not accept inline suppressions. `JsonValue`, explicitly
mutable lifecycle helpers, and private raw-input decoders remain supported.

This is a syntax-level guard, not a replacement for `ty` or protocol tests.
SYN004 understands the SDK's WireModel base and inheritance within a file; it
is not a general cross-module Python type resolver. SYN005 reserves Pydantic's
construction and copy-with-update method spellings within the SDK. Generated
contract drift and behavioral tests still enforce event kinds, valid states,
strict scalar parsing, and completion semantics.


## Type-hint verification

`uv run ty check` covers `synth_ai/`, every file in `scripts/`, and
`typing_tests/accepted.py`. Tooling uses typed JSON accessors for the open schema
vocabulary, typed mutable smoke-observation records, and explicit typed transport
parameters. These are checked by ty and by Ruff's annotation rules.

`uv run python scripts/check_type_contracts.py` checks the SDK from a consumer's
perspective. Accepted sync/async examples use `assert_type` to check return values,
stream items, records, and enums. Rejected examples must produce their exact
expected diagnostic codes and locations: missing budgets, wrong input types,
removed override fields, and wrong return types. Unexpected, duplicate or missing
diagnostics fail the check; these comments never suppress errors. The invalid
fixture is deliberately outside the default passing `ty check` source list.

Both checks run in CI and before package publication. They are offline checks;
none of the consumer examples or live smoke operations are executed.

## Notification-driven observation

`wait()` opens the durable event feed before its first authoritative `getRun`.
Events wake the observer; only a fresh snapshot can prove a milestone. Idle
connections reconcile at `max(30 seconds, poll_interval)` to cover missing
notifications. Heartbeats do not trigger reads. One monotonic `timeout` covers
subscription, reads, reconnect delays and fallback, and cancels observation only.

Reconnect attempts are bounded by `RetryPolicy.attempts`; delays respect both its
backoff and `poll_interval`. Unexpected EOF or a transport disconnect does not
prove completion. Once the server closes a feed for an execution-closed run,
accounting and cleanup continue through explicit polling at
`max(1 second, poll_interval)`. The same polling fallback applies when opening a
feed returns a valid 406, 415 or 501 Problem. Authentication, cursor expiry,
404, malformed feeds, and other server refusals propagate; they never silently
switch to polling. No snapshot or history is discarded to recover an expired
cursor. Exhausting reconnect attempts raises `TransportFailureError` with the
checkpoint. `WaitTimeoutError` retains the latest snapshot and run identity.

`poll_interval` remains accepted by sync and async methods, but now controls the
minimum reconnect/fallback cadence, not continuous polling while a feed is open.
Use `contextlib.aclosing` if stopping `follow()` before exhaustion:

```python
from contextlib import aclosing
from synth_ai import StreamCheckpoint

checkpoint = StreamCheckpoint()
async with aclosing(client.swarms.follow(run_id, checkpoint=checkpoint)) as events:
    async for event in events:
        await consume(event)
        save_checkpoint(checkpoint.to_dict())
```

A checkpoint means **received**, not **durably applied**. It advances before an
event is yielded, and its cursor commits on the final frame of an expanded
sequence. Pending event IDs suppress replay of a partially received expansion;
a replayed final frame with the currently committed cursor is also suppressed.
The SDK preserves owner-feed order within one run; it neither sorts event IDs
nor promises global ordering. Unknown semantic kinds remain readable. It does
not offer exactly-once application processing or an acknowledgement protocol.
Persist the checkpoint only after applying the event if recovery must replay
unprocessed deliveries; persist application state and checkpoint atomically if
that is required by the application. Do not persist the live mutable checkpoint
concurrently with unfinished processing. A crash after application but before
checkpoint persistence can replay an event, so application handlers must tolerate
that duplicate. Retention expiry remains an explicit owner error.

Each frame/line is bounded to 1 MiB, a checkpoint to 4096 pending IDs and each
identity to 1024 characters. Invalid/oversized frames do not advance a checkpoint.
Checkpoint construction and serialization copy pending collections so mutation
of caller-owned sets or serialized records cannot alter a live checkpoint.

## Recovery and internal ownership

The operation executor serializes a validated request once before awaiting I/O.
Automatic retries retain identical payload bytes, request ID and idempotency key.
A mutation retries 429/503 only when the operation permits it and a valid Problem
with matching HTTP status explicitly says `not_accepted`. Malformed error bodies
and ambiguous outcomes are not replayed. Connection/pool failures before send
report `not_accepted`; write/read failures and malformed successful mutation
responses report `unknown`, including operations without durable keys. Retain the
original inputs and identity for caller-directed recovery; generating a fresh key
is a new operation, not recovery. Credentials and entire request payloads are not
attached to exceptions as recovery records.

The public facade composes focused internal modules: `_requests` constructs
launch requests; `_observer` owns milestone observation through a typed Protocol;
`_operations` owns invocation/identity/decoding; `_retry` owns retry policy;
`_transfers` owns artifact integrity and atomic download completion; `_stream`
and `_event_stream` own checkpoint/framing and typed event decoding. `_transport`
owns the shared HTTP client and stream connection lifetime. Injected clients
remain caller-owned. Sync delegates remain generated from the async facade.

## Contract and installed cohort check

`uv run python scripts/check_cohort.py` verifies the local source pin and hash.
Before installing against Core, run:

```sh
uv run python scripts/check_cohort.py --core-checkout /path/to/core --core-revision FULL_COMMIT
```

This verifies the pinned source bytes against Git and separately compares the
frozen candidate's exact contract bytes. A mismatch exits nonzero; it does not
rewrite generated models or assume that an older/newer commit implies
compatibility. Run both generators in `--check` mode as well. This offline
verification does not replace installed SDK/MCP and live lifecycle acceptance.

## Profiles and authority (0.31)

Profile IDs remain extensible strings. Core publishes each known profile's
`requires_research_task` and default in its contract; generation supplies the
SDK's local construction rules. Explicit `profiles()` discovery refreshes these
rules for new catalog entries without adding a network request to every launch.
Availability is advisory at discovery time: launch admission checks it again.
The SDK never substitutes a different profile or relaxes an only/exclude set.
An unknown ID remains representable and is sent to Core for a named decision.

Core's policy now authorizes a 32768-byte UTF-8 message for this coordinated
cohort. The generated SDK, CLI and MCP use that exact byte boundary. Install the
qualified Core cohort first, then the SDK. This does not change guidance text
limits or permit a newer SDK to assume an older server accepts larger messages.

Resource authority narrows from organization/service ceilings to a run's
explicit authorization to child reservations. Child allocation cannot widen a
parent or multiply its money. Objectives describe desired outcomes; budgets,
execution deadlines and concurrency bound authorized work. `timeout` on SDK
observation only bounds client waiting; it is neither an execution deadline nor
a billing ceiling. Core and Gateway, not SDK estimates, enforce admission and
financial constraints. Requested routing, admitted assignments and observed
providers remain separately readable in generated records.

## Bounded transfers and iteration

`upload_file` and `resume_upload` hash and stream regular files in 64 KiB chunks.
Each client permits at most four outstanding disk operations; cancellation joins
an outstanding operation before closing its file. Size remains bounded to 32 MiB.
Device/inode, size, modification/change timestamps and a streaming digest must
match the prepared source. Replacing or editing the file is refused before
finalization. Prepare/finalize and retry requests retain the original upload key;
an interrupted transfer can be reconciled with `resume_upload` and that key.
Committed manifest size/digest must match the prepared source. Injected HTTP
clients stay caller-owned; closing the SDK is idempotent and prevents new calls.
Signed storage requests do not inherit API credentials or cookies. API, stream
and download calls disable redirects even on an injected redirect-enabled
client; unexpected redirect statuses remain available in the typed error.
Download writes and the final durability flush use the same bounded I/O owner.

`iter_runs`, `iter_tasks`, `iter_outputs`, `iter_files`, `iter_events`,
`account.api_keys.iter_keys`, `account.iter_usage_records`,
`billing.iter_history` and `billing.iter_statements` are lazy async/sync
iterators. They preserve filters, fetch only on demand, and stop on an empty
final page. Invalid or repeated cursors raise `PaginationError` with the cursor;
no background prefetch is performed. Each traversal is capped at 10000 pages
and cursor identities at 1024 characters to bound cycle-detection memory. Use
the retained page-returning methods and explicit cursor continuation for larger
traversals. Closing/cancelling an iterator does not mutate the resource.

## Optional diagnostics

Pass `diagnostic_hook` to either client to receive immutable
`OperationDiagnostic` values from `synth_ai.diagnostics`. Operation, phase,
attempt count and elapsed seconds describe generated API invocations; correlation
values are SHA256 hashes of request/idempotency IDs. There are no prompts,
response bodies, tokens, signed URLs or payment data in these records. No hook
means no logging. A hook exception cannot change the operation's result. This
hook is synchronous and should be cheap; it is not an exporter thread or a
billing estimator. Transfer bytes and stream metrics remain independently
measured in qualification rather than represented as API billing facts.

## Open-data audit

The remaining JSON boundaries follow Core's declared schemas rather than
inventing SDK-only records:

| Boundary | Why JSON remains | Validation/ownership |
| --- | --- | --- |
| `RunSnapshot.method`, `guidance_coverage`, `problems`, `failure` | Core intentionally publishes open historical projections | Strict recursive JSON; known event payloads use named models |
| `RunEvent.usage` | Extensible owner usage facet | Event kind/payload pairing is validated separately |
| `RunConfiguration.configuration` | Published program-specific configuration | Core owns its schema and historical versions |
| Metadata and compute extensions | Genuine extension fields | Generated declared properties retain strict types |
| Error extensions | Owner diagnosis beyond standard Problem fields | Stable Problem fields and outcome remain typed |

Generated records remain shallowly frozen for compatibility. Operation identity
uses serialized request snapshots; checkpoint constructors and serialization
copy mutable sets/lists. There is no blanket tuple/map conversion and no
`object` replacement for schema-defined domain data. Architecture regression
checks prevent execution/observation/transfer strategies from importing public
facades; generation and semantic fault tests retain the existing standards bar.

See [executable recipes](../examples/README.md) for launch, profiles, routing,
result/finality, timeout/cancel, restartable events, upload recovery, verified
artifacts and usage readback.
