# Changelog

## 0.36.0

- Adopt Core contract 1.12.0 and expose assignment limits, parked actor state and revision history through the limits projection.

## 0.35.0

- Add revision-fenced `update_limits` and `synth limits set`, with typed revision, entitlement and exhausted-limit errors (Core contract 1.11.0).
- Launch accepts `maximum_tokens` and `pause_retention_seconds`; CLI flags are `--tokens` and `--pause-retention`. Run time supports up to 86400 unpaused seconds; pause retention defaults to 86400 seconds.
- Launch and budget raises now check plan daily/weekly spend entitlements, including open funding holds, as well as wallet credit. Existing customers may receive `spend_entitlement_exceeded` at launch.

## 0.34.0

- `wait` stops on a paused run with `RunPausedError` carrying its pause record. CLI pause has a distinct exit status (Core contract 1.10.0).

- Add `swarms.limits(run_id)` and the `swarms_run_limits` MCP tool, using Core contract 1.9.0. Reads preserve unknown quantities and expose owner observation freshness and funding separately from usage.

## 0.33.0

Contract: the SDK pins Core public contract 1.9.0 (Core `cf85aa84`, SHA-256
`96ebcdea…d2a3b3`), which adds `GET /v1/runs/{run_id}/result` (`getRunResult`)
and the MCP tool `swarms_run_result`, so an MCP client reads a run's answer in
one call.

Breaking (no support window): `swarms.result(run)` is now that one read and
returns Core's `RunResult`: `run`, `state`, `answer`, `answer_output`,
`objective_answer`, `charge`. `outputs` and `content` and the `max_outputs`,
`max_bytes` and `max_json_outputs` arguments are gone; list outputs with
`outputs()` / `iter_outputs()` and fetch bytes with `output_content()` or
`download_output()`. Core chooses the answer in publication order; the old
client-side choice used listing order, which Core sorts by output id, so with
several text outputs it could return an older one. `ObjectiveAnswerBody` and
`ObjectiveAnswerSource` are replaced by the generated `ObjectiveAnswerAnswer`
(its `sources` are plain objects). `synth result RUN --json` prints the
`RunResult` as Core returned it.

`swarms.wait` outlasts a Core deploy: 502, 503 and 504 on its snapshot reads or
event feed are retried with backoff until the caller's `timeout`, then the
original error is raised. A Problem marked `retryable: false` still ends the wait
at once. Launch and other mutations are never retried.

`synth run` validates the budget, limits, labels and profile before uploading
any `--file`, so an out-of-range `--parallel-workers` or `--wall` uploads nothing.

## 0.32.3

Contract: the SDK pins Core public contract 1.8.0 (Core `f2f9ae9f`, SHA-256
`bebc837c…42a32e`). Launch accepts `limits.parallel_workers`, an integer from 1
to 10 (default 1): the planner may run up to that many workers at once, one per
task. `swarms.launch` and `swarms.run` (async and sync) take
`parallel_workers=`, `synth run` takes `--parallel-workers N`, and the MCP launch
tool schema carries the field. The generated `Limits` model enforces the range,
so the SDK rejects an out-of-range value with a `ValidationError` before
sending; Core refuses it with 422 `invalid_launch_request`, cause
`limits.parallel_workers`. The budget must fund the planner, the reviewer and
every parallel worker at once; a smaller budget is refused with 422
`launch_budget_below_actor_floor`, now mapped to
`LaunchBudgetBelowActorFloorError` (an `InvalidRequestError`). Research profiles
do not accept `parallel_workers`.

## 0.32.2

Contract: the SDK pins Core public contract 1.7.0 (Core `3a62a087`, SHA-256
`7f60260f…1dcab`). The run snapshot gains a read-only `charge` (`RunCharge`:
`amount` Money, `final` bool) on launch, single-run and wait reads (list pages
omit it), and `started_at` is now populated for new runs. `RunResult.charge`
exposes it; `synth result RUN` and `synth run --wait` print one line on stderr,
`charge  AMOUNT USD (final|not final)` (stdout stays the answer alone), and
`--json` includes `charge`.

Contract 1.6.0 (Core `867e0819`) bounded `limits.wall_time_seconds` is now at most 3600 (one hour),
down from 86400: Agents refuses actor deadlines more than an hour away, so a
longer run used to be admitted and then fail at start with `deadline_invalid`.
Core now refuses it at launch with 422 `invalid_launch_request`, cause
`limits.wall_time_seconds`. The generated `Limits` model enforces the bound, so the SDK rejects a larger
value with a `ValidationError` before sending. `synth run --wall` help and
`docs/swarms.md` state the bound.

SDK (SYN-5583): `swarms.result()` reads an objective answer. A run whose only
output is an `application/json` `synth.objective-answer.v1` document (empty
filename) used to return `answer=None`. Without a text output, `result()` now
uses the most recent objective answer: `answer` is its `answer.text`, `content`
the raw JSON, `objective_answer` the typed `ObjectiveAnswer`, and the source
output moves to the new `answer_output` field (`answer` is now the answer text,
not the output record). `synth result RUN` and `synth run --wait` print it.

SDK (SYN-5583): unsupported upload formats are one typed refusal,
`UploadFormatUnsupportedError` (`InvalidRequestError`, code
`upload_format_unsupported`, 422, `supported_formats`). It replaces the plain
`ValueError` from `upload_file`, the pydantic `ValidationError` from
`upload_text` (and `prepare_upload`), and the stdio MCP schema text; the stdio
MCP server now returns an `isError` Problem for it, and `tool_arguments_invalid`
for other invalid input. The server's own 422 maps to the same class. The CLI
prints it like any refusal and exits 4.

## 0.32.1

Contract: the SDK pins Core public contract 1.5.0 (Core `12ba622f`). Core now
refuses `synth-ai` clients older than its minimum version with HTTP 426 and code
`client_upgrade_required`; the body carries `minimum_version` and `client_version`.

SDK: `UpgradeRequiredError` (a `SwarmsAPIError`, exported from `synth_ai`) is raised
for that refusal and is never retried. The CLI prints the refusal and one line
ending `run: pip install -U synth-ai`, and exits 4.

## 0.32.0

Contract: the SDK pins Core public contract 1.4.0 (Core `817056b9`, SHA-256
`4c3e45cc…f365`). The generated `Problem` model has `retryable` and
`last_event_id`; stream reconnects read the typed `retryable` member and keep the
`event_owner_unavailable`/503 fallback for servers that omit it. The new read-only
`GET /v1/runs/{run_id}/wait` (`waitRun`, `RunWait`) backs the MCP
`swarms_run_wait` tool; the typed client keeps `wait()`, which already follows
the event stream to completion, and adds no method for it.

MCP: the stdio server adds `swarms_run_pause`, `swarms_run_resume`,
`swarms_run_tasks`, `usage_for_run` and `swarms_run_wait`, with the same names,
scopes and argument schemas as Core's hosted catalog (20 tools; the catalog file
is byte-identical to Core's). The generator honours `x-synth-mcp-required`.

CLI: `synth result RUN` prints the answer text, or the `RunResult` as JSON.
`synth run --wait` ends with the answer text in human mode.

Tests: `tests/test_journey.py` drives one customer journey (profiles, upload,
launch, watch through a retryable stream drop, pause/resume, outputs, answer,
usage) through the SDK, the CLI and the MCP tools, and fails naming any step a
surface lacks.

Docs: `docs/quickstart.md` (one per surface), an error guide in
`docs/errors.md`, and the support policy: only the latest release is supported.

Breaking: the `synth` CLI is replaced by task verbs (SYN-5564), with no
compatibility period. Commands: `login [--no-browser]`, `whoami`, `logout`,
`profiles`, `upload`, `run`, `status`, `wait`, `logs [-f]`, `tasks`, `ls`, `pause`,
`resume`, `cancel`, `outputs`, `download`, `report`, `result`, `usage`, `billing`,
`limits`.
Output is aligned text; `--json` on every command prints the SDK record as one
JSON document. Exit codes: 0 success, 1 client or usage error, 2 wait timeout,
3 run failed or cancelled, 4 server refusal. `logs -f` resumes from a 0600
checkpoint under `~/.config/synth/streams/` (`SYNTH_STATE_DIR`). See
`docs/cli.md`. `auth.login(open_browser=False, redirect_source=...)` accepts a
pasted loopback redirect.

| Removed | Replacement |
| --- | --- |
| `synth swarms status` | `synth status RUN` |
| `synth swarms report` | `synth report RUN` |
| `synth swarms sandboxes` | removed, no replacement |
| `synth swarms list` | `synth ls` |
| `synth swarms events` | `synth logs RUN` |
| `synth swarms cancel` | `synth cancel RUN` |
| `synth swarms pause` | `synth pause RUN` |
| `synth swarms resume` | `synth resume RUN` |
| `synth swarms program` | removed, no replacement |
| `synth swarms configuration` | removed, no replacement |
| `synth swarms tasks` | `synth tasks RUN` |
| `synth swarms outputs` | `synth outputs RUN` |
| `synth swarms output` | `synth outputs RUN` |
| `synth swarms output-content` | `synth download RUN OUTPUT` |
| `synth swarms families` | removed, no replacement |
| `synth swarms profiles` | `synth profiles` |
| `synth swarms upload-text` | `synth upload PATH` |
| `synth swarms uploads-prepare` | `synth upload PATH` |
| `synth swarms uploads-get` | removed, no replacement |
| `synth swarms uploads-finalize` | `synth upload PATH` |
| `synth swarms files` | removed, no replacement |
| `synth swarms file` | removed, no replacement |
| `synth swarms labels` | `synth run --label k=v (at launch)` |
| `synth swarms download` | `synth download RUN [OUTPUT] [-o DIR]` |
| `synth swarms upload` | `synth upload PATH...` |
| `synth swarms launch` | `synth run "goal" --budget B` |
| `synth swarms run` | `synth run "goal" --budget B --wait` |
| `synth swarms wait` | `synth wait RUN` |
| `synth swarms follow` | `synth logs RUN -f` |
| `synth swarms attach` | `synth logs RUN -f` |
| `synth swarms watch` | `synth logs RUN -f, then synth wait RUN` |
| `synth account get` | `synth whoami` |
| `synth account accept-terms` | removed, no replacement |
| `synth api-keys list` | removed, no replacement |
| `synth api-keys mint` | removed, no replacement |
| `synth api-keys update-grants` | removed, no replacement |
| `synth api-keys rotate` | removed, no replacement |
| `synth api-keys revoke` | removed, no replacement |
| `synth billing balance` | `synth billing` |
| `synth billing history` | removed, no replacement |
| `synth billing statements` | removed, no replacement |
| `synth billing plans` | removed, no replacement |
| `synth billing checkout` | removed, no replacement |
| `synth billing checkout-status` | removed, no replacement |
| `synth usage summary` | `synth usage` |
| `synth usage records` | `synth usage RUN` |
| `synth usage export` | removed, no replacement |
| `synth usage export-status` | removed, no replacement |
| `synth limits list` | `synth limits` |
| `synth auth status` | `synth whoami` |
| `--json REQUEST bodies` | removed; typed flags only |
| `--task PATH (launch/run)` | `synth run --file PATH` |
| `--until, --providers-*/--models-*, --idempotency-key, --last-event-id` | removed, no replacement (logs -f keeps its own checkpoint) |

Breaking: `message` is required on run launch. Core public contract 1.3.0
(synth-backend `5403b03b1c30d59ab2aeb38ca28260827412f1c7`, SHA-256
`94d95075bf0815e6d70363dc53b01c348db5bf4f75c90416f97a05f779f0a027`) makes
`LaunchRequest.message` a required string (32 KiB UTF-8, not blank, no NUL);
Core refuses a launch without one with 422 `invalid_launch_request`, cause
`message`, before any run or hold. `swarms.launch()`/`run()` take `message: str`
with no default and raise `ValueError` locally for a missing or blank message;
the generated `LaunchRequest` and MCP `swarms_run_launch` schema require it; the
CLI requires the goal.

`wait()` and `follow()` survive a dropped event stream. A feed refusal returned
on open or in-stream now reconnects from the checkpoint when its Problem carries
`retryable: true`, or, for servers that predate that member, when it is
`event_owner_unavailable` with status 503 (seen during server deploys). Reconnects
share the `RetryPolicy.attempts` budget and the single `timeout`, never
re-deliver an event, and `wait()` rereads the run before declaring a milestone.
All other refusals still propagate at once. Exhausted reconnects raise
`TransportFailureError` (`event_stream_interrupted`) whose message states the
attempts made, with `connection_attempts`, `checkpoint` and the last `snapshot`.

New `swarms.result(run)` (async and sync) returns a `RunResult` with the finished
run's snapshot, all outputs and the UTF-8 text of its last `text/*` output
(`content` is `None` when there is none). An unfinished run raises
`ResultUnavailableError`; outputs (1000) and answer bytes (1 MiB) are bounded.

## 0.31.0 (released 2026-11-09)

Aligns with the integrated Core public contract and its 32 KiB UTF-8 message
limit. Profile eligibility comes from generated metadata and explicit discovery.
Adds typed lazy iterators, payload-free diagnostic hooks, bounded streaming
uploads with source identity/digest verification, and runnable fixture-tested
recipes. Closing either client prevents new requests and preserves injected
HTTP ownership.

## 0.30.0 (not published; changes ship in 0.31.0)

`wait()` now subscribes before reading run state, uses durable events as wakeups,
and periodically reconciles dropped notifications. EOF, authentication failures,
reconnect limits and the final accounting/cleanup polling phase are explicit.
Observation cancellation closes its stream without cancelling execution.

Request construction, operation execution, retries and artifact transfers have
focused internal owners. Retry bodies are immutable snapshots; malformed mutation
responses retain uncertainty and identity. Stream checkpoints explicitly mean
receipt, isolate mutable collections, and suppress replayed committed frames.
Checkpoint restoration accepts read-only mappings as well as dictionaries.

A cohort checker verifies source provenance and exact Core contract bytes.
Isolated wheel qualification now exercises notification-based SDK observation and
the optional MCP server over stdio, in addition to imports and CLI help.

The release smoke now checks a bounded two-run Swarms journey instead of an
unreachable full-API gate. Its v4 receipt reports `smoke_complete`, verified output,
final accounting and cleanup; launch and cancellation identities survive resume.
Unrelated account, API-key and checkout mutations are removed from the smoke.
Historical receipts remain intact and cannot be resumed with the v4 runner.

Generated Python types now include only supported SDK operation models, public
exports and their dependencies. Removes unused profile-override, OAuth-server and
MCP-transport schemas from the private generated module; the full pinned Core
contract remains unchanged. Drops an unread transport retry field.

Type verification now covers all SDK tooling and accepted consumer examples.
Explicit negative type contracts guard budgets, models and sync/async return
types. Generators and smoke receipts have typed boundaries; scripts require
annotations. No provider-backed smoke run is needed for these checks.

Strict, frozen generated records reject implicit scalar coercion. Nested file
inputs, upload grants, resource selections and tasks have named models. Schema
enums are `StrEnum` classes. Public clients and run helpers have explicit typed
parameters; `run` no longer accepts an arbitrary launch keyword bag.

`CompletionMilestone`, `RunSnapshot.finished` and `known_state` expose lifecycle
semantics without hiding unknown server states. Event payloads are validated
against their declared event kinds. Checkpoints have a typed, versioned persisted
format and still read the original unversioned format. Local errors expose stable
codes and categories. Credential records and auth status now use attribute access.

CI uses `ty`, expanded Ruff annotation/suppression rules, and an AST standards
lint for Any aliases, explicit API signatures, typed maps, strict frozen records,
and validation bypasses. Production modules contain no `Any`
annotations; intentional JSON extensions remain `JsonValue`.

## 0.29.0 (unreleased)

Every run requires a budget. Public launch selection uses published named
`profile` values with optional `research_task`; `program`, `method` and
`profile_overrides` are removed. Profiles can be discovered through the SDK and
CLI. Routing supports fixed or prohibited provider/model lists. Finished waits
require execution closure, final accounting and cleanup. Uploads and streams
retain recovery identities.

## 0.26.0

Public API v1: generated models and operations, opaque resource IDs, decimal-string
money, typed Problem Details, cursor pagination, resumable event streams, OAuth
PKCE, and matching local/hosted MCP tools. Async and synchronous clients share
one transport. API-key management requires a signed-in session.

Adds owner resource intents, provider presets through profile overrides, and CLI
`watch` with terminal snapshots and bounded streaming.

Adds execution method selection for Status Quo, Fork & Flush and Forgetting through
`method=` and `--method`. The two research methods require a published project
pairing. Adds file upload and run attachment through `upload_file`, `upload_text` and
`synth swarms upload`.

Adds `run.selection` for research run results and the selected snapshot output. Removes
`method.forge_profile_digest` from run snapshots.

Earlier releases and their changelog remain at tag `legacy/0.24.0`.
