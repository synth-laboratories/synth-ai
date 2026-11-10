# Changelog

## 0.32.0 (unreleased)

Breaking: the `synth` CLI is replaced by task verbs (SYN-5564), with no
compatibility period. Commands: `login [--no-browser]`, `whoami`, `logout`,
`profiles`, `upload`, `run`, `status`, `wait`, `logs [-f]`, `tasks`, `ls`, `pause`,
`resume`, `cancel`, `outputs`, `download`, `report`, `usage`, `billing`, `limits`.
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
