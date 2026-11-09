# Changelog

## 0.31.0 (unreleased)

Aligns with the integrated Core public contract and its 32 KiB UTF-8 message
limit. Profile eligibility comes from generated metadata and explicit discovery.
Adds typed lazy iterators, payload-free diagnostic hooks, bounded streaming
uploads with source identity/digest verification, and runnable fixture-tested
recipes. Closing either client prevents new requests and preserves injected
HTTP ownership. Publication still requires installed-cohort acceptance.

## 0.30.0 (unreleased)

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
