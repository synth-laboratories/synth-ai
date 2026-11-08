# Changelog

## 0.30.0 (unreleased)

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
