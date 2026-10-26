## 0.23.8

Swarms adds the `code_native` project template with typed `CodeValidation` in synchronous and asynchronous project creation. Code projects use the existing workspace-write worker, governed push, registered verifier, host-grade, and task/run review authorities. Report projects keep their default template.

# Changelog

## 0.23.9

- Swarms run create retries the backend's pre-acceptance `owner_unavailable` refusal (503, `outcome=not_accepted`, bounded `Retry-After`) under a fresh idempotency key, the same way as `launch_admission_busy`; a transient owner readiness probe failure no longer fails the create (SYN-5091).
- Contracts (core.v1, index.v1, swarms.v1) are unchanged from 0.23.8.

## 0.23.7

- Swarms launches can grant `rhodes.pool`: a top-level `selection.pool_ids` (1-8 distinct ids), the `rollouts` limit, and up to three resource grants per launch.
- swarms.v1 contract fingerprint changes to match the backend export; core.v1 and index.v1 are unchanged.

## 0.23.6

- Preserve negotiated `synth.refusal.v1` failure chains in SDK HTTP errors and MCP results.
- Reject unsupported refusal versions, oversized envelopes, and mismatched refusal codes.
- Keep existing Index and Swarms contract fingerprints unchanged.

## 0.23.5

Contracts: core.v1 bbc5c42b, index.v1 378b8af4, swarms.v1 32b0b01e.

- `SwarmRun.watch()` / `AsyncSwarmRun.watch()` stream verified `RunSnapshot`s from
  `GET /swarms/runs/{run_id}/stream` (`swarms.stream`). Each snapshot's event id is its version
  token (`watch.last_event_id`); the watch reconnects with `Last-Event-ID` on rotation and on
  bounded transient failures, and ends when the backend reports the run final.

## 0.23.4

Patch release on 0.23.1 (contracts: core.v1 bbc5c42b, index.v1 378b8af4, swarms.v1 22eb54d2).
Swarms is not part of the production release; `native-swarms` is the only deferrable
retained check (SYN-4887).

- Versioned account-only credential and cloud companion (#495).
- Exact Refs, provenance links, bounded repository archives and artifact uploads/downloads.
- Atomic run bundles, revisioned labels, artifact inputs, message/guidance and objective results.
- Pause/resume controls and typed resource grants with itemized agent/workload/tool usage.
- Response models decode tolerantly: `Artifact.labels`, `RunSnapshot.labels` and
  `ResourceGrant`/`ResourceGrantSet` accept what a newer server stores; request types stay
  strict. `ArtifactManifest` stays strict because its digest covers every field.
- `run.download()` requires a final run (`execution_closed`, `cleanup_complete`,
  `accounting_final`) and selects the exact run branch across ref pages; uploads in a
  refused or expired state raise a typed error before finalize.
- Consistent transport errors, request identifiers, jittered read retries, injected HTTPX clients,
  and shipped contract fingerprints across supported Pydantic versions.

### Breaking

The legacy `objective` launch argument emits a deprecation warning; use `message`. The
alias is sent on the wire as `message` (the backend accepts either field, never both).
Absent `inputs`, `providers`, `models`, `labels`, `guidance` and `resources` are omitted
from launch requests rather than sent as `null`.
The legacy file/output facades remain readable with deprecation warnings.

### 0.22 to 0.23 migration

| Previous | Current |
| --- | --- |
| `PublicIndexClient` | `SynthClient.anonymous()` |
| `client.research.swarms` / `SwarmSpec` | `client.swarms.create(message=..., project_id=..., limits=...)` |
| Legacy file uploads | `client.artifacts.upload(...)` and exact input Refs |
| Separate MCP executables | `synth-ai-mcp` |
| Implicit collection iteration | One page, followed by its explicit cursor |
| `SYNTH_USER_KEY`, `SYNTH_BACKEND_API_KEY` | `SYNTH_API_KEY` (aliases removed in 0.25.0) |
| `SYNTH_BACKEND_URL`, `SYNTH_API_BASE_URL` | `SYNTH_API_URL` (aliases removed in 0.25.0) |
| Broad research/optimizer/container namespaces | Product-specific packages |

- `RetentionModeNotAllowed.allowed_modes` also reads the `X-Synth-Retention-Allowed-Modes` header when the body omits it; `SearchContentPurged` exposes `search_id`/`purges_at`/`purged_at` from `X-Synth-Search-Id`/`X-Synth-Purges-At`/`X-Synth-Purged-At` when sent.
- Includes organization retention, bounded DEEP `deadline_seconds`, and the Tunnel v2 control surface from the prior unpublished 0.24.0.dev20261018 tag (index.v1 fingerprint unchanged).

### Tunnel v2 (previously 0.24.0.dev20261018)


- Restore typed Tunnel v2 connector, lease custody, capability and checkpoint/session control through the current sync/async SDK transports.
- Keep canonical UUID recovery and immutable deadlines; mutations are never automatically replayed after uncertain delivery.
- Preserve the existing Index/Swarms schema fingerprints. Tunnel catalog v2 uses its separate versioned backend protocol and does not claim Index/Swarms release qualification.

## 0.23.1

Patch release on 0.23.0. No API surface or contract bytes change.

- Retained-history lifetime errors are the same over hosted MCP as over HTTP. A history
  entry that was deleted, has expired or is unavailable now raises a typed, public exception
  (for example `search_history_deleted`) instead of a generic `transport_error`
  (SYN-4635). Unknown status/code pairs and uncertain mutations keep their behavior.
- HTTP 408 replies while observing a search or run are classified as `RequestTimeoutError`, so
  bounded observation recovery retries them, and `Retry-After` hints are carried on the error
  (SYN-4560).
- The release verifier now accepts the attestation files that PyPI publication adds
  (SYN-4373) and the SDK contracts shipped in 0.23.0 are reconciled with these fixes
  (SYN-4354). Swarms is still not part of the production release.

## 0.23.0 (2026-10-16)

### Supported source changes

- Reduced typed `SynthClient` / `AsyncSynthClient` facade for Index, matching
  sync/async methods and durable operation handles. The SDK includes the Swarms
  client facade; Swarms is not enabled in the v0.23 production release and calls
  return a typed `operation_unavailable` until it is.
- Explicit request deadlines, bounded lazy streams, integrity-checked downloads,
  and cleanup of owned client resources without cancelling durable server work.
- Canonical `SYNTH_API_KEY` and `SYNTH_API_URL`, with value-free warnings for
  deprecated aliases and refusals for conflicting values.
- Per-organization entitlement and capability discovery.

### Authentication and transport

- Browser PKCE login and API-key login over stdin, origin-bound private credential
  storage, verified auth status, refresh, and logout.
- Unified `synth-ai-mcp` with public/read/action profiles and backend scope checks.
- Usage quantities, list-price observations, customer charges and exact funding
  attribution as separate fields. Unknown/pending values remain explicit.

### Breaking changes and migration

- Replace `PublicIndexClient` with `SynthClient.anonymous()` and typed options.
- Replace `client.research.swarms` / `SwarmSpec` with `client.swarms.create(...)`,
  an existing project, explicit limits and immutable input references.
- Use `synth-ai-mcp` instead of the separate Index/Research MCP executables.
- The broad legacy Research, optimization, container and tunnel namespaces are
  removed from this package. Use the respective product packages where available.
- List calls return one page. Pass the returned opaque cursor explicitly.
- Preserve mutation identity across uncertain outcomes; resolve before retrying.
  Output availability, execution closure, settlement and cleanup are independent.

### Limits

Device login returns `device_authorization_unavailable` before contacting the
provider. Factory is not part of the reduced public surface. Swarms requires organization admission. No per-operation price
is implied; current backend entitlements and charge ceilings are authoritative.

## 0.22.1 — 2026-09-30

### Fixed

- Complete typed Index client coverage for exact-revision rights attestation and
  resealed research correction registration, with explicit retry keys and sealed
  response bindings. Both operations preserve backend authority; neither grants
  publication consent or QA approval.
- Mirror the joined backend contract, including both missing revision operations.


## 0.22.0 — 2026-09-30

### Added

- Contribution archive upload sessions and explicit local file transfer helpers.
- Typed QA case, review and lifecycle SDK APIs, CLI commands and local MCP tools.
- Fenced appeal, escalation, adjudication and internal-note operations with visibility and reviewer policy.
- Explicit bearer-token Clerk OAuth Index transports.

- Hosted and local MCP registries expose the typed `index_qa_*` tools from
  `synth_ai.mcp.research.tools.index_qa` (the backend hosted MCP imports this module).
- Index v0.2 intake: manifests, classification, value, scopes and scope errors, research
  capture/build/release helpers, and the vendored `openapi/index-v1.json` refreshed to the
  backend 0.24.0 contract.

### Changed

- Swarms: `openrouter` is an accepted public inference provider selection
  (previously a duplicated `modal` entry). Additive; no other Swarms/SMR SDK surface changed.

Same Python package source as the `0.22.0.dev663` prerelease (synth-ai `dev` `681d0ce2`);
only version metadata and the README long description differ.

