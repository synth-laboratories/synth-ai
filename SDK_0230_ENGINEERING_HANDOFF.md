# Synth SDK/MCP v0.23 engineering and live local-slot validation handoff

**Prepared October 13, 2026. This is an implementation handoff, not a release or test-pass certificate.**

The owner asked for a narrow, high-quality SDK and one MCP covering **auth, Index and Swarms**, with real backend changes in scope. The current SDK/MCP source is committed locally. Backend/service adoption and native qualification are incomplete. Nothing in this handoff authorizes publishing, production deployment, taking another owner's slot, or bypassing admission gates.

The required sequence is:

1. Finish SDK/MCP v0.23 and the backend/service changes necessary to implement its contract.
2. Run **live end-to-end local-slot tests**, fix failures, then run **live staging** against the same reviewed candidate family.
3. Add and qualify **cached inference** execution of the same business assertions.
4. Finish product docs/examples, release work and other deferred items.

Older planning documents and the original Index handoff sometimes put cached tests first. **That ordering is superseded.** The latest owner instruction is live local/staging before cached. This document is the explicitly requested handoff; product documentation/examples remain deferred.

No live, mock or protocol tests were run while preparing this handoff. Read-only source, artifact and controller inspection was performed. No slot was claimed, rebuilt, restarted or released.

## 1. Read this first

- The current package metadata is **0.22.2**, even though this is the v0.23 implementation branch. A version string alone cannot identify this candidate. Pin **source commit + wheel SHA256 + contract hashes**.
- The SDK's source-level contracts are **provisional SDK-first targets**. Backend pure contracts must adopt/reconcile them. An exported schema does not mean a route or guarantee exists.
- **`gpt-5.6-luna` is the first supported caller-supplied Swarms model override.** It is not an Index search option, a universal backend model ID restriction, or an instruction that every internal role must use Luna.
- Omitted `models`, `providers` and `inputs` preserve backend project/profile defaults. Explicit selections must be honored or refused. Do not silently inject Luna into omitted options.
- Only expose what can be proven. Storage of a PNG is not vision support. A terminal run status is not a canonical result. A 200 is not exactly-once accounting. A healthy container is not a qualified slot.
- New SDK/MCP behavior has passed static lint/type/policy checks and package builds. **It has not passed behavioral or native tests.** The earlier 78- and 96-test results cover older snapshots.
- Use the maintained `synth-dev` runtime manager. Do not start an ad hoc backend/Redis/DB stack to manufacture a green E2E result.
- All file work must remain under `/Users/joshuapurtell/GitHub`, with the standing OrbStack-config exception only. Historical documents that mention `Documents/GitHub` are not permission to write there.

## 2. Exact local inputs and provenance

### 2.1 Main implementation

| Item | Location / identity | Meaning |
| --- | --- | --- |
| Current SDK worktree | `/Users/joshuapurtell/GitHub/worktrees/synth-ai-sdk-0230-swarms` | Owned v0.23 SDK/MCP implementation |
| Branch | `feature/sdk-0230-swarms` | Local development branch; not a release branch |
| Source snapshot | `2d8348568a970f8108c22bc5e6783c80f6ee197f` | Last implementation commit before this handoff |
| Prior implementation commit | `3b7e86c3` | Swarms output/file SDK and unified MCP adapter |
| Earlier core commit | `13e2c3e2` | Initial typed Swarms core |
| Preserved legacy ref | `legacy/sdk-pre-0230-20261013` at `ea39eb39ee369cb28d75d69e8839258962de0591` | Historical recovery reference; do not restore it into the new API |
| Package metadata | `0.22.2` | Unpublished development artifact; not an assertion of compatibility with released 0.22.2 |
| Python MCP dependency | `mcp>=1.30,<2`; current lock resolves 1.30.0 | Optional `mcp` extra; official MCP Python implementation |

At preparation time the built wheel and source artifacts were:

| Artifact | SHA256 |
| --- | --- |
| `dist/synth_ai-0.22.2-py3-none-any.whl` | `68eeba9569e52fd38950a3c4b727a6aa5f18080fb094c12bc211d3e50354ee18` |
| `contracts/index-v1.json` | `38538ed5579193bc70373f3627835fd82d568b6535977f272958870371bdfcd2` |
| `contracts/swarms-v1.json` | `c2920af952913d6356787c0b6107dca17d053d605d7c14351514fea1edf77b5c` |
| `contracts/mcp-v1.json` | `9f2118d928e2a981d62dd4e7d8bab994b359bbb8c06393f39c6ad40523671453` |
| `uv.lock` | `cfdbc79929bf4966cd22cd2ffec6bac71ab7cbce323f6af3232ff38ed1dda6d6` |

A rebuild may produce a different wheel hash. Capture the actual rebuilt artifact; never copy the historical hash into a new receipt without checking its bytes. Preserve failed candidates and their receipts separately.

### 2.2 Tests and backend notes

| Input | Current location / snapshot | Caveat |
| --- | --- | --- |
| Frozen Index verification | `/Users/joshuapurtell/GitHub/worktrees/testing-sdk-0230`, commit `14d63907a0f0c94676036c98013c1df1aa9240d9` | Has the native readiness suite and the completed engineer receipt |
| Swarms contract tests | `/Users/joshuapurtell/GitHub/worktrees/testing-sdk-0230-swarms`, HEAD `0e9cf13503e4d4398b131bb8d0093af2611ed5b2` | `sdk/contracts/test_swarms_0230.py` has uncommitted owner-clarification edits; preserve/review them before consolidation |
| Backend design worktree | `/Users/joshuapurtell/GitHub/worktrees/backend-sdk-0230-notes`, commit `4c9156d43` | Notes only; does not implement the new backend |
| Frozen Index SDK | `/Users/joshuapurtell/GitHub/worktrees/synth-ai-sdk-0230`, commit `6a450c199c1424064719d520d1565b1b52a9428e` | Older target verified by the other engineer; do not confuse with the current source |

Read the retained verification:

- [Index verification receipt](../testing-sdk-0230/sdk/receipts/index-0230-verification-20261013.md)
- [Original Index handoff](../testing-sdk-0230/sdk/INDEX_0230_TEST_HANDOFF.md)
- Backend notes: `notes/specifications/sdk/{surface,index_contract,swarms_contract,swarms_outputs,mcp_adapter,backend_changes,files_images_review}.md`.

The original handoff is useful for tooling/provenance, but its source pins and cached-first ordering are historical.

### 2.3 What the completed test engineer actually established

At the frozen Index SDK snapshot:

- **78 installed-wheel tests passed**, with zero failures/skips.
- Package file bytes, import provenance, type markers and schema export were checked.
- A three-check native readiness suite ran on each of slots1/2: **three failures each**.
- All 30 candidate Index/common method/path pairs were absent from the observed native inventory. Invalid-key auth-context requests returned unversioned 404s, not the required typed 401 contract.
- Aggregate preflight reported **20 failures across 54 checks**.
- Provider spend was $0. The engineer stopped; no unattended test job remained.

That assignment is complete. **Native acceptance is not.** Later source changes, including new lifetimes, policy metadata, uploads and MCP, need fresh evidence.

## 3. Product contract that must be preserved

### 3.1 Auth and visibility

- `SynthClient` and `AsyncSynthClient` use an explicit API key or `SYNTH_API_KEY`.
- `.anonymous(...)` ignores ambient credentials and permits only admitted public Index operations.
- API calls use `Authorization: Bearer ...` at one explicit origin. No credential-bearing redirects, host fallback or client-selected tenant impersonation.
- Hosted MCP accepts properly verified OAuth or registered Synth API keys. Stdio obtains its key from process configuration; secrets never become tool parameters.
- Effective authority intersects credential grants, current membership, ownership/resource grants, entitlements and operation policy. A scope is not unlimited budget.
- Public search/citation visibility does not grant private content, artifact, export or preview access.
- Public contribution metadata does not imply owner draft, review, publish or administration permissions.

### 3.2 Index

- Anonymous search returns one terminal `SearchResponse`; no anonymous durable search handles, 202/polling-token recovery or keyed resolver promise.
- Keyed callers can perform a bounded terminal search or explicitly create a durable `SearchRun`.
- Exact content references bind contribution, revision, asset and digest; current rights are checked on every delivery.
- Owner contribution flow is draft/update, prepare/transfer/finalize, attest, submit and correction. Upload finalization is not submission or publication.
- Account and collection setup is read-only in the initial surface.

### 3.3 Swarms

- Existing projects only; preflight must not launch work or reserve budget.
- Caller override groups are **limits, inputs, providers, models**.
- `Models(default="gpt-5.6-luna")` and `Providers(default="openai")` are the initial selectable overrides. Omission remains unspecified, currently represented as null on the wire.
- Resolved model/provider metadata has separate bounded types and may report backend defaults outside the override allowlist.
- Finite charge, wall-time, task and concurrency limits must be admitted atomically. The backend cannot widen requested limits.
- Durable create/get/list, status/configuration/events, canonical result, task outcomes, output delivery, explicit cancellation and usage are the core.
- A run may finish execution before settlement or cleanup. Keep these states separate.
- Files are immutable initial inputs. Ready, selected, consumed-by-actor and used-by-model are distinct facts.
- Do not expose mid-run file adoption, arbitrary provider keys/endpoints, per-role palettes, pause/resume, steering, human-action workflows, branching/checkpoints, vision, OCI administration, Rhodes or Optimizers until separately admitted.
- An admitted profile must not deadlock on a required human action that the v0.23 surface cannot answer. Exclude that profile or explicitly implement/qualify the missing capability before admission.

### 3.4 Recovery, accounting and timing

- A keyed mutation identity binds key, operation, concrete resource path and exact payload SHA256, within the authenticated principal's scope.
- `MutationOptions(idempotency_key=...)` derives the remaining identity in the SDK. `identity=...` reuses a complete retained identity. Providing both is invalid.
- Same key and intent must not repeat execution/reservation/charging. Changed intent must conflict.
- Recovery distinguishes accepted, pending, confirmed not accepted, unavailable and conflict. Unavailable/expired is **not** permission to blindly resubmit paid work.
- Physical usage, provider cost and customer charge can differ. Preserve exact decimal quantities and integer coefficient/scale money, provenance, rate version, corrections and pending/unknown/final states.
- Customer-visible cost must respect disclosure policy; do not expose internal provider/rate details merely because a model has fields for them.
- Request, queue, first-useful-output, active, wall and settlement timing identify clock/source and uncertainty. SDK monotonic elapsed time is a separate measurement.

## 4. Current code layout and completed source work

| Area | Source |
| --- | --- |
| Root clients and auth | `synth_ai/client.py`, `types/auth.py`, `options.py`, `errors.py` |
| Shared operation/policy definitions | `contracts/operations.py`, `contracts/swarms.py`, `contracts/registry.py` |
| Contract exporters | `contracts/schema.py`, `contracts/swarms_schema.py`, `contracts/mcp_schema.py` |
| Bounded HTTP / lifetimes | `core/http/{client,async_client,request,response,lifetime}.py` |
| Creation recovery / observation | `core/recovery.py`, `core/observation.py` |
| Local files / signed uploads | `core/local_files.py`, `core/transfer.py` |
| Index resources | `sdk/index/` |
| Swarms resources | `sdk/swarms/{client,run,files,file_uploads,outputs,output_stream}.py` |
| Shared result vocabulary | `types/{base,identity,receipts,money,usage,uploads}.py` |
| One MCP catalog | `mcp/catalog.py`, `mcp/contracts.py` |
| Stdio execution | `mcp/__main__.py`, `mcp/runtime.py`, `mcp/server.py` |
| Hosted auth/execution integration | `mcp/hosted.py` |

The latest source pass added:

- One source of truth for operation scope, mutation, charging and retry-safety facts; MCP no longer hand-copies these flags.
- Operation/path validation and shared typed recovery for search, Swarm and upload creation.
- Public type exports and product-neutral auth context.
- Grant-aware local MCP discovery and invocation, using fresh backend auth context rather than treating a selected profile as authority.
- Explicit control-receipt tools in MCP.
- Single-consumer event/output streams, client-owned response cleanup and no buffered events yielded after closure.
- Credential-isolated bounded upload transfer, verified output text, and atomic no-clobber downloads.

Remaining source-level review obligations are real: exercise those new paths; reconcile protocol error behavior and supported-client expectations; prove cleanup/cancellation under faults; review the current numerical bounds; and reconcile the SDK-first schema with backend contract authority. Static checks did not prove these behaviors.

## 5. Cross-repository implementation work packages

Use clean owned feature worktrees based on each repository's appropriate reviewed integration ref. Do not assume every service has a local branch named `dev`. Do not edit a dirty shared primary checkout.

At inspection, `/Users/joshuapurtell/GitHub/backend` was on `codex/sublinear-production-config`, with 241 changed paths including 3 unmerged paths. Its local `dev` ref was `49595605414bcbe4c4380d9662444f8c7cd0f379`. This is context, **not a backend candidate to build**. Earlier notes audited another dev snapshot and some paths have since moved. Locate current owners before editing.

### WP1 — Pure public contracts, operation registry and HTTP composition

**Owner:** backend packages/application composition.

1. Adopt/reconcile the concrete Index/Swarms DTOs and policy facts into a pure package, using existing domain contracts where they meet the target. `packages/customer_api` is a suggested location, not an existing guarantee.
2. Preserve package/service/application boundaries: pure validation and state semantics in packages; persistence/external I/O in services; FastAPI auth/routing/error translation in app.
3. Add exact `/api/v1` dispatch for the candidate operations. Negotiate `X-Synth-Contract` on successful and error responses; reject incompatible versions before admission/provider work.
4. Supply safe typed errors, request IDs, logical operation IDs and observed metadata. Bound requests, pages and responses.
5. Export deterministic schemas/operation metadata from the authority and compare with the SDK artifact. Decide corrections jointly; do not weaken either side until a 200 happens.
6. Preserve active legacy/internal consumers through explicit adapters where needed. Do not bulk-delete unrelated runtime code to match the reduced public SDK.

**Completion evidence:** contract parity, actual route registration without duplicate dispatch ambiguity, typed refusals, and native service calls through the new boundary. Presence in OpenAPI alone is insufficient.

### WP2 — Shared principals and real hosted MCP auth

**Owner:** backend auth/services plus the authorization-server configuration owner.

1. Implement concrete API-key validation and OAuth verification, including active state, expiry, issuer/audience, organization membership, client attribution and revocation.
2. Avoid treating an OAuth identity as an unrestricted API key. Preserve concrete provenance throughout audit and service calls.
3. Implement `AuthorizationBackend.verify()` and `invoke()` from `synth_ai/mcp/hosted.py` using the shared principal/service facade. This interface exists; its production implementation does not yet exist in this workstream.
4. `verify()` returns a typed `VerifiedPrincipal` or None for invalid credentials. Authority outage must fail unavailable, never fall through to another verifier or anonymous access.
5. `invoke()` rechecks current resource grants, entitlements, budgets and idempotency and dispatches to shared services. Do not substitute an administrator key or forward an MCP-audience browser token to an unrelated resource.
6. Mount the one Streamable HTTP app with explicit resource URL, issuer, allowed origins and qualified tool allowlist. Publish working protected-resource metadata and authorize/PKCE/consent/refresh behavior at the actual authorization authority.
7. Keep secrets outside tool inputs, schemas, outputs and logs. Verify scope upgrades do not automatically replay a mutation.

**Immediate migration hazard:** the inspected backend `dev` `app/api/v1/routes_mcp.py` still imports `synth_ai.sdk.research.errors` and `synth_ai.mcp.research.server`, which are removed from the new package. Replace those consumers before changing its SDK dependency. Preserve internal worker/agent MCPs; they are not a second customer catalog to delete.

### WP3 — Durable identity and exactly-once effects

**Owner:** backend mutation/admission/persistence owners.

- Persist principal/org, operation, concrete path, key, payload digest and accepted outcome atomically enough to avoid double admission.
- Same-intent retries resolve to the original search/run/control/upload; changed payload or scope conflicts.
- Cover terminal blocking search as well as durable create, upload prepare/finalize, contribution mutation and cancellation.
- Retain typed known-resource references and an explicit expiry policy. Never return not-accepted merely because a receipt expired.
- Couple reservation/debit/settlement to logical operations rather than network attempts.
- Support fresh-process recovery and return the original identity on ambiguous responses where possible.

**Completion evidence:** dropped acknowledgements, concurrent duplicate requests, restart/recovery and changed-payload tests with one durable effect and one correct customer charge.

### WP4 — Index application, retrieval and reviewed delivery

**Owners:** backend Index/contributions services; `synth-index`; Artifact; OpenMonitor.

- Implement terminal public/keyed search projections and durable keyed search state/events/result/cancel/control.
- Public Deep must complete within its declared request contract or refuse; it cannot silently use a legacy 202/token polling flow.
- Preserve exact citations and content identities through retrieval, answer synthesis and reviewed delivery.
- Enforce current collection/content/revision rights at every read, including a second read after revocation.
- Make public and keyed results obey the same typed result vocabulary without broadening visibility.
- Join real retrieval, model execution, Monitor review, artifact custody and accounting. A mocked final answer or an SDK-side aggregation is not the new backend contract.
- Keep mode/configuration readiness explicit. Enabled flags without enrolled workers/dependencies are not readiness.
- Complete read-only account/collection inspection and the admitted owner contribution flow.

Likely current reuse areas: `app/api/v1/index`, `services/contributions`, `packages/contributions`, `services/artifact_platform`. Reconfirm paths in the selected candidate; do not copy old directory assumptions blindly.

### WP5 — Swarms admission and execution ownership

**Owners:** backend Swarms/SMR, Core/Horizons runtime, Orchestra, Sublinear as applicable to the admitted execution profile.

- Normalize only the four public override groups into an immutable resolved launch configuration.
- Retain profile/runtime/image/source/model/provider identities. Keep internal placement, harness and credential resolution backend-owned.
- Preflight must be non-reserving, return real blockers and refuse unsupported configurations before work starts.
- Admit budget/tasks/concurrency/wall-time atomically and persist a usable run identity before acknowledgement.
- Maintain ordered, resumable events, bounded task outcomes and a saved canonical result for success/partial/failure/cancellation.
- Propagate cancellation to actual execution; track accepted/applied/refused control receipt, worker closure, settlement and cleanup separately.
- Preserve runtime-owned authority; do not add a parallel terminal-state reducer in the SDK.
- Confirm the profile does not require an unexposed human action. Do not claim every legacy runbook/runtime is supported.

Changes in Core/Orchestra/Sublinear are required **where those guarantees are absent**, not as a blanket rewrite. Existing reliable primitives may be adapted.

### WP6 — File admission, artifact custody and output delivery

**Owners:** backend file/output services, Artifact and admitted runtime consumers.

- Prepare an expiring scoped upload grant bound to exact project/manifest; validate bytes, media type, size and digest on finalize.
- Preserve scanning, ready, rejected and deleted states. Only a ready file returns a selectable immutable `FileRef`.
- Prove the selected initial file is fetched and consumed by the executing actor using authoritative digest/consumption evidence, not just an answer that repeats a fixture phrase.
- Supply one canonical paginated output list and exact descriptor/content route; consolidate duplicate legacy output declarations.
- Current byte-delivery rights must be checked independently of the earlier metadata read.
- Match all required content headers, content length and digest. Missing storage is a typed failure, never an empty successful string.
- Do not expose raw internal/S3 URLs to make a failed managed download appear successful.
- Defer model image interpretation and dynamic input adoption. Separate feature branches contain relevant image-input work but were not proven integrated by the earlier audit.

### WP7 — Usage, billing, provenance and timing

**Owners:** Index metering/settlement; runtime accounting; backend public projections; Monitor attribution.

- Supply physical quantities, provider-cost valuation and customer charge as distinct facts, with exact units, provenance and rate versions.
- Include itemized operation/attempt/parent identities and corrections without counting snapshots, retries, child tasks and Monitor work twice.
- Respect customer-visible versus internal-only cost policy.
- Reserve/settle/release atomically against the admitted charge ceiling. Verify cancellation and partial/failure outcomes, not only successful completion.
- Provide server/worker timing with explicit missing/pending/uncertain values; never fabricate zeros or claim SDK elapsed equals backend duration.

### WP8 — Runtime candidate and test harness

**Owners:** synth-dev runtime/slot authority and testing.

- Bind exact clean source and image identities for backend, retrieval, Artifact, Monitor and the admitted Swarms worker/runtime closure.
- Reconcile migration/schema head, auth enrollment, service health, config checksums, resource admission and test tenant seeding.
- Build first-class installed-client live suites under `testing/sdk` and `testing/mcp`, register them in the maintained suite catalog and produce scoped receipts.
- Preserve the current native readiness suite as a prerequisite gate, not a substitute for customer journeys.
- Implement required fault cases and route/tool/acceptance coverage evidence.
- Add staging execution after local qualification; cached inference execution follows live local/staging.

## 6. Exact output and transfer protocol

For output content, the SDK first obtains a fresh descriptor, then requests:

`GET /api/v1/swarms/runs/{run_id}/outputs/{output_id}/content?revision_id=...&digest_sha256=...`

with `If-Match: "<sha256>"`. Success must be 200 complete bytes with:

- `X-Synth-Contract: swarms.v1`
- `X-Synth-Run-Id`, `X-Synth-Output-Id`, `X-Synth-Output-Revision`
- `X-Synth-Content-SHA256`
- `ETag: "<sha256>"`
- `Content-Type` matching the descriptor; supported text decoding is UTF-8
- identity/no content encoding; `Content-Length`, when supplied, equal to the descriptor

The SDK checks complete byte length and SHA256. An empty file is valid only with declared zero size and the empty digest. Stream chunks are provisional until EOF verification. Text reads return complete verified text, not an unverified prefix.

Downloads write a private temporary sibling, flush/fsync, then link into place without overwriting. On failure, remove the temporary file and preserve any existing destination. Filesystems without the required link semantics must fail safely, not silently overwrite. Async HTTP is native async; bounded local file I/O is synchronous and must be included in runtime responsiveness review.

Upload transfer is a distinct operation using a scoped grant and explicit host allowlist. No Synth API credential is forwarded to the upload destination; no automatic redirect/retry. Source bytes are rehashed during transfer. A prepare/finalize lost acknowledgement must be resolved, not blindly repeated under a new key.

## 7. Local runtime preparation: use authority, ownership and aggregate preflight

### 7.1 Observed serving authority

Reconfirmed read-only during preparation:

- Serving root: `/Users/joshuapurtell/GitHub/worktrees/integrate-synth-dev-20260922`
- Paired client: `/Users/joshuapurtell/GitHub/artifacts/hotspot-repair-20261010/controller/releases/eea57321f8fd9010/synth-local-slotctl`
- Authority executable: same release directory, `synth-local-slot-managerd`
- Matching client/authority build ID: `3a653f79bbe578bb`
- Reported compatible/current: true
- Manager reported other active work, including slot6 prebuild activity and slot4/5 evaluations. **Do not interrupt it.**

These are observations, not permanent routing instructions. Always re-resolve the serving authority before work. Do not rebuild/activate the dirty primary synth-dev client over the running authority just because a wrapper is stale.

### 7.2 Read-only inspection commands

Run from the future test engineer's owned workspace. Keep raw receipts private if they contain environment metadata; do not print secrets.

```bash
export UV_CACHE_DIR=/Users/joshuapurtell/GitHub/.cache/sdk0230-uv
SDK023_AUTHORITY_ROOT=/Users/joshuapurtell/GitHub/worktrees/integrate-synth-dev-20260922
SDK023_SLOTCTL=/Users/joshuapurtell/GitHub/artifacts/hotspot-repair-20261010/controller/releases/eea57321f8fd9010/synth-local-slotctl
export SYNTH_SLOT_MANAGER_CANONICAL_ROOT="$SDK023_AUTHORITY_ROOT"
export SMR_SLOT_MANAGER_CANONICAL_ROOT="$SDK023_AUTHORITY_ROOT"
export SMR_SLOTCTL="$SDK023_SLOTCTL"

"$SDK023_SLOTCTL" manager-status
"$SDK023_SLOTCTL" next
"$SDK023_SLOTCTL" claims
"$SDK023_SLOTCTL" resource-status --await-fresh --max-wait 30
```

Choose a slot only after evaluating current claims, active jobs, runtime compatibility and capacity. `next` is advisory; silence and an unclaimed slot do not prove safety/readiness. Do not hardcode slot1 merely because older tests used it.

### 7.3 Create a real candidate workspace

Use the maintained workspace configuration flow from the serving root's AGENTS/runbooks. Bind exact reviewed repo roots/commits and image inputs in `synth-workspace.toml`. Do not edit another owner's workspace or `config/sources.toml` to slip a branch through admission.

The backend dependency must resolve the intended candidate bytes reproducibly. A bare dependency `synth-ai==0.22.2` may install the public historical package, not this candidate. Do not use a dirty editable checkout as proof of image dependency parity. Choose an approved hash-pinned candidate artifact/build input, update the actual backend dependency/build declaration, and record it.

Before any rebuild, run **the aggregate preflight first** with the complete service closure and selected private seed source. The exact service names and candidate workspace are inputs to resolve from current runtime configuration; they are not invented by this document.

```bash
# Fill these from the reviewed candidate workspace/runtime inventory.
export SYNTH_WORKSPACE_ROOT=/Users/joshuapurtell/GitHub/workspaces/REPLACE_WITH_OWNED_WORKSPACE
SDK023_SLOT=REPLACE_WITH_SELECTED_SLOT
SDK023_SERVICES=REPLACE_WITH_COMMA_SEPARATED_SERVICE_CLOSURE
SDK023_SEED=REPLACE_WITH_AUTHORIZED_PROJECT_LOCAL_ENV_FILE

python3 "$SDK023_AUTHORITY_ROOT/local_dev/scripts/slot-preflight.py" "$SDK023_SLOT" \
  --services "$SDK023_SERVICES" \
  --seed "$SDK023_SEED" \
  --build-timeout-seconds 1800
```

This is a **template**, not a ready-to-run selection. Include DB and migration readiness and the actual Index/Artifact/Monitor/Swarms dependencies. Empty `--services` skips critical checks and cannot qualify a candidate. Match the build timeout to the intended managed build; also check execution-lease duration.

`--published-sdk-wheel`, if required by the current checker, must refer to the artifact expected by the actual backend lock/dependency inspection. Do not pass an unrelated candidate wheel to silence a published-dependency mismatch.

The current aggregate preflight may launch a short isolated host-health probe container; it is not merely a text-file parser. It must still preserve all existing slots/jobs. An exit zero means the inspected prerequisites passed, not that the later lifecycle admission is waived.

### 7.4 Previously observed blockers to recheck

The retained engineer receipt reported:

- Free disk below the 30 GiB raw floor and fresh-bundle reserve/growth requirement.
- Private credential projection and seed/inventory disagreement.
- Candidate source not admitted; backend dependency inspection pending.
- Missing cached seed image receipts and incomplete native Task configuration.
- Missing exact Index, Artifact and Monitor source/revision bindings and Index model files.
- Dirty serving source policy.
- DB connection budget 130 > 90, plus migration-head mismatch.
- Missing reviewed broker/Monitor activation bind files.
- Missing SMR/Artifact worker enrollment; protected Index disabled and Monitor capability unverified.

Do not assume these remain identical, and do not waive them. Re-run aggregate diagnosis against the actual candidate and fix all independent failures before building. Preserve other owners' active jobs. Use supported cleanup only for proven inactive owned scratch. Keep the configured mbx cache budget and host free-space floor.

### 7.5 Managed lifecycle only after prerequisite closure

After the actual candidate/ownership configuration is in place and preflight passes, use the serving client's supported lifecycle commands. Confirm semantics in its current `--help` and runbooks before mutation.

```bash
"$SDK023_SLOTCTL" claim "$SDK023_SLOT" \
  --purpose "sdk023-live-local" \
  --expect-minutes 90 --preemptible after_checkpoint

# Space-separated services here, selected from the same reviewed closure.
"$SDK023_SLOTCTL" reconcile "$SDK023_SLOT" REPLACE_WITH_SERVICE_NAMES \
  --reason "sdk023 live local candidate" --timeout-seconds 1800

"$SDK023_SLOTCTL" status "$SDK023_SLOT"
"$SDK023_SLOTCTL" ready-report "$SDK023_SLOT"
```

These templates are not authorization to take over a conflicting claim. If an unfinished bundle owns the lane, resume **that same** bundle with `--bundle-id`; do not mint a new bundle to compete with it. Use the managed launcher for Rust builds and retain per-target storage and build-lane guards. Do not bypass these with direct Cargo/Docker builds.

Derive the backend endpoint from current controller `ports.python` and the current binding receipt, not remembered port constants. Capture the exact status JSON, container/image/source/config identities and readiness report before running tests.

## 8. Build and install the exact SDK/MCP candidate

The commands below are for the receiving engineer after creating/reviewing an owned test worktree. They were **not executed as tests during preparation**.

### 8.1 Consolidate the test inputs without losing edits

The complete current testing base is `14d63907...` in the frozen Index verification worktree. The Swarms worktree has its own commit history and an uncommitted model-override clarification. Preserve those edits as an owned patch/commit and review them; do not overwrite them with checkout/reset. Merge/cherry-pick the required test source into a clean owned test worktree. Do not claim the old test count as a required count for the new source.

Keep `.test-artifacts` under that test worktree. Never put fixture scratch, credentials or output under Documents.

### 8.2 Candidate build and isolated installation

**Critical:** every `uv pip install` must use `--no-config`. The testing root has an override that previously replaced an explicit local wheel with a public release. Do not run root testing `uv sync`, do not use editable SDK installation for acceptance, and unset `PYTHONPATH` for installed-client execution.

```bash
SDK023_SDK_ROOT=/Users/joshuapurtell/GitHub/worktrees/synth-ai-sdk-0230-swarms
SDK023_TEST_ROOT=/Users/joshuapurtell/GitHub/worktrees/REPLACE_WITH_OWNED_TEST_WORKTREE
SDK023_RUN_ROOT="$SDK023_TEST_ROOT/.test-artifacts/sdk023-live-local-REPLACE_WITH_RUN_ID"
mkdir -p "$SDK023_RUN_ROOT"

# Review source identity and unexpected edits before building.
git -C "$SDK023_SDK_ROOT" rev-parse HEAD
git -C "$SDK023_SDK_ROOT" status --short
git -C "$SDK023_TEST_ROOT" rev-parse HEAD
git -C "$SDK023_TEST_ROOT" status --short

(cd "$SDK023_SDK_ROOT" && uv lock --check)
(cd "$SDK023_SDK_ROOT" && uv build --out-dir "$SDK023_RUN_ROOT/wheels")
(cd "$SDK023_SDK_ROOT" && uv export --locked --no-dev --extra mcp \
  --no-emit-project --format requirements-txt) > "$SDK023_RUN_ROOT/runtime.txt"
uv venv --python 3.13 "$SDK023_RUN_ROOT/venv"
uv pip install --no-config --python "$SDK023_RUN_ROOT/venv/bin/python" \
  -r "$SDK023_RUN_ROOT/runtime.txt" 'pytest==9.1.1'
```

Install exactly one selected wheel using a direct file URL with its hash. This Python snippet avoids shell quoting mistakes and ambiguous wheel selection:

```bash
export SDK023_RUN_ROOT
python3 - <<'PY'
import hashlib, os, subprocess
from pathlib import Path
root = Path(os.environ['SDK023_RUN_ROOT'])
wheels = list((root / 'wheels').glob('synth_ai-*.whl'))
if len(wheels) != 1:
    raise SystemExit('Expected exactly one candidate wheel')
wheel = wheels[0].resolve()
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
(root / 'wheel.sha256').write_text(digest + '\n')
subprocess.run(['uv', 'pip', 'install', '--no-config', '--python',
                str(root / 'venv/bin/python'), '--no-deps', '--reinstall',
                f'synth-ai @ {wheel.as_uri()}#sha256={digest}'], check=True)
print('Installed candidate SHA256', digest)
PY
```

Then verify installed provenance **before** running acceptance. Run from an owned scratch directory, not the SDK source root:

```bash
cd "$SDK023_RUN_ROOT"
env -u PYTHONPATH "$SDK023_RUN_ROOT/venv/bin/python" - <<'PY'
import hashlib, importlib.metadata, json, os, zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, unquote
import synth_ai
root = Path(os.environ['SDK023_RUN_ROOT'])
distribution = importlib.metadata.distribution('synth-ai')
provenance = json.loads(distribution.read_text('direct_url.json') or '{}')
url = urlsplit(provenance.get('url', ''))
if url.scheme != 'file':
    raise SystemExit('SDK was not installed from the selected local artifact')
wheel = Path(unquote(url.path))
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
recorded = provenance.get('archive_info', {}).get('hashes', {}).get('sha256')
fragment = parse_qs(url.fragment).get('sha256', [None])[0]
assert recorded == digest or fragment == digest
assert digest == (root / 'wheel.sha256').read_text().strip()
assert synth_ai.__file__ and 'site-packages' in Path(synth_ai.__file__).parts
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith('/') or name.endswith('/RECORD'):
            continue
        assert distribution.locate_file(name).read_bytes() == archive.read(name), name
print('Candidate provenance verified', digest)
PY
```

`direct_url.json` may contain the SHA256 only in the URL fragment, not `archive_info`. Check both. Checking only `__version__` is insufficient. Preserve the original wheel after installation so it cannot silently be overwritten at the same file path.

### 8.3 Static and installed boundary checks

From the reviewed test worktree, run the retained contract suite after updating it for the new implementation. Failures are work to resolve; do not change expected counts or weaken assertions solely to get green.

```bash
cd "$SDK023_TEST_ROOT"
env -u PYTHONPATH "$SDK023_RUN_ROOT/venv/bin/python" -m pytest \
  -c sdk/contracts/pytest.ini --confcutdir sdk/contracts sdk/contracts \
  --basetemp "$SDK023_RUN_ROOT/contract-pytest" -q

uv run --project sdk/lints --locked python sdk/lints/check.py \
  --sdk-root "$SDK023_SDK_ROOT" --format json
(cd "$SDK023_SDK_ROOT" && uv run --project "$SDK023_TEST_ROOT/sdk/lints" --locked \
  ruff check synth_ai setup.py)
(cd "$SDK023_SDK_ROOT" && uv run --project "$SDK023_TEST_ROOT/sdk/lints" --locked \
  ruff format --check synth_ai setup.py)
(cd "$SDK023_SDK_ROOT" && uv run --project "$SDK023_TEST_ROOT/sdk/lints" --locked \
  ty check --python "$SDK023_RUN_ROOT/venv/bin/python" synth_ai --error-on-warning)
```

Source and installed tests must be type-checked from their respective roots. The original combined command caused misleading relative-import failures. Keep deliberate negative typing cases separate and require their intended diagnostics, not an arbitrary nonzero exit.

These checks are prerequisites, **not live E2E**.

## 9. Native readiness before spending on model execution

The existing suite is `sdk023-index-native-readiness-v1` at testing commit `14d63907...`. It checks the 30 Index/common method/path definitions and typed invalid-key auth refusal through installed sync/async clients. It does not cover Swarms, successful authentication, worker execution, billing or MCP.

After selecting an owned ready slot and installing the candidate:

```bash
export INDEX023_INSTALLED_PYTHON="$SDK023_RUN_ROOT/venv/bin/python"
# Derive this value from the fresh controller ports.python receipt.
export INDEX023_LOCAL_BACKEND_URL=http://127.0.0.1:REPLACE_WITH_OBSERVED_PORT
cd "$SDK023_TEST_ROOT"
env -u PYTHONPATH "$INDEX023_INSTALLED_PYTHON" -m pytest \
  -c sdk/native/pytest.ini --confcutdir sdk/native sdk/native \
  --basetemp "$SDK023_RUN_ROOT/native-readiness" -q
```

This command is valid only after the native suite has been brought into the owned consolidated test worktree. Extend readiness to both contract inventories and exact dispatch/negotiation; do not treat these existing three checks as full readiness for 52 operation definitions.

**Stop paid tests if:** candidate routes are absent; auth gives unversioned 404s; service/image/config identity differs; a prerequisite is unavailable; the slot is unowned/unready; or a required scope/worker/migration cannot be verified. Fix the dependency, then rerun readiness. Do not let required tests skip green.

## 10. Build the live test harness that does not exist yet

There is no complete current v0.23 live SDK/MCP suite to invoke. Do not report a command against an invented directory as an existing test runner.

Recommended new source layout in the **testing** repo:

```text
sdk/live/
  pytest.ini                   # isolate this suite from root config/import effects
  conftest.py                  # explicit target, candidate, tenant, budget and receipt fixtures
  test_auth_and_isolation.py
  test_index_search.py
  test_index_content.py
  test_index_contributions.py
  test_swarms_run.py
  test_swarms_files_outputs.py
  test_recovery_and_accounting.py
mcp/live/
  pytest.ini                   # explicit configuration for installed-client execution
  conftest.py                  # real stdio/HTTP ClientSession drivers
  test_discovery_and_auth.py
  test_index_journeys.py
  test_swarms_journeys.py
  test_protocol_errors.py
```

These paths are a **proposal to implement**, not current files. Keep common business assertions/seed builders in a shared testing module so SDK and MCP exercise equivalent effects without sharing a fake API implementation.

Register bounded live suites in the maintained suite catalog. Add reviewed IDs for SDK and MCP local live execution, and separate staging live execution. Declare exact inputs, target type, required secrets, model/compute cap, deadline, ownership and no reuse of stale acceptance. Existing LLM-free E2E suites explicitly do not cover model-backed search/Monitor/Swarm acceptance.

**Current runner limitation:** `testing/scripts/run_e2e_board.py` selects only maintained suites with `max_cost_usd == 0`. It cannot run paid model-backed qualification unchanged. Do not label paid suites zero-cost to pass the filter. Integrate the new suites with the maintained paid eval/release-guardrail execution path, or extend the existing runner with explicit aggregate cost admission, per-suite caps, process/remote cleanup and receipts. Review the chosen runner before execution; registering a TOML entry alone does not make paid qualification executable. The existing board runner also displays skipped cases but can still return zero: the new acceptance path must reject required skips.

The existing zero-cost prerequisite can be run through the board runner after its environment is configured:

```bash
cd "$SDK023_TEST_ROOT"
python3 scripts/run_e2e_board.py --group SDK \
  --suite sdk023-index-native-readiness-v1 \
  --python "$SDK023_RUN_ROOT/venv/bin/python" \
  --log-dir "$SDK023_RUN_ROOT/readiness-board"
```

After the proposed live files/configuration exist, their test payloads use the installed candidate interpreter, for example:

```bash
# FUTURE payloads, not currently existing suites; run only through the admitted
# budget/deadline/ownership wrapper once the harness has been implemented.
env -u PYTHONPATH "$SDK023_RUN_ROOT/venv/bin/python" -m pytest \
  -c sdk/live/pytest.ini --confcutdir sdk/live sdk/live \
  --basetemp "$SDK023_RUN_ROOT/sdk-live-pytest" -q
env -u PYTHONPATH "$SDK023_RUN_ROOT/venv/bin/python" -m pytest \
  -c mcp/live/pytest.ini --confcutdir mcp/live mcp/live \
  --basetemp "$SDK023_RUN_ROOT/mcp-live-pytest" -q
```

A target fixture must resolve and record:

- Local or staging explicitly; never environment fallback.
- Backend origin and MCP origin/audience, CA trust where applicable.
- Candidate manifest and installed-wheel identity.
- Owned run namespace and test org/project IDs.
- Independent test personas and API keys or OAuth contexts, without logging values.
- Aggregate and per-run spend ceilings; expected actual cost.
- Expected live execution mode. Assert no inference-cache route is selected.

Use project-local authorized credential sources/Workshop's secrets proxy. Do not access non-GitHub credentials through macOS Keychain by default, and do not import secrets into a Keychain-backed registry. Never print `.env` contents or bearer headers.

### 10.1 Personas and deterministic seeded data

Prepare through authorized fixtures/admin setup, not client-supplied impersonation fields:

- Anonymous with an ambient key present in the parent environment.
- Organization A owner with only the required scopes.
- Organization A read-only caller and a caller missing one required scope.
- Organization B caller with valid credentials but no access to A's resources.
- Revoked/expired key and OAuth grants as applicable.
- Service-principal identity distinct from interactive-user identity.

Seed a public contribution and a private contribution with unique fact/nonce content; an exact canonical asset; a second revision; ready and unavailable file cases; a compatible existing Swarm project; and bounded test wallet/entitlements. Seed a small, known corpus through the real ingestion/indexing path and wait for its authoritative readiness. A seed script bypassing business validation must be separately controlled and cannot substitute for testing public contribution operations.

Use fixture identity receipts and cleanup namespaces. Do not delete or mutate other users' data to create negative cases.

## 11. Required live local journeys and assertions

Run through installed sync and async SDKs. Exercise each admitted MCP transport where meaningful. Keep exact-text model output assertions rare; verify factual content, references, state invariants and authoritative effects. A model answer alone is not proof of consumption or accounting.

### A. Authentication, negotiation and isolation

1. `.anonymous()` under an environment containing a key sends no credential and cannot access private/durable/account/write/Swarm operations.
2. Valid A credentials report the correct org/subject/client and current grants through `auth_context()`.
3. Invalid/revoked/expired credentials return a typed 401; insufficient scope returns the appropriate denial without widening grants.
4. B cannot read A's run, output, private content, upload or usage by guessing an ID.
5. Wrong contract is refused before reservation/execution. Error responses negotiate the expected contract and expose safe IDs, not secret/private bodies.
6. Revocation and membership removal take effect on subsequent operations, including between descriptor and content reads.

### B. Public Index terminal search

1. Submit one bounded public query against seeded public evidence.
2. Assert a terminal `SearchResponse`, no durable search ID/recovery token, truthful completeness and safe citations.
3. Prove private corpus facts do not appear in response/citations/metadata.
4. Retrieve the exact admitted public citation using current content policy.
5. Verify actual retrieval/model/review execution and live provenance. Public cost/admission policy must be explicit even if the caller is not charged.
6. For unsupported modes or deadline exhaustion, require the defined refusal/terminal behavior; do not accept a legacy asynchronous token flow.

### C. Keyed Index search and durable recovery

1. Run a terminal keyed query with an exact customer-charge ceiling and authorized collection IDs.
2. Create a durable search with a stable key; capture the returned handle and retained request identity.
3. List/get/status/event pages, then reattach from a **new process/client** and obtain the canonical result.
4. Assert event IDs/order and cursor behavior; compare exact citation identities, usage, charge and timing.
5. Verify a small ceiling refuses admission cleanly without execution or a leaked reservation.
6. Distinguish output-ready, execution-closed, accounting-final and cleanup-complete states.

### D. Content, privacy and corrections

- Exact revision/asset/digest succeeds; changed digest or stale revision conflicts/refuses as defined.
- Bounded text honors UTF-8 byte limits and complete/truncated semantics without inventing a substring digest.
- Withdraw/revoke access between reads and prove no protected text/locator/derived preview leaks.
- New revision/correction cannot retroactively change the bytes identified by an old immutable reference.

### E. Contribution delivery

- Draft/create/update, prepare upload, transfer exact package bytes, finalize, attest and submit through real routes.
- Confirm finalize does not auto-submit/publish. Confirm owner cannot gain review/admin authority.
- Exercise expired grants, bad digest, changed local source, unsupported media and wrong-project/owner access.
- Drop prepare/finalize acknowledgements and recover their identity rather than creating duplicate packages or debits.
- Validate correction lineage and exact accepted revision. Observe downstream ingestion/readiness rather than assuming transfer completion implies searchable content.

### F. Swarm preflight and successful execution

1. Read a compatible existing project.
2. Preflight finite limits with omitted model/provider overrides. Prove no run, budget reservation or worker was created.
3. Check resolved defaults without assuming Luna; record exact profile/runtime/image/model/provider identities.
4. Preflight and run a second case with explicit `Models(default="gpt-5.6-luna")` and admitted provider override. Prove the override is honored or clearly refused.
5. Create with a stable key. Observe real worker execution, tasks and ordered events.
6. Reattach from a new process and obtain a saved canonical result and its output references.
7. Verify actual budget/limit enforcement and independent accounting/cleanup finality.
8. A model may fail an objective. The harness must distinguish an honest terminal failed/partial result from a broken control plane; the chosen happy-path fixture must still meet its acceptance assertion.

### G. Immutable initial file consumption and output delivery

1. Create a bounded local text fixture containing a unique fact and known SHA256.
2. `FileSource.from_path`, `files.uploads.prepare`, `transfer`, `finalize`, then observe file state until ready using bounded reads.
3. Select the returned immutable `FileRef` in initial `Inputs`. Reject cross-project or unready refs before execution.
4. Require runtime evidence that the correct actor received the exact file digest. Correlate useful output with that consumption; a guessed nonce alone is insufficient.
5. List all output pages and confirm every advertised output has one consistent identity.
6. Read complete verified text; stream binary bytes; download without clobbering an existing destination.
7. Exercise truncated/extra/wrong bytes, MIME/encoding mismatch, stale revision, inaccessible storage and revocation between metadata/content.
8. Closing a stream/client or cancelling a local reader must release local resources and never cancel the remote run. An interrupted download must not publish a partial final file.
9. Include a legitimate empty output if the runtime supports it; verify zero size and the actual empty digest, not the historical empty-success fallback.

Do not promote image upload to vision acceptance. If later admitted, require a separate joined backend/Core/Orchestra model-consumption proof with exact image/digest and capability checks.

### H. Cancellation, partial/failure and cleanup

- Cancel an actually running owned Swarm and a durable search. Capture accepted/applied/refused receipt and read it without reissuing the cancellation.
- Verify actual workers stop or reach the specified cancellation boundary; no orphaned actor/container continues spending.
- Confirm queued/startup/running/finalizing/terminal races have a consistent response.
- Validate partial outputs where allowed, terminal result, released/settled budget and eventual cleanup independently.
- A local SDK wait timeout or MCP disconnect must **not** itself trigger remote cancellation.

### I. Idempotency and fault injection

Use an owned client-side fault boundary or scoped test hook. Do not destabilize shared services to drop one acknowledgement.

- Allow a mutation to reach the real server, then lose its response. Recover with the retained identity/key from a fresh client.
- Concurrently replay identical key/intent and prove one durable resource and one reservation/charge.
- Reuse the key with a changed payload or concrete resource and require conflict.
- Test unavailable/expired receipt behavior without assuming nonacceptance.
- Interrupt local observation; retain and resume the durable handle.
- Where restart resilience is claimed, perform only an owner-approved restart of the isolated owned worker/service and validate recovery. A fresh-process client test does not prove backend crash recovery.

### J. Usage, charges and timing

- Match returned summary/items to authoritative ledger and worker/provider observations using IDs, not an approximate total alone.
- Include multiple attempts, child tasks, Monitor costs, corrections, failure and cancellation cases.
- Verify no duplicate customer charge after retry/recovery and no final settlement while required items are pending.
- Check exact money scale and decimal quantities; do not use binary-float equality for monetary truth.
- Respect disclosure policy for internal costs. Unknown/pending is not zero.
- Verify live provenance and elapsed spans are nonfabricated, correctly scoped and sane relative to the recorded execution. Numerical latency thresholds must come from an admitted profile, not a made-up SLA.

## 12. Real MCP validation

### 12.1 Stdio

Launch the **installed candidate** interpreter as a child process from an owned scratch cwd:

`<candidate-python> -m synth_ai.mcp --profile action --base-url <observed-local-backend-origin>`

Provide the API key through the child environment, not CLI args or tool inputs. For the public profile, deliberately retain an ambient key and prove anonymous behavior. Use the official MCP `ClientSession`/stdio transport: initialize, list tools, read the schema resource and call tools over actual protocol I/O. Do not call Python handler functions and label it MCP acceptance.

Assert:

- Negotiated protocol/client behavior and parseable stdout with diagnostics only on stderr.
- Discovery matches profile **and fresh grants**. Calling a hidden/disabled tool directly still fails safely.
- Typed argument rejection, output-schema conformance, error codes/request IDs and no secret reflection.
- Search/run mutations require stable keys and preserve recovery after disconnect/timeouts.
- Event tools return bounded pages. Output text respects limits; no local filesystem path or unbounded base64 tool escape hatch.
- Revoked grants change behavior without restarting the server process.
- `auth_context` preserves org/client/service-user distinctions.

### 12.2 Hosted Streamable HTTP

Mount the real concrete backend auth/service adapter. The current constructor requires **HTTPS resource and issuer URLs**. A local backend `http://127.0.0.1:port` is acceptable for SDK testing, but it is not itself a valid hosted MCP resource URL for this implementation.

Use the managed local TLS/reverse-proxy arrangement with explicit audience and test CA trust, or implement a reviewed loopback-only development accommodation before testing. Do not disable TLS verification globally or silently change audience semantics.

Test API keys and OAuth independently:

- Missing/invalid token → HTTP auth challenge and protected-resource metadata.
- Correct issuer/audience/expiry, client attribution, org scope and live revocation.
- Wrong audience, cross-org, missing scope, expired and revoked grants fail without verifier fallback.
- Verifier outage returns unavailable, not anonymous or an API-key fallback.
- Per-tool scope challenge carries useful scope information without replaying a prior mutation automatically.
- Actual SDK-vs-hosted-MCP operations produce equivalent authorized service effects.
- No incoming browser token is sent to a different backend audience; no admin-key substitution.

A controlled local OAuth authority proves protocol/policy behavior, not live Clerk interoperability. Live staging must include the actual admitted authorization-server/client flow with authorized credentials.

## 13. Route, tool and acceptance coverage

The current SDK artifacts contain **30 Index/common definitions and 22 Swarms definitions**. They include two product-specific resolver definitions sharing the HTTP method/path `/api/v1/mutations/resolve`, distinguished by contract. The MCP catalog currently contains **32 tools**. Do not equate these counts with 52 distinct URLs or 52 MCP tools.

Build coverage from the pinned artifacts and actual selected release allowlist:

- HTTP identity: method + route template + operation ID + contract where needed.
- MCP identity: tool name + operation mapping + active authorization profile.
- SDK helper identity: helper/method plus the underlying operations it drives.

Capture actual handler/service dispatch correlated to test/request/operation/run IDs. A middleware-only 401 is an auth-boundary observation, not successful handler coverage. A tool may call shared services without an HTTP dispatch; measure both layers rather than pretending the route was exercised.

For every operation, retain the disposition: admitted and passed, admitted but failing/pending, intentionally absent from MCP, or explicitly deferred. Do not silently remove a failed operation from the denominator. Require policy/outcome assertions as well as observed execution.

The initial wire inventories are appended below; regenerate them from the exact candidate after changes. These are implementation targets, not a claim of deployed support.


### Index wire operation inventory

| Operation | Method and route | Keyed | Required scope | May charge |
| --- | --- | --- | --- | --- |
| `auth.context` | `GET /api/v1/auth/context` | yes | original-operation scope / authenticated context | no |
| `mutations.resolve` | `GET /api/v1/mutations/resolve` | yes | original-operation scope / authenticated context | no |
| `index.public.search` | `POST /api/v1/index/public/search` | no | public policy | no |
| `index.search` | `POST /api/v1/index/search` | yes | `index:search` | yes |
| `index.public.contents.get` | `POST /api/v1/index/public/contents` | no | public policy | no |
| `index.contents.get` | `POST /api/v1/index/contents` | yes | `index:read` | no |
| `index.searches.create` | `POST /api/v1/index/searches` | yes | `index:search` | yes |
| `index.searches.get` | `GET /api/v1/index/searches/{search_id}` | yes | `index:read` | no |
| `index.searches.list` | `GET /api/v1/index/searches` | yes | `index:read` | no |
| `index.searches.events` | `GET /api/v1/index/searches/{search_id}/events` | yes | `index:read` | no |
| `index.searches.result` | `GET /api/v1/index/searches/{search_id}/result` | yes | `index:read` | no |
| `index.searches.cancel` | `POST /api/v1/index/searches/{search_id}/cancel` | yes | `index:search` | no |
| `index.searches.control` | `GET /api/v1/index/searches/{search_id}/controls/{operation_id}` | yes | `index:read` | no |
| `index.searches.usage` | `GET /api/v1/index/searches/{search_id}/usage` | yes | `index:read` | no |
| `index.searches.usage_operations` | `GET /api/v1/index/searches/{search_id}/usage/operations` | yes | `index:read` | no |
| `index.usage.get` | `GET /api/v1/index/usage/{operation_id}` | yes | `index:read` | no |
| `index.usage.operations` | `GET /api/v1/index/usage/{operation_id}/operations` | yes | `index:read` | no |
| `index.account.get` | `GET /api/v1/index/account` | yes | `index:read` | no |
| `index.collections.list` | `GET /api/v1/index/collections` | yes | `index:read` | no |
| `index.collections.get` | `GET /api/v1/index/collections/{collection_id}` | yes | `index:read` | no |
| `index.contributions.list` | `GET /api/v1/index/contributions` | no | public policy | no |
| `index.contributions.get` | `GET /api/v1/index/contributions/{contribution_id}` | no | public policy | no |
| `index.contributions.create` | `POST /api/v1/index/contributions` | yes | `index:intake` | no |
| `index.contributions.update` | `PATCH /api/v1/index/contributions/{contribution_id}` | yes | `index:intake` | no |
| `index.contributions.attest` | `POST /api/v1/index/contributions/{contribution_id}/attest` | yes | `index:intake` | no |
| `index.contributions.submit` | `POST /api/v1/index/contributions/{contribution_id}/submit` | yes | `index:intake` | no |
| `index.contributions.correct` | `POST /api/v1/index/contributions/{contribution_id}/corrections` | yes | `index:intake` | no |
| `index.contributions.uploads.prepare` | `POST /api/v1/index/contributions/{contribution_id}/uploads` | yes | `index:intake` | no |
| `index.contributions.uploads.get` | `GET /api/v1/index/uploads/{upload_id}` | yes | `index:intake` | no |
| `index.uploads.finalize` | `POST /api/v1/index/uploads/{upload_id}/finalize` | yes | `index:intake` | no |

### Swarms wire operation inventory

| Operation | Method and route | Keyed | Required scope | May charge |
| --- | --- | --- | --- | --- |
| `swarms.projects.get` | `GET /api/v1/swarms/projects/{project_id}` | yes | `swarms:read` | no |
| `swarms.preflight` | `POST /api/v1/swarms/preflight` | yes | `swarms:run` | no |
| `swarms.create` | `POST /api/v1/swarms/runs` | yes | `swarms:run` | yes |
| `swarms.get` | `GET /api/v1/swarms/runs/{run_id}` | yes | `swarms:read` | no |
| `swarms.list` | `GET /api/v1/swarms/runs` | yes | `swarms:read` | no |
| `swarms.configuration` | `GET /api/v1/swarms/runs/{run_id}/configuration` | yes | `swarms:read` | no |
| `swarms.result` | `GET /api/v1/swarms/runs/{run_id}/result` | yes | `swarms:read` | no |
| `swarms.events` | `GET /api/v1/swarms/runs/{run_id}/events` | yes | `swarms:read` | no |
| `swarms.tasks` | `GET /api/v1/swarms/runs/{run_id}/tasks` | yes | `swarms:read` | no |
| `swarms.cancel` | `POST /api/v1/swarms/runs/{run_id}/cancel` | yes | `swarms:control` | no |
| `swarms.control` | `GET /api/v1/swarms/runs/{run_id}/controls/{operation_id}` | yes | `swarms:control` | no |
| `swarms.usage` | `GET /api/v1/swarms/runs/{run_id}/usage` | yes | `swarms:read` | no |
| `swarms.usage_operations` | `GET /api/v1/swarms/runs/{run_id}/usage/operations` | yes | `swarms:read` | no |
| `swarms.resolve` | `GET /api/v1/mutations/resolve` | yes | original-operation scope / authenticated context | no |
| `swarms.outputs.list` | `GET /api/v1/swarms/runs/{run_id}/outputs` | yes | `swarms:read` | no |
| `swarms.outputs.get` | `GET /api/v1/swarms/runs/{run_id}/outputs/{output_id}` | yes | `swarms:read` | no |
| `swarms.outputs.content` | `GET /api/v1/swarms/runs/{run_id}/outputs/{output_id}/content` | yes | `swarms:read` | no |
| `swarms.files.list` | `GET /api/v1/swarms/projects/{project_id}/files` | yes | `swarms:read` | no |
| `swarms.files.get` | `GET /api/v1/swarms/projects/{project_id}/files/{file_id}` | yes | `swarms:read` | no |
| `swarms.files.uploads.prepare` | `POST /api/v1/swarms/projects/{project_id}/uploads` | yes | `swarms:write` | no |
| `swarms.files.uploads.get` | `GET /api/v1/swarms/uploads/{upload_id}` | yes | `swarms:read` | no |
| `swarms.files.uploads.finalize` | `POST /api/v1/swarms/uploads/{upload_id}/finalize` | yes | `swarms:write` | no |

### MCP tool inventory

| Tool | Underlying operation |
| --- | --- |
| `index_search_control` | `index.searches.control` |
| `swarms_control` | `swarms.control` |
| `auth_context` | `auth.context` |
| `index_public_search` | `index.public.search` |
| `index_search` | `index.search` |
| `index_search_create` | `index.searches.create` |
| `index_search_status` | `index.searches.get` |
| `index_search_result` | `index.searches.result` |
| `index_search_list` | `index.searches.list` |
| `index_search_events` | `index.searches.events` |
| `index_search_usage` | `index.searches.usage` |
| `index_search_usage_items` | `index.searches.usage_operations` |
| `index_search_cancel` | `index.searches.cancel` |
| `index_content` | `index.contents.get` |
| `mutation_resolve` | `mutations.resolve` |
| `swarms_project` | `swarms.projects.get` |
| `swarms_preflight` | `swarms.preflight` |
| `swarms_create` | `swarms.create` |
| `swarms_list` | `swarms.list` |
| `swarms_status` | `swarms.get` |
| `swarms_configuration` | `swarms.configuration` |
| `swarms_result` | `swarms.result` |
| `swarms_events` | `swarms.events` |
| `swarms_tasks` | `swarms.tasks` |
| `swarms_usage` | `swarms.usage` |
| `swarms_usage_items` | `swarms.usage_operations` |
| `swarms_cancel` | `swarms.cancel` |
| `swarms_outputs` | `swarms.outputs.list` |
| `swarms_output` | `swarms.outputs.get` |
| `swarms_output_text` | `swarms.outputs.content` |
| `swarms_files` | `swarms.files.list` |
| `swarms_file` | `swarms.files.get` |

## 14. Receipt and evidence contract

Create one immutable run directory per attempt under the owned testing worktree. Keep credentials out of it. Required evidence should include:

| Evidence group | Required contents |
| --- | --- |
| Candidate | SDK code commit, wheel/source-distribution hashes, installed provenance, dependencies, Python/uv/MCP versions, contract inventory hashes |
| Services | Backend/retrieval/Artifact/Monitor/runtime source and image digests, observed build IDs, configuration checksums, migration head, controller identity |
| Target | Slot/claim/lease/workspace identity, explicit origins, actual ports, auth persona labels, test namespace, before/after readiness |
| Execution | Test code/assertion commit, suite IDs, exact commands with secrets redacted, UTC start/end, per-test statuses and logs |
| Business effects | Fresh search/run IDs, mutation identities, event cursors, task/output/file refs and digests, authoritative admission/control/settlement records |
| Coverage | Matched route/service dispatch, tool invocations, operation inventory denominator, policy/outcome obligations and unresolved gaps |
| Spend | Aggregate approval/tranche, per-run caps, expected cost, actual provider/compute spend or unresolved reconciliation |
| Cleanup | Final owned-run state, released reservations, closed clients/processes, worker/container cleanup, released claim/lease and retained debug artifacts |

Example machine-receipt shape to implement in the harness, not a current emitted schema:

```json
{
  "schema_version": "synth.sdk023.live_receipt.v1",
  "status": "failed",
  "target": {"kind": "local_slot", "slot": "slotN", "workspace": "owned-workspace"},
  "candidate": {"sdk_code_sha": "...", "wheel_sha256": "...", "contracts_sha256": "..."},
  "services": [],
  "tests": [],
  "coverage": {"operations": [], "tools": [], "pending_obligations": []},
  "spend": {"cap_usd": "...", "actual_usd": null},
  "cleanup": {"complete": false, "remaining_owned_resources": []},
  "blockers": []
}
```

Use decimal strings/exact representations for money in receipts. Do not transform unknown actual spend into zero. Keep the full detailed candidate manifest rather than collapsing multiple service versions into one SDK version.

### Pass criteria

A live local qualification pass requires all of these:

1. Exact installed candidate and actual running service/config/schema identities agree before and after.
2. The selected owned slot is admitted/ready; no skipped prerequisite is being counted as passed.
3. Every required journey and relevant policy/recovery obligation passes through the actual wire and services.
4. Live model/worker execution is demonstrated where the scenario requires it; no response replay or hidden inference-cache fallback.
5. No duplicate logical execution/charge under the required fault cases.
6. Required accounting settles and owned resources clean up; unresolved cleanup is a failure, not a footnote.
7. Route/tool coverage uses the actual selected inventory and distinguishes missing evidence from exclusions.
8. Evidence is retained, spend is reconciled, and limitations are stated explicitly.

Separate labels are useful: `static_pass`, `installed_boundary_pass`, `native_readiness_pass`, `live_local_pass`, `live_staging_pass`, `cached_pass`, `release_ready`. Never promote one to the next by implication.

## 15. Paid execution and safety boundaries

This handoff did not run paid work. Before the receiving engineer starts model/compute-backed tests:

- Check the existing $50 spend-tranche ledger and available headroom.
- State one aggregate maximum and expected actual cost for the bounded experiment; do not ask repeatedly for each small arm/retry inside already authorized scope.
- Set per-run budget, task/concurrency and timeout limits, plus a suite-level stop condition. Limits should reflect the chosen simple journeys, not a large generic stress test.
- Record each paid run in `/Users/joshuapurtell/GitHub/Jstack/.jstack/records/spend/ledger.md`, including cap and actual/unknown until reconciled.
- If actual plus outstanding commitments would exceed the open tranche, get the next tranche before exceeding it. Do not silently reclassify cloud compute as free because inference was cached.
- Provider credentials remain backend-owned. A client cannot set a provider API key, URL or fallback policy through the new SDK/MCP surface.

Do not deploy `index-monitor-prod` or staging Index Monitor on your own initiative. Monitor changes require the owner's deployment go, pinned identity and a receipt. Do not use an unpinned `railway up`. Do not restart the shared OrbStack engine during somebody else's work. Preserve live inference services and clean up only resources owned by the test.

## 16. Staging comes after live local

After local acceptance, curate the reviewed backend/service candidate through the existing staging policy. Inspect actual deploy triggers before pushing. Git `dev`, Git `staging` and a Railway environment named `dev` are not interchangeable.

Use:

- `/Users/joshuapurtell/GitHub/synth-dev/deployment/release/README.md`
- `/Users/joshuapurtell/GitHub/synth-dev/deployment/release/sdlc.md`
- `/Users/joshuapurtell/GitHub/synth-dev/deployment/synth-deploy/ships/README.md`
- `/Users/joshuapurtell/GitHub/synth-dev/deployment/synth-deploy/checklists/staging.md`
- Product-specific current shipping guidance and actual deployment configuration.

Staging validation must use explicit staging origins, isolated test orgs/resources, staged service identities and production-like auth. Re-run the selected live journeys including real OAuth/client behavior, rate/refusal paths and accounting. Do not reuse a local pass as evidence of cloud connectivity or authorization correctness.

The owner previously requested an SDK channel flow `dev -> staging -> nightly -> main`, with nightly package versions following `0.x.y.devN`. Treat this as intended SDK release-channel work, not a claim that current shared policy/publishing automation already implements it. Reconcile the package-specific policy and actual workflow before changing or pushing release branches. No publication is part of this handoff.

## 17. Cached execution comes after live local and staging

Once the live journeys are correct, adapt **the same assertions** to sealed inference-only replay via `synth-dev/local_dev/ai_cache`.

- Re-execute auth, API/MCP, DB, orchestration, file transfers, policy and accounting afresh.
- Cache only the supported inference boundary, not final API responses or saved run results.
- Record fixtures explicitly under authorized bounded spend; review and seal them. Replay must never record or fall back to a live provider.
- Give each independent test/worker its own replay occurrence state and diagnostics namespace.
- Bind fixtures to model/tool/prompt/canonicalizer identities. Changed contracts may need new recording; do not weaken matching just to conceal drift.
- Require expected hits and zero misses/forwards/recordings/bypasses for AI-consuming replay cases; declare zero-inference cases explicitly.
- Keep replay provenance separate from physical provider spend and customer charge policy.
- Add cloud replay only with authorized test-tenant/run-scoped routing; no ordinary customer/MCP provider-URL or cache override escape hatch.

Existing cache documentation is useful, but no new cached v0.23 fixture or native replay pass was created by the SDK implementation work.

## 18. Common traps and their remedies

| Trap | Correct response |
| --- | --- |
| Public 0.22.2 or an editable source replaces the candidate | Install with `uv pip --no-config`, an exact file URL/hash; verify installed member bytes and import origin |
| A source import makes tests pass | Run from the testing/scratch cwd with `PYTHONPATH` unset; assert site-packages provenance |
| Green OpenAPI inventory treated as E2E | Require real handler/service dispatch, model/worker evidence and business assertions |
| All models forced to Luna | Restrict caller overrides only; preserve omitted project defaults and report actual resolved configuration |
| Legacy `.research` imports crash the backend | Migrate concrete consumers before changing backend package dependency; retain needed internal adapters |
| API key/OAuth treated as interchangeable strings | Preserve credential kind, issuer/audience, org/client attribution and current grants |
| Missing permission repaired by a fallback admin key | Fail with a typed denial; correct grants through the authorized owner flow |
| Metadata access assumed to permit download | Revalidate current rights on the exact content request |
| Empty storage response accepted | Check declared size and digest; missing storage is a failure |
| Repeated mutation under a fresh key after timeout | Reconcile retained identity; same intent/key must converge without duplicate charge |
| Closed observer cancels the run | Only explicit control mutates the run; observation close is local |
| Cached controller status treated as current capacity | Obtain a fresh resource snapshot and actual admission result |
| Slot1 assumed free from a historical handoff | Re-read claims/jobs/current authority before choosing a slot |
| Rebuild discovers one gate at a time | Aggregate preflight first, full closure, then fix every reported prerequisite |
| An old suite silently skips absent credentials | Required live target/personas must fail configuration explicitly, not skip green |
| Cached-first instructions copied from the old handoff | Current order is live local, live staging, then cached |
| Broad no-Any lint bypassed for runtime adapters | Keep first-party interfaces typed; isolate unavoidable third-party protocol plumbing without introducing `Any` into first-party APIs |

## 19. What the receiving engineer should deliver

### Source deliverables

- Reviewed backend contract/auth/idempotency implementation on the correct dev-based branch.
- Required retrieval/Artifact/Monitor/runtime changes, each with exact source pins and ownership.
- A concrete hosted `AuthorizationBackend` and migrated customer MCP composition.
- Correct backend dependency on the actual candidate SDK/MCP artifact.
- Current installed-client and live SDK/MCP suites in testing, registered in the maintained runner.
- Any SDK/MCP corrections discovered through actual integration, with regenerated contract artifacts.

### Evidence deliverables

- Static/installed-boundary results for the exact new source, without recycling the historical 78/96 counts.
- Fresh native readiness and aggregate preflight outputs.
- Live local journey, policy, recovery, accounting, route/tool coverage and cleanup receipts.
- Live staging results only after the authorized curated staging candidate is running.
- Then separate cached-fixture and cached-run evidence.
- Exact outstanding blockers with attempted remedies, responsible owner and required external change.

### Completion rule

Do not stop at “routes added”, “SDK installed”, “first happy path passed” or “five of six tickets done”. Carry the selected acceptance scope through its final required evidence or identify a concrete external blocker, while continuing independent authorized work. A status update is not cancellation.

If blocked, state exactly what cannot proceed, what was tried, and what input/state change is required. Do not imply an unattended job continues unless a supported process actually exists and has been verified.

## 20. Coordination and final cleanup

At substantive task start, check the local MQ session named `agents`. During this handoff it reported no private session state. Do not start/reconfigure a broker solely to satisfy the check. If available with MCP tools, use your own logical identity, receive messages before overlap/handoff, and send scoped ownership/findings.

Use Pressure under the shared workspace binding `/Users/joshuapurtell/GitHub/.pressure/workspace.json`. Create your own session for the receiving coding conversation and actual worktree. Do not reuse this author's session for new work. Preserve mutation UUIDs, committed receipts and readbacks for feedback writes. Relevant existing records:

- `295b371f-4121-4807-bc3b-89ab953dce93`: corrected Luna-as-override decision.
- `b94f4b6e-9de5-42aa-bccb-baf0f8c0e6ef`: source-only SDK/MCP work and live-before-cached sequence.
- `17943453-fe71-40e2-afb1-ab16be04bd4c`: uv override replaced the intended SDK artifact.
- `a7989fed-f2b3-5e59-ac27-8db8559d8eb4`: historical output-content empty-success problem; new source alone did not resolve it.

After testing, close SDK/MCP clients and child processes; stop only owned runs/workers/paid compute; confirm reservation/settlement cleanup; release the owned claim/leases through the manager; retain failure evidence. Do not delete shared volumes or broad caches. Follow the workspace's Modal cleanup rules if that provider was used.

Return source commits, immutable evidence locations, actual cost, completed versus pending acceptance, and exact blockers. No release/publish/deploy claim should be made from this handoff alone.
