# 0.23.0 implementation and removal plan

October 13, 2026. Checked out on `feature/sdk-0230-index-swarms`, based on
preserved dev `ea39eb39`. Package metadata remains 0.22.2 until qualification
and release preparation. The initial owner-authorized local retirement cut is now applied. It does not
tag/publish or change the original dirty checkout.

Backend `notes/specifications/sdk/api_outline.md` owns the candidate API sketch;
`v0.23.0.md` owns guarantees and qualified-operation admission. Only Index/Swarms
ship. Candidate implementation tasks are not public support promises.

## Target shipped shape

```text
synth_ai/
  __init__.py          explicit typed exports and truthful installed version
  client.py            SynthClient / AsyncSynthClient, owned transports
  core/                concrete auth/options/errors/http/JSON boundaries
  sdk/index/           admitted search/content/contributions/owner resources
  sdk/swarms/          minimal setup/runs/events/results/outputs/usage
  mcp/                 one registry, stdio/HTTP adapters
  cli/                 index/swarms/auth/mcp/version, stable JSON/exit behavior
  py.typed
```

Exact internal names/public type imports freeze with contracts. Do not create
empty advertised namespaces. Sync/async match; generated repetitive models and
bindings sit below deliberate resource handles, workflows and lifetime semantics.

## Concrete cuts

`SDK_0230_REMOVAL.json` records tracked paths in review groups. It is not an
executable deletion script. Replacement/import/consumer and reduced-wheel proof
precede removal.

| Current code | Disposition |
| --- | --- |
| `__init__.py`, `client.py` | Rewrite typed exports/client. Remove `_Any` dynamic export loader and Research/Messaging/Optimizers/pool properties; no hidden aliases. Preserve version and transport ownership. |
| `sdk/research/session/` | Port proven admitted behavior into focused Swarms contracts/resources; then remove the monolithic session/client/generic forwarding package. Review file/source/output delivery individually. |
| `sdk/research/facade.py`, `advanced_swarms.py`, `client.py` | Replace with typed Swarms facade; retire ResearchClient and dynamic advanced/session fallback. |
| `sdk/research/swarms.py`, selected contracts | Port concrete launch/status/events/usage/task/control parsers; implement canonical result, managed streams/pages. Only qualified controls/profiles enter facade. Retire old path after tests. |
| Remaining `sdk/research/` | Remove Intern/program, Factory/Effort/Tag, wiki/knowledge/messaging/integration, generic dataset/model/PR/visual/cloud/project-computer admin and container-pool public implementations. Keep only audited domain helpers needed by admitted output/intake. |
| `sdk/optimizers/`, `sdk/messaging/`, `pools.py` | Remove shipped deferred products/exports; source survives in preservation branch. |
| `sdk/index/qa*.py`, `qa_preflight.py`, `forge_notices.py` | Retire public QA/review/coordination/admin flows; audit shared utility imports first. |
| Remaining `sdk/index/` | Consolidate public/keyed/search/lifecycle/upload/transfer around frozen contracts. Keep exact citations/rights/contribution delivery/corrections and admitted owner reads. Audit legacy-named intake helpers before cuts. |
| `mcp/research/server.py`, parsing and broad tool builders | Replace with one typed admitted registry and adapters. Remove unused definitions and dispatcher branches, not merely discovery; stop importing retired session. |
| `cli/research*.py` | Replace with minimal Swarms commands, then remove Research/admin registration/help/imports. Rework retained Index commands. |
| `pyproject.toml` | Keep CLI `synth-ai`, one new `synth-ai-mcp`; remove Research/Index MCP entrypoints and pool extra. Audit compatibility extra/dependency closure. |
| `ty.toml`, Ruff, Makefile/AGENTS/docs links | Remove three shipped MCP exclusions; add annotation/Any/layer/export/wheel gates. Repair absent delegate/doc paths and point Synth Style to backend/tigerstyle.md. Explicit worktree roots; no guessed Documents paths. |

User-directed breaking scope supersedes old instructions to keep every public
Research shim. Retain migration instructions/preservation source; no broad dynamic
compatibility package inside the reduced wheel. Backend services/storage/native
owners and active internal/archival consumers are not blanket-deleted. Retire
public aliases/duplicate routing only after actual consumer/migration review.

## uv and dependency plan

The repo already has `uv.lock`. Reconcile it for the reduced package. Published
runtime dependencies stay separate from local dependency groups; pin tools and
Python identity in release receipts. Keep HTTP/model/schema/CLI libraries when
justified by the reduced import graph and installed tests.

- `pynacl` currently serves Research `_internal/crypto.py`; remove only after
  its surviving auth/transfer consumers are ruled out or replaced.
- Remove `synth-containers` pool extra for this SDK cut. Other testing repo
  products keep their own pool/optimizer/backend dependencies.
- Consolidate duplicated dev extra/group; review notebook/kernel/papermill/
  nest_asyncio/keyring against surviving code. Docs tooling has its own named
  locked group or docs-owner project.
- Use owned uv environments and locked commands. Isolated acceptance installs
  the exact wheel; testing's editable `../synth-ai` source must not shadow it.
- Consumer dependency ranges come from package metadata, not the development
  lockfile or local `tool.uv.sources`.

[uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/),
[locking](https://docs.astral.sh/uv/concepts/projects/sync/).

## Lint and type gates

- Locked Ruff format/lint across shipped package and new/touched test helpers;
  add relevant annotation rules including implicit Any checks.
- ty covers all shipped SDK/MCP/CLI/private/generated modules. Remove current
  server/request/workspace-input exclusions; no broad ignores or casts/Unknown
  forwarding that conceal unsound boundaries.
- AST/source gate rejects explicit Any imports/aliases/casts, untyped signatures/
  callbacks, bare authoritative containers and dynamic facade return forwarding.
  Bounded raw JSON narrows to concrete models at one deliberate boundary.
- Installed-wheel positive/negative typing for sync/async handles/pages/streams,
  options/results/errors; reject wrong fields/options/return use, not only accept
  examples. Check import origin and packaged py.typed.
- Layer/export/deep-import/wheel-content/entrypoint/dependency gates prove retired
  surfaces absent. Direct MCP calls to hidden names must also be uncallable.
- Synth Style: bounded content/retries/queues, monotonic deadlines/UTC, causal
  errors, stable idempotency and accounted async cancellation/background tasks.

Proposed commands after implementation, not green baseline claims:

```bash
uv lock --check
uv run --locked ruff format --check synth_ai
uv run --locked ruff check synth_ai
uv run --locked ty check
uv build
```

Add maintained testing gates and installed/wire acceptance via existing catalogs;
no new universal runner. [ty config](https://docs.astral.sh/ty/reference/configuration/),
[Ruff annotations](https://docs.astral.sh/ruff/rules/#flake8-annotations-ann).

## Cutover order

1. Source/preservation pins, API/profile/migration targets and removal inventory.
2. Pure contracts and shared auth/admission/errors; backend result/output/accounting.
3. Focused sync/async resources and installed tests, without importing old session.
4. One MCP registry and small CLI; protocol/auth/schema/rights/client journeys.
5. Required files, cached/live flows, identity/coverage and generated tested docs.
6. Delete retired exports/modules/entrypoints/dependencies in reviewable commits
   after replacements pass; prove absence and rerun admitted regression checks.
7. Fresh exact reduced-wheel/backend qualification before release; production
   acceptance follows authorized deployment. Old receipts cannot certify new bytes.

SDK release promotion is dev → staging → nightly → main. Nightly X.Y.Z.devN and
stable X.Y.Z are distinct final-version artifacts and each is qualified before
publication. Current workflow publishes from main/dev and needs explicit channel
cutover work; no automatic publication is enabled by these planning commits.

The initial testing/sdk/lints checker and locked regression suite are implemented.
Its current SDK baseline fails with 2,754 diagnostics; that is evidence of work
remaining, not an allowed-violations list. Installed/backend/E2E gates remain pending.

Testing `sdk/IMPLEMENTATION.md` and `contracts/sdk_0230_scenario_plan.json` contain
the planned suites/obligations. Bodies/catalog registration and numeric/cost
profiles remain pending; no skipped-green placeholders or paid jobs run here.

## Initial retirement receipt — October 13, 2026

Owner requested removal before building the replacement API. The local branch
now removes the complete old Research/session/Factory/Intern/admin package,
Messaging, Optimizers, pools, compatibility import hook, old CLI and split MCP.
The old CLI/MCP entrypoints and pool/research/dev compatibility extras are gone.
Swarms and the unified MCP/CLI are deliberately absent until implemented; no
placeholder facade or broken entrypoint is exposed.

Index QA/review/classification/publication, reward/contest/profile/tag and
collection-grant facades are cut, together with unused dispatch entries and
retired DTO modules. Shared exact reference DTOs move into Index without wire
constraint changes. Index intake/upload/search/rights/accounting helpers and
shared auth/HTTP remain implementation material pending consolidation; this cut
does not certify them as the final method surface.

Old SDK-local tests/export scripts and packaged guides referring to retired APIs
are removed; their history remains on the legacy branch. New installed retirement
checks live in testing/sdk/contracts/test_0230_retirement.py. Setuptools now ships
only the retained Index snapshot; legacy Research/Messaging snapshots and the
broad YAML are removed. The Index snapshot remains legacy material, not the
frozen 0.23 route-coverage denominator. uv.lock is reconciled after pruning pool,
crypto, CLI, notebook/kernel and Keychain development dependencies.

Preserved source: legacy/sdk-pre-0230-20261013 at ea39eb39; 0.23 planning pre-cut
commit c53b830e. No backend/native owner/data/customer deployment is changed.
Installed-wheel retirement tests verify retained imports/transport lifetime,
removed deep imports/exports/entrypoints/admin facades, py.typed/schema packaging
and moved reference identity/validation. Full no-Any/ty and backend acceptance
remain pending; retirement evidence cannot qualify release readiness.

Verification: 387 tracked files removed, including 339 Python files across the
package and old test/script trees; every removed path exists on the preservation
ref. 60 package Python files remain. Reduced sdist/wheel build succeeds; 25
isolated installed-wheel retirement tests pass. New tests pass Ruff and ty with
warnings as errors. Changed entry/reference/build modules pass Ruff; package-wide
undefined-name checks and uv lock check pass. Source-policy findings drop from
2,754 to 148 (99 Any, 32 annotations, 16 bare generics, 1 dynamic facade), with
zero remaining retired-import or ty-exclusion findings. These are remaining
implementation failures, not allowed violations or end-to-end qualification.

## Proposed implementation tree — Index first

October 13, 2026. This sketches file ownership, not existing implementations.
Keep the current package root; no src-layout migration or empty advertised modules.
Create files when their implemented responsibility warrants them.

```text
synth_ai/
  __init__.py                  explicit public exports
  client.py                    SynthClient / AsyncSynthClient composition and lifetime
  options.py                   ClientOptions / ReadOptions / MutationOptions
  errors.py                    concrete safe public exceptions
  py.typed
  types/                       shared typed vocabulary, no product execution
    identity.py                IDs and mutation identity
    pagination.py              Page[T] and distinct cursor types
    money.py                   exact units/currency, no float charges
    usage.py                   summary/items/provenance/finality
    timing.py                  named measurements and source/clock
    receipts.py                control, transfer and resolution states
  core/                        internal shared mechanisms
    auth.py                    API-key resolution and explicit anonymous policy
    http/
      client.py                owned/borrowed sync HTTP boundary
      async_client.py          native async HTTP boundary
      request.py               typed operation/request encoding
      response.py              bounded decoding and safe response metadata
      retry.py                 deadline/retry/idempotency policy
    observation.py             bounded managed observation
    transfer.py                bounded transfer and digest/expiry checks
  contracts/
    operations.py              schema-derived operation IDs/paths/authority metadata
  sdk/
    index/
      __init__.py              intentional typed exports
      client.py                compose IndexClient / AsyncIndexClient
      search.py                single-call public/keyed search
      searches.py              keyed create/get/list/creation recovery
      search_run.py            SearchRun / AsyncSearchRun lifetime and observation
      contents.py              exact currently authorized content
      contributions.py         owner draft/attest/submit/correct workflow
      uploads.py               contribution preparation/transfer/finalization
      account.py               read-only account inspection
      collections.py           read-only authorized collection inspection
      usage.py                 operation summary/items
      types/
        search.py              options/response/citation/snapshot
        contents.py            ContentRef/options/response
        contributions.py       drafts/revisions/rights/submission/correction
        uploads.py             manifest/session/receipt references
        account.py             account and collection projections
    swarms/                    later: same resource/handle/type discipline
  mcp/                         later: one admitted registry, two transports
    registry.py
    auth.py                    connection auth; backend owns grants/admission
    tools/index.py
    tools/swarms.py
    stdio.py
    http.py
  cli/                         later: thin typed client commands
    main.py
    index.py
    swarms.py
    auth.py
    mcp.py
```

Sync/async resources share model definitions, validation, operation metadata and
pure helpers. Both classes live in their resource module; separate transport
implementations preserve native async ownership. No duplicated sync/async package
trees, reflection facade, mega-client, generic forwarding or catch-all utils file.
Handles retain identity; response snapshots are data and never hide network work.
Shared types remain independent of clients/MCP/CLI. Imports point from bindings
through resources to common mechanisms/types; products never import each other.

Generate repetitive contract models/operation facts from selected authoritative
backend schemas only once generator output passes zero-Any/ty/doc gates. Do not
add a second hand-maintained wire DTO tree when SDK and wire models agree.
Behavioral workflows, handles/retry policy and public docstrings are deliberate
code. HTTP authority remains backend-owned; MCP/CLI are bindings, not new business
or accounting implementations. OAuth connection auth does not widen SDK auth.

Testing stays in the dedicated repository, with sdk/unit/index, sdk/typing/index,
sdk/contracts, sdk/journeys/index and sdk/examples for tested guide sources;
mcp/protocol, mcp/auth and mcp/journeys/index exercise actual wire behavior.
Shared sdk/support and sdk/scenarios hold typed target/artifact/fixture/receipt
contracts and business assertions, including explicit live/sealed cache modes.
Reuse maintained catalogs/runners. Generated MDX/reference belongs to the docs
repository; docstrings/types/signatures and tested examples are its inputs.

Reference observations, October 13: [official Exa Python repo](https://github.com/exa-labs/exa-py)
exposes Exa/AsyncExa search/content calls and product packages; api.py is large,
so borrow the customer simplicity while splitting our implementation by resource.
[Cursor Python SDK](https://cursor.com/docs/sdk/python) documents matching
sync/async clients, agents and run handles with explicit async ownership.
[OpenAI Agents results/state](https://developers.openai.com/api/docs/guides/agents/results)
distinguishes final output, history, resumable state and pending interruptions.
These are design analogues, not copied package trees or extra 0.23 features.

Index implementation order: concrete search/content models and current backend
contract delta → single-call search/exact contents with bounded sync/async
transport and explicit auth → keyed durable handles/recovery/accounting →
contribution workflow → wire/live/cached/docs admission. Current backend public
Deep returns 202 plus a search token; the selected single-call public contract
requires an explicit backend decision/update, not a hidden anonymous durable
handle in the new SDK. Current content routes identify revisions and returned
asset digests; exact expected digest binding must be designed and tested rather
than inferred from citation visibility. No runtime implementation is added by
this tree sketch.

## Index SDK-first implementation — October 13, 2026

The new Index facade, closed models, sync/async transport, keyed search handles,
managed event observation, content/digest checks, usage, owner reads and
contribution/upload lifecycle are implemented against the provisional index.v1
contract. Native backend adoption has NOT occurred. See backend's
`notes/specifications/sdk/index_contract.md` for the required changes.

`contracts/index-v1.json` is a deterministic schema/operation bundle exported by
`python -m synth_ai.contracts.schema`; it is not OpenAPI or native route evidence.
The wheel ships the same bundle. API prefix is /api/v1 with explicit index.v1
negotiation. PageCursor and EventCursor are distinct types; event pages use
Page[SearchEvent, EventCursor]. Transfers currently admit zero automatic retries;
use status/reconciliation for uncertainty. Anonymous clients have no durable
search handles. Async lifetime uses aclose.

Verification of this SDK-first cut: 78 installed-wheel tests pass; full package
Ruff/format and ty (warnings are errors) pass; the source gate reports zero
violations, including zero Any and no type-check exclusions. Two negative typing
examples correctly fail (public options for keyed create; page cursor for event
read). These checks do not establish native backend/worker/auth/cache behavior.
Package metadata remains 0.22.2; no publication or deployment is authorized here.

Slot testing is handed to another engineer. The primary synth-dev checkout is
not the serving authority. The running controller serves
worktrees/integrate-synth-dev-20260922. Read-only status found slot1/slot2 healthy
but unlatched/unclaimed; preflight also found disk headroom below its 30 GiB floor,
private projection/authority-key problems and config/label drift. No slots were
claimed, rebuilt, stopped or mutated. Use the handoff's fresh preflight rather
than these observations as permission to start work.

## Swarms worktree — first core slice

This separate branch/worktree starts at the frozen Index handoff commit
6a450c199c1424064719d520d1565b1b52a9428e. It does not modify the handed-off
Index worktree or its installed test environment.

Owner clarification: **gpt-5.6-luna is a Swarms caller model override**;
`openai` is the initial provider override. Typed override options reject other
selections locally. Optional model/provider/input groups remain unspecified when
omitted, preserving backend project/profile defaults. Explicit selections must
be honored or refused; preflight must not silently substitute them. Read-only
resolved model/provider metadata uses separate bounded types and can report
backend defaults outside the caller override allowlist. This is not an Index
search option or a rule forcing every internal agent role to use Luna. Backend
admission still resolves and reports the exact qualified execution configuration.

Implemented core: projects.get, preflight/create/get/list/reconcile, run status,
configuration/result/tasks, events/pages/wait, cancel/control and usage. Shared
bounded observation and parameter helpers now sit under core instead of one
product importing another. Each operation selects and verifies its product
contract header/body. Outputs/file transfer facades, native backend adoption,
MCP/CLI and real live/cached slot qualification remain pending.

Swarms build verification: 93 installed boundary/retirement tests pass, including
all inherited Index tests; full SDK/test Ruff/format and ty pass; source gate
has zero violations. These are mock HTTP/unit checks, not native acceptance.


## Output/file delivery and unified MCP implementation

Owner instruction: build SDK and MCP without running tests. The prior 96 passing
mock/contract checks predate this slice; they do not qualify the code below.
Only static lint/type checks, schema generation and package builds are performed.

`SwarmRun.outputs` and `AsyncSwarmRun.outputs` now expose `list`, `get`, lazy
`stream`, bounded `read_text` and no-clobber `download`. Every content read obtains
a fresh authorized descriptor and requests exact revision/digest bytes from the
backend. Streams verify identity headers, MIME, identity content encoding, total
length and SHA256; chunks are provisional until EOF. Timeout/close never cancels
the remote run. The transport closes active responses, including those created
using an injected/borrowed HTTP client, without closing the borrowed client.

Downloads write a private sibling temporary file and link it into place only
after complete verification and fsync. Existing destinations and symlinks are
not overwritten. Failed/interrupted downloads remove their scratch file. Text
reads return complete verified UTF-8 only; oversize/non-text outputs are refused.
There is no empty-success fallback, raw signed download URL, automatic resume,
image rendering or implicit filesystem path from a server filename. Async HTTP
is native; local file reads/writes are synchronous, bounded by transfer chunks.

`client.swarms.files` provides project-scoped `list/get`, plus
`files.uploads.prepare/get` handles with `status/transfer/finalize`. `FileSource`
accepts an explicit MIME and bounded regular-file source. File descriptors are
opened without following final symlinks where O_NOFOLLOW exists; non-regular
files are refused. Transfers share the credential-isolated, host-allowlisted
upload implementation with Index through structural protocols. Finalization
seals the exact manifest; only backend-ready immutable `FileRef` values can be
selected as run inputs. PDF/image upload support means stored bytes, not a claim
that every worker or model consumes them. Backend qualification remains required.

`MutationOptions(idempotency_key=...)` lets callers retain a stable key while the
SDK derives the exact payload digest and route. Existing full `identity=...`
recovery remains supported. Supplying both is refused. MCP mutation tools require
a key so disconnect/retry does not silently invent a new logical mutation.

The optional `synth-ai[mcp]` package exposes one stdio entrypoint and hosted
Streamable HTTP adapter with one typed operation catalog. See
[SDK_0230_MCP.md](SDK_0230_MCP.md) for implemented tools, principal/authorization
boundaries, build artifacts and the explicitly pending native auth qualification.
The backend integration owns current grants, revocation, resources, billing and
service execution; this package does not introduce an alternate policy engine.

The test engineer completed frozen Index verification at testing commit
`14d63907`; installed 78-case coverage passed. Native probes on both slots failed
because all 30 candidate routes/contract negotiation are absent. Twenty runtime
preflight blockers remain. That handoff is complete; backend adoption and native
acceptance are not. No test reruns or slot mutations are part of this slice.
