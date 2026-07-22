# Synth SDK Research migration and refactor plan

- **Status:** engineer review required — unmerged implementation candidate exists
- **Date:** 2026-07-21
- **Last updated:** 2026-07-22
- **Scope:** `synth-ai`, `backend`, and `evals`
- **Quality bar:** `strong_option`
- **Primary public entrypoint:** `SynthClient().research`

## Review packet

This is the single review document for the Research SDK consolidation. It
combines the original cross-repository refactor guide with the subsequent SDK
benchmark, Synth Style type critique, post-migration failure analysis, and the
repeatable Jesterky quality rubric. Earlier working notes are evidence for this
document, not parallel implementation plans.

The document deliberately separates three kinds of statement:

| Label | Meaning | Reviewer action |
|---|---|---|
| **Constraint** | Existing ownership or authority decision that this refactor must preserve | Reject changes that violate it |
| **Proposal** | Recommended product/API/package design | Approve, amend, or reject explicitly |
| **Evidence** | Current-tree measurement, external SDK comparison, or quality proof | Verify material claims and use them to judge the proposal |

An implementation candidate now exists, but it is not an approved final API.
The reviewing engineer must use [Part VII](#part-vii--engineer-review-and-handoff)
to compare the landed shape with the product and protocol decisions here,
resolve the remaining gaps, and amend this plan before integration.

### Consolidated source map

This document supersedes the separate working notes. It preserves their useful
content as named sections rather than asking the reviewer to reconcile several
documents:

| Source material | Consolidated here |
|---|---|
| Original `synth-ai` / `backend` / `evals` refactor guide | Parts I–IV: authority, target graph, source dispositions, public API |
| Tinker, OpenAI, Claude, and Modal SDK review | Part V: comparative guarantees and post-migration failure analysis |
| Synth Style type critique | Parts I, IV, and V: typed boundaries, invalid states, failures, concurrency, naming |
| Repeatable Gemini/Jesterky proposal | Part VI: 11-dimension rubric, thresholds, evidence selection, scan log |
| Phase 0–9 implementation notes | Parts VI–VII: ratchets, commits, deterministic evidence, open defects |
| Engineer handoff | Part VII: review order, decision table, exact worktrees, integration constraints |

The original attachment that established the dedicated worktree and mandatory
before/after scan is implementation provenance, not a second plan. The
capability ledger, OpenAPI artifact, quality manifests, and Jstack goal record
are evidence attachments to this document.

### Executive handoff snapshot

| Area | Current checkpoint | Review implication |
|---|---|---|
| `synth-ai` | Clean candidate at `320be4ba` with projects/swarms/factories, typed failures/events, native async, activity, transcripts, workspace inputs, and project repository/dataset APIs | Review the product/API contract in the candidate; do not equate source structure with release readiness |
| `backend` | Clean candidate at `64361869d`; the source allowlist has 64 operations and a fresh temporary export has 197 schemas, but the checked-in and vendored artifacts still contain only 49 operations / 155 schemas | Treat contract publication as stale until the backend artifact and SDK vendor are regenerated atomically and parity is re-proved |
| `evals` | Clean candidate at `5301844d` with no deep imports and typed activity, transcript, usage, configuration, and durable-evidence consumers | Fix five active workspace call sites across ReportBench, SwarmBench, and SwarmGameBench that target nonexistent stable methods, then prove representative workflows rather than relying on source inspection |
| Compatibility | `synth_ai.managed_research` is 150 generated warning-only re-export files; 117 implementation files remain temporarily under `core/research/_legacy` | Enforce the 0.18.0 / no-earlier-than-2026-09-01 deletion contract and prevent stable imports from loading `_legacy` |
| Public surface | Candidate discovery exposes `projects`, `swarms`, and `factories`; operator breadth is grouped under `advanced`; old `Research*`, `runs`, and `smr_*` names are compatibility-only | Finalize the smallest hero lifecycle, the stable/advanced line, and alias visibility before release |
| Resources | Workspace inputs and project repository/dataset contracts exist end to end; Environment catalog is backend-only; image releases, DevEnvironments, and cloud deployments remain ungraduated | Settle resource taxonomy and batching semantics before adding another broad resource namespace |
| Delivery | Source discovery finds 46 allowlisted stable noun-first MCP definitions and 245 advanced definitions; the ledger does not yet contain runtime invocation receipts for either set; CLI exposes resolved configuration, typed usage, durable evidence, activity, transcript, and project data | Approve adapter scope and require every advertised stable tool to conform to the same operation/error contract at runtime |
| Quality | Baseline 5.64 → relocation 6.45 → first candidate 7.09 → import-isolated candidate 7.09 → strongest accepted repeat 7.27; post-usage verification 7.18 with minimum 6 and no holds | v11 predates the later activity/transcript/workspace/project-data changes and is not current-head proof; rerun only after review decisions and contract synchronization |

The implementation worktrees and exact review order are recorded in
[Current handoff state](#current-handoff-state). This document is the only
migration plan; scan manifests and generated ledgers are evidence attachments,
not competing plans.

### What is already constrained

- `synth-ai` owns the Python SDK, its public Research facade, transport,
  contracts, errors, lifecycle handles, CLI adapter, and MCP adapter.
- `backend` remains the authority for routes, state machines, policy,
  persistence, orchestration, billing, and generated service contracts.
- `evals` owns benchmark definitions, task setup, scoring, experiment policy,
  and GameBench-specific behavior. General SDK behavior may not depend on it.
- `synth_ai.managed_research` and the standalone `managed-research` package are
  temporary compatibility surfaces, not places for new product authority.
- The customer entrypoint remains `SynthClient().research`; compatibility
  imports must resolve to the same implementation rather than a second client.
- OpenCode-specific product code does not belong under `synth_ai`. Skills live
  under the repository-level `skills/` tree, and benchmark-specific code lives
  in `evals` unless it is genuinely general SDK functionality.
- The refactor may not bypass backend authority or add persistence, policy, or
  eval orchestration to the SDK.

### What this review must decide

- Whether **project → swarm → result/event** is the irreducible customer loop.
- Whether factories are a first-class public resource in the initial stable
  surface or a later/advanced capability.
- Which backend operations are public, operator-only, adapter-only, or removed.
- The exact public noun and verb map, including the `run` → `swarm` cutover.
- The authoritative contract artifact and generation/conformance workflow.
- The native sync/async, streaming, cursor, cancellation, and deadline model.
- Workspace upload semantics above the backend's 100-file request limit,
  including partial completion and idempotency.
- The distinction between bootstrap source repositories, project external
  repositories, Environment catalog entries, operator DevEnvironments, and
  admitted image releases.
- Whether the customer-stable observability set needs one small authoritative
  swarm-status projection and a binary workspace-archive operation, while raw
  logs/traces and giant poll-summary payloads remain advanced or disappear.
- The supported/deprecated/removed schedule for every legacy import and entrypoint.
- Whether the release thresholds and proof matrix in Part VI are sufficient.

### Recommended review order

1. Approve constraints and product nouns in Parts I and IV.
2. Review the target dependency graph in Part II.
3. Verify every source disposition in Part III against current consumers.
4. Use Part V to challenge the proposed API—not merely the directory layout.
5. Amend and approve the vertical migration sequence and release gates in Part VI.
6. Record final decisions and owners in the Part VII handoff tables.

## Part I — Decision and boundaries

### Decision summary

Consolidate reusable Python SDK implementation under `synth_ai/core`, organized
by explicit concerns. Keep `synth_ai.research` as the small, documented public
facade whose first-class product nouns are **projects**, **swarms**, and
**factories**. Convert `synth_ai.managed_research` into a time-bounded
compatibility package and then remove it.

This is not permission to copy backend or eval implementation into the SDK.
The ownership boundary remains:

```text
backend
  authoritative routes, policy, state machines, persistence, billing,
  resource admission, orchestration, and projections
       |
       | versioned OpenAPI + stable operation/error/state vocabularies
       v
synth_ai/core
  typed contracts, boundary codecs, transport, client operations,
  lifecycle handles, polling/streaming, and error classification
       |
       +--> synth_ai.research       public Python facade
       +--> synth_ai.cli            terminal adapter
       +--> synth_ai.mcp.research   MCP adapter
       |
       v
evals
  benchmark definitions, task materialization, scoring, acceptance policy,
  evidence evaluation, and experiment orchestration
```

The core package provides one correct typed path for each supported operation.
It must not become a second backend, an eval framework, or a flat utilities
graveyard.

### Why this migration is necessary

The June 11 package consolidation moved the full Managed Research SDK into
`synth-ai`, but it preserved its old structure. A second public facade was then
placed in front of it. Today:

- `synth_ai.managed_research`, `synth_ai.research`, and `synth_ai.core` contain
  approximately 66,650 lines across 185 Python files.
- `synth_ai.managed_research.models` contains approximately 23,812 lines;
  `sdk` contains 20,046; and `mcp` contains 12,298.
- `synth_ai.research` adds another 4,181 lines, mostly wrapping or aliasing the
  Managed Research implementation.
- `synth_ai.managed_research.sdk.client.ManagedResearchClient` is 6,207 lines.
- The backend has approximately 483 research-related route operations across
  49 active route modules, but only 11 operations currently have stable public
  operation IDs in `openapi_contract.py`.
- Active eval code still imports `synth_ai.managed_research`, deep model
  modules, `sdk.client`, `sdk.runs`, and internal launch enums directly.
- SDK, MCP, eval harnesses, and backend-hosted MCP therefore bind to different
  levels of the same legacy tree.

The result is two apparent public authorities, a monolithic client, duplicated
domain facades, deep consumer imports, and no mechanically complete statement
of which backend capabilities are supported public SDK operations.

### Governing decisions and constraints

This proposal preserves the existing architecture decision recorded in
`Jstack/.jstack/records/decisions/synth-ai/2026-06-11-retire-managed-research-sdk-package.md`:

- `synth-ai` owns Python SDK, client, MCP, models, transport, and schemas.
- Backend routes and OpenAPI remain authoritative.
- The standalone `managed-research` distribution is only a compatibility shim.
- `SynthClient().research` is the public Python story.

It also applies the July 8 product decisions and the July 20 public-noun cutover:

- Open Research is frontend-only. The SDK's
  `synth_ai.managed_research.open_research` package is not migrated into core;
  it is removed.
- Managed Research (SMR) is the product umbrella. Its first-class public
  capabilities are **Managed Factories** and **Managed Swarms**, plus supporting
  nouns such as **projects**, limits, economics, and evidence.
- A **swarm** is one bounded multi-agent execution. It is the first-class
  synth-ai public noun for launch, wait, control, and readout.
- **`run` / `run_id` are internal and wire terms.** The backend protocol and
  OpenAPI may keep run vocabulary; the public SDK, CLI, MCP, and docs must use
  `swarm` / `swarm_id`. `research.runs` and `run_id=` remain temporary
  compatibility aliases only.
- `Effort` is the approved public noun for the hosted Factory unit of work
  (it is also the canonical cloud noun; `Stack` is deprecated as of
  2026-07-20). Factories produce and manage Efforts. Do not create a new
  public synonym during this refactor.
- Prefer `evidence`, `evidence packet`, `artifact`, and `usage record` according
  to meaning. Do not introduce new public `receipt` names.

### Synth Style interpretation

The migration follows these rules:

1. **One authority.** Backend owns truth. Core decodes and invokes backend
   contracts; it does not reimplement eligibility, billing, lifecycle, or
   projection policy.
2. **Push interface complexity inward.** Customers express intent through a
   small `ResearchClient`; server-owned defaults and algorithm internals remain
   hidden.
3. **General foundations, targeted affordances.** HTTP, errors, pagination,
   streaming, idempotency, deadlines, artifacts, and resource identities are
   general core foundations. Research-specific operations compose them.
4. **Sparse interconnects.** Backend-to-SDK coupling is a versioned contract,
   not imports of backend code. Public facade-to-core coupling is a small set of
   typed clients and models.
5. **Clear noun hierarchies.** Parent noun is Research; first-class children are
   Factories and Swarms (with Projects and supporting domains). Prefer
   `swarm` over `run` in every public Python name, package path, and docs
   surface. Avoid `helpers.py`, `types.py`, and another all-purpose client.
6. **Parse once at the boundary.** Wire mappings become typed models inside the
   transport boundary. Domain code uses attributes, enums, and exhaustive
   matches—not `.get()` shape probing.
7. **No fallback authority.** No route-prefix guessing, legacy-route retry,
   database fallback, raw Redis access, or alternate persistence path.
8. **Precise errors.** Every external failure has a stable code, category,
   operation, request context, and causal exception. Human messages are not
   classifiers.
9. **Correct time.** UTC for wire timestamps; monotonic deadlines for waits,
   retries, and stream reconnects.
10. **Concern-based modules.** Core is a package root, not a god module. Files
    split on trust and ownership boundaries, not arbitrary line limits.

## Part II — Target architecture

### Target package structure

```text
synth_ai/
  client.py                         # SynthClient composition only
  research/                         # documented public facade
    __init__.py                     # public exports
    client.py                       # small intent-level facade
    models.py                       # intentional public aliases
    errors.py                       # intentional public aliases

  core/
    README.md                       # boundary and dependency rules
    auth/
      credentials.py               # explicit credential resolution
      context.py                   # typed auth/tenant context
    http/
      transport.py                 # sync transport
      async_transport.py           # true async transport
      request.py                    # HttpRequest and operation metadata
      response.py                   # typed decode boundary
      retry.py                      # one classified retry policy
      errors.py                     # HTTP-to-Synth error mapping
    contracts/
      json_value.py                 # recursive typed JSON only where open data is real
      pagination.py
      errors.py
      resources.py
      artifacts.py
      usage.py
    research/
      README.md
      client.py                     # composition root, not operation soup
      context.py                    # ResearchContext and identifiers
      operations.py                 # stable operation registry
      contracts/                    # generated/maintained wire contracts
        projects.py
        swarms.py                   # public swarm models; decode wire run fields
        factories.py
        economics.py
        resources.py
        artifacts.py
        collaboration.py
      projects/
        client.py
        handles.py
      swarms/                       # first-class public domain (not runs/)
        client.py
        handles.py
        lifecycle.py
        readouts.py
        streams.py
      factories/
        client.py
        handles.py
      economics/
        client.py
      artifacts/
        client.py
        manifests.py
      resources/
        client.py
        environments.py
        images.py
        deployments.py
      collaboration/
        client.py
        messages.py
      admin/                         # explicitly unstable/operator-only
        client.py

  mcp/
    research/                       # thin delivery adapter over core operations
      server.py
      registry.py
      tools/

  cli/                              # thin terminal adapter over public/core clients

  managed_research/                 # temporary compatibility package only
    __init__.py
    models/
    sdk/
    mcp/
```

#### Dependency rule

Imports point inward exactly once:

```text
synth_ai.client -> synth_ai.research -> synth_ai.core.research
synth_ai.cli -------------------------> synth_ai.core / public facade
synth_ai.mcp.research ----------------> synth_ai.core.research
synth_ai.managed_research ------------> synth_ai.core.research
```

Code under `synth_ai/core` may not import from `synth_ai.research`,
`synth_ai.managed_research`, `synth_ai.cli`, or `synth_ai.mcp`. A checker must
enforce this rule.

## Part III — Cross-repository migration inventory

### What moves from each repository

#### From `synth-ai`

| Current source | Target | Treatment |
|---|---|---|
| `managed_research/auth.py`, `sdk/config.py` | `core/auth`, `core/http` | Merge into one credential and endpoint authority |
| `managed_research/transport/*`, `sdk/transport.py`, `sdk/_base.py` | `core/http` | Replace duplicate transports with one typed transport |
| `managed_research/errors.py` | `core/contracts/errors.py`, `core/http/errors.py` | Split domain errors from delivery mapping |
| `managed_research/models/*` | `core/research/contracts/*` | Group by domain; remove `types.py` and giant export files incrementally |
| `managed_research/sdk/client.py` | `core/research/client.py` plus domain clients | Decompose by operation ownership; no compatibility methods in core |
| `managed_research/sdk/runs.py` and `_run_authority_mixin.py` | `core/research/swarms/*` | One swarm lifecycle, handle, readout, and stream hierarchy; map wire `run` fields at the codec boundary |
| `research/swarms.py`, `swarm_readouts.py` | `core/research/swarms/*` plus thin facade | Keep `research.swarms` as the documented public surface |
| `research/runs.py`, `run_readouts.py` | compatibility aliases only | Deprecated `research.runs` → same objects as `research.swarms` |
| `managed_research/sdk/project.py`, `projects.py` | `core/research/projects/*` | One project client and project-bound handles |
| Factory, billing, artifact, resource SDK modules | matching `core/research/*` domains | Preserve typed operations; remove parallel wrapper methods |
| `research/*` | small facade over core | Retain customer nouns (`projects`, `swarms`, `factories`); delete duplicated implementation |
| `managed_research/mcp/*` | `synth_ai/mcp/research/*` | MCP only binds schemas to core operation calls |
| `managed_research/open_research/*` | nowhere | Delete per frontend-only product decision |
| `managed_research/factory_standup.py` | CLI/evals or delete | Keep only if it is a general customer operation; benchmark/orchestration policy belongs in evals |
| `managed_research/schemas/smr_openapi.yaml` | generated contract input | Replace whole-backend snapshot with a public research contract artifact |

The current `synth_ai/sdk` container, tunnel, and pool surfaces should reuse the
same `core/http`, auth, error, pagination, and resource primitives. They should
not be nested beneath `core/research`.

#### From `backend`

Backend implementation does not move. The migration extracts only explicit
cross-boundary contracts:

- Stable public operation IDs and paths.
- Request and response schemas.
- Closed enums for state, kind, role, lifecycle, denial, and error codes.
- Idempotency requirements and mutation semantics.
- Pagination, streaming, cursor, and freshness contracts.
- Error categories and retryability metadata.
- Artifact, usage, resource, and authority identities.

The following remain exclusively in backend:

- FastAPI routers and dependency injection.
- Database and Redis access.
- Lifecycle, finalization, execution, and billing authorities.
- Resource admission, leases, scheduling, and reapers.
- Runtime actors, workers, sandboxes, and provider credentials.
- Projection builders and persistence models.
- Policy for launch eligibility, budgets, privacy, publication, and grading.

Backend must publish a bounded public research OpenAPI artifact instead of
forcing the SDK to vendor every internal SMR route. Each supported SDK
operation requires a stable operation ID. Internal-only routes remain in the
backend schema but outside the SDK input artifact.

#### From `evals`

Move code from evals into core only when it is a general client capability for
an authoritative backend route and is useful outside a benchmark. Candidates
include:

- Typed builders for canonical project/swarm launch intent that currently live
  in smoke drivers.
- General wait, stream, artifact download, evidence readout, and cleanup
  operations that currently duplicate SDK calls.
- Typed environment, image, repository, and workspace-input request models
  where backend already owns the corresponding route.
- General source/artifact identity types when they are part of the backend API.

Keep these in evals:

- Benchmark inventories, lanes, tasks, suites, and frozen seeds.
- Scoring, grading, thresholds, held-out policy, and candidate selection.
- GameBench and other benchmark-specific scorer clients and models.
- Experiment orchestration and matrix runners.
- Eval artifact rendering and acceptance decisions.
- Local fixture materialization and benchmark environment code.
- Evidence interpretation that decides pass/fail for a benchmark.

After migration, evals may construct public SDK request types and call public
SDK operations. It must not import `synth_ai.core` or compatibility modules
directly.

## Part IV — Public product and contract model

### Capability model

Every existing route, SDK method, MCP tool, and eval helper must enter one row
in a migration ledger with exactly one disposition:

| Disposition | Meaning |
|---|---|
| `public` | Stable customer operation exposed through `SynthClient().research` |
| `advanced` | Supported operator operation, explicitly labeled unstable |
| `internal` | Backend-only; no Python SDK exposure |
| `frontend` | Open Research/public-web projection only |
| `eval` | Benchmark or acceptance logic owned by evals |
| `retire` | Legacy, duplicate, unused, or superseded operation |

No operation is migrated merely because it exists. This prevents the current
483-route backend surface from becoming a 483-method public SDK.

#### Proposed public capability groups

1. **Projects:** create runnable project, retrieve/list, setup, workspace,
   repositories, code source, datasets, objectives, and project-bound swarms.
2. **Swarms:** launch, retrieve/list, wait, pause/resume/cancel, branch,
   checkpoints, tasks, actors, messages, timeline, logs, traces, runtime state,
   and terminal outcome. Public identifiers use `swarm_id`; wire `run_id` is
   decoded at the transport boundary and exposed only via temporary aliases.
   Launch parameters — including model/harness overrides, roles, limits,
   images, and kickoff — live on a single typed swarm creation request (see
   below), not on a separate overrides API.
3. **Evidence and artifacts:** manifests, content, work products, trained
   models, hosted artifacts, evaluations, source identity, and lineage.
4. **Factories:** factories, projects, candidates, results, champion state,
   messages, wake operations, and **Efforts** — the hosted Factory unit of
   work, exported under that approved noun. Factories compose or graduate
   swarms and Efforts; they do not reintroduce `run` as a public product noun.
5. **Economics:** preflight, limits, budgets, usage, costs, allowance, and
   typed denial details.
6. **Resources:** environments, dev environments, runtime images, image
   releases, cloud deployments, claims, readiness, and cleanup evidence.
7. **Collaboration:** objectives, milestones, tasks, interactions, message
   threads, and operator evidence where these are public product contracts.

### Core contract patterns

#### Operation identity

Each operation has a closed identity, not a string assembled at call sites:

```python
class ResearchOperation(StrEnum):
    CREATE_PROJECT = "create_project"
    CREATE_SWARM = "create_swarm"
    RETRIEVE_SWARM = "retrieve_swarm"
    CANCEL_SWARM = "cancel_swarm"
```

Operation metadata binds method, canonical path template, request type,
response type, mutation/idempotency semantics, and allowed error codes. Path
templates may still contain backend wire segments such as `/runs` while the
public operation and Python names remain swarm-first. Path aliases and prefix
probing are forbidden.

#### Context and identifiers

Loose strings become explicit immutable types at the boundary:

```python
@dataclass(frozen=True, slots=True)
class ResearchContext:
    organization_id: OrganizationId
    project_id: ProjectId
    swarm_id: Optional[SwarmId]

@dataclass(frozen=True, slots=True)
class SwarmRef:
    project_id: ProjectId
    swarm_id: SwarmId
```

Wire payloads that carry `run_id` decode into `SwarmId` once at the HTTP
boundary. Optional values represent actual domain absence, not fallback paths.
Mutation operations take explicit idempotency keys or typed request identities.
Public method kwargs use `swarm_id=`; `run_id=` is accepted only as a
deprecated alias during the compatibility window.

#### Swarm launch request (typed params, not free-form overrides)

Customer launch intent is one typed request object. Overrides are fields on
that request — there is no `research.overrides` namespace.

```python
research.swarms.create(
    project_id,
    request=ResearchSwarmLaunchRequest(
        work_mode=ResearchWorkMode.DIRECTED_EFFORT,
        timebox_seconds=3600,
        limit=UsageLimit(...),
        roles=ResearchRoleBindings(...),
        actor_model_overrides=(
            ResearchActorModelAssignment(
                actor_type=ResearchActorType.WORKER,
                actor_subtype=ResearchWorkerSubtype.ENGINEER,
                agent_model=ResearchAgentModel.GPT_5_6_LUNA,
                agent_harness=ResearchAgentHarness.CODEX,
            ),
        ),
        actor_image_overrides=...,
        kickoff_contract=...,
    ),
)
```

Public launch fields must be closed catalogs or nested dataclasses, synced from
backend-authored policy — not open `dict[str, Any]` / `WireMapping` escape
hatches. At minimum the swarm create contract exposes typed selectors for:

| Concern | Public type (illustrative) | Notes |
|---|---|---|
| Agent model | `ResearchAgentModel` (`StrEnum`) | Closed catalog; actor-policy may further restrict by role/subtype |
| Agent harness | `ResearchAgentHarness` (`StrEnum`) | e.g. `codex`, `opencode_sdk` |
| Role policy | `ResearchRoleBindings` / `ResearchRoleBinding` | Per-role model, harness, params, provider |
| Actor model override | `ResearchActorModelAssignment` | Actor type/subtype + model + optional harness |
| Actor image override | typed image-release bindings | Admitted release IDs only; never raw Docker tags |
| Work mode / host / runtime / env | closed enums | Project + launch selection |
| Runbook | `ResearchRunbookPreset` / runbook kind | Named presets from backend |
| Usage / timebox | `UsageLimit`, `timebox_seconds`, run policy | Complements project economics |
| Execution target | typed execution-target union | e.g. `platform_resolved` for slot launches |
| Kickoff | typed kickoff / kickoff-contract models | Tasks, required work products |

Project setup may still bind durable defaults (for example `agent_profiles`).
Per-swarm deviations belong on `ResearchSwarmLaunchRequest`. Preflight and
create take the same typed request (or its `to_client_kwargs()` projection)
so evals and customer code share one validation path.

Today's `ResearchRunLaunchRequest` and related `Smr*` / Research aliases are
the starting point. Migration renames the public noun to swarm, moves the
types under `core/research/contracts`, and ratchets away loose mappings on
public fields until the SDK answer to "what can I override to?" is
autocomplete + validation against backend catalogs.

#### Errors

Use one hierarchy across sync SDK, async SDK, CLI, and MCP:

```text
SynthError
  AuthenticationError
  AuthorizationError
  ValidationError
  ConflictError
  ResourceExhaustedError
  RateLimitedError
  TransientServiceError
  ContractMismatchError
  ResearchOperationError
```

Every error carries a stable `error_code`, operation, HTTP status when
applicable, request/correlation identifiers, retryability, and human detail.
CLI and MCP translate these errors at their delivery edges without changing
their classification.

#### Sync and async

Sync and async clients share contracts and operation definitions, not event
loop tricks. The async transport is genuinely async. The sync transport is
genuinely sync. Both use the same codecs and error mapping.

#### Waits, streams, and retries

- Waits take explicit monotonic deadlines and polling policies.
- Streams expose typed cursors, event enums, reconnect limits, and terminal
  semantics.
- Retry policy is centralized and permits only classified transient failures.
- Cancellation always propagates truthfully.
- Durable mutations are retried only with idempotency proof.

## Part V — External benchmark and failure analysis

### External SDK benchmark

This migration is not judged only against the current repository. It is judged
against four unusually strong Python SDKs that solve adjacent problems:

- **Tinker** for a small set of research primitives and an explicit split
  between core service operations and a higher-level cookbook.
- **OpenAI Responses and Agents SDKs** for the boundary between raw model work,
  managed agent loops, typed semantic events, resumable state, and progressive
  documentation.
- **Claude Agent SDK and Claude Managed Agents** for versioned agent
  configuration, environment/session separation, event-driven lifecycles, and
  migration guidance.
- **Modal** for Python-native resource nouns, lazy remote handles, related
  local/remote/background/parallel invocation modes, deployment versioning, and
  unusually explicit runtime semantics.

The comparison was refreshed on 2026-07-21 from first-party documentation. It
is an API-design review, not a claim about vendor reliability, model quality,
price, or production uptime. Scores below are deliberately demanding and are
useful only as a consistent rubric for this migration.

#### Evaluation rubric

| Dimension | What good looks like |
|---|---|
| Conceptual compression | A user can predict the resource hierarchy and complete the hero workflow after learning a few nouns. |
| Naming | Names are domain nouns and precise verbs, without redundant package prefixes, transport vocabulary, or internal implementation terms. |
| Parameter design | Requiredness is honest; mutually exclusive states are modeled; configuration is grouped by concern; arbitrary mappings are rare. |
| Type contract | Inputs, outputs, errors, states, events, and cursors are closed and statically useful across the public path. |
| Lifecycle guarantees | Creation, readiness, terminality, cancellation, cleanup, version pinning, and consistency are documented and mechanically represented. |
| Concurrency | Sync, async, background work, polling, streaming, resume, cancellation, and deadlines form one coherent model. |
| Failure contract | Typed errors, stable codes, request IDs, retryability, backoff, timeouts, and partial-success behavior are explicit. |
| Observability | A user can correlate requests, events, logs, usage, artifacts, traces, and server-side work without raw transport access. |
| Documentation | A five-minute quickstart grows into concepts, recipes, API reference, migration guides, limits, and troubleshooting without contradiction. |
| Compatibility | Versioning, deprecation, generated contracts, release notes, and server/client compatibility are public and enforceable. |
| Reach | The contract is usable outside one Python implementation, normally through generated SDKs or a stable protocol. |

#### Tinker: narrow primitives and an honest abstraction boundary

Tinker's strongest decision is scope. `ServiceClient` creates concern-specific
`TrainingClient`, `SamplingClient`, and `RestClient` objects. The training loop
is visible in a handful of verbs: `forward_backward`, `optim_step`,
`save_state`, and `load_state`; sampling is `sample`. The user owns data,
rendering, loss choice, evaluation, and loop policy, while the service owns
distributed training and reliability. The official cookbook then adds
training pipelines, environments, renderers, evaluation, and recipes without
pretending those abstractions are server primitives.

The parameter names are usually operationally precise. In particular,
`create_training_client_from_state` and
`create_training_client_from_state_with_optimizer` make checkpoint semantics
visible in the method name. `APIFuture[T]` gives each submitted operation one
handle that can be awaited or resolved synchronously with `result(timeout=...)`.
The progressive tutorial track moves from API basics through core concepts,
cookbook abstractions, advanced methods, and deployment; tutorials also state
the minimum SDK version they require.

Tinker is not a perfect Synth Style model:

- `ServiceClient(..., **kwargs)` is an untyped extension point at the primary
  boundary.
- `create_sampling_client(model_path=None, base_model=None, ...)` expresses a
  mutually exclusive choice as two optional strings instead of a sum type.
- Base models, loss functions, and Tinker paths are mostly strings rather than
  validated identifiers or closed catalog values.
- `train_mlp`, `train_attn`, and `train_unembed` are boolean configuration
  flags that would compose better as a named adaptation policy.
- `user_metadata` and parts of loss configuration remain dictionaries.
- Sync/async method variants coexist with `APIFuture`, which creates more than
  one concurrency story.
- `RestClient` is named after delivery rather than a domain, and a deprecated
  no-op `name` parameter on a save method is an API scar.

The lesson for Synth is not to copy Tinker's classes. It is to make the hosted
research loop similarly legible and to keep batteries-included experiment
policy in `evals` or a separate cookbook layer. A customer should be able to
name the five or six core operations without seeing 100 backend capabilities.

Primary references: [Tinker ServiceClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/),
[APIFuture](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/apifuture/),
[tutorials](https://tinker-docs.thinkingmachines.ai/tutorials/), and the
[Tinker Cookbook boundary](https://tinker-docs.thinkingmachines.ai/cookbook/).

#### OpenAI: a clear product-layer choice and typed semantic events

OpenAI's strongest SDK decision is explaining which layer owns the loop. The
Responses API is the lower-level response lifecycle; the Agents SDK owns an
agent run loop. The Agents quickstart keeps the first program to `Agent` plus
`Runner.run`, returns a result containing final output and run history, and
adds function tools, specialist handoffs, sessions, guardrails, approvals, and
tracing incrementally. The documentation explicitly routes readers by intent:
define one agent, understand the runtime, add tools, add human review, inspect
results, or evaluate traces.

Long-running Responses have a concrete object lifecycle: create with
`background=True`, retrieve while queued or in progress, cancel, and inspect a
terminal response. Repeated cancellation is documented as idempotent. Streaming
uses a documented union of semantic event types rather than arbitrary event
mappings. Background streams carry sequence numbers that can act as resume
cursors. Importantly, the documentation also states the current limitation:
SDK-level stream resume support is still forthcoming, even though the protocol
supports `starting_after`. That candor is part of the quality bar.

OpenAI also has Synth Style liabilities:

- The Responses create request is a large, evolving product surface with many
  optional fields and several unions; static typing does not make the cognitive
  load disappear.
- Model IDs, prompt IDs, tool names, and metadata still use open strings or
  mappings at extensible boundaries.
- The direct API SDK and Agents SDK intentionally expose different layers, but
  users still need to understand response, run, session, conversation, and
  trace state.
- Server features can precede ergonomic SDK support, as stream resume currently
  demonstrates.

The lesson for Synth is the documented layer choice: public users need a clear
answer to “do I own the swarm loop, or does Synth?” They also need one closed
event algebra shared by Python, MCP, CLI, webhooks, and the backend—not
independent payload dictionaries for each adapter.

Primary references: [Agents SDK guide](https://developers.openai.com/api/docs/guides/agents),
[Agents quickstart](https://developers.openai.com/api/docs/guides/agents/quickstart),
[background mode](https://developers.openai.com/api/docs/guides/background),
and [typed streaming events](https://developers.openai.com/api/docs/guides/streaming-responses).

#### Claude: versioned configuration, environments, sessions, and events

Claude Managed Agents has the cleanest resource hierarchy for a managed agent
service in this comparison:

```text
Agent -> immutable versions
Environment -> sandbox/runtime policy
Session -> one agent version in one environment
Session events -> work, progress, custom tool calls, approvals, interruption
Memory store / Vault / Skill -> independently managed supporting resources
```

An agent is a reusable configuration containing model, system prompt, tools,
MCP servers, skills, and optional multi-agent declarations. Updating changed
configuration creates a version; a no-op update does not. A session can use the
latest version, pin a version, or apply explicit per-session overrides. Session
creation provisions the sandbox, while sending a user event starts work. The
docs state which fields can change during an idle session, how archive/delete
interact with running state, and which resources survive session deletion.
There is also a first-class migration guide from both a hand-written Messages
API loop and the local Claude Agent SDK.

The general Anthropic SDK adds a documented error hierarchy, request IDs on
responses, two default retries for classified transient failures, exponential
backoff, `retry-after` handling, client/request timeout overrides, and both sync
and async clients. The local Claude Agent SDK separately offers a one-shot
`query()` iterator and a bidirectional `ClaudeSDKClient`, typed message/content
classes, hooks, custom in-process MCP tools, and explicit process/CLI error
classes.

Claude's Synth Style costs are real:

- Managed Agents is a beta contract with dated beta headers. SDK convenience
  hides header plumbing but does not make the contract mature.
- `agent` can be an ID string, a pinned reference, or an override object. This
  is expressive but wide; using “latest” by default weakens reproducibility.
- The local Agent SDK wraps a bundled CLI/subprocess, so process discovery,
  connection, JSON decoding, and working-directory behavior enter the public
  failure model.
- `ClaudeAgentOptions` is a broad configuration object spanning model, prompt,
  tools, permissions, working directory, MCP, hooks, and CLI behavior.
- Hook inputs, custom tool schemas, metadata, and some content payloads remain
  dictionary-shaped.
- `allowed_tools` controls automatic permission, not tool availability; the
  name can surprise a reader who has not learned the permission system.
- “Claude Agent SDK” and “Claude Managed Agents” are adjacent but different
  products, which requires unusually careful documentation.

The lesson for Synth is durable configuration identity. A swarm must record the
exact project, factory, role/model/tool policy, environment/image, and contract
versions it ran with. A mutable “current” server configuration plus a run ID is
not a reproducible research object.

Primary references: [define an agent](https://platform.claude.com/docs/en/managed-agents/agent-setup),
[start a session](https://platform.claude.com/docs/en/managed-agents/sessions),
[session operations](https://platform.claude.com/docs/en/managed-agents/session-operations),
[Managed Agents migration](https://platform.claude.com/docs/en/managed-agents/migration),
[API error guarantees](https://platform.claude.com/docs/en/api/errors), and the
[official Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python).

#### Modal: Python-native resource composition and explicit remote semantics

Modal's public vocabulary is exceptionally economical: `App`, `Function`,
`Cls`, `Sandbox`, `Image`, `Secret`, `Volume`, `Dict`, `Queue`, `Environment`,
and `Workspace`. These nouns compose in ordinary Python. `App.function`
registers a function; the resulting `Function[P, R]` exposes visibly related
execution modes: `local`, `remote`, `spawn`, and `map`. `FunctionCall` is the
background handle. Async uses the same operation with `.aio`, so the user does
not have to discover a parallel tree of async resource clients.

Remote resources use lazy handles. `Function.from_name(...)` and
`Volume.from_name(...)` can be constructed without an immediate network call,
then hydrate when used; `.hydrate()` is explicit when metadata is needed.
Deployed Functions can be version-pinned. Deployment docs state rolling versus
recreate semantics, including what “latest” can mean during a rolling cutover.
Volume docs state the consistency model directly: commits and reloads are
explicit, concurrent same-file writes are last-write-wins, and reload has
open-file constraints. The error reference maps transport status to Modal
exception types and distinguishes quota, conflict, auth, timeouts, serialization,
remote execution, and version errors.

Modal is also a warning about API growth:

- `App.function(...)` and especially `Sandbox.create(...)` have very large
  keyword surfaces containing compute, networking, security, storage,
  lifecycle, and experimental concerns. This is optional-parameter overload,
  even when the names are individually good.
- GPU, cloud, region, image commands, tags, environment variables, and
  experimental options include open strings and dictionaries.
- Arbitrary `cloudpickle` inputs make Python ergonomics excellent but weaken
  language-neutral and long-term serialization guarantees.
- A `Function.from_name(...)` lookup cannot preserve the original Python
  callable signature for type checkers; callers must narrow the result.
- The `.aio` synchronization machinery is ergonomic magic with a separate
  event-loop/thread implementation. It is better surfaced than Synth's current
  proxy, but it still has runtime and typing complexity.
- Modal's 1.0 migration guide documents prior ambiguity such as eager `lookup`
  versus lazy `from_name`; even strong SDKs accumulate duplicate paths and must
  deliberately remove them.

The lesson for Synth is to make handles do useful work. A `Swarm` handle should
carry identity and expose `status`, `events`, `wait`, `cancel`, `artifacts`, and
`usage` with one consistent sync/async story. Users should not navigate a tree
of classes named `ResearchRunEventsObjectivesAPI` to recover one execution.

Primary references: [Modal Python SDK reference](https://modal.com/docs/sdk/py/latest),
[Function invocation and version pinning](https://modal.com/docs/guide/trigger-deployed-functions),
[async usage](https://modal.com/docs/guide/async),
[Volume consistency](https://modal.com/docs/sdk/py/latest/modal.Volume),
[errors](https://modal.com/docs/sdk/py/latest/modal.exception), and the
[Modal 1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration).

#### Comparative scorecard

Scores are 1–10, equally weighted, and intentionally subjective. “Synth after
Phase 10” means the migration above is completed exactly as currently written,
without the additional product-contract requirements introduced below. It is
therefore a forecast of structural improvement, not an earned score.

| Dimension | Tinker | OpenAI | Claude | Modal | Synth now | Synth after Phase 10 |
|---|---:|---:|---:|---:|---:|---:|
| Conceptual compression | 9 | 8 | 8 | 9 | 2 | 5 |
| Naming | 8 | 8 | 9 | 9 | 3 | 6 |
| Parameter design | 7 | 8 | 7 | 6 | 2 | 6 |
| Type contract | 7 | 9 | 8 | 7 | 2 | 7 |
| Lifecycle guarantees | 7 | 8 | 9 | 9 | 3 | 6 |
| Concurrency | 7 | 9 | 9 | 8 | 2 | 6 |
| Failure contract | 6 | 8 | 9 | 8 | 3 | 7 |
| Observability | 6 | 9 | 9 | 9 | 4 | 7 |
| Documentation | 9 | 9 | 9 | 9 | 3 | 6 |
| Compatibility | 6 | 8 | 7 | 9 | 2 | 5 |
| Reach | 2 | 10 | 10 | 8 | 1 | 1 |
| **Unweighted aggregate** | **6.7** | **8.5** | **8.5** | **8.3** | **2.5** | **5.6** |

The uncomfortable conclusion is that a perfectly executed directory migration
still leaves Synth roughly three points behind the strongest references. The
gap is not polish. It is missing public contract, missing conceptual
compression, and missing guarantees.

### Where Synth falls horribly short—even after this migration

#### 1. The migration fixes ownership, not the product model

Today the customer encounters Research, Managed Research, SMR, project, run,
swarm, launch, job, Factory, Effort, Tag session, objective, milestone, task,
actor, work product, artifact, hosted artifact, visual, receipt, evidence,
environment, dev environment, image, image release, cloud deployment, and
claim. Moving each noun into a concern-based directory does not prove that the
noun belongs in the customer model.

The current public facade alone defines 60 classes named `Research*API`.
Examples include `ResearchFactoriesTagSessionsMessagesAPI`,
`ResearchRunMessageQueueInteractionsAPI`, and
`ResearchRunEventsObjectivesAPI`. These names reveal the backend route tree
instead of compressing it. Prefixing every visible type with `Research` inside
`synth_ai.research` and suffixing namespace objects with `API` adds words but
not meaning.

Phase 0 must therefore produce a customer noun map, not only an operation
ledger. Every public noun needs one sentence answering: what is it, who creates
it, what owns it, how long does it live, and why can it not be represented by
an existing noun? Any noun without a distinct lifecycle is a view or field, not
a top-level client namespace.

#### 2. There is no tiny, memorable hosted-research loop

Tinker can be explained as forward/backward, optimizer step, sample, and save.
Modal can be explained as define an App/Function and call local, remote, spawn,
or map. OpenAI Agents can be explained as define an Agent and run it. Claude
Managed Agents can be explained as define/version an Agent, choose an
Environment, create a Session, and send/receive Events.

Synth's hero example currently performs economics reads, preflight, project
setup, a run launch, polling, readouts, billing drawdown, and cleanup through
different namespace layers. The migration still does not declare the smallest
irreducible loop. The target should be teachable as:

```python
project = client.research.projects.create(ProjectCreate(...))
swarm = project.swarms.create(SwarmCreate(...))

for event in swarm.events.stream(after=cursor):
    render(event)

result = swarm.wait(deadline=deadline)
```

Top-level lookup can coexist for reattachment:

```python
swarm = client.research.swarms.get(swarm_id)
```

Economics, limits, images, collaboration, and evidence remain discoverable, but
they cannot be prerequisites for understanding the four-line loop. Preflight
that only restates server validation should disappear into `create`; preflight
that reserves resources or incurs cost must be named as such.

#### 3. Parameter requiredness and invalid states are not trustworthy

The current facade still accepts `Mapping[str, Any]`, `dict[str, Any]`, and
`**kwargs: Any` at customer boundaries. A source snapshot on 2026-07-21 found
117 occurrences of those broad shapes under `synth_ai/research` and 622 broad
mapping/`Any` return annotations across the Research facade and managed SDK.
Several methods accept a typed request *or* a mapping, which makes the typed
path advisory rather than authoritative.

The migration's proposed dataclasses help, but Synth Style requires more:

- Mutually exclusive choices become tagged unions or distinct constructors,
  not multiple optional fields.
- Required server identifiers use nominal types (`ProjectId`, `SwarmId`,
  `FactoryId`) rather than interchangeable `str` values.
- Model, harness, role, runtime, and image selection use typed catalog
  references with explicit escape hatches for forward-compatible values.
- Durations use `timedelta` or named duration types internally; public numeric
  seconds are not mixed as `timeout`, `timeout_seconds`, `poll_seconds`, and
  `poll_interval` without a standard.
- Creation intent, server state, update patches, and read projections are
  different types. One giant model must not serve all four roles.
- Booleans that alter lifecycle or authority become explicit policy enums when
  there are more than two meaningful states.
- Open metadata is `Mapping[str, JsonValue]`, never `Any`, and is confined to a
  documented metadata field.
- Compatibility mappings are parsed by isolated shim functions and never
  accepted by the new public signature.

#### 4. The async API is not an async contract

`AsyncResearchClient` currently proxies arbitrary attributes through
`__getattr__`, types everything as `Any`, and runs sync calls with
`asyncio.to_thread`. This prevents useful autocomplete, hides which operations
are actually awaitable or streaming, consumes threads for network waits, and
cannot provide principled cancellation or backpressure. It is explicitly
documented in source as a parity stub.

Phase 1's true async transport is necessary but insufficient. Sync and async
must be generated or implemented from the same operation specifications and
return the same domain models. Every long-running operation needs exactly one
documented relationship among:

- immediate create response;
- durable handle identity;
- polling snapshot;
- typed event stream;
- resume cursor and event ordering;
- cancellation/interrupt acknowledgement;
- terminal result;
- cleanup/archive/delete.

Modal's `.aio` syntax is not required, but its parity is. Tinker's future model
is not required, but its single durable abstraction is. A second hand-written
async namespace tree will drift; a dynamic `Any` proxy has already surrendered.

#### 5. Lifecycle and consistency guarantees are scattered implementation facts

The repository contains good local behaviors—monotonic polling deadlines,
some idempotency keys, SSE cursors, typed denial errors, and operation-specific
reconciliation—but no single public guarantee matrix says which operations
have them. Defaults vary by concern and include 120-second, 600-second, and
1,800-second timeout conventions. Idempotency appears on selected creates and
MCP tools, sometimes with compatibility aliases. Event and projection freshness
rules are route-specific.

Before Phase 3, every public operation needs a contract row containing:

| Guarantee | Required declaration |
|---|---|
| Idempotency | key location, deduplication scope, retention window, replay response, and whether the SDK generates a key |
| Retry | safe status/error codes, maximum attempts, backoff/jitter, `retry-after`, and whether a timed-out write may have committed |
| Timeout | connect/read/write/pool or operation deadline semantics; whether server work continues after client timeout |
| Ordering | event ordering key, sequence behavior, duplicate policy, and gap detection |
| Resume | cursor type, cursor retention, stale-cursor error, and snapshot/replay relationship |
| Consistency | authoritative read, projection lag behavior, freshness metadata, and read-your-write guarantee |
| Cancellation | requested versus acknowledged versus terminal state; idempotency and race semantics |
| Cleanup | archive/delete/release ownership, cascade behavior, irreversibility, and retained evidence |
| Version | API contract, SDK, agent/config, image/environment, and model/catalog versions captured by the resource |
| Failure | stable error code/category, retryability, request ID, operation ID, and partial-result fields |

Without this matrix, `core/http/retry.py` merely centralizes guesses.

#### 6. Swarms are not reproducible research objects

Claude versions agent definitions; Modal versions deployments and supports
version-pinned Function handles. The candidate now exposes a dedicated
`retrieve_swarm_configuration` operation: the backend loads the exact durable
run-bound configuration version, secret-redacts it at the public boundary,
hashes the returned projection, and the SDK recursively freezes it. The review
must still verify that the snapshot covers every effective value after server
defaults and policy resolution:

- public contract and server build version;
- project/repository/workspace input revisions;
- Factory and policy revision;
- model/provider/catalog identities and effective model parameters;
- role, harness, tool, MCP, skill, and permission configuration;
- environment, image, dependency, and secret *references* (never secret
  values);
- limits, budget, admission decision, and resource allocation;
- parent/fork/replay provenance;
- artifact/evidence schema versions.

The snapshot is immutable, typed, serializable, hashable, and retrievable. A
replay either pins it or returns a typed incompatibility. “Uses whatever the
project currently says” is not sufficient for research, evaluation, incident
analysis, or billing disputes.

#### 7. Events are payloads, not a closed protocol

OpenAI documents a semantic streaming union. Claude treats events as the
session protocol. Synth currently has multiple event, transcript, runtime
message, message queue, snapshot, progress, objective, task, actor, and timeline
views, many returning dictionaries. These projections may all be useful, but
they do not form one exhaustive algebra.

Core needs a versioned `SwarmEvent` tagged union whose variants include stable
identity, swarm ID, monotonically comparable cursor/sequence, server timestamp,
causation/correlation IDs, schema version, and typed payload. Snapshot APIs
must state how they relate to the event log. Unknown future events decode to a
single explicit `UnknownSwarmEvent` containing `JsonValue`, not to `Any` and not
to a false known type. MCP, CLI, webhooks, and Python consume the same variants.

#### 8. Error types are still a taxonomy, not a guarantee

The existing denial classes are a positive start, but the SDK has multiple
error layers and dictionary-shaped `detail`, `body`, and `cause` fields.
Traditional exception subclasses alone do not satisfy Synth Style when callers
still need to inspect strings or raw mappings.

Every public exception should carry a frozen typed error record:

```python
@dataclass(frozen=True, slots=True)
class SynthFailure:
    code: SynthErrorCode
    category: SynthErrorCategory
    operation: OperationId
    request_id: RequestId | None
    retry: RetryDirective
    resource: ResourceRef | None
    detail: SynthErrorDetail
```

Exception classes can remain ergonomic catch points, but code/category—not the
English message or HTTP status—drive behavior. The SDK must publish which
errors each operation can raise, preserve unknown server codes, expose request
IDs on successes and failures, and never retry a write merely because the
transport failed.

#### 9. Documentation is a generated reference shelf, not a learning system

The local docs contain useful README material, generated Research reference
pages, and two Research recipes. They do not yet match the progressive systems
of the references. The current README also demonstrates vocabulary that this
proposal retires (`research.runs`, `run_id`, and `receipt`), illustrating that
source, docs, and migration decisions are already out of phase.

The public documentation deliverable must include:

1. **Choose a starting point:** create one swarm, integrate an existing
   evaluation, operate a Factory, inspect evidence, or administer resources.
2. **Five-minute quickstart:** fresh environment to terminal swarm and one
   typed artifact, using only stable public imports.
3. **Concepts:** the noun hierarchy and ownership/lifecycle diagrams.
4. **Core workflows:** project/swarm, Factory, events/streaming, evidence,
   collaboration, and resource configuration.
5. **Guarantees:** errors, retries, idempotency, timeouts, cancellation,
   consistency, pagination, event ordering, versioning, and data retention.
6. **Cookbook:** realistic `evals`-backed examples that state required SDK and
   backend contract versions and expected outputs.
7. **Generated API reference:** every public symbol, signature, model field,
   enum member, exception, and return type.
8. **Migration guides:** managed-research imports, `run` to `swarm`, raw
   mappings to typed requests, sync-proxy to real async, and every removal
   version.
9. **Troubleshooting:** request IDs, diagnostic collection, common terminal
   states, stale cursors, quota/admission failures, and cleanup.
10. **Changelog and compatibility table:** package version, public contract
    version, minimum backend version, deprecations, and breaking changes.

Every quickstart is executed from a built wheel against an intended service
boundary. Generated symbol pages are necessary, but they are not a substitute
for teaching the model.

#### 10. Python-only reach guarantees adapter drift

OpenAI and Anthropic publish broad language SDK families. Modal supports Python
plus deployed Function invocation from JavaScript and Go. Tinker is much more
Python-centric, but its deliberately tiny remote primitive set limits the
damage. Synth proposes Python, CLI, and MCP adapters but no language-neutral
public SDK strategy.

The backend-authored public contract must be sufficient to generate at least a
conformance client and models in another language, even if Python remains the
only polished SDK initially. MCP cannot substitute for an SDK contract: tool
descriptions, pagination, streaming, error handling, and typed composition have
different ergonomics. Phase 2 should emit an OpenAPI/JSON Schema artifact plus
language-neutral lifecycle and event semantics, then run protocol conformance
fixtures independent of Python implementation classes.

#### 11. `core` can become the next god package

Consolidating 66,000 lines under `synth_ai/core` creates a strong temptation to
call every shared-looking behavior “core.” Modal's nouns and Tinker's split
show the better test: a foundation belongs in general core only when at least
two real domains use the same semantics, not merely similar code.

`core/http`, resource identity, pagination, error envelopes, deadlines, JSON,
and artifact primitives can be general. Swarm state, Factory candidates,
research economics, actor readouts, and collaboration semantics belong under
`core/research`. Benchmark scoring, recipes, and orchestration policy remain in
`evals`. A module promoted to general core needs named consumers and one shared
contract; otherwise it stays domain-specific until reuse is proved.

#### 12. Capability breadth is being mistaken for SDK quality

Approximately 483 backend research-related route operations and a large MCP
tool inventory are liabilities until curated. Great SDKs do not win because
every backend action is public. They win because common workflows are obvious,
advanced workflows are composable, and operator-only capabilities are not mixed
into customer autocomplete.

The public ledger should target the smallest complete surface, not maximum
parity. “Python/MCP/CLI parity” applies only after an operation is accepted as
public. Internal and operator operations can have separate generated clients
or remain backend-only. The migration should delete or demote more operations
than it promotes.

### Additional release bar derived from the benchmark

Phase 10 is not sufficient for a stable SDK release unless all of the following
are also true:

- A new user can complete the project → swarm → event/result flow with no raw
  dictionaries, private imports, preflight ritual, or backend vocabulary.
- The public noun map contains at most the approved resource hierarchy, with no
  `*API` namespace class names in user-facing documentation or type hints.
- Every public request rejects invalid combinations at construction time where
  Python's type system can express them.
- Every public operation has a guarantee-matrix row and a stable operation ID.
- Every swarm stores and returns an immutable resolved configuration snapshot.
- Every event belongs to the versioned `SwarmEvent` union; resume and gap
  behavior are proved.
- Sync and async are statically typed, behaviorally conformant, and use native
  transports; the dynamic thread-offload proxy is deleted.
- Every success and failure exposes request/operation correlation; every retry
  follows the published directive.
- Python, CLI, MCP, and a language-neutral conformance fixture agree on request,
  response, event, error, and pagination behavior.
- The built-wheel quickstart, async quickstart, reconnect/resume example,
  cancellation race, idempotent create replay, and version-pinned replay are
  exercised against the intended service boundary.
- Documentation has the full progressive structure above and contains no
  legacy `managed_research`, `SMR`, `run`, or `receipt` vocabulary except in an
  explicitly labeled migration guide.
- The supported/deprecated/removed matrix names package versions and dates;
  aliases do not persist indefinitely.

If these conditions are omitted, the migration should be described honestly as
an internal architecture cleanup. It should not be presented as a top-tier SDK
redesign.

## Part VI — Execution, compatibility, and acceptance

### Migration sequence

#### Phase 0 — freeze and ledger

- Freeze new public work in `synth_ai.managed_research`.
- Create the capability ledger covering backend routes, SDK methods, MCP tools,
  public Research methods, and active eval imports.
- Assign every row a source disposition and canonical noun, plus a concrete
  migration target or an explicit `unresolved` marker.
- Keep source definition, runtime availability, and successful eval invocation
  as separate evidence fields.
- Record the compatibility removal version and date.
- Add architecture checks forbidding new deep eval imports and new legacy
  implementation files.

Exit: no missing source classification, no unresolved target disposition, and
runtime receipts for every claimed public operation and delivery adapter; no
new legacy growth.

Implementation evidence:

- `specifications/sdk/research_capability_ledger.json` v2 records **2,807
  source inventory rows**: 2,332 definitions, 204 compatibility aliases, and
  271 backend/eval references across routes, core/public/compatibility SDK
  methods, legacy aliases, MCP tools, and imports. Its AST pass has
  zero missing source disposition/noun classifications, but it intentionally
  reports **2,807 unresolved runtime-availability rows**, **1,304 unresolved
  target dispositions**, **1,301 unresolved canonical targets**, and **11
  eval-import rows without a
  statically discoverable invoked capability**. Those unresolved magnitudes
  mean Phase 0 is not closed. Target-unresolved rows break down as 887 internal
  compatibility methods, 245 MCP tools, 83 core methods, 45 eval references,
  43 backend routes, and one public `advanced` entrypoint.
- The ledger distinguishes the checked-out HEAD from a dirty
  `head_plus_worktree` snapshot instead of attributing uncommitted source to a
  commit. It maps all 64 backend-authored operation IDs to uniquely discovered
  handlers, including the billing handler outside the Managed Research route
  root.
- All 46 stable MCP source definitions map to one or more bounded operation IDs;
  the 245 advanced MCP definitions remain target-unresolved. All 150 deprecated
  `synth_ai.managed_research` module aliases and 54 lazy public compatibility
  exports record their exact target and removal in 0.18.0. Row count may change
  as source definitions are added or retired; it is an inventory magnitude, not
  a completion score.
- The baseline freezes 156 legacy implementation files / 61,081 lines, 42
  deep backend/eval consumer imports, and 11 stable public operation IDs.
- `scripts/check_research_migration_boundaries.py` rejects new legacy files,
  increased legacy lines/files, increased deep consumer imports, forbidden
  core imports, or a missing source disposition/noun classification. That
  stable check does not convert unresolved runtime or target fields into proof.
- `scripts/generate_research_capability_ledger.py` refreshes the ledger from
  explicit synth-ai, backend, and eval worktrees.

#### Phase 1 — core foundations

- Establish `core/auth`, `core/http`, and general contracts.
- Unify endpoint normalization, headers, timeouts, error mapping, pagination,
  retries, and sync/async transports.
- Migrate containers, pools, and research to these primitives without changing
  their public APIs.

Exit: one transport/error/auth path; no behavior fallback.

Implementation evidence (in progress):

- `synth_ai/core/auth`, `core/contracts`, and `core/http` now own credential
  resolution, request metadata, recursive JSON, resource identities, cursor
  pagination, sync/async transports, strict SSE decoding, and retry policy.
- Infrastructure `SynthBaseClient` and the deprecated Managed Research transport
  both delegate to the same core transport while retaining their declared error
  compatibility at the adapter boundary.
- `AsyncResearchClient` uses `httpx.AsyncClient` directly; the prior dynamic
  `asyncio.to_thread` namespace proxy has been removed.

#### Phase 2 — bounded backend contract

- Add stable backend operation IDs for approved public and advanced operations.
- Generate the bounded public research OpenAPI artifact.
- Generate or mechanically verify SDK wire contracts from that artifact.
- Fail CI on route, enum, requiredness, or error-code drift.

Exit: every migrated operation has a single backend-authored contract.

Implementation evidence (in progress):

- Backend `scripts/export_research_openapi.py` filters the real application
  OpenAPI graph by explicit `PUBLIC_OPERATION_IDS`, verifies operation ID drift,
  and emits the transitive schema closure.
- Backend source now allowlists **64 operations**. A fresh export from
  `64361869d` produced **64 operations / 197 schemas** in
  `/tmp/research-openapi-environments.json`.
- The checked-in backend `research_openapi.json` and SDK
  `openapi/research-v1.json` are both still the older **49-operation /
  155-schema** artifact. Their mutual byte parity is therefore stale evidence,
  not proof that either file represents current backend source.
- Before another vertical is accepted, regenerate the backend artifact, vendor
  the exact bytes in `synth-ai`, and run
  `scripts/check_research_openapi_contract.py` against the fresh artifact and
  `core/research/operations.py`. The source allowlist, authoritative artifact,
  SDK operation registry, and vendored artifact are one atomic contract change.

#### Phase 3 — projects and swarms vertical set

- Move project and swarm contracts, clients, handles, lifecycle, waits, and
  basic readouts into `core/research` under `projects/` and `swarms/`.
- Land `ResearchSwarmLaunchRequest` (and nested catalog types for models,
  harnesses, roles, actor overrides, limits, images, kickoff) as the sole
  public launch intent object; keep `ResearchRunLaunchRequest` only as an
  alias to the same class during compatibility.
- Close public launch fields to enums/dataclasses; forbid new `WireMapping`
  or free-string selectors for models, harnesses, and actor assignments.
- Rewire `SynthClient().research.projects` and `.swarms` directly to core.
- Keep `research.runs` as a deprecated property alias to the same
  `ResearchSwarmsAPI` object; do not preserve a second implementation.
- Preserve legacy managed-research imports as aliases to the exact same classes.
- Cut active eval launch/smoke drivers to the public swarm facade and typed
  launch request (same pattern SwarmBench uses today).

Exit: the primary create-project to terminal-swarm workflow never imports
legacy implementation; new code uses `swarm` / `swarm_id` only; evals can
express model/harness/role/limit overrides only through typed launch fields.

Implementation evidence (in progress):

- `core/research/contracts` defines opaque IDs, closed model/harness/work-mode/
  lifecycle enums, typed project create/patch/setup models, and the sole
  `ResearchSwarmLaunchRequest`. `ResearchRunLaunchRequest` is the same class
  object as the temporary compatibility alias.
- Typed launch intent covers actor model assignments, providers, resource
  limits, kickoff messages, local execution identity, and execution profiles;
  no public launch mapping is accepted by the core API.
- `SwarmHandle.configuration()` and the sync/async swarm APIs resolve the exact
  run-bound configuration version through one backend-owned operation. The
  returned `ResolvedSwarmConfiguration` is frozen, versioned, secret-redacted,
  hash-identified, and exposed through matching CLI and stable MCP reads. The
  backend fails explicitly for historical unversioned runs or missing durable
  snapshots instead of reading the project's current configuration.
- Sync and native-async project/swarm APIs support create, retrieve/list,
  setup/preflight, wait, pause/resume/cancel, branch, and typed live events.
  `research.runs` warns and returns the exact same swarm API object.
- The deep-import cutover is complete, but active eval launch drivers still use
  compatibility names and explicit `research.advanced` capabilities; those
  remaining dependencies are tracked under Phase 8 rather than hidden here.

#### Phase 4 — evidence, observability, and collaboration

- Move artifacts, work products, usage, logs, events, tasks, actors, messages,
  timelines, traces, and evidence readouts.
- Establish one typed projection freshness contract.
- Remove mapping probes and fallback parsing from readout code.

Exit: active eval evidence collection uses only public typed models.

Implementation evidence (in progress):

- Evidence/readout, artifact, control, collaboration, and project-namespace
  implementation has moved behind `synth_ai.core.research`; matching
  `synth_ai.research` modules are now import-only public facades.
- Active eval code no longer imports `synth_ai.core` or
  `synth_ai.managed_research`. Capabilities that do not yet have a stable hero
  namespace use the documented, explicitly unstable
  `synth_ai.research.advanced` bridge rather than a deep package path.
- `retrieve_swarm_usage` now collapses the legacy run-usage and actor-usage
  probes into one backend-owned projection with exact aggregate cents/pico-USD,
  closed token partitions, cent-granular actor attribution, and explicit
  source/as-of/record-count/terminal-state freshness. Sync and native-async
  swarm APIs, handles, CLI, and stable MCP use the same operation and strict
  decoder; unknown fields and invalid token partitions fail.
- Three active ReportBench evidence collectors now record `SwarmUsage` through
  the stable facade instead of making two legacy-backed dictionary reads.
- `SwarmHandle.evidence()` now returns one backend-authored, exact-field
  artifact and WorkProduct index with closed WorkProduct kind/status/readiness
  and artifact-role enums, opaque identifiers, count invariants, and observable
  read time/terminality. Artifact and WorkProduct content use operation-aware
  sync/native-async byte transports, and the bounded OpenAPI declares binary
  responses rather than incorrectly advertising JSON.
- ReportBench artifact listing, WorkProduct listing, and both content paths now
  use the stable swarm API. The hero driver no longer probes project-scoped
  fallback routes or reads these surfaces through the compatibility session.
- `retrieve_swarm_activity` adds a strict, frozen activity projection for
  actors, tasks, messages, and typed timeline events. ReportBench's principal
  observability adapter now consumes that model rather than locally probing
  dictionary shapes.
- Transcript reads and runtime SSE now have versioned page, cursor, freshness,
  participant, event-envelope, heartbeat, and snapshot contracts. Sync and
  native-async swarm APIs, CLI, stable MCP, and active ReportBench transcript
  collection share those contracts. The active eval path records per-page
  cursor and freshness evidence.
- The remaining active observability gaps are the legacy poll-summary snapshot,
  objective-event reads, workspace archive content, and advanced session launch/
  preflight. The reviewer must decide whether activity supersedes each read or a
  separate backend-authored projection is genuinely necessary.

#### Phase 5 — factories and economics

- Move Factory operations, candidates, results, messages, evidence, limits,
  budgets, costs, and denial errors.
- Export the hosted Factory unit under its approved public noun, `Effort`
  (typed `EffortId`, contracts under `core/research/contracts/factories.py`
  or a sibling `efforts.py`).
- Keep backend-only maintenance and scheduling policy internal.

Exit: SDK and MCP have parity through the same core operation registry.

Implementation evidence (in progress):

- Seven bounded limits/economics operations and twelve Factory/Effort operations
  now use shared core sync/native-async transports and typed contracts. Stable
  Factory lifecycle and Effort CRUD no longer open the legacy session client.
- Operator-only Factory projections still use `advanced_factories.py` and the
  compatibility session. Each must either graduate to bounded operation
  metadata or receive an explicit adapter-only/backend-only disposition before
  this phase exits.

#### Phase 6 — resources and deployment

- Move environments, dev environments, images, image releases, repositories,
  datasets, workspace inputs, cloud deployments, and claims.
- Use common resource, lease, artifact, cleanup, and authority types where the
  backend contract genuinely shares them.

Exit: no domain-specific duplicate transport or resource identity remains.

Implementation evidence (in progress):

- Three backend-authored workspace-input operations now use strict request and
  response DTOs: retrieve inputs, set the bootstrap source repository, and
  upload files. The SDK exposes matching sync/native-async workspace APIs with
  frozen contracts. Each upload request accepts 1–100 unique paths and the
  backend recognizes only `base64`, `text`, and canonical `utf-8` encodings.
- Four project external-repository operations and three project-dataset
  operations are backend-bounded and exposed through typed sync/native-async
  SDK, CLI, and stable MCP adapters. Dataset content is binary-safe. A bootstrap
  source repository is intentionally a workspace input; it is not an external
  repository record.
- Four Environment catalog operations are bounded in backend source: list,
  create, retrieve, and preflight. Their schemas are strict and extra-forbid,
  but no matching core SDK, CLI, or MCP implementation was committed before
  this review handoff. Environment catalog entries are not the same lifecycle
  as operator DevEnvironments.
- The five customer image-release routes remain outside
  `PUBLIC_OPERATION_IDS`. Their declaration model still permits an
  optional-field soup, and declaration/artifact/inspection/materialization/list
  responses retain `dict[str, Any]` islands. The next contract should use a
  discriminated scorer-versus-actor declaration union and strict nested models
  where backend shape is closed.
- DevEnvironment lifecycle, runtime image materialization/control, and cloud
  deployments remain advanced/operator surfaces until separately classified.
- ReportBench currently calls nonexistent `projects.repos.attach` and
  `projects.workspace.upload` methods. The stable targets are
  `projects.workspace.set_source_repository(...)` and
  `projects.workspace.upload_files(...)` with typed request objects. The old
  client silently chunked large directories, while the new route caps one
  request at 100 files; the plan must not hide that semantic regression.

#### Phase 7 — delivery adapters

- Move MCP to `synth_ai/mcp/research`.
- Generate MCP tool schemas from core request models and operation metadata.
- Make CLI commands call the same core/public operations.
- Delete connector-only and CLI-only business rules.

Exit: Python, CLI, and MCP differ only in presentation and serialization.

Implementation evidence (in progress):

- The MCP delivery package is now `synth_ai.mcp.research`; backend-hosted MCP
  imports `ResearchMcpServer` from that path, while the old class and module
  paths resolve to the same objects through generated shims.
- The Factory stand-up console script now targets
  `synth_ai.cli.research_factory_standup` instead of the compatibility package.
- Open Research SDK/MCP modules were removed as required by the frontend-only
  product decision. Source discovery finds 46 allowlisted stable noun-first
  tools, each mapped to one or more bounded operation IDs, and 245 advanced
  definitions with unresolved migration targets. The generator does not load
  the server or invoke a tool, so runtime registration, schema parity, auth,
  failure behavior, and reachability remain unproved for both sets.
- The stable set now includes activity, transcript/watch, workspace, repository,
  and dataset adapters. That source coverage does not cure the stale checked-in
  OpenAPI artifact or substitute for actual MCP registration and invocation
  receipts.

#### Phase 8 — eval cutover

- Replace all active `synth_ai.managed_research` imports in evals.
- Move general client behavior found in eval drivers into the SDK.
- Leave benchmark-specific policy and scoring in evals.
- Rename active scripts whose names imply the retired package when compatibility
  naming is no longer required.

Exit: active eval code imports only documented `synth_ai` public surfaces.

Implementation evidence:

- The deep-consumer source ratchet decreased from 42 to **0** across the backend
  and active eval tree. Evals now import `synth_ai.research`,
  `synth_ai.research.advanced`, or other documented `synth_ai` surfaces;
  graveyard paths remain excluded from the active inventory. This proves source
  boundaries, not successful eval execution.
- GameBench-specific scorer models, client, and recipe now live under
  `evals/swarmbench/gamebench`; `synth_ai/gamebench` has been removed. Remaining
  compatibility client imports and advanced dependencies still require an
  owner and stable-or-advanced disposition before the phase is complete.
- Active evals no longer import the standalone `managed_research` package or
  the top-level `ManagedResearchClient` alias. Of 246 active eval import rows,
  the ledger labels **45** explicit `synth_ai.research.advanced` rows separately
  from 201 supported-public rows. It discovers call expressions for 235 rows;
  11 have `invoked_capability_status=unresolved`, and none has a runtime receipt
  merely because a call expression exists.
- ReportBench's hero adapter exposes `get_swarm_usage`, and its principal SDK,
  SwarmBench, and compatibility-driver evidence collectors all consume the
  stable typed usage/freshness projection.
- ReportBench also consumes typed swarm activity and transcript pages. Its
  source-repository attachment and workspace-file upload methods have not yet
  been cut over correctly: they name methods that do not exist on the stable
  project API. Directory-backed lanes may exceed the new 100-file request
  bound, so fixing the method names without an explicit batching contract is
  insufficient.

#### Phase 9 — compatibility shell

- Reduce `synth_ai.managed_research` to re-exports and entrypoint aliases.
- Every compatibility module carries its removal version and emits one clear
  deprecation warning at the import or construction boundary.
- Unknown legacy wire shapes fail; compatibility does not mean fallback
  parsing.

Exit: zero implementation logic under `synth_ai.managed_research`.

Implementation evidence:

- `scripts/generate_managed_research_compatibility.py` generates and verifies
  exact warning-only re-export modules. The package is currently 150 Python
  shims / 754 lines versus the frozen 156-file / 61,081-line implementation
  baseline; reusable code resides under core and the MCP/CLI delivery roots.

#### Phase 10 — removal

- Remove `synth_ai.managed_research` and legacy MCP entrypoint aliases after the
  declared compatibility window.
- Remove the standalone `managed-research` shim release.
- Delete compatibility tests and launch-checklist requirements.
- Retain a migration guide mapping old imports to public Research imports.

Exit: one implementation under core and one public Research facade.

### Compatibility policy

Compatibility aliases must point to the same class objects where possible:

```python
# Public core contract
from synth_ai.core.research.contracts.swarms import ResearchSwarm

# Temporary compatibility aliases (same object)
from synth_ai.core.research.contracts.swarms import ResearchSwarm as ResearchRun
from synth_ai.core.research.contracts.swarms import ResearchSwarm as ManagedResearchRun
```

Do not maintain parallel subclasses or duplicate serialization logic. A shim
may rename, warn, and delegate. It may not repair arbitrary shapes, try retired
routes, or silently select a second implementation. Public packages must not
grow new `Run*` / `runs` APIs except as explicit deprecation aliases over
swarm types.

The compatibility window ends only after published-package telemetry and
workspace grep show no supported consumers. The removal deadline is set before
Phase 3, not after migration work becomes inconvenient.

| State | Package line | Date | Contract |
|---|---|---|---|
| Deprecated | `synth-ai` 0.16.0 | 2026-07-21 | New code uses `synth_ai.research`; no new legacy implementation |
| Last compatibility line | 0.17.x | Through 2026-08-31 | Exact aliases and legacy entrypoint delegation only |
| Removed | 0.18.0 | Not before 2026-09-01 | `synth_ai.managed_research` and standalone shim removed after consumer proof |

### Enforcement and acceptance

#### Structural checks

- `synth_ai/core/**` cannot import `research`, `managed_research`, `cli`, or
  `mcp`.
- Active `evals/**` cannot import `synth_ai.core` or
  `synth_ai.managed_research`.
- `synth_ai.managed_research/**` may contain only approved compatibility
  patterns after Phase 9.
- Every public backend operation has a stable operation ID.
- MCP tool schemas derive from the same typed request contracts as Python.
- Public `__all__` contains approved Research nouns only; no new `Smr*` names
  and no new public `Run*` / `runs` surfaces except documented deprecation
  aliases to swarm types.
- Domain modules do not use `dict[str, Any]` as internal application contracts.
- Public swarm launch request fields for models, harnesses, actor assignments,
  roles, work modes, and related catalogs are closed enums or nested
  dataclasses; new open mappings on those fields are rejected.

#### Ratchets

Track at least:

- Legacy implementation lines and files, monotonically decreasing.
- Deep legacy imports in backend and evals, monotonically decreasing.
- Public operations with stable backend operation IDs, monotonically
  increasing until ledger-complete.
- Core type-check errors, held at zero per migrated subtree.
- Jstack HIGH violations in migrated code, held at zero.
- Public/MCP/CLI capability parity, increasing to 100% of public ledger rows.

#### Repeatable SDK product-quality scan

The canonical one-line local review is:

```bash
./synth-dev/quality ./synth-ai
```

The wrapper and SDK-design configuration currently exist as untracked files in
the `synth-dev` main worktree: `quality` and `config/quality/`. They could not be
committed during this migration because that worktree already contains an
unrelated unresolved merge in `local_dev/sltop-rs/src/main.rs`. The next
`synth-dev` owner must resolve that merge without absorbing these migration
changes, then review and commit the quality wrapper/configuration as a separate
tooling change. The accepted v6 receipt was produced with those exact local
files, including the strict verdict schema and validator.

For repositories with a `synth_ai/research` surface, that command includes a
`jesterky-sdk-design` lane driven by Gemini 3.5 Flash Lite. The lane creates one
parallel audit job for each of the 11 benchmark dimensions above. Its evidence
builder selects bounded, line-numbered excerpts from active SDK source,
contracts, documentation, CLI/MCP adapters, and the migration specification;
it excludes graveyards, generated output, caches, build artifacts, and test
fixtures. Agents may judge only those supplied excerpts and may not explore the
repository or network. This keeps repeated scans comparable and prevents an
obsolete `old/` subtree from dominating the review.

The SDK-design gate fails when any of these conditions holds:

- fewer than all 11 dimensions produce a numeric score;
- an auditor emits a hold or any dimension scores 3 or lower;
- the mean score is below 7.0; or
- the minimum dimension score is below 5.0.

The thresholds may be raised with `QUALITY_SDK_MIN_AGGREGATE` and
`QUALITY_SDK_MIN_DIMENSION`; lowering them is not release evidence. The final
terminal scorecard and `summary.json` include every dimension score, severity,
hold, finding, aggregate, range, finding count, model usage, wall time, and the
Jesterky manifest path. The manifest retains detailed evidence, violations,
and next actions. Provider failure, malformed output, or an incomplete map is
a failed quality lane, not an unavailable score disguised as a pass.

The 2026-07-21 baseline proof completed 11/11 dimensions in 16 seconds with
92k tokens and scored the current SDK **5.64/10**. Naming scored **3/10** and
the concurrency contract **4/10**; those results reinforce the migration's
public-noun and native-async release conditions rather than replacing them.

The scan is also this migration's before/after instrument. The implementing
engineer must treat the 2026-07-21 baseline as the "before" measurement,
re-run `./synth-dev/quality ./synth-ai` at every phase exit, and append the
date, aggregate, per-dimension scores, and manifest path to the scan log
below. The aggregate must not decrease across consecutive phase exits; a
decrease blocks the phase from being accepted until explained and fixed.
Phase 10 acceptance additionally requires the release thresholds above
(mean ≥ 7.0, every dimension ≥ 5.0, no holds).

##### Scan log

| Date | Milestone | Aggregate | Weakest dimensions | Manifest |
|---|---|---:|---|---|
| 2026-07-21 | Baseline (before migration) | 5.64 | naming 3, compression 4, concurrency 4 | `.quality/runs/sdk-design-proof-20260721T211647Z/summary.md` |
| 2026-07-21 | Core relocation checkpoint | 6.45 | naming 4, compression 4, failure 6, observability 6, reach 6 | `.quality/runs/migration-sdk-design/manifest.json` |
| 2026-07-21 | Stable API candidate | **7.09** | compatibility 6, naming 6, reach 6 | `.quality/runs/migration-sdk-design-v2/manifest.json` |
| 2026-07-21 | Candidate after stable-import isolation | **7.09** | compression 6, failure 6, naming 6, reach 6 | `.quality/runs/migration-sdk-design-v6/manifest.json` |
| 2026-07-21 | Candidate after GameBench ownership and contract parity fixes | **7.18** | compression 6, naming 6, reach 6 | `.quality/runs/migration-sdk-design-v8/manifest.json` |
| 2026-07-21 | Latest repeat after advanced-consumer classification | **7.09** | compression 6, documentation 6, naming 6, reach 6 | `.quality/runs/migration-sdk-design-v9/manifest.json` |
| 2026-07-21 | Final handoff repeat after resolved-configuration vertical and final capability ownership | **7.27** | naming 6, parameter design 6, reach 6 | `.quality/runs/migration-sdk-design-v10/manifest.json` |
| 2026-07-22 | Post-usage-contract verification | **7.18** | compression 6, naming 6, reach 6 | `.quality/runs/migration-sdk-design-v11/manifest.json` |

The final handoff repeat completes all 11 dimensions, has no holds, and clears
the declared qualitative bar: mean **7.27** (80/110), minimum **6**, maximum
**8**, with 20 recorded violations. The strict manifest validator passed. It
completed in **21 seconds** using approximately **88k tokens** (83k input and
5.3k output) through Gemini 3.5 Flash Lite. Relative to v9, conceptual
compression, concurrency, documentation, and observability each rose one point,
while parameter design fell two points; the aggregate therefore rose 0.18.
The parameter-design finding is substantive rather than hidden as sampling
noise: public `ResourceLimit` and `BudgetPolicy` shapes still permit too many
optional combinations. Retain both receipts so reviewers can distinguish
judge variance from a repeated design concern. The final receipt is evidence
for API review, not a substitute for deterministic or vertical integration
proof.

Its complete scorecard is:

| Dimension | Score | Severity | Principal remaining gap |
|---|---:|---|---|
| Compatibility | 8 | low | Automate enforcement of the public compatibility manifest and versioned deprecation contract |
| Conceptual compression | 7 | low | Hide low-level core namespaces and present one flat, noun-led Research hierarchy |
| Concurrency contract | 8 | low | Document transcript-cursor recovery and resume behavior for native async streaming |
| Documentation | 7 | low | Complete generated public reference pages and link each back to executable recipes |
| Failure contract | 8 | low | Complete generated documentation for every public typed error class |
| Lifecycle guarantees | 8 | low | Make archive/delete terminality explicit and consistent across resource APIs |
| Naming | 6 | medium | Remove internal `Smr*` prefixes and compatibility aliases from public discovery before 0.18.0 |
| Observability | 8 | low | Move legacy readout diagnostics directly onto native core Research types |
| Parameter design | 6 | medium | Replace optional-soup resource and budget records with validated combinations or explicit builders |
| Reach | 6 | medium | Add CI conformance proving Python, CLI, and MCP schemas mirror the bounded OpenAPI artifact |
| Type contract | 8 | low | Replace fallback preflight-blocker dictionary parsing with strict typed union decoding |

The longitudinal scorecard is:

| Dimension | Baseline | Relocation | First candidate | Import-isolated | Latest | Net |
|---|---:|---:|---:|---:|---:|---:|
| Compatibility | 6 | 8 | 6 | 8 | 8 | +2 |
| Conceptual compression | 4 | 4 | 7 | 6 | 7 | +3 |
| Concurrency contract | 4 | 8 | 8 | 8 | 8 | +4 |
| Documentation | 6 | 8 | 7 | 8 | 7 | +1 |
| Failure contract | 6 | 6 | 8 | 6 | 8 | +2 |
| Lifecycle guarantees | 8 | 7 | 8 | 8 | 8 | 0 |
| Naming | 3 | 4 | 6 | 6 | 6 | +3 |
| Observability | 7 | 6 | 7 | 7 | 8 | +1 |
| Parameter design | 6 | 7 | 7 | 7 | 6 | 0 |
| Reach | 6 | 6 | 6 | 6 | 6 | 0 |
| Type contract | 6 | 7 | 8 | 8 | 8 | +2 |

The import-isolated rerun restored compatibility to 8 after stable imports
stopped loading `_legacy`. The final handoff rerun lifts compression,
concurrency, documentation, and observability, but exposes parameter design as
the clearest remaining type-design weakness. Its aggregate is 1.63 points above
baseline and no dimension falls below the release minimum. The v8 and v9
receipts remain useful corroborating evidence, but the handoff uses the 7.27
result as its strongest accepted ratchet. All aggregates above are computed
directly from the eleven manifest scores; no hand-entered aggregate is accepted
as evidence.

The later v11 verification was run after the typed swarm-usage vertical. It
completed 11/11 dimensions in **21 seconds**, used approximately **88k tokens**
(83k input and 5.1k output), recorded 18 violations, emitted no holds, and
passed the strict 7.0/5.0 manifest validator. Its mean is **7.18** (79/110),
minimum **6**, and maximum **8**. Compared with v10, compatibility moved 8→7,
conceptual compression 7→6, and parameter design 6→7; every other dimension was
unchanged. The relevant source facts did not regress: the compatibility
timeline and shim architecture, stable discovery boundary, and public resource
tree are unchanged, while the new usage contract adds a strict typed read. The
judge instead surfaced two existing debts more strongly: wildcard imports in
generated compatibility shims and the breadth visible beneath the Research
resource tree. Those are valid findings and are retained below. Because the
aggregate nevertheless decreased by 0.09, v11 proves the absolute release floor
but is **not accepted as a monotonic phase-exit improvement**. The reviewer must
either approve those debts as bounded compatibility/advanced-only exceptions or
require their source fix and a new receipt before accepting the phase exit.

The complete latest verification scorecard is:

| Dimension | Score | Severity | Finding / required review action |
|---|---:|---|---|
| Compatibility | 7 | low | Generated compatibility shims still rely on wildcard legacy imports; decide whether generation must emit explicit symbol exports |
| Conceptual compression | 6 | medium | Auxiliary resource namespaces still expose too much backend-shaped breadth; keep the stable hero surface focused on projects, swarms, and factories |
| Concurrency contract | 8 | low | Native sync/async separation and cursored events are strong; document resume and cancellation propagation |
| Documentation | 7 | low | Recipes are coherent but need explicit version and expected-output annotations |
| Failure contract | 8 | low | Typed failures preserve operation/request/correlation/retry metadata; complete public error documentation |
| Lifecycle guarantees | 8 | low | Resource transitions and immutable snapshots are explicit; add version-pinned replay guidance |
| Naming | 6 | medium | Remaining `Smr*` and compatibility client suffixes must stay out of stable discovery and be removed on schedule |
| Observability | 8 | low | Correlated events and typed usage/readouts are strong; graduate remaining legacy-backed observability projections |
| Parameter design | 7 | medium | Required fields are typed, but `ResourceLimit` and `BudgetPolicy` still allow optional-soup configurations |
| Reach | 6 | medium | Python, CLI, and MCP need authoritative cross-surface conformance against the bounded OpenAPI artifact |
| Type contract | 8 | low | Strict decoders reject malformed/unknown fields; remove remaining fallback dictionary parsing at optional wire blocks |

#### Required proof per vertical set

Each vertical set records:

1. Backend contract artifact and operation IDs.
2. Generated or parity-checked request/response models.
3. Public SDK invocation against the intended local/staging boundary.
4. Equivalent MCP invocation where the capability is connector-supported.
5. Active eval consumer using only the public SDK.
6. Stable failure proofs for auth, validation, conflict, quota, and transient
   service errors relevant to the operation.
7. Import-boundary, architecture, Ruff, type, docs, and package checks.

No phase is accepted by line movement alone.

### Rollout order and change discipline

- Migrate vertical capabilities, not directory trees wholesale.
- A change adds the core path, rewires public callers, adds compatibility
  aliases, and deletes the old implementation in one bounded change set.
- Do not temporarily run old and new implementations behind a fallback.
- Do not change backend route semantics and SDK structure in the same change
  unless the contract version explicitly requires both.
- Preserve `dev -> staging -> main` promotion and the production launch
  checklist.
- Publish no package that exposes partially duplicated public models for the
  same operation.

### Explicit non-goals

- Moving backend persistence or runtime authority into `synth-ai`.
- Making all backend SMR routes public.
- Moving GameBench, FactoryBench, ReportBench, or other benchmark policy into
  core.
- Replacing backend service models with SDK models.
- Introducing a universal repository/service abstraction.
- Renaming every backend `smr` or wire `run` identifier during structural
  migration; public swarm nouns are required, wholesale backend renames are not.
- Reopening Open Research as an SDK surface.
- Preserving undocumented deep imports indefinitely.

### Completion definition

The migration is complete when:

- `SynthClient().research` is the only documented customer entrypoint, with
  hero namespaces `projects`, `swarms`, and `factories`.
- All reusable SDK implementation lives in concern-specific packages beneath
  `synth_ai/core`.
- `synth_ai.research`, CLI, and MCP are thin adapters over the same operations
  and use swarm-first public nouns (`swarm_id`, `ResearchSwarm`, …).
- Backend publishes the bounded authoritative contract consumed by the SDK;
  wire `run` vocabulary is decoded at the boundary and is not the public story.
- Active evals use public SDK types and operations only.
- Benchmark-specific code remains in evals.
- `synth_ai.managed_research` contains no implementation and is removed after
  its declared window.
- No operation depends on route guessing, loose-shape fallback, swallowed
  errors, or a second authority path.

### Current completion audit

This matrix is the requirement-by-requirement state of the committed candidate,
not a projection from the intended design. `PARTIAL` means current evidence
proves some exit conditions and contradicts or cannot prove the rest.

| Phase | Status | Evidence proved now | Evidence still required |
|---|---:|---|---|
| 0 — freeze and ledger | NOT PROVED | v2 inventories 2,807 source definitions/aliases/references; source disposition/noun classification is complete; all 64 operation IDs map to handlers; frozen growth/deep-import ratchets and dated compatibility targets exist | Resolve 1,304 migration targets, classify every unresolved runtime field and attach receipts for the public claims, and resolve 11 eval invocation rows |
| 1 — core foundations | PARTIAL | Shared credential, sync/async HTTP, retry, JSON, pagination, and error primitives exist; Research uses native async | Prove every container/pool/Research path uses one behaviorally identical error/retry/auth boundary and no domain transport remains |
| 2 — bounded backend contract | NOT PROVED — STALE | Backend source allowlists 64 operations and a fresh temporary export has 197 schemas | Regenerate the checked-in backend and SDK artifacts, which still contain 49 operations / 155 schemas; then prove byte, operation, enum, requiredness, error, event, pagination, and state parity in authoritative CI |
| 3 — projects and swarms | PARTIAL | Stable typed sync/async project and swarm APIs, same-object run aliases, event protocol, immutable resolved-configuration read, zero deep eval imports | Replace remaining active compatibility launch names and prove the hero loop plus resolved-configuration read against a real backend |
| 4 — evidence and collaboration | PARTIAL | Typed activity, transcript/SSE, usage/freshness, durable artifact/WorkProduct evidence, and binary content are shared by sync/async Python, CLI, stable MCP, and active ReportBench collectors | Replace giant poll-summary consumption with a small status projection plus existing resources; graduate the required binary workspace archive; remove or strictly type objective events; keep raw logs/traces advanced; attach live cursor/reconnect/failure proof |
| 5 — factories and economics | PARTIAL | Seven economics plus twelve Factory/Effort operations are bounded and typed with sync/async peers | Classify or graduate every legacy-session Factory projection and prove lifecycle/idempotency semantics against backend |
| 6 — resources and deployment | PARTIAL | Workspace inputs plus project external repositories/datasets are bounded end to end; four strict Environment operations exist in backend source | Add the Environment SDK adapters; design strict image-release unions and five operation contracts; classify DevEnvironments/cloud deployments; fix ReportBench's nonexistent workspace calls and explicitly handle uploads above 100 files |
| 7 — CLI and MCP | PARTIAL | MCP package moved; 46 stable definitions have bounded operation mappings; CLI/MCP source adapters cover configuration, activity, transcript, usage, durable evidence, workspace, repositories, and datasets | Synchronize the 64-operation artifact, attach runtime registration/invocation/failure receipts, and replace or remove the 245 target-unresolved advanced definitions and legacy request models |
| 8 — eval cutover | PARTIAL — BROKEN CALLS | Deep and standalone-package imports are zero; GameBench scoring is eval-owned; typed activity/transcript/usage/evidence consumers landed | Replace five nonexistent workspace call sites across ReportBench/SwarmBench/SwarmGameBench, resolve >100-file batching and mutation atomicity, resolve 11 ledger invocation rows, and run representative project/swarm/Factory workflows; source calls are not execution proof |
| 9 — compatibility shell | PASS | 150 generated warning-only re-export files / 754 lines; no implementation under `synth_ai.managed_research` | Preserve exact aliases and warnings through the declared window |
| 10 — removal | NOT DUE | Version/date matrix is recorded | After supported-consumer and published-package proof, remove compatibility in 0.18.0 no earlier than 2026-09-01 |

Accordingly, the migration is not complete merely because the stable discovery
surface and qualitative threshold pass. Phases 1–8 require the proofs or source
closures above, Phase 10 is intentionally date-gated, and integration still
requires the user-triggered `/ultrareview`.

## Part VII — Engineer review and handoff

### Review objective

Turn this draft and its unmerged implementation candidate into an approved
migration contract before further broad implementation. The review is
successful only when it resolves the product and protocol questions, not when
it merely agrees that files belong under `core`.

The implementing engineer should edit this document in place:

1. Treat each adopted recommendation below as the working implementation
   decision authorized by the user on 2026-07-21.
2. Record any deviation and rationale before implementing it.
3. Amend the target tree and phases when current code proves the recommendation
   incomplete.
4. Name an owner for every cross-repository contract and migration phase.
5. Change the document status to `complete` only after SDK, backend, and eval
   integration evidence proves the completion definition.

### Decisions requiring explicit closure

| ID | Decision | Current recommendation | Required reviewers | Status |
|---|---|---|---|---|
| R1 | Minimal customer lifecycle | `project → swarm → typed events/result`; hide administrative route breadth | SDK/product, backend | adopted |
| R2 | Public Factory timing | Keep the noun, but admit it to the stable surface only after the swarm loop is small and complete | SDK/product, factory owner | adopted |
| R3 | Public vocabulary | Use projects, swarms, factories, and efforts (the hosted Factory unit — decided 2026-07-21); keep `run`/`smr` at wire or migration boundaries only | SDK/product, docs | adopted |
| R4 | Public operation ledger | Explicit allowlist; everything else is operator-only, adapter-only, deprecated, or removed | SDK, backend, MCP | adopted |
| R5 | Contract authority | Backend-authored bounded OpenAPI/protocol artifact with stable operation, error, state, event, and pagination identities | Backend, SDK | adopted |
| R6 | Python contract generation | Generate or conformance-check boundary models; preserve hand-authored ergonomic domain types only where they add semantics | SDK, backend | adopted |
| R7 | Async and streaming | Native async transport with typed cursors, ordering, duplicate/gap semantics, deadlines, cancellation, and behavioral sync parity | SDK, backend runtime | adopted |
| R8 | Resolved swarm identity | Persist and return an immutable, versioned resolved configuration snapshot for replay and audit | Backend, SDK, evals | adopted |
| R9 | Compatibility window | Publish an exact package/version/date matrix and delete aliases on schedule; no indefinite shims | SDK release owner, docs | adopted |
| R10 | Language-neutral reach | Make the protocol suitable for conformance clients now; decide separately which additional language SDK ships first | SDK, backend | adopted |
| R11 | Quality release bar | Require Jesterky mean ≥7.0, every dimension ≥5.0, no holds, plus deterministic and vertical proof gates | SDK, release engineering | adopted |
| R12 | `core` boundary | Allow only typed contracts, codecs, transport, operations, and lifecycle handles; reject policy/orchestration and generic utility dumping | SDK architecture | adopted |
| R13 | Usage evidence contract | One concise swarm projection with exact aggregate money, typed tokens/actors, honest actor precision, and explicit freshness; keep raw ledger entries advanced | Backend, SDK, evals | candidate implemented; reviewer approval required |
| R14 | Durable evidence contract | One strict swarm evidence snapshot for artifacts and WorkProducts, plus operation-aware binary content reads; backend remains index/content authority and the SDK does not expose storage URIs as authority | Backend, SDK, evals | candidate implemented; reviewer approval required |
| R15 | Contract publication atomicity | A vertical is incomplete unless the backend allowlist, generated backend artifact, SDK operation metadata, and byte-identical vendored artifact land and validate together | Backend, SDK, release engineering | open; current 64-source/49-artifact split violates the recommendation |
| R16 | Workspace uploads above 100 files | Keep the single backend mutation bounded at 100 files; add an explicit typed batch operation with deterministic partitions, per-batch idempotency, partial-completion receipts, and fail-loud semantics rather than silently chunking one call. Execute the authoritative git mutation before its stored-file projection so a failed git write cannot leave rows claiming content was committed; the same idempotency key must safely resume the projection write | SDK, backend, evals | open; required by directory-backed ReportBench lanes, and the current backend route writes its projection before the git mutation |
| R17 | Repository taxonomy | Keep bootstrap source repository configuration under `projects.workspace`; keep reusable/attached project external repositories under `projects.repositories`; do not alias one into the other | SDK/product, backend, evals | candidate distinction exists; reviewer approval and eval cutover required |
| R18 | Environment taxonomy | Stable `research.environments` is the declarative catalog/preflight resource; DevEnvironment materialization/control is operator-only unless separately approved | SDK/product, backend runtime | open; backend catalog contract exists, SDK adapter does not |
| R19 | Image-release declarations | Replace the optional-field declaration record with a discriminated scorer-versus-actor union and strict nested artifact/inspection/materialization contracts before exposing five image-release operations | SDK, backend images | open; no bounded operation IDs yet |
| R20 | Observability and run outputs | Prefer the typed activity/transcript/evidence/usage set; replace giant poll summary with one small authoritative status projection; graduate the existing binary workspace archive; retain objective events only through an exact typed list contract when historical transitions are a real evidence requirement; keep raw logs/traces operator-only | SDK/product, backend runtime, evals | open; active advanced ReportBench reads and archive downloads remain |

### Residual observability disposition

The final read-only consumer audit found that the remaining advanced reads are
not one capability. Review them separately so the stable SDK does not inherit a
single unbounded poll payload:

| Residual | Proposed disposition | Why |
|---|---|---|
| Poll summary | Add one strict `retrieve_swarm_status` operation containing state, liveness, terminal/finalization/recovery outcome, last progress, typed issues/invariants, failure classification, and freshness only | Active evals need a cheap authoritative terminal projection; actors, tasks, messages, evidence, usage, and transcripts already have separate typed resources |
| Workspace archive | Graduate existing `GET /smr/runs/{run_id}/workspace/archive` as `retrieve_swarm_workspace_archive`, with byte-return and destination-download forms plus a typed checksum/commit/media receipt | ReportBench grading, code churn, staged-output hydration, Frontier, DeepSWE, and Crafter require the exact run-owned archive; a project-current fallback can grade the wrong generation |
| Objective events | Remove the periodic/raw pass-through unless historical objective transitions are an explicit evidence requirement; if retained, expose one strict typed list operation | Current active code stores/counts the records but does not consume their arbitrary payload fields |
| Raw logs | Keep generic log search advanced; use activity plus typed status issues for normal progress/failure interpretation | Customer correctness must not depend on parsing log prose |
| Raw traces | Keep under `research.advanced` as an operator/debug capability | The collector requests raw redaction mode and reconstructs actor configuration |
| Launch/preflight | Cut evals to existing `research.swarms.preflight(...)` and `.create(...)`; replace the preflight blocker loose-shape decoder with a closed blocker/denial union | No new backend operation is needed; the current hero adapter simply calls the advanced compatibility API |

The status projection must not repeat actors, tasks, messages, artifacts,
WorkProducts, logs, traces, trained models, or container packages. Those belong
to their existing or separately approved resources. The workspace archive must
remain run/swarm-owned and operation-aware; do not revive a project-current
fallback in the stable path.

### Claims the reviewer should re-measure

The counts in this draft describe the 2026-07-21 tree and are evidence, not
permanent constants. Before approving the plan, refresh and annotate:

- Python files and lines under `synth_ai/core`, `synth_ai/research`, and
  `synth_ai/managed_research`;
- public `*API` classes and exported `Research*` / `Smr*` names;
- public/open-boundary `Any`, `dict`, and `Mapping` return paths;
- backend research routes versus stable public operation IDs;
- active backend and eval imports of deep or legacy SDK modules;
- current docs recipes and generated reference coverage; and
- the Jesterky SDK-design baseline using the canonical quality command.

If a refreshed number changes, update the number and date here. Do not weaken a
design conclusion solely because the exact count moved.

### Required review passes

| Pass | Reviewer focus | Required output in this document |
|---|---|---|
| Product/API | Smallest usable workflow, nouns, verbs, progressive disclosure, Factory timing | Approved public resource tree and hero example |
| Backend contract | Operation IDs, schemas, errors, states, events, idempotency, consistency | Authoritative artifact and versioning decision |
| Python/Synth Style | Dataclasses/unions/enums, invalid states, native async, error hierarchy, package boundaries | Approved type and concurrency patterns |
| Evals | Consumer imports, experiment policy, GameBench ownership, proof workloads | Cutover inventory and representative eval proof |
| CLI/MCP | Thin-adapter rule and parity with public operations | Adapter scope and conformance obligations |
| Docs/DevEx | Five-minute quickstart, concepts, cookbook boundary, migration guide, troubleshooting | Documentation information architecture |
| Release engineering | Compatibility dates, ratchets, wheel proof, quality thresholds | Phase gates and removal schedule |

### Next engineering action

Phases 0–6 and the structural portions of Phases 7–9 now have an unmerged
implementation candidate. The next engineer should review and amend it in this
order rather than repeat the mass move:

1. Start from the three clean heads recorded below. Read this document before
   source review, then verify every `Evidence` claim against those exact heads.
   Preserve unrelated main-worktree changes.
2. Close R1–R20 with named owners. Pay particular attention to the hero loop,
   Factory timing, stable/advanced scope, workspace batching, resource taxonomy,
   image declaration unions, and whether poll summary still deserves a public
   projection.
3. Review the candidate public tree: `projects`, `swarms`, and `factories` are
   stable; `advanced` contains operator breadth; `runs`, `Research*`, `Smr*`,
   and `smr_*` are compatibility only. Preserve stable import isolation: normal
   discovery loads zero `_legacy` modules.
4. Repair contract publication before more SDK breadth. Regenerate the
   64-operation / 197-schema backend artifact, vendor it byte-identically in
   `synth-ai`, and prove source-operation parity. If the reviewer removes an
   operation from the stable set, remove it from source and operation metadata
   before generation rather than editing generated JSON.
5. Finish Phase 6 as narrow verticals: add the four Environment catalog SDK
   operations; design and bound the five image-release operations; explicitly
   classify DevEnvironments/cloud deployments; and keep repository identities
   distinct.
6. Fix all five active ReportBench/SwarmBench/SwarmGameBench workspace call
   sites using the typed stable methods and decide R16 before changing code. A
   method-name correction without a >100-file, idempotency, partial-completion,
   and projection-atomicity contract is a hidden behavior regression. In the
   backend route, perform the authoritative git mutation before materializing
   stored-file projection rows and make retry resume the projection safely.
7. Review `SynthFailure`, activity/transcript SSE, usage, evidence, and Factory/
   Effort semantics as public protocol contracts. Require operation, request,
   correlation, retry, unknown-event, cursor, idempotency, cancellation, and
   freshness behavior to remain observable through Python, CLI, and MCP.
8. Classify every remaining eval `research.advanced` use and the 245 advanced
   MCP definitions. Start with the residual disposition above: status and the
   run-owned workspace archive are proposed stable contracts; objective events
   require product evidence; raw logs/traces stay advanced. Each remaining use
   is an accepted operator dependency with an owner, a missing stable
   capability, or removal—not an implicit compatibility waiver.
9. After review-driven edits, run the deterministic boundary/contract/package
   proofs and a new bounded Gemini 3.5 Flash Lite scan. Do not lower the
   7.0/5.0/no-hold thresholds. v10's 7.27 is the strongest accepted historical
   ratchet; v11's 7.18 predates current HEAD and cannot accept this phase.
10. Request the required user-triggered `/ultrareview` before integrating this
    high-blast-radius change into `dev`.

Do not perform another mass move. Finish one customer-visible vertical with its
backend contract, typed Python API, CLI/MCP parity where appropriate, active
eval consumer, documentation, and failure behavior.

### Reviewer completion checklist

- [ ] R1–R20 have outcomes, rationale, and named owners.
- [ ] The public hero workflow fits in one short, executable example.
- [ ] The operation ledger has a current source disposition, concrete target,
      and runtime receipt for every claimed capability; no `unresolved` field is
      being counted as closure.
- [ ] Backend authority and the generated/conformance boundary are unambiguous.
- [ ] Sync, async, streaming, cancellation, and retry guarantees are written as
      observable contracts.
- [ ] Compatibility includes package versions, dates, aliases, and deletion
      conditions.
- [ ] Each phase has entry criteria, exit evidence, owner, and rollback/cutover
      behavior.
- [ ] Active eval consumers and adapter parity obligations are enumerated.
- [ ] The quality scan and deterministic gates are accepted or amended with
      recorded rationale.
- [ ] The document status is changed only after all required owner reviews.

### Current handoff state

- Canonical plan: this file.
- Goal record: `Jstack/.jstack/records/goals/2026-07-21-019f8645.md`.
- Goal id: `019f8645`; the goal remains active pending engineer review,
  implementation completion, review gates, and integration.
- `synth-ai` worktree:
  `/Users/joshpurtell/Documents/GitHub/synth-ai-worktrees/sdk-core-research-migration`
  on `sdk-core-research-migration-20260721`.
- `backend` worktree:
  `/Users/joshpurtell/Documents/GitHub/backend-worktrees/sdk-core-research-migration`
  on `sdk-core-research-migration-20260721`.
- `evals` worktree:
  `/Users/joshpurtell/Documents/GitHub/evals-worktrees/sdk-core-research-migration`
  on `sdk-core-research-migration-20260721`.
Review these exact clean **source baselines**; the consolidation documentation
commit sits on top of the Synth baseline, and no documentation-pivot agent left
uncommitted source changes:

| Repository | Head | Relative to `origin/dev` | Latest committed scope |
|---|---|---:|---|
| `synth-ai` | `320be4ba` | 38 ahead | ledger receipt over `56ddecd2` project data, `bf4e8c68` transcript/SSE, `f7c89c65` workspace inputs, and `aced5585` activity |
| `backend` | `64361869d` | 25 ahead | strict Environment catalog over `941408134` project data, `5b9092ef7` workspace inputs, `009e3a9bb` transcript/SSE, and `4d9bf5dd5` activity |
| `evals` | `5301844d` | 36 ahead / 8 behind | typed transcript over `d52b69cf` usage and `381ff1a0` activity; five active workspace call sites still use invalid stable methods |

Earlier foundational checkpoints remain in each branch history: `9e992da0` and
`1a050cef` establish/consolidate the core; `adc56a5b` isolates stable imports;
`7088c78f` and eval `e16d0bf2` move GameBench ownership to evals; `e0b889c8`,
`d2efdc44`, and `4f8ad880` add resolved configuration, usage, and durable
evidence; backend `6b5944bb9` through `86cc5ce12` author the matching bounded
Factory/configuration/usage/evidence contracts. Git history is the detailed
change log; this document records review-significant checkpoints only.
- The current capability ledger v2 inventories **2,807 source
  definitions/aliases/references**. It
  reports zero missing source disposition/noun classifications and zero active
  backend/eval deep imports, but also **2,807 unresolved runtime-availability
  rows**, **1,304 unresolved target dispositions**, **1,301 unresolved canonical
  targets**, and **11 unresolved eval invocation rows**. It records 64 stable
  operation IDs with 64 discovered handlers, 43 advanced backend routes, 45
  advanced eval import rows, 46 stable MCP definitions, 245 target-unresolved
  advanced MCP definitions, 150 generated compatibility-shim files / 754
  lines, 54 lazy
  public compatibility exports, and 117 relocated `_legacy` implementation
  files / 46,295 lines. Re-measure these figures if review changes the branch.
- Current candidate magnitude (physical Python lines, including compatibility
  code) is **414 files / 81,974 lines** under `synth_ai`; **158 / 58,170** under
  `synth_ai/core/research`; **20 / 672** in the thin `synth_ai/research` facade;
  and **150 / 754** in the generated `synth_ai/managed_research` shims. The
  largest remaining file is `_legacy/sdk/client.py` at 6,124 lines, followed by
  the stable Research MCP server at 3,504 lines. These are magnitude indicators,
  not permission to split files without a concern boundary.
- The cumulative `synth-ai` candidate implements typed projects/swarms/
  factories, configuration, activity, transcript/SSE, usage, durable evidence,
  workspace inputs, external repositories, and project datasets; native
  sync/async parity; typed failures; a thin advanced bridge; CLI; 46 stable MCP
  definitions; and the refreshed ledger. These changes are review material,
  not an approved final API.
- The cumulative backend candidate source allowlists 64 operations and adds the
  matching activity/transcript/workspace/project-data contracts plus four strict
  Environment catalog operations. Its checked-in 49-operation artifact is
  stale and must not be described as current contract proof.
- The cumulative eval candidate contains the stable/advanced classifications
  plus ReportBench configuration, activity, transcript, usage, and durable-
  evidence cutovers. Workspace source/upload cutover is incomplete and names
  two nonexistent methods.
- Cross-repository commits `7088c78f` (synth-ai) and `e16d0bf2` (evals) remove
  `synth_ai/gamebench` and place the exact scorer contracts, client, and recipe
  under `evals/swarmbench/gamebench`.
- The newest handoff defects are recorded in Jstack: the workspace
  method/batching defect under `evals` at `reportbench/hero_driver.py:139`
  (observed at `5301844d`), the workspace projection-before-git integrity defect
  under `backend` at
  `app/api/v1/managed_research/collaboration.py:2529` (observed at
  `64361869d`), and the recurring stale Research OpenAPI artifact under
  `backend` at `research_openapi.json` (observed at `64361869d`).
- Main worktrees may contain unrelated user changes and must not be reset,
  stashed, or folded into this migration.
- Baseline SDK-design verdict: **FAIL, 5.64/10**, 11/11 dimensions, two holds,
  30 findings, 16 seconds, Gemini 3.5 Flash Lite.
- Relocation checkpoint verdict: **FAIL, 6.45/10**, 11/11 dimensions, no holds,
  13 seconds, approximately 89k tokens. Naming and conceptual compression were
  both **4/10**.
- Strongest accepted stable API candidate verdict: **PASS, 7.27/10**, 11/11
  dimensions, minimum 6, maximum 8, no holds, and 20 recorded violations. The
  later typed-usage verification also passes the absolute floor at **7.18/10**,
  minimum 6, no holds, and 18 violations, but is 0.09 lower and is retained as
  a ratchet-review receipt. Both 21-second Gemini 3.5 Flash Lite runs used
  approximately 88k tokens. The complete scorecards, variance explanation, and
  findings are in Part VI.
- Repeatable command: `./synth-dev/quality ./synth-ai` from the workspace root.
- Repeatable-scan tooling is not yet committed: `synth-dev/quality` and
  `synth-dev/config/quality/` remain untracked because the `synth-dev` main
  worktree has a pre-existing unresolved merge in
  `local_dev/sltop-rs/src/main.rs`. The `synth-dev` owner must resolve that
  independently and commit the reviewed tooling; do not stage the unrelated
  slot-UI conflict with it.
- Baseline proof:
  `synth-ai/.quality/runs/sdk-design-proof-20260721T211647Z/summary.md`.
- Checkpoint Jesterky manifest:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design/manifest.json`.
- Candidate Jesterky manifest:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v2/manifest.json`.
- Accepted Jesterky manifest after stable-import isolation:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v6/manifest.json`.
- Strongest Jesterky repeat after GameBench ownership and contract-parity fixes:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v8/manifest.json`.
- Latest Jesterky repeat after advanced-consumer classification:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v9/manifest.json`.
- Final Jesterky handoff repeat after the resolved-configuration vertical and
  final capability ownership:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v10/manifest.json`.
- Post-usage-contract verification receipt:
  `synth-ai-worktrees/sdk-core-research-migration/.quality/runs/migration-sdk-design-v11/manifest.json`.
- Earlier deterministic candidate checks passed generated compatibility-shim
  conformance, architecture boundaries, cross-repository migration boundaries,
  49-operation OpenAPI parity, public symbol/identity smoke, typed-failure
  smoke, unknown-event protocol smoke, and diff integrity. Later verticals each
  passed targeted source compilation/import/diff checks, but no unified
  current-head validation was run. The earlier OpenAPI proof is now invalidated
  by the 64-source/49-artifact split; all checks must be rerun after review.
- Clean-process import isolation now passes: all stable manifest modules load
  zero `_legacy` modules, while deliberately accessing `.advanced` loads the
  compatibility implementation. Compatibility-only error aliases remain lazy.
- Ruff, `ty`, and tests were not run in this handoff pass. Workspace policy
  prohibits Ruff/`ty` unless the user asks for those validations, and prohibits
  tests during normal development unless requested. This is an explicit
  validation gap, not a green result.
- The canonical full quality wrapper currently includes Ruff and `ty`; the
  candidate qualitative scan was therefore run directly through Jesterky with
  Gemini 3.5 Flash Lite. Release evidence still requires the user-authorized
  deterministic lanes plus the qualitative lane.

The next owner is the reviewing engineer. Their deliverable is an edited,
decision-complete version of this document, named owners, and a review verdict
on the existing candidate. Implementation should resume only from those
resolved decisions; completion still requires the release gates, user-triggered
ultrareview, `dev` integration, worktree cleanup, and compatibility follow-through.

### Handoff scorecard

**Overall verdict: REVIEW REQUIRED — NOT READY TO MERGE.** The qualitative SDK
design bar passes; the migration and release bar do not yet pass.

| Gate | Status | Current evidence / required next proof |
|---|---:|---|
| Authority and source disposition | REVIEW | 2,807 source rows have a noun/current disposition and all 88 advanced backend/eval rows have owners, but 1,304 target dispositions and 1,301 canonical targets remain unresolved; backend still owns policy/state/persistence and evals owns benchmark behavior |
| Bounded backend contract | **FAIL — STALE** | Backend source has 64 operation IDs and a fresh 197-schema export; checked-in backend and SDK artifacts still have 49 / 155. Regenerate and re-prove atomic parity before more breadth |
| Active deep-import boundary | PASS (source proof) | Zero active backend/eval deep imports in the generated migration ledger; runtime eval success is tracked separately and remains unproved |
| Compatibility shim conformance | PASS | 150 generated warning-only files / 754 lines; removal scheduled for 0.18.0 no earlier than 2026-09-01 |
| Public architecture ratchets | PASS | Concern-based core, thin public facade, architecture and cross-repo migration checkers green |
| Stable import isolation | PASS | Clean stable manifest import loads zero `_legacy` modules; explicit `.advanced` access loads the compatibility implementation |
| Public API qualitative design | HISTORICAL PASS / CURRENT HEAD UNPROVED | Strongest accepted repeat: 7.27 mean (80/110), minimum 6, no holds. v11: 7.18 (79/110), minimum 6, no holds, 18 violations, 21 seconds. Both predate the activity/transcript/workspace/project-data head and cannot accept it |
| Parameter / naming / reach quality | WARN | Each is 6/10; eliminate optional-soup budget/resource states, enforce alias removal, and add cross-surface OpenAPI conformance |
| Typed failure and event protocols | PASS (source proof) | Typed codes/categories/retry/request/correlation metadata and unknown-event preservation; reviewer must approve them as stable protocol |
| GameBench ownership | PASS | No `synth_ai/gamebench` package or SDK recipe remains; scorer contracts/client/recipe and the active grader live under `evals/swarmbench/gamebench` |
| Factory/Effort vertical | REVIEW | 12-operation candidate is typed and bounded; owner must approve lifecycle semantics and prove representative backend behavior |
| Resolved swarm identity | PASS (source proof) | Exact run-bound config version, redacted immutable snapshot, public digest, typed sync/async API, CLI command, and stable MCP tool; live backend proof remains required |
| Typed swarm usage evidence | PASS (source proof) | One strict backend projection; exact aggregate units; honest cent-only actor precision; closed tokens; explicit freshness; sync/async Python, CLI, stable MCP, and three ReportBench consumers; live backend proof remains required |
| Durable swarm evidence | PASS (source proof) | One strict artifact/WorkProduct index; closed states/roles; count invariants; operation-aware binary content; sync/async Python, CLI, stable MCP, and ReportBench cutover; live backend proof remains required |
| Resource verticals | PARTIAL | Workspace inputs and project repository/dataset APIs are typed end to end; Environment is backend-only; image releases, DevEnvironments, and cloud deployments remain ungraduated; >100-file workspace semantics are undecided |
| Eval consumer cutover | **REVIEW — BROKEN CALLS** | Deep, standalone-package, and top-level compatibility-client imports are zero, and typed evidence consumers landed. Five active ReportBench/SwarmBench/SwarmGameBench call sites use nonexistent `projects.repos.attach` or `projects.workspace.upload`; 11 ledger rows and all runtime receipts remain unresolved |
| Package/docs/live vertical proof | NOT RUN | Required before release; no wheel/import matrix, generated docs proof, or live local/staging Factory/Swarm proof in this handoff |
| Ruff / `ty` / tests | NOT RUN | Explicitly skipped under workspace policy because the user did not request those validations |
| User-triggered `/ultrareview` | PENDING | Mandatory before integration for this high-blast-radius migration |
| `dev` integration and worktree cleanup | PENDING | Merge only after review; prove commits reachable from `dev`, then remove all three worktrees and feature branches |

## Pause handoff — 2026-07-22T03:48:02Z

This is the authoritative freeze point for the next engineer. It supersedes the
older exact-head and incomplete-workspace statements in the historical handoff
above. The goal remains **active**, not blocked or complete; implementation was
paused at the user's request before review, release proof, or integration.

### Frozen worktrees

All three worktrees use branch `sdk-core-research-migration-20260721`. Do not
reset, stash, or replace their main sibling worktrees.

| Repository | Worktree | Frozen source head before this handoff commit | `origin/dev` delta | Worktree state |
|---|---|---|---:|---|
| `synth-ai` | `/Users/joshpurtell/Documents/GitHub/synth-ai-worktrees/sdk-core-research-migration` | `710583f5` | 44 ahead / 0 behind | This documentation commit sits on top; two interrupted image-release files remain untracked; every other file is clean |
| `backend` | `/Users/joshpurtell/Documents/GitHub/backend-worktrees/sdk-core-research-migration` | `4add92832` | 27 ahead / 0 behind | Clean |
| `evals` | `/Users/joshpurtell/Documents/GitHub/evals-worktrees/sdk-core-research-migration` | `2f0b11df` | 38 ahead / 8 behind | Clean; rebase/merge disposition belongs to review |

Goal record:
`Jstack/.jstack/records/goals/2026-07-21-019f8645.md` (`019f8645`). All
implementation subagents were interrupted before this freeze.

### New committed work after the previous handoff

`synth-ai`:

- `d34292ed` — deterministic workspace batches: 100 files per server mutation,
  10,000 files per composite request, content-derived idempotency, complete
  receipts, and exact partial-progress errors for sync/native async.
- `0cdaaaf0` — strict four-operation Environment catalog in sync/native async
  Python, CLI, and stable MCP; deliberately does not expose DevEnvironment
  lifecycle as an Environment concern.
- `6de7b40e` — public workspace batch exports and idempotent operation metadata.
- `710583f5` — routes the three existing workspace MCP names through the typed
  core API, adds the composite bound and exact parsing/scopes, removes the
  obsolete dictionary request wrapper, exposes the tools in stable discovery,
  and adds binary-safe CLI workspace inspect/source/upload commands.

`backend`:

- `e910fc69c` — performs the authoritative workspace git mutation before stored
  file projection, propagates a bounded idempotency key, and allows projection
  completion to resume after an existing idempotent git result.
- `4add92832` — bounds the five customer image-release operations with strict
  request/response schemas. The implementer's latest temporary export reported
  69 operations / 211 schemas; this is not a replacement for the checked-in
  contract artifact or a current-head parity receipt.

`evals`:

- `fe2ae068` — converts five ReportBench/SwarmBench/SwarmGameBench workspace
  upload sites to exact `WorkspaceFileUpload` values and deterministic batches,
  including binary-safe base64 handling.
- `2f0b11df` — preserves an explicit advanced compatibility route for broad
  ReportBench launch arguments that the current stable `SwarmSpec` cannot yet
  represent. This prevents data loss during migration; it is not the final
  launch cutover.

The matching integrity defects and fixes are recorded in
`Jstack/.jstack/records/bug_fixes/2026-07.md`, including workspace mutation
ordering, eval launch compatibility, and legacy MCP workspace dispatch.

### Quarantined, uncommitted image SDK draft

The interrupted image-release SDK agent left exactly two untracked files:

- `synth_ai/core/research/contracts/image_releases.py` — 1,187 lines.
- `synth_ai/core/research/image_releases.py` — 221 lines.

Do **not** stage them as-is. They were interrupted before compile/import/wire
proof and before operation-registry, client composition, public exports, CLI,
or MCP integration. The 1,187-line contract draft also fails the intended
concern/magnitude posture and needs reviewer-led decomposition or deletion.
Inspect it only as recoverable design material against backend `4add92832`.
The paused status/archive and shared retry tasks left no source edits.

### Release-significant launch audit

The remaining launch gap is correctness, not naming polish:

- Public `ResearchRunLaunchRequest` resolves to stable `SwarmSpec`, but the
  active SwarmBench and SwarmGameBench builders pass unsupported `roles`,
  `kickoff_contract`, `execution_target`, and, for SwarmBench,
  `actor_image_overrides`.
- `research.runs` is only a deprecated alias for `SwarmsAPI`; it has no
  `check_preflight` or `create_configured`. The stable shape is one immutable
  spec reused by `research.swarms.preflight(spec, project_id=...)` and
  `research.swarms.create(spec, project_id=...)`.
- Central ReportBench still prefers raw `_request_json`. Its adapter currently
  drops the staged `kickoff_contract`, while its `workflow` field does not exist
  in the backend request and is silently ignored by Pydantic's current default.
- Do not solve this by adding raw dictionaries to `SwarmSpec`. The minimum
  reviewed additions are typed role bindings, platform-resolved execution
  target, admitted image-release bindings, required resource capabilities,
  an Environment reference, canonical `idempotency_key_run_create` wire output,
  and a kickoff strategy based on typed messages plus a workspace artifact.
- Keep signed bound-runtime attestations, raw sandbox/cache overrides, inline
  parent mutation, and unclosed kickoff dictionaries in advanced/control-plane
  ownership unless an explicit product decision promotes a closed contract.
- Backend `actor_model_overrides` and `actor_image_overrides` are different
  authorities and must not be collapsed into a generic override type.

The full field-by-field backend/SDK disposition was captured during the final
read-only audit; re-derive it from `services/smr/api_schemas.py` and
`synth_ai/core/research/contracts/swarms.py` before editing because neither
contract is approved as final.

### Ordered closeout for the next engineer

1. Review the three frozen heads and decide whether to split/salvage or delete
   the two untracked image files. Do not mix them into the first review commit.
2. Finish the image-release vertical concern by concern: strict SDK contracts,
   sync/native-async resource, operation metadata, client composition, public
   exports, CLI, stable MCP, and exact backend OpenAPI parity.
3. Add the small backend-owned swarm-status projection and graduate the existing
   run-owned `application/gzip` workspace archive with stable operation IDs;
   project only authoritative public-run fields and do not duplicate recovery
   or finalization policy in a route or SDK decoder.
4. Close the launch-contract decisions above, then rebuild SwarmBench,
   SwarmGameBench, and central ReportBench around the same immutable typed spec
   for preflight and create. Stage kickoff data as a typed workspace artifact,
   preserve the objective, and delete the nonexistent `workflow` field.
5. Wire the existing `RetryPolicy` into sync/native-async `execute` with exact
   parity: retry only metadata-declared idempotent operations, require an
   idempotency key for unsafe methods, retry only classified transient failures,
   cap deterministic exponential/`Retry-After` delays, and never retry contract
   decode failures. Bytes and SSE need explicit, separately reviewed semantics.
6. Regenerate the bounded backend artifact and vendor it byte-identically in
   `synth-ai`; regenerate the capability ledger and prove operation/handler,
   Python, CLI, MCP, compatibility, and stable-import parity at the same heads.
7. Run only the validations the user authorizes. This freeze has targeted
   compilation/import/schema/diff receipts for landed verticals, but no unified
   current-head gate, package/docs/live proof, Ruff, `ty`, or tests. The latest
   Gemini/Jesterky receipts predate these heads and cannot accept them.
8. Ask the user to trigger `/ultrareview`. Only after it clears should the
   engineer integrate all three branches into `dev`, prove reachability, remove
   the worktrees, and delete the temporary branches.

### Pause scorecard delta

| Gate | Frozen status | Exact next proof |
|---|---:|---|
| Workspace batching / projection integrity | PASS (targeted source proof) | Live retry and >100-file receipt proof |
| Environment vertical | PASS (targeted source proof) | Checked-in OpenAPI parity and representative backend call |
| Image-release backend | PASS (targeted backend source proof) | Reviewed/split SDK vertical and checked-in parity |
| Image-release SDK | **UNCOMMITTED / UNREVIEWED** | Decide salvage/delete; then complete every adapter and proof |
| Swarm status / workspace archive | NOT IMPLEMENTED | Backend-authoritative DTO/download plus SDK/adapters |
| Launch contract and eval cutover | **FAIL** | One immutable typed spec; no raw fallback or dropped fields |
| Shared retry guarantee | NOT IMPLEMENTED | Sync/native-async classified retry receipt |
| Checked-in OpenAPI / capability ledger | **STALE** | Same-head byte and operation parity |
| Current-head qualitative score | UNPROVED | Gemini 3.5 Flash Lite scan at mean >=7, every dimension >=5, no holds |
| Release review / integration | PENDING | User-triggered ultrareview, authorized gates, merge and cleanup |

### Resume continuation after pause — 2026-07-22T04:30:00Z

Goal remained active. Implementation resumed from frozen heads `synth-ai ea2eea9e`, `backend 4add92832`, `evals 2f0b11df`.

#### Disposition of quarantined image drafts

- Deleted the bloated untracked `contracts/image_releases.py` draft (~1187 lines).
- Kept a lean contracts module (~409 lines) plus the existing resource API (~221 lines).
- Completed the full vertical: ops, core/async clients, research facade, CLI (`research image-releases`), MCP (`smr_*` renamed to stable `research_*` via registry).

#### Closed in this continuation

| Vertical | Result | Heads |
|---|---|---|
| Image-release SDK | PASS (source-complete) | synth-ai `1d102df9` |
| Shared RetryPolicy on sync/async execute | PASS (source-complete) | synth-ai `1d102df9` |
| Swarm status + workspace archive | PASS (backend + SDK) | backend `7aa3646b9`, synth-ai `1d102df9` |
| SwarmSpec launch fields + SwarmBench cutover | PASS (source-complete) | synth-ai `1d102df9`, evals `fa103a6b` |

#### Still open before merge

1. Checked-in OpenAPI + capability ledger same-head regen/vendor (artifacts still stale vs source allowlist now at 71 ops).
2. Remaining eval cutovers: SwarmGameBench + ReportBench/hero_driver still on advanced/compat launch paths.
3. Authorized gates: Ruff, ty, tests, package/docs, live proofs — not run this continuation.
4. Current-head Gemini/Jesterky qualitative scan (prior receipts predate these heads).
5. User `/ultrareview`, then merge to `dev`, prove reachability, remove worktrees.

#### Current committed heads

- synth-ai: `1d102df9`
- backend: `7aa3646b9`
- evals: `fa103a6b`

No branches merged. No worktrees removed. Validation not run (user gate authorization required).

### Continuation — OpenAPI parity + eval cutovers — 2026-07-22T04:45:00Z

- Regenerated backend `research_openapi.json` and vendored byte-identically to `openapi/research-v1.json` (71 operations / 223 schemas; SHA-256 `c8954b0bd209a47ac868aa36fa7fcbfae6c12f2db8a632ec80a40a304b1ef25e`).
- Regenerated `specifications/sdk/research_capability_ledger.json` at the same heads.
- Exported `LocalExecution` / `ExecutionProfile` from `synth_ai.research`.
- SwarmBench now stages `kickoff_contract.json` into the workspace before `KickoffArtifact` reference.
- SwarmGameBench cut over to typed `SwarmSpec` + `research.swarms.preflight/create` (including cloud slot-local execution).
- ReportBench `hero_driver` now requires `SwarmSpec` only (compat kwargs / advanced aliases closed). `reportbench/client_api.py` still has a raw `_request_json` trigger path that remains to cut over.

Still open: ReportBench `client_api` raw trigger, authorized gates, Gemini/Jesterky current-head scan, `/ultrareview`, merge + worktree cleanup.

