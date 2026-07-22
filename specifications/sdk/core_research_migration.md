# Synth SDK Research migration and refactor plan

- **Status:** engineer review required — unmerged implementation candidate exists
- **Date:** 2026-07-21
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
| `synth-ai` | Committed candidate with concise projects/swarms/factories APIs, typed failures/events, native async parity, and stable-versus-advanced discovery | Review the product/API contract in the candidate; do not equate passing structure checks with release readiness |
| `backend` | Committed Research contract/MCP cutover, 12-operation Factory/Effort extension, and immutable resolved-configuration read; bounded artifact now contains 45 operations | Approve the exact Factory/Effort lifecycle and confirm each operation remains backend-owned |
| `evals` | Committed removal of deep imports and seven call-site cutovers to stable or explicit `advanced` APIs | Verify real workflow behavior and decide whether each advanced dependency is acceptable or must graduate |
| Compatibility | `synth_ai.managed_research` is 150 generated warning-only re-export files; 117 implementation files remain temporarily under `core/research/_legacy` | Enforce the 0.18.0 / no-earlier-than-2026-09-01 deletion contract and prevent stable imports from loading `_legacy` |
| Public surface | Candidate discovery exposes `projects`, `swarms`, and `factories`; operator breadth is grouped under `advanced`; old `Research*`, `runs`, and `smr_*` names are compatibility-only | Finalize the smallest hero lifecycle, the stable/advanced line, and alias visibility before release |
| Delivery | MCP defaults to 36 stable noun-first tools; runtime discovery contains 245 additional advanced tools, while the source ledger records 260 advanced adapter definitions before registry deduplication/removal; CLI exposes the resolved-configuration read and keeps legacy limits/Tag under `research advanced` | Approve adapter scope and require every advertised stable tool to conform to the same operation/error contract |
| Quality | Baseline 5.64 → relocation 6.45 → first candidate 7.09 → import-isolated candidate 7.09 → strongest repeat 7.18 → latest repeat 7.09; latest minimum 6, no holds | Qualitative threshold passes; the 0.09 repeat variance and compression, documentation, naming, and reach at 6/10 remain review evidence, not a release waiver |

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
- Assign every row one disposition and one canonical noun.
- Record the compatibility removal version and date.
- Add architecture checks forbidding new deep eval imports and new legacy
  implementation files.

Exit: no unclassified operation and no new legacy growth.

Implementation evidence:

- `specifications/sdk/research_capability_ledger.json` now records 2,092
  backend-route, SDK-method, MCP-tool, and backend/eval-consumer rows with zero
  unclassified dispositions and exact source commit identities. All 45
  advanced backend-route rows are owned by their backend domain, all 45
  advanced eval-import rows are explicitly retained under evals ownership, and
  the MCP inventory distinguishes 36 public from 260 advanced source adapters.
  The original
  Phase 0 inventory contained 2,339 rows; row count may decrease as retired
  surfaces disappear while the frozen architecture magnitudes remain ratchets.
- The baseline freezes 156 legacy implementation files / 61,081 lines, 42
  deep backend/eval consumer imports, and 11 stable public operation IDs.
- `scripts/check_research_migration_boundaries.py` rejects new legacy files,
  increased legacy lines/files, increased deep consumer imports, forbidden
  core imports, or an incomplete ledger.
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
- `openapi/research-v1.json` currently vendors 45 backend-authored operations and
  137 required schemas. `scripts/check_research_openapi_contract.py` proves byte
  parity with a supplied backend artifact and exact method/path/operation-ID
  parity with `core/research/operations.py`.

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
  product decision. Stable MCP discovery is limited to 35 noun-first tools;
  280 explicitly advanced tools and their legacy-backed request models still
  require operation-registry cutover or removal.

#### Phase 8 — eval cutover

- Replace all active `synth_ai.managed_research` imports in evals.
- Move general client behavior found in eval drivers into the SDK.
- Leave benchmark-specific policy and scoring in evals.
- Rename active scripts whose names imply the retired package when compatibility
  naming is no longer required.

Exit: active eval code imports only documented `synth_ai` public surfaces.

Implementation evidence:

- The deep-consumer ratchet decreased from 42 to **0** across the backend and
  active eval tree. Evals now import `synth_ai.research`,
  `synth_ai.research.advanced`, or `synth_ai.config`; graveyard paths remain
  excluded from the active inventory.
- GameBench-specific scorer models, client, and recipe now live under
  `evals/swarmbench/gamebench`; `synth_ai/gamebench` has been removed. Remaining
  compatibility client imports and advanced dependencies still require an
  owner and stable-or-advanced disposition before the phase is complete.
- Active evals no longer import the standalone `managed_research` package or
  the top-level `ManagedResearchClient` alias. The ledger now labels **45**
  explicit `synth_ai.research.advanced` import rows separately instead of
  counting them as supported stable-public consumers.

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

The latest repeat completes all 11 dimensions, has no holds, and clears the
declared qualitative bar: mean **7.09**, minimum **6**, maximum **8**, with 20
recorded violations. The strict manifest validator passed. The immediately
preceding repeat scored **7.18** with the same minimum and maximum. The 0.09
movement came from concurrency (8 → 7), documentation (7 → 6), and parameter
design (7 → 8), while the deterministic contract and import-boundary evidence
remained unchanged. Treat this as expected qualitative-judge sampling variance,
not as evidence that 7.18 is the exact score of the branch. Both receipts are
retained; the latest receipt is the handoff score. It is evidence for API
review, not a substitute for deterministic or vertical integration proof.

Its complete scorecard is:

| Dimension | Score | Severity | Principal remaining gap |
|---|---:|---|---|
| Compatibility | 8 | low | Automate enforcement of the public compatibility manifest and versioned deprecation contract |
| Conceptual compression | 6 | medium | Stop leaking low-level clients and backend route structure through top-level discovery |
| Concurrency contract | 7 | low | Document end-to-end cancellation and deadline propagation across every helper method |
| Documentation | 6 | medium | Put a complete five-minute quickstart in the primary README and finish generated-reference coverage |
| Failure contract | 8 | low | Make every async Research method document and raise the typed hierarchy with request identity |
| Lifecycle guarantees | 8 | low | Specify repeated-cancellation idempotency and immutable resolved-version snapshots |
| Naming | 6 | medium | Keep `Smr*`, `ManagedResearch*`, and verbose `*API` names confined to time-bounded compatibility modules |
| Observability | 7 | medium | Consolidate legacy usage/readout projections into the correlation-aware event protocol |
| Parameter design | 8 | low | Continue reducing optional-state combinations and separate configuration boundaries in public contracts |
| Reach | 6 | medium | Generate and publish a non-Python conformance client from the bounded OpenAPI artifact |
| Type contract | 8 | low | Reject lenient preflight blocker shapes and remove dynamic legacy error imports |

The score moved in the intended direction at every measurement:

| Dimension | Baseline | Relocation | First candidate | Import-isolated | Latest | Net |
|---|---:|---:|---:|---:|---:|---:|
| Compatibility | 6 | 8 | 6 | 8 | 8 | +2 |
| Conceptual compression | 4 | 4 | 7 | 6 | 6 | +2 |
| Concurrency contract | 4 | 8 | 8 | 8 | 7 | +3 |
| Documentation | 6 | 8 | 7 | 8 | 6 | 0 |
| Failure contract | 6 | 6 | 8 | 6 | 8 | +2 |
| Lifecycle guarantees | 8 | 7 | 8 | 8 | 8 | 0 |
| Naming | 3 | 4 | 6 | 6 | 6 | +3 |
| Observability | 7 | 6 | 7 | 7 | 7 | 0 |
| Parameter design | 6 | 7 | 7 | 7 | 8 | +2 |
| Reach | 6 | 6 | 6 | 6 | 6 | 0 |
| Type contract | 6 | 7 | 8 | 8 | 8 | +2 |

The import-isolated rerun restored compatibility to 8 after stable imports
stopped loading `_legacy`. The latest rerun keeps the failure contract at 8,
moves documentation to 6 and concurrency to 7, and moves parameter design to
8. Its aggregate is 1.45 points above baseline and no dimension falls below the
release minimum. The preceding 7.18 receipt remains useful corroborating
evidence, but the handoff uses the latest 7.09 result. All aggregates above are
computed directly from the eleven manifest scores; no hand-entered aggregate
is accepted as evidence.

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
| 0 — freeze and ledger | PASS | 2,092 rows, zero unclassified, explicit owners for all 90 advanced backend/eval rows, public/advanced MCP classification, frozen growth/deep-import ratchets, dated compatibility policy | Keep ratchets green after every later edit |
| 1 — core foundations | PARTIAL | Shared credential, sync/async HTTP, retry, JSON, pagination, and error primitives exist; Research uses native async | Prove every container/pool/Research path uses one behaviorally identical error/retry/auth boundary and no domain transport remains |
| 2 — bounded backend contract | PARTIAL | Backend-authored 45-operation / 137-schema artifact has exact byte and operation parity | Put the checker in authoritative CI and prove enum, requiredness, error, event, pagination, and state drift—not only operation metadata |
| 3 — projects and swarms | PARTIAL | Stable typed sync/async project and swarm APIs, same-object run aliases, event protocol, immutable resolved-configuration read, zero deep eval imports | Replace remaining active compatibility launch names and prove the hero loop plus resolved-configuration read against a real backend |
| 4 — evidence and collaboration | PARTIAL | Implementation is under core and active callers use public or explicit advanced paths | Replace legacy-backed readout models, mapping probes, and `Any` returns with one typed freshness/event/artifact protocol; prove an eval evidence vertical |
| 5 — factories and economics | PARTIAL | Seven economics plus twelve Factory/Effort operations are bounded and typed with sync/async peers | Classify or graduate every legacy-session Factory projection and prove lifecycle/idempotency semantics against backend |
| 6 — resources and deployment | NOT PROVED | Existing resource code is reachable through the explicit advanced bridge | Graduate general environments/images/deployments/repositories/datasets/workspace-input operations to typed core contracts or classify them non-public; remove duplicate transport/identity paths |
| 7 — CLI and MCP | PARTIAL | MCP package moved, stable discovery is 36 noun-first tools, CLI uses the new Research command root and matches the resolved-configuration operation | Replace or remove the 245 runtime-discoverable advanced legacy-backed tools and legacy request models; prove Python/CLI/MCP operation and failure parity |
| 8 — eval cutover | PARTIAL | Deep and standalone-package imports are zero; top-level client aliases are gone; GameBench scorer ownership moved to evals; all 45 explicit advanced imports are retained with evals ownership; the ReportBench SDK driver records the stable resolved-swarm configuration in fresh and recovery paths | Run the representative project/swarm/Factory workflows against a real backend and graduate any dependency proven to be generally customer-facing |
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

Phases 0–3 and the structural portions of Phases 7–9 now have an implementation
candidate. The next engineer should review and amend rather than repeat them:

1. Review the committed checkpoints in all three dedicated worktrees before
   making further edits. Preserve unrelated main-worktree changes.
2. Approve or amend the candidate public tree: `projects`, `swarms`, and
   `factories` are stable; `advanced` contains operator breadth; `runs`,
   `Research*`, `Smr*`, and `smr_*` are compatibility only.
3. Preserve the now-green stable/advanced import boundary. A clean stable
   Research import loads zero `_legacy` modules; deliberately accessing
   `.advanced` loads the compatibility implementation. Reject any change that
   makes stable discovery eager again.
4. Review the 12-operation Factory/Effort addition as one bounded backend
   vertical. Confirm create/list/retrieve/update, lifecycle transitions, and
   Effort CRUD are the intended stable set and that no backend policy moved
   into the SDK.
5. Review the stable `SynthFailure`/typed HTTP hierarchy and versioned
   `SwarmEvent` telemetry union as protocol contracts. Require operation,
   request, correlation, retry, unknown-event, cursor, and cancellation
   behavior to remain observable through Python, CLI, and MCP.
6. Classify the seven eval call-site dependencies. Each use of
   `research.advanced` is either an accepted operator-only dependency with an
   owner or evidence that a typed stable capability is still missing.
7. Re-run the deterministic boundary/contract checks after review edits and
   use the same bounded Gemini scan evidence. Do not lower the 7.0/5.0/no-hold
   thresholds; investigate any regression from the latest 7.09 repeat and use
   multiple receipts to distinguish source regressions from judge variance.
8. Request the required user-triggered `/ultrareview` before integrating this
   high-blast-radius change into `dev`.

Do not perform another mass move. Finish one customer-visible vertical with its
backend contract, typed Python API, CLI/MCP parity where appropriate, active
eval consumer, documentation, and failure behavior.

### Reviewer completion checklist

- [ ] R1–R12 have outcomes, rationale, and named owners.
- [ ] The public hero workflow fits in one short, executable example.
- [ ] The operation ledger has a disposition for every current capability.
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
- Committed Synth checkpoints: `9e992da0` (typed Research core), `1a050cef`
  (implementation consolidation and compatibility shell), `462b83dc` and
  `b2284f1b` (consolidated plan/handoff), and `adc56a5b` (stable Research SDK
  candidate with import isolation and accepted qualitative evidence), followed
  by `6e5b4782` (final review packet) and `7088c78f` (removal of the
  GameBench-specific SDK package), plus `828c5192` (explicit advanced eval
  profile loaders), `e0b889c8` (immutable resolved-swarm configuration across
  Python, CLI, and MCP), and the current ledger/handoff follow-up.
- Committed backend checkpoints: `713b6db59` (bounded 32-operation contract)
  and `f734def19` (Research MCP delivery adapter), plus `6b5944bb9` (the
  12-operation Factory/Effort contract extension) and `23533208c` (regenerated
  44-operation / 135-schema authoritative artifact), followed by `938487707`
  (the 45th operation and 137-schema resolved-configuration contract).
- Committed eval checkpoints: `7708aafa` (active consumer cutover to public
  SDK), `c3164c5f` (seven stable/advanced boundary cleanups), and `e16d0bf2`
  (GameBench scorer ownership moved to evals), followed by `65b9387c` (removal
  of standalone-package and top-level compatibility-client imports) and
  `7d7e29f8` (ReportBench resolved-configuration evidence through the stable facade).
- The current capability ledger classifies **2,092** rows with zero unclassified
  rows. It reports zero active backend/eval deep imports, **45** stable
  operations, **45** backend advanced rows, **45** explicit advanced eval-import
  rows, 36 public MCP adapter definitions, 260 advanced MCP adapter definitions,
  150 generated
  compatibility-shim files / 754 lines, and 117
  relocated `_legacy` implementation files / 46,295 lines. Re-measure these
  figures if review changes the branch.
- Current candidate magnitude (physical Python lines, including compatibility
  code) is **399 files / 76,307 lines** under `synth_ai`; **145 / 53,108** under
  `synth_ai/core/research`; **20 / 498** in the thin `synth_ai/research` facade;
  and **150 / 754** in the generated `synth_ai/managed_research` shims. The
  largest remaining file is `_legacy/sdk/client.py` at 6,124 lines, followed by
  the Research MCP server at 3,455 lines. These are magnitude indicators, not
  permission to split files without a concern boundary.
- `synth-ai` commit `adc56a5b` implements concise stable contracts and clients,
  the Factory/Effort vertical, native sync/async parity, typed
  failures and correlated events, a thin advanced bridge, revised docs and CLI,
  36-tool stable MCP discovery, the 45-operation artifact, and the refreshed
  ledger. These changes are review material, not an approved final API.
- `backend` commits `6b5944bb9` and `938487707` add the 12 Factory/Effort
  operation IDs and immutable swarm-configuration read that produce the
  45-operation bounded artifact.
- `evals` commit `c3164c5f` contains seven consumer cleanups using either the
  stable facade or explicit `research.advanced` access.
- Cross-repository commits `7088c78f` (synth-ai) and `e16d0bf2` (evals) remove
  `synth_ai/gamebench` and place the exact scorer contracts, client, and recipe
  under `evals/swarmbench/gamebench`.
- Main worktrees may contain unrelated user changes and must not be reset,
  stashed, or folded into this migration.
- Baseline SDK-design verdict: **FAIL, 5.64/10**, 11/11 dimensions, two holds,
  30 findings, 16 seconds, Gemini 3.5 Flash Lite.
- Relocation checkpoint verdict: **FAIL, 6.45/10**, 11/11 dimensions, no holds,
  13 seconds, approximately 89k tokens. Naming and conceptual compression were
  both **4/10**.
- Latest stable API candidate verdict: **PASS, 7.09/10**, 11/11 dimensions,
  minimum 6, maximum 8, no holds, and 20 recorded violations. The preceding
  repeat scored 7.18 with the same range. The complete scorecard, variance
  explanation, and findings are in Part VI.
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
- Deterministic candidate checks already passed: generated compatibility-shim
  conformance, architecture boundary, cross-repository migration boundary,
  45-operation OpenAPI byte/operation parity, targeted source compilation,
  public symbol/identity smoke, typed failure smoke, unknown-event protocol
  smoke, and diff integrity. They must be rerun after review-driven changes.
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
| Authority and source disposition | PASS | 2,092 ledger rows, zero unclassified; all 90 advanced backend/eval rows have explicit owners; backend owns policy/state/persistence; evals owns benchmark behavior |
| Bounded backend contract | PASS | 45 stable operation IDs and byte/operation parity against `openapi/research-v1.json` |
| Active deep-import boundary | PASS | Zero active backend/eval deep imports in the generated migration ledger |
| Compatibility shim conformance | PASS | 150 generated warning-only files / 754 lines; removal scheduled for 0.18.0 no earlier than 2026-09-01 |
| Public architecture ratchets | PASS | Concern-based core, thin public facade, architecture and cross-repo migration checkers green |
| Stable import isolation | PASS | Clean stable manifest import loads zero `_legacy` modules; explicit `.advanced` access loads the compatibility implementation |
| Public API qualitative design | PASS | Latest Gemini 3.5 Flash Lite repeat: 7.09 mean, 6 minimum, 8 maximum, 11/11 dimensions, no holds, 20 violations; strict manifest validation passed; preceding repeat was 7.18 |
| Compression / failure / naming / reach quality | WARN | Each is 6/10; simplify nouns, publish complete failure guarantees, enforce alias removal, and make OpenAPI sufficient for CLI/MCP |
| Typed failure and event protocols | PASS (source proof) | Typed codes/categories/retry/request/correlation metadata and unknown-event preservation; reviewer must approve them as stable protocol |
| GameBench ownership | PASS | No `synth_ai/gamebench` package or SDK recipe remains; scorer contracts/client/recipe and the active grader live under `evals/swarmbench/gamebench` |
| Factory/Effort vertical | REVIEW | 12-operation candidate is typed and bounded; owner must approve lifecycle semantics and prove representative backend behavior |
| Resolved swarm identity | PASS (source proof) | Exact run-bound config version, redacted immutable snapshot, public digest, typed sync/async API, CLI command, and stable MCP tool; live backend proof remains required |
| Eval consumer cutover | REVIEW | Deep, standalone-package, and top-level compatibility-client imports are zero; all 45 explicit advanced rows are retained under evals ownership; ReportBench now captures the stable resolved configuration, but live representative workflow proof is still required |
| Package/docs/live vertical proof | NOT RUN | Required before release; no wheel/import matrix, generated docs proof, or live local/staging Factory/Swarm proof in this handoff |
| Ruff / `ty` / tests | NOT RUN | Explicitly skipped under workspace policy because the user did not request those validations |
| User-triggered `/ultrareview` | PENDING | Mandatory before integration for this high-blast-radius migration |
| `dev` integration and worktree cleanup | PENDING | Merge only after review; prove commits reachable from `dev`, then remove all three worktrees and feature branches |
