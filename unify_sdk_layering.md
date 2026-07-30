# Unify synth-ai onto one layering DAG

- **Status:** phases 0–4 done and shipped in 0.17.5; phase 5 deliberately deferred
- **Date:** 2026-07-29 (implemented same day)
- **Scope:** `synth-ai` package layering refactor (sibling repos only for import updates)
- **Kind:** refactor — package placement only
- **Public entrypoint unchanged:** `SynthClient().research`

## Implementation record

| Phase | State | Notes |
|-------|-------|-------|
| 0 Decision and docs | done | `synth_ai/README.md`, `core/README.md`, `sdk/README.md` carry one diagram and the import table |
| 1 Boundary ratchet | done | `check_sdk_layering.py`, in **`testing/scripts/`** — not synth-ai/scripts, which no longer exists. Caught the 3 documented `core → sdk.pagination` sites before the move and passes after |
| 2 Mechanical relocate | done | 178 files `core/research` → `sdk/research`, 858 import sites rewritten; alias shim left behind |
| 3 Rewire adapters | done | Imports repointed. The plan's central worry — MCP running "a second ad hoc HTTP path" — did not exist: neither `cli/` nor `mcp/` imports httpx or requests at all |
| 4 Cross-repo callers | **done** | `testing`, `docs`, `evals` (160 refs / 48 files, incl. `old/`), `backend` (1 ref, committed on its `dev`) |
| 5 Shim retirement | deferred on purpose | Phase 4 is done, so it is now *possible* — but see below |

### Deviations from the plan as written

- The reverse edge needed no separate fix. The plan offered "move pagination into
  `core`, or complete the Research relocate first"; relocating made
  `sdk/research → sdk.pagination` a legal peer import, so the edge dissolved.
- The shim is **one finder**, not 178 stub modules. It resolves the real module and
  registers it under both names, so `synth_ai.core.research.X` and
  `synth_ai.sdk.research.X` are the *same object* — `isinstance` across the two
  paths works, which per-file stubs would not guarantee.
- `evals` and `backend` were swept on the second attempt, not the first. Both were
  mid-flight in other people's work — evals had 250+ dirty files and moved
  `containers/common/pool_runner.py` to `containers/nonsensitive/common/` *between
  two checks in one session*. The sweep was applied, reverted by inverse
  substitution (160 references restored exactly), and redone once evals came clean.
  `evals/old/` was included: the alias is scheduled for deletion, and a reference
  left in a deprecated tree breaks as loudly as one in live code.
- Adapter discipline's remaining item is a *preference*, not a violation: MCP
  reaches `sdk.research` directly rather than through `SynthClient().research`.
  The import table marks `mcp → client` "preferred yes", and the plan explicitly
  permits deep contract imports for request parsing.

### Why phase 5 is deferred even though phase 4 is done

Deleting the alias now would give external consumers **no deprecation window at
all**. 0.17.4 is on PyPI with `synth_ai.core.research` as the documented path;
0.17.5 introduces the alias. Removing it in the same release that introduces it
makes the deprecation theatre. Our own repos are clean, which is what the plan
meant by "once siblings are clean" — but pip installers are not a sibling repo.

Remove it in a later release, and when that happens add `core/research` to the
retired-tree absence assertions in `check_research_migration_boundaries.py`
(which already names both spellings in preparation).

### Three things the move nearly shipped

- `pyproject.toml` package-data still globbed `core/research/factory_plans/*.json`
  and `core/research/schemas/*`. Those globs are the only thing putting the data
  files in the wheel; a stale path ships a package that imports and then fails at
  runtime. Verified by building the wheel and reading its manifest.
- A stale `build/` directory caused `python -m build` to package **169**
  pre-move modules alongside the new ones. `build/` is gitignored, so nothing
  flagged it. Any release must clean `build/` first.
- The docs repo's production generator named `synth_ai.research` — a package
  retired *before* this refactor — so the published reference documented code
  nobody could import. Fixed there; regenerating is a separate editorial job
  because 6 links have no valid target. See
  `docs/misc/synth-ai-sdk/REGENERATE-BEFORE-YOU-DO.md`.

## Why

Two stories currently coexist:

- **Infra** (documented in `synth_ai/README.md`): `core/ → sdk/ → client.py → cli/`
- **Research** (from `specifications/sdk/core_research_migration.md` Part I/II):
  Research *implementation* lives under `synth_ai/core/research`, with CLI/MCP as
  adapters over that tree

The prior migration correctly killed `managed_research` as a product authority and
made `SynthClient().research` the front door — but it also made `core/` mean “the
Research SDK,” which conflicts with the infra meaning of `core/` as shared
plumbing. Result today:

- ~169 Research Python files under `core/research/`; only ~14 under `sdk/`
- MCP and research CLI import `core.research.client` / `session` / `public`
  directly, often skipping `SynthClient` and the Research facade
- Reverse edge: `core/research` imports `sdk.pagination`
- Package README claims a clean DAG that Research does not follow

This refactor is **package-layering only**. It does not reopen product nouns,
backend authority, or the `SynthClient().research` public entrypoint.

## Target architecture (single DAG)

```text
backend (authority)
    |
    v
synth_ai/core/          # plumbing only: auth, http, errors, utils, shared contracts
    |
    v
synth_ai/sdk/           # ALL public HTTP clients + domain contracts
  ├── containers, tunnels, pools, managed_agents, ...
  └── research/         # moved from core/research
    |
    v
synth_ai/client.py      # SynthClient / AsyncSynthClient composition
    |
    +--> synth_ai/cli/            # thin terminal adapter (via client/sdk)
    +--> synth_ai/mcp/research/   # thin MCP adapter (via sdk.research)
```

### Import rules

| From \ To | `core` | `sdk` | `client` | `cli` | `mcp` |
|-----------|--------|-------|----------|-------|-------|
| `core` | yes | **no** | no | no | no |
| `sdk` | yes | yes | no | no | no |
| `client` | yes | yes | — | no | no |
| `cli` | yes | yes | yes | — | no |
| `mcp` | yes | yes | preferred yes | no | — |

No parallel Research tree. No “Research is special and lives in core.”

## What stays in `core/`

Keep only true shared runtime plumbing:

- `core/auth`, `core/http`, `core/errors.py`, `core/utils`
- `core/contracts` (generic JSON/error contracts only)

Once Research lives under `sdk/`, peer imports such as `sdk.pagination` are fine.
Until then, the existing `core → sdk.pagination` reverse edge must be fixed
(move pagination into `core`, or complete the Research relocate first).

## What moves to `sdk/research/`

Essentially today’s `synth_ai/core/research/**`:

- Transport client (`client.py`), public facade (`facade.py`), domain APIs
- `contracts/`, `session/`, schemas, factory plans

`SynthClient.research` continues to return the same facade type; only the module
path changes (`synth_ai.core.research.*` → `synth_ai.sdk.research.*`).

## Adapter discipline

After the move:

- **MCP** binds tools to `sdk.research` client/facade/session APIs — not a
  second ad hoc HTTP path
- **CLI** research commands go through `SynthClient().research` (or explicitly
  documented `sdk.research` clients), same as containers/tunnels/pools
- Deep contract imports in adapters are OK for request parsing; business calls
  must not invent a parallel client

## Compatibility

Default: temporary re-export shims at `synth_ai.core.research.*` →
`synth_ai.sdk.research.*` with deprecation warnings, then delete in a follow-up
(same ratchet pattern as retired `managed_research` / `_legacy`).

Rationale: the previous migration directed consumers (`evals`, etc.) at
`core.research`. A hard cut without shims thrashes those trees for no product gain.

## Phases

### 0 — Decision and docs
- This document is the authoritative layering plan
- Supersede Part II of `core_research_migration.md` on *package placement only*;
  preserve product constraints
- Update `synth_ai/README.md`, `core/README.md`, `sdk/README.md` to one diagram

### 1 — Boundary ratchet first
- Extend `scripts/check_research_migration_boundaries.py` (or add
  `check_sdk_layering.py`) to enforce the import table above
- Fix `core → sdk.pagination` before or as part of the relocate

### 2 — Mechanical relocate
- Move `core/research` → `sdk/research` + rewrite internal imports
- Behavior identical; no API redesign
- Leave thin shim package at `core/research` re-exporting `sdk.research`

### 3 — Rewire composition and adapters
- `client.py` / `__init__.py` → `sdk.research`
- MCP and research CLI → `sdk.research` / `SynthClient().research`
- Update boundary checker and any ty overrides naming `core.research`

### 4 — Cross-repo callers
- Sweep `evals`, `testing`, `docs`, and backend-adjacent scripts
- Prefer `synth_ai.sdk.research` or `SynthClient().research`

### 5 — Shim retirement
- Delete `core/research` shims once siblings are clean
- Add `core/research` to retired-tree absence assertions
- CHANGELOG / migration note

## Non-goals

- Changing public product nouns (`projects` / `swarms` / `factories`)
- Splitting the PyPI package
- Redesigning MCP tool names or CLI command names
- Moving business logic into CLI/MCP
- Reopening the `managed_research` deletion schedule except where shim paths collide

## Success criteria

- One documented DAG; README and boundary script agree
- `core/` contains no Research domain implementation
- `sdk/research` is the only Research implementation tree
- CLI and MCP call into `sdk` / `client`, not a parallel stack
- No `core → sdk` imports
- `SynthClient().research` remains the supported public Python entrypoint
- Existing Research behaviors covered by current checks remain green

## Relationship to prior work

| Document | Role |
|----------|------|
| `specifications/sdk/core_research_migration.md` | Historical + product constraints; Part II package graph **superseded** for placement |
| `unify_sdk_layering.md` (this file) | Authoritative layering refactor plan |
| `scripts/check_research_migration_boundaries.py` | Extend or replace to enforce the unified DAG |
