# Handoff: look at all of `synth_ai/`

**Date:** 2026-07-29  
**Audience:** Eng reviewing / cleaning up `synth-ai` after a Research-first trim  
**Repo:** `synth-laboratories/synth-ai` (siblings: `backend`, `testing`, `docs`, `synth-dev`)  
**Scope:** the entire installable package tree `synth_ai/` (~14MB; ~12MB is `sdk/research`)

Backend remains authority for Managed Research / SMR. This package is the Python
SDK + CLI + MCP adapter. Infra clients (containers / tunnels / pools / managed
agents) are **not** live on `SynthClient` today — see §2 and repo-root `old/`.

Related docs: [`unify_sdk_layering.md`](unify_sdk_layering.md), `old/README.md`,
`backend/specifications/tanha/references/synthstyle.md`.

---

## 0. How to read this package (one diagram)

```text
synth_ai/
├── __init__.py, client.py, config.py, __main__.py, py.typed
├── core/                 # plumbing (+ deprecated core.research alias)
├── sdk/
│   ├── pagination.py     # shared list pages (Research uses this)
│   └── research/         # ★ THE PRODUCT (~174 modules)
├── cli/                  # Click / standup adapters
└── mcp/research/         # MCP adapter over sdk.research
```

```text
core/  →  sdk/  →  client.py  →  cli/ + mcp/research/
```

`SynthClient().research` is the public hero. Deep imports of
`synth_ai.sdk.research.*` are OK for adapters; do not invent a second HTTP stack.

---

## 1. Main product surfaces (keep / invest)

```python
from synth_ai import SynthClient
r = SynthClient().research
```

Also: `AsyncSynthClient().research`, CLI `synth-ai research …`,
`synth-ai-research-mcp`, and `synth-ai-research-factory-standup`.

### Hero nouns

| Noun | SDK | Notes |
|------|-----|--------|
| **Projects** | `r.projects` | Org research workspaces |
| **Swarms** | `r.swarms` | Multi-agent runs under a project |
| **Factories** | `r.factories` | Factory lifecycle + typed Efforts |
| **Intern** | `r.intern` | Durable Research Intern + Magi — **shipping / soon**; first-class |

Still live Research (not “legacy infra”): environments, image releases, files,
wiki, knowledge, experiments, visuals, traces, advanced/economics billing, etc.
Discover via `ResearchClient` / `facade.py` rather than new entrypoints.

---

## 2. Legacy / parked (outside live composition)

| Surface | Status |
|---------|--------|
| **Managed agents** (+ OpenAI Agents SDK sibling) | **Deleted** from package. Research session still has loud retirement stubs. |
| **Containers / tunnels / pools** (+ horizons_private, base, openai_tools, container auth, CLI, container OpenAPI fragment) | Archived under **gitignored** repo-root `old/` — may return later; do not re-export on `SynthClient` until product asks. |

`old/` is not under `synth_ai/`. Restoring = move back + rewire composition/docs.
Backend/OpenAPI may still list `/v1/containers|pools|tunnels`; that ≠ a shipped
Python client today.

---

## 3. Full tour of `synth_ai/`

### 3.1 Top-level modules

| Path | Role | Verdict |
|------|------|---------|
| `__init__.py` | Version, lazy public exports (`SynthClient`, Research types), log filter | **keep** |
| `client.py` | Front door — `.research` only | **keep** |
| `config.py` | Thin re-export of env/URL helpers | **keep** (or fold into `core` later) |
| `__main__.py` | `python -m synth_ai` → CLI; has leftover `train`-arg debug noise | **review** |
| `py.typed` | Typing marker | **keep** |
| `README.md` | Package DAG | **keep** (keep honest) |

**Entrypoints** (`pyproject.toml` scripts):

| Script | Target |
|--------|--------|
| `synth-ai` | `synth_ai.cli:cli` |
| `synth-ai-research-mcp` | `synth_ai.mcp.research.server:main` |
| `synth-ai-research-factory-standup` | `synth_ai.cli.research_factory_standup:main` |

---

### 3.2 `core/` — plumbing (~keep)

Must **not** import `sdk/` (layering ratchet).

| Piece | Role | Verdict |
|-------|------|---------|
| `errors.py` | Shared error taxonomy | **keep** |
| `auth/` | API credentials, request context | **keep** |
| `http/` | Sync/async transport, retry, SSE | **keep** |
| `contracts/` | Generic JSON/pagination/resource types | **keep** |
| `utils/` | env, urls, json, paths, workspace, secure files, log filter | **keep** |
| `research/` | **Only** MetaPathFinder alias → `sdk.research` (+ deprecation warning) | **park until phase 5**, then delete |
| `__init__.py` / `README.md` | Docs still may mention containers/tunnels/pools | **doc cleanup** |

---

### 3.3 `sdk/` — live clients

| Piece | Role | Verdict |
|-------|------|---------|
| `pagination.py` | `SyncPage` / `AsyncPage` / `page_from_wire` | **keep** (Research lists) |
| `__init__.py` | Exports pagination only | **keep** |
| `research/` | Entire Managed Research implementation | **★ product** |

No live `sdk/containers|tunnels|pools|base|…` — those are in `old/sdk/`.

#### `sdk/research/` map (~12MB)

| Area | Approx | Role | Product vs internal |
|------|--------|------|---------------------|
| `facade.py` | small | `ResearchClient` = `SynthClient().research` | **hero** |
| `client.py` | small | Typed HTTP client over `core.http` | product |
| `projects.py`, `swarms.py`, `factories.py`, `research_intern.py` | large | Hero namespaces | **product** |
| `environments.py`, `image_releases.py`, `visuals.py`, `traces.py`, `wiki.py`, `knowledge.py`, `experiments.py`, … | mid | Stable / supporting APIs | product |
| `advanced.py`, `advanced_swarms.py`, `advanced_factories.py`, `economics.py`, … | mid | Unstable / operator / billing | product-but-unstable — don’t treat as stable public story |
| `public.py` | large | Aggregated type re-exports | product |
| `errors.py` | mid | Research errors (+ some `Smr*` aliases) | product |
| `contracts/` | ~86 files | Domain DTOs / wire models | **product contracts** (adapters may deep-import) |
| `session/` | ~43 files | `ResearchSession` + operator namespaces (runs, factories, files, github, billing, …) | product-adjacent / advanced; MCP + standup use this |
| `session/compat.py` | tiny | Retired MA bridge attrs → raise | **delete when safe** |
| `transport/` | ~5 | Research-local HTTP/retry/stream wrappers | **internal** |
| `_internal/` | crypto, env, urls | Sealed-box (`pynacl`), helpers | **internal** |
| `factory_plans/` | JSON plan(s) | Builtin Factory standup plans | **product data** (must ship in wheel) |
| `schemas/` | `public_models.json`, `smr_openapi.yaml` (~2MB) | Vendored registry / OpenAPI | **product data**; watch wheel weight |

**Cleanup lens inside Research:** prefer clear nouns; avoid new aliases; don’t
grow `advanced.*` into a second public front door; keep Intern/Factories/Swarms
consistent with backend SMR contracts.

---

### 3.4 `cli/` — thin adapters (~review weight)

`main.py` registers: **`research`**, **`dev_envs`** only.

| Module | ~LOC | Role | Registered? | Verdict |
|--------|------|------|-------------|---------|
| `main.py` | 28 | Root Click group | — | **keep** |
| `research.py` | 364 | `synth-ai research` group | yes | **keep** |
| `research_projects.py` | 405 | projects subcommands | via research | **keep** |
| `research_environments.py` | 143 | environments | via research | **keep** |
| `research_image_releases.py` | 149 | image-releases | via research | **keep** |
| `dev_envs.py` | 1162 | `synth-ai dev-envs …` | **yes** | **hard look** — ty-excluded; operator/dev tooling, not first-mile Research |
| `research_factory_standup.py` | 1429 | One-shot Factory standup | **own console script**, not Click | **hard look** — keep if customers use; else park |
| `README.md`, `AGENTS.md` | — | CLI conventions (flat files, logic in sdk) | — | **keep** |

Stale local `cli/__pycache__` may still mention containers/pools/tunnels — wipe;
not part of source.

---

### 3.5 `mcp/research/` — MCP delivery (~keep, heavy)

Thin adapter over `sdk.research` (often deep session/client imports rather than
`SynthClient` — allowed; prefer client when easy).

| Piece | Notes | Verdict |
|-------|--------|---------|
| `server.py` (~3.5k) | MCP server + `main()` | **keep** (ty-excluded) |
| `registry.py` | Tool registry / scopes | **keep** |
| `request_models.py` | Boundary parsing | **keep** (ty-excluded) |
| `objective_tools.py` | Objective tools | **keep** |
| `tools/` (~30 modules) | Per-domain builders; largest include runs / factories / projects | **keep** — cleanup for consistency with facade nouns |
| no `mcp/__init__.py` | Namespace package only | fine |

Wire tool names may still use `smr_` prefixes; product language is Managed Research.

---

## 4. Where to put new / retired code

| Kind | Location |
|------|----------|
| Research APIs, contracts, session, transport | `sdk/research/` |
| Shared pagination | `sdk/pagination.py` |
| Front door | `client.py` |
| CLI wiring only | `cli/` (one file ≈ one command) |
| MCP tools | `mcp/research/` |
| Auth / HTTP / errors / env / URLs | `core/` |
| Maybe-return infra (tunnels, pools, …) | repo-root **`old/`** (gitignored), then remove from tracked tree |
| Dead MA-style surfaces that must not return | **Delete** |
| `core.research.*` import paths | Alias until phase 5; then remove |

Do **not** put product Research back under `core/`. Do **not** add
`managed_research/` paths — that tree is gone.

---

## 5. Synth Style (review bar)

Canonical: `backend/specifications/tanha/references/synthstyle.md`.

Short form for this cleanup:

1. Push complexity **inward** (backend contracts); keep SDK/CLI simple.  
2. **API-first**; UX/CLI second.  
3. **Clear nouns / hierarchies** — projects / swarms / factories / intern.  
4. **Minimal user config** — hide algorithm internals.  
5. Meet production code where it is — no forced lock-in.  
6. Design for the **long horizon** (Intern + Factories).  
7. Comments = why; code = what; specs = full story.  
8. Never swallow exceptions.  
9. No abbreviations; units last (`timeout_seconds`).  
10. Do exactly the cleanup asked — no drive-by refactors.

Honesty: don’t document parked infra as supported; fix packaging strings that still say “containers, tunnels, pools.”

---

## 6. Cleanup queue (reviewing eng)

Ordered by honesty / leverage across **all of `synth_ai/`**:

1. **Packaging truth** — `pyproject` description; empty extras (`schemas`, `research`); ghost ruff/ty excludes (`v0`, `demos`, `environments`, `utils/container_discovery.py`, `http.py`, `agent_demos`); **`MANIFEST.in` still points at deleted `managed_research/`** while `package-data` correctly uses `sdk/research/factory_plans|schemas`.  
2. **Docstring honesty** — `core/__init__`, package READMEs, OpenAPI `synth-api-v1.yaml` infra routes.  
3. **CHANGELOG Unreleased** — MA delete + infra → `old/` + Research-only client.  
4. **CLI product boundary** — keep vs park `dev_envs` and `research_factory_standup`.  
5. **Research session MA stubs** — delete `session/compat` / `ResearchControlSession` mixin when safe.  
6. **`__main__.py` train debug** — strip noise.  
7. **`core/research` alias** — phase 5 per `unify_sdk_layering.md` (PyPI deprecation window).  
8. **Wheel weight** — `schemas/smr_openapi.yaml` ~2MB; confirm it must ship.  
9. **MCP / advanced surface consistency** — nouns match facade; no parallel client.  
10. **Local hygiene** — wipe `build/`, stale `__pycache__` before release.  
11. **Sibling `testing/`** — still has containers/pools/tunnels SDK tests; coordinate separately (this trim left tests alone in a later pass).

Non-goals unless product asks: restoring infra clients; redesigning Intern;
backend proxy deletion.

---

## 7. Verify

```bash
cd ~/Documents/GitHub/synth-ai
uv run python -c "
from synth_ai import SynthClient
c = SynthClient(api_key='sk_x', base_url='http://127.0.0.1:8000')
assert hasattr(c, 'research')
assert not hasattr(c, 'pools')
assert not hasattr(c, 'tunnels')
assert not hasattr(c, 'containers')
"
# uv run python ../testing/scripts/check_sdk_layering.py
```

Local backend: sibling **`synth-dev`** (`./scripts/local.sh`).

---

## 8. What we already did (context)

1. Removed managed-agents (+ OpenAI Agents SDK sibling) from the package.  
2. Moved containers → then tunnels/pools (+ related helpers/CLI/OpenAPI fragment) into gitignored `old/`.  
3. `SynthClient` is Research-only; CLI keeps `research` (+ `dev_envs` pending your call).

Intent: a clear Managed Research package (swarms, factories, intern) that an eng
can walk top-to-bottom without mistaking parked infra for product.
