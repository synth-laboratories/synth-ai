# Review: `synth_ai/` cleanup decisions (update / move-to-old / delete)

**Date:** 2026-07-29
**Method:** six parallel review agents over the full tree (core+top-level, sdk surface,
session/contracts/internals, cli, mcp, packaging+hygiene), reviewed against
`backend/specifications/tanha/references/synthstyle.md`. Companion to
`HANDOFF_SDK_SURFACES.md`. Every claim below was verified with evidence
(file:line quoted or behavior reproduced).

---

## 0. Bugs first — fix before any cleanup

1. **The trim is uncommitted.** git HEAD (`c0493148`) still contains all 10
   infra/MA files (`sdk/pools.py`, `cli/tunnels.py`, `sdk/managed_agents/`, …).
   Anything built from a checkout ships the pre-trim package. Also
   `pyproject.toml:3` is `0.17.5`, already on PyPI; `publish-dev.yml` refuses a
   `.dev` of a released base → **commit the trim, bump the version** (and
   `README.md:3` badge — `ci.yml:36-57` gates on them matching).
2. **`core/research` alias corrupts the modules it forwards to**
   (`synth_ai/core/research/__init__.py:63-67`). `create_module` returns the
   real module, then CPython overwrites its `__spec__` with the alias spec.
   Reproduced: after `import synth_ai.core.research.factory_plans`,
   `importlib.resources.files("synth_ai.sdk.research.factory_plans").iterdir()`
   returns `[]` and reading the builtin plan raises
   `FileNotFoundError: Can't open orphan path`. Breaks
   `synth-ai-research-factory-standup --plan builtin:…` for any process that
   touched the deprecated path first. Fix the finder (restore the original
   `__spec__` or use a loader that doesn't re-init attrs) — don't wait for phase 5.
3. **Broken sdist** (`MANIFEST.in:28,31`). Re-includes point at deleted
   `synth_ai/managed_research/`, so the `*.json` exclude wins:
   sdist ships **without** `factory_plans/rsi_synth_on_synth.plan.json` and
   `schemas/public_models.json` (verified via `SOURCES.txt`; the yaml does reach
   the sdist). Any `--no-binary` install fails on `--plan builtin:…`.
   Exact fix — replace lines 26-31 (keep ordering after the line-13 exclude):
   ```
   recursive-include synth_ai/sdk/research/factory_plans *.json
   include synth_ai/sdk/research/schemas/public_models.json
   ```
   Also delete line 5 (`recursive-include synth_ai *.toml` matches nothing);
   add `recursive-include synth_ai *.md` if in-package READMEs should ship
   (today they reach neither sdist nor wheel).
4. **`public.AsyncResearchClient` raises `AttributeError`** (verified) —
   `public.py:688` points at `client.AsyncResearchClient`, which only exists in
   `sdk/research/__init__.py:7`.
5. **MCP scope table fails open** (`mcp/research/registry.py:198-209`): a tool
   missing from the smr_-keyed table silently gets `required_scopes=()`.
   48 of 311 tools are unscoped, including advertised writes
   `research_pause_run`, `research_resume_run`, `research_start_one_off_run`.
   Make an empty lookup raise at registry build.
6. **MCP entrypoint can never expose advanced tools** (`server.py:3533` +
   `:350`): `main()` hardcodes `include_advanced_tools=False` with no
   flag/env override, and `call_tool` also goes through the advertised set, so
   **247 of 311 built tool definitions are unreachable and uncallable** via
   `synth-ai-research-mcp`. Decide: add an override, or shrink the tree to what
   ships.
7. **Stale `build/`** — 6.0MB of pre-move modules (`build/lib/synth_ai/sdk/pools.py`,
   `sdk/managed_agents/`). `rm -rf build/` before release (documented hazard in
   `unify_sdk_layering.md:59-61`). Also wipe `synth_ai/cli/__pycache__` +
   `synth_ai/sdk/__pycache__` (11 orphan `.pyc` for deleted modules — they make
   `import synth_ai.sdk.pools` still succeed locally) and stale `synth_ai.egg-info/`.

---

## 1. Move to `old/`

| What | Detail |
|------|--------|
| `cli/dev_envs.py` (1162 LOC) | Operator/worker console, not product: `seed-topology-manifest`, `claim-materialization --worker-id`, `materialize` are the control-plane worker's side of the protocol; 28/29 commands have no help text; ty-excluded. Zero capability loss — everything is reachable via `SynthClient().research` and MCP `smr_*_dev_environment*` tools. **Before parking:** lift `_verify_cloud_s0_evidence_payload` (dev_envs.py:167) into `sdk/research/session/dev_environments.py` as `verify_cloud_s0_evidence(...)`. Unregister from `cli/main.py:5,26`. Coordinate: delete the two `dev_envs` entries in `testing/.../internal_vocabulary_allowlist.json` (that gate fails on stale entries — land both repos together). |
| `cli/research_factory_standup.py` (1429 LOC) | Synth's internal RSI activation gate wearing a console-script hat: hardcodes `factory.name == "synth-rsi"`, seven internal milestone IDs, and Synth's private repo remotes as an allowlist — a customer provably cannot use it. ~750/1429 lines are gate + evidence-packet logic that exists nowhere in `sdk/research`. **Before parking:** lift `execute_factory_standup` (lines 1277-1363) + payload builders into `sdk/research/factories.py` as a typed `standup(plan, ...)`. Drop the `synth-ai-research-factory-standup` script from `pyproject.toml:21`. Optional follow-up: a ~60-line Click `synth-ai research factories standup` over the SDK. |
| `sdk/research/schemas/` (both files) | **§6.8 answered definitively:** nothing in the package reads `smr_openapi.yaml` (2.0MB) or `public_models.json` at runtime — no `importlib.resources`/`open()`/path reference anywhere. The yaml ships `/v1/tunnels/*` (10 paths) and `/v1/pools/*` (~60 paths) inside the Research-only wheel; the json is a point-in-time snapshot (`observed_at: 2026-07-09`) that goes stale silently. Move both to `old/`, drop `pyproject.toml:91` glob (~2.1MB/wheel saved). **Coordinate:** `testing/scripts/validate_synth_ai_contract.py:108` and `sync_smr_schemas.py:610` read them from the source tree — update those paths (or keep files tracked but out of the wheel, if the gates must stay). |
| `core/errors.py:229-282` | `UsageLimitError` + `PlanGatingError` — pools-era ("Environment Pools requires a … plan"), zero references. Both are also latently broken (`@dataclass` on `Exception` bypasses `SynthError.__init__`). |
| `core/utils/urls.py:155-271` | Tunnel/ngrok/container trust predicates (`is_cloudflare_tunnel_url`, `is_synthtunnel_url`, `is_local_http_container_url`, `is_synth_managed_ngrok_url` + env escape hatches) and `resolve_synth_interceptor_base_url` — zero references anywhere including `old/` and `testing/`. |
| `core/utils/paths.py:14-16, 66-89` | `SYNTH_CONTAINER_CONFIG_PATH` and the container import-path shims (`configure_import_paths`/`temporary_import_paths`, which mutate `sys.path` process-wide). Dead. |
| `core/utils/workspace.py` (whole module) | Multi-repo workspace-manifest resolver; monorepo dev tooling, zero consumers; still maps a `managed-research` repo (line 17). |
| MCP `tools/dev_environments.py` (25 tools) + `tools/cloud_deployments.py` (27 tools) | Same operator boundary as `cli/dev_envs.py` — claims, fencing tokens, exec. Already unadvertised. **Make this one decision across CLI + MCP**, not two. If SDK session namespaces stay (they should — live `/smr/dev-environments/*` routes), parking the MCP tool builders is optional; parking the CLI is not. |

`old/` bookkeeping: inventory is complete and 1:1 with deletions (8 sdk + 3 cli +
1 openapi files). **Update `old/README.md`** to state the restore procedure —
today that sentence lives only in `HANDOFF_SDK_SURFACES.md:71`, which doesn't
travel with the gitignored directory.

## 2. Delete (verified zero live consumers unless noted)

**Retired-surface remnants — one commit, pure subtraction:**
- `session/compat.py` + `ResearchControlSession`. Complete importer map:
  `session/client.py:195` (import), `:6066-6095` (class), `:275` (`__all__`);
  `session/__init__.py:329,723,823`; `session/README.md:55` (doc line).
  Nothing in MCP, CLI, facade, or `public.py` references any bridge attribute
  (verified repo-wide).
- Retired OpenAI transport-mode surface: `session/config.py:13-18,83-98`
  (`resolve_openai_transport_mode` has zero callers) + re-export layers
  `session/client.py:200-202,271-273`, `session/__init__.py:326-328,491-493`.
- `__main__.py:10-24` — three dead `train` debug probes (no `train` command is
  registered anywhere); also fix `:27-36` double-traceback-then-reraise.
- MCP `server.py:433-653` — 14 "hosted compat" verb-last aliases, all duplicates
  of noun-first tools, all unadvertised; last home of "SMR" product language.
- MCP unregistered raise-only/dead handlers: `server.py:787,897-901,1356-1387,
  2548-2561,2563,2659-2663,3345,3431`; duplicate registrations
  `tools/files.py:58,71` (same handlers as `resources.py:31,44`).
- `request_models.py:99,113,176,706-729` and `objective_tools.py` dead enums/helpers.

**Dead alias soup:**
- `sdk/research/enums.py` — whole module: no importers, contains no-op
  `SmrWorkMode = ResearchWorkMode`, and conflicts with `public.py`'s mapping of
  the same names to *different* enums.
- `errors.py:443` `ManagedResearchError`; `events.py:16-18` aliases;
  unreferenced `Research*`/`Async*` aliases: `swarms.py:1199` (async pair),
  `projects.py:378` (3 of 4), `visuals.py:636`, `economics.py:152`,
  `advanced_swarms.py:718`, `factories.py:1208` (name-collides with
  `advanced_factories.ResearchFactoriesAPI`).
- `public.py:481-497` duplicate `__all__` entries (`ActorHarness`, `ActorModel`,
  `ActiveActorModel`, `DeprecatedActorModel`, `WorkMode`).
- `session/__init__.py:397` — the first, dead `__all__` (~347 names, silently
  overwritten at line 746).
- `session/integrations.py` `IntegrationsAPI` — exported but never mounted
  (`hasattr(ResearchSession, "integrations")` is False).
- `session/client.py:1311-1320` `credentials` shim (unflagged, no caller).
- `synth_ai/config.py` — zero references repo-wide; third spelling of
  `core.utils.env`/`urls`.

**Dead core plumbing:**
- `core/contracts/resources.py` (`ResourceId`/`ResourceRef` — also name-collides
  with a *different* `ResourceRef` in `core/errors.py:39`); `core/contracts/pagination.py:17-22`
  generic `Page` (keep `build_query_params`/`extract_next_cursor` — live).
- `core/errors.py`: `JobError`, `StorageError`, `ModelNotSupportedError`,
  `ConfigError`, `ValidationError` (zero importers); stray `pass` at `:96`.
- `core/utils/env.py`: `mint_demo_api_key`/`ensure_synth_api_key`/
  `LOCAL_DEV_SYNTH_API_KEY` (:161-231, mutate `os.environ` silently),
  `write_env_var_to_json`, `get_backend_url`, `mask_str`, `ensure_env_var`,
  `PROD_BASE_URL` alias.
- `core/utils/urls.py`: `normalize_inference_base`, `normalize_base_url`,
  `local_backend_url`, `backend_health_url`, `backend_me_url`,
  `backend_demo_keys_url`, `BACKEND_URL_SYNTH_RESEARCH_OPENAI/_ANTHROPIC`.
- `core/utils/paths.py`: `REPO_ROOT = Path.cwd()` (import-time cwd capture) and
  the ~10 unused file helpers incl. `cleanup_paths` (an rmtree nothing calls).
- `core/utils/json.py`, `core/utils/dict.py` — only imported by the
  `core/utils/__init__.py` aggregator, which is what keeps all of the above alive;
  prune the aggregator to live names.
- `core/auth/context.py` `RequestContext`/`OrganizationId` (zero consumers;
  `OrganizationId` independently redefined in `contracts/common.py:78`).

**Dead transport/contract tail:**
- `sdk/research/transport/retries.py` — dead duplicate `RetryPolicy` whose field
  names (`attempts`, `base_delay_seconds`) contradict the real
  `core/http/retry.py` policy; `transport/streaming.py` dead half
  (`BinaryPayloadPreview`); `transport/__init__.py` aggregator (all consumers
  deep-import).
- `sdk/pagination.py:21` `AsyncPage` (exported, used nowhere).
- Contracts: `factory_operations.py:3109` `FactoryCostsLimits`;
  `smr_providers.py:362,607` helpers; `smr_resource_kinds.py:17`
  `coerce_smr_resource_kind`.
- `session/py.typed` (nested `partial` marker — ignored by checkers, never ships).

**Dead config:**
- `pyproject.toml`: `:24` `schemas = []` extra; `:25-29` `providers`
  (anthropic/google-genai — zero imports since MA delete); `:39-43` `all`
  (verbatim copy of dead providers); `:44-46` `analytics` (no pandas import);
  `:127` per-file-ignore for missing `synth_ai/http.py`; `:131-139` dead
  `[tool.ty.*]` blocks (`ty.toml` takes precedence); `:76` `agent_demos` guard.
  Keep `research = []` (documented compat) and the ruff `old` exclude.
- `ty.toml:11-14`: ghost excludes `synth_ai/v0`, `synth_ai/demos`,
  `synth_ai/environments`, `synth_ai/utils/container_discovery.py`. Keep :15-19.
- `.gitignore`: `:152-153` (`synth_ai/learning/` gone), `:176` (`tui/` gone),
  `:177-178` (root `contracts/`, `skills/` gone).

## 3. Update

**Packaging / release truth:**
- `pyproject.toml:4` description — still sells "containers, tunnels, pools".
- `CHANGELOG.md` `## Unreleased` is empty — record MA delete, infra→`old/`,
  Research-only `SynthClient`.
- Root `README.md:52,93,118` — all three quickstarts teach the deprecated
  `synth_ai.core.research.public` import path (which also triggers bug #2 above).
- `openapi/synth-api-v1.yaml`: keep in tree (live backend contract; handoff-
  endorsed split) but fix the two `x-surface-note`s at `:1724,:1745` telling
  callers to "use SDK tunnel abstractions" that no longer ship, and the stale
  `x-sdk-package-version: 0.9.7` at `:5`.
- `.gitignore:159` `**/*.md` matches `CHANGELOG.md`; `:162` `!**/agents.md` is
  case-wrong on Linux (`AGENTS.md` falls back to ignored on case-sensitive FS);
  `:180` `uv.lock` ignored-yet-tracked.

**Docstring honesty (the "most misleading strings"):**
- `core/__init__.py:3-5` — "only exposes the pieces used by containers, tunnels,
  and pools". Also: its 60-line re-export facade has zero `from synth_ai.core import`
  consumers — trim or delete the facade.
- `client.py:28-29` and `synth_ai/README.md:12-13` — point customers at
  gitignored `old/` they can't see; per style, don't document parked infra.
- `cli/README.md:4,32` — same `old/cli/` pointers; also omits registered
  commands. `cli/AGENTS.md:13` — dead incident-log path with wrong username.
- `session/README.md:28-68` — links into `/Users/joshpurtell/...` and the
  deleted `managed-research` repo; `:55` references the bridge being deleted.
- `transport/http.py:268` — says "Deprecated compatibility adapter"; it is the
  *only* transport the session uses.
- `contracts/traces.py:3` (owned by `synth-containers`), `contracts/runtime_image.py:3`
  + `visuals.py:3` (pointers into personal `Jstack/.jstack/daily_notes`).
- MCP `README.md:13,37,47-58` — three central claims contradicted by code
  (`smr_` is never advertised; no hosted transport here; 9 of 17 documented
  tools unreachable).
- `synth_ai/README.md:44-49` — the one "Supported Imports" example demos
  unstable `advanced.economics`; `sdk/README.md:27` omits hero noun `intern`.

**Structural (the real §6.9 story):**
- **Collapse the `advanced.*` second front door.** `advanced_factories`
  reimplements `FactoryCandidates/Champions/Lenses` over the same routes as
  stable `factories.py` (and calls the *stable* surface "legacy");
  `advanced.runs.create` hits the same `POST /smr/projects/{id}/trigger` as
  `swarms.create`; `facade.runs` (deprecated→swarms) vs `advanced.runs` makes
  one noun mean two things. Pick the stable module as owner; make advanced
  delegate or shrink.
- **`smr_` → `research_` rename in MCP is free**: `registry.py:231` already
  rewrites every name before discovery and `resolve_tool` keeps the old alias,
  so the ~300 source-level `smr_` literals, the smr_-keyed scope table, and the
  cross-references in tool descriptions (`files.py:18`, `factories.py:42`,
  `projects.py:545`, `runs.py:1988`, …) can all be renamed with **zero wire
  impact**. Key the scope table on final names and fail closed (bug #5).
- **`Smr*` error aliases are load-bearing internally** — `transport/http.py:149`
  *raises* `SmrLimitExceededError`; `session/client.py` imports aliases 21×;
  `mcp/server.py:80` imports `SmrApiError`. Migrate internal call sites to
  `Research*`, then add `DeprecationWarning` + removal version to the 11 aliases
  (`errors.py:444-454`), and drop them from `__all__`.
- **Client-name collisions**: `research.Client` (transport) vs `public.Client`
  (facade); `public.AsyncClient` is not the async peer of `public.Client`.
  `public.py`'s 110-name `_COMPATIBILITY_EXPORTS` table has no warnings and no
  removal note; `public.ResearchWorkMode` and `enums.ResearchWorkMode` were two
  different enum classes (moot once `enums.py` is deleted, but pick one `WorkMode`).
- **`session/__init__.py:860`** — `__all__` is a runtime globals dump publishing
  380 names incl. 38 submodules (incl. `compat`). Write an explicit list.
- **One pagination idiom**: cursor-in/tuple-out currently discards the next
  cursor (`projects.py:139`, `swarms.py:329`); shared `SyncPage` is used only by
  deprecated surfaces. Pick typed pages or cursor-returning tuples, apply to heroes.
- **Repos vs repositories** (`session/repos.py` vs `session/repositories.py`) —
  same backend routes, different DTOs; one noun must win.
- **Second HTTP stacks**: `session/_run_authority_mixin.py:1357` raw
  `httpx.get` bypasses transport (no retry/error mapping); MCP builds both
  `ResearchSession` and `CoreResearchClient` with different default timeouts
  (30 vs 120) for overlapping nouns.
- Unflagged compat shims need removal notes (the bar `core/research` meets):
  hidden `Smr*` aliases in `synth_ai/__init__.py:201-210`,
  `client.py:97-105` `async_research`, `facade.py:181` nine-entry compat map,
  `sdk/research/auth.py`, `_internal/urls.py`, `efforts.py` ("LegacyResearchSession").
- `client.py:37` front-door `timeout` → `timeout_seconds` (units-last), and the
  same in `advanced_swarms.py:537`, `readouts.py:209`; `client.py:57-60`
  `close()` doesn't clear `_research_client` (post-close `.research` returns a
  closed transport).
- Error-swallowing sweep (style: never swallow): `synth_ai/__init__.py:22-23`
  (log-filter install), `core/utils/env.py:234-251` (`_load_user_env` — corrupt
  config reads as "no API key"), `core/utils/urls.py` six bare
  `except Exception: return False`, `log_filter.py:32-48` (lies about bytes
  written), `session/dev_environments.py:194,213` (exception → string field),
  `research_factory_standup.py:1420` (flattens to `{"error","message"}` —
  relevant if any part survives), `wiki.py:50`/`knowledge.py:33`
  (`.get() or ""` on required ids), `artifacts.py:13` (contract break ≡ empty list).
- `research_intern.py:316,558` — `ProjectComputerAPI`/`ProjectDataBindingsAPI`
  are project nouns; move within-package to `projects.py`/`project_runtime.py`.
- `account.py` — org balance/credits/API keys are platform `/api/v1` surfaces
  under the research noun; flag for a future `client.account`, don't grow it.
- Duplicated CLI auth/URL helper block (5 copies) → `cli/lib/`;
  `research_projects.py:375` base64 upload packing → a
  `WorkspaceFileUpload.from_path` contract; `research.py` `smoke` commands are
  dev-harness verbs in the customer CLI; no `intern` CLI group despite hero status
  (same gap in MCP: zero `research_intern_*` tools because `ResearchSession`
  has no `.intern`).
- MCP `tools/tag.py` — leaks unclosed sessions (only tool module without `with`),
  third naming family (`tag_*`), all 6 unadvertised.
- `factory_plans/rsi_synth_on_synth.plan.json:26` — plan text depends on
  unlanded backend work ("after backend issue 340 lands").
- `core/http/async_transport.py:28` + `session/_run_authority_mixin.py:55` +
  `wiki.py:15` — private cross-module imports (`_decode_json_value`,
  `_raise_for_error_response`, `_optional_datetime`); promote or localize.
- Two `RetryPolicy` classes exist until `transport/retries.py` is deleted (§2).

## 4. Sibling `testing/` coordination (will fail once trim lands)

Fail at collection (dead imports): `test_sdk_containers.py:4`,
`test_sdk_pools.py:3`, `test_sdk_tunnels.py:5`, `test_evals_usage_paths.py:11-12`,
`public_surface/test_public_surface.py:44-45`, `end_to_end/test_live_backend.py:9,35-40`;
support fixture `fake_backend.py:29-30`. These sit exactly where
`ci.yml:150` probes to decide whether to run pytest. Plus: two
`internal_vocabulary_allowlist.json` entries for `cli/dev_envs.py` (gate fails
on stale entries), and `validate_synth_ai_contract.py`/`sync_smr_schemas.py`
read `sdk/research/schemas/*` from the source tree (see schemas decision, §1).

## 5. Suggested sequencing

1. Fix bugs §0 (alias `__spec__` corruption, MANIFEST, scope fail-open,
   `AsyncResearchClient`), wipe `build/`+pycache, commit the trim, bump version,
   write CHANGELOG.
2. Land §2 deletes — start with the compat/bridge commit (pure subtraction),
   then dead aliases, dead core, dead config.
3. Land §1 moves (dev_envs + standup after lifting their SDK-worthy pieces;
   schemas decision with testing-repo update), coordinated with `testing/` fixes (§4).
4. §3 renames/docs: smr_→research_ in MCP, docstring honesty pass,
   advanced.* collapse (largest item — own PR).
