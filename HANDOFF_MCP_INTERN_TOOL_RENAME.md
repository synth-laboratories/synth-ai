# Handoff: rename the 22 Research Intern MCP tools to the `research_` prefix

**Date:** 2026-07-31
**Audience:** Eng picking up the Research Intern alpha MCP surface
**Repo:** `synth-laboratories/synth-ai` (siblings: `backend`, `testing`, `evals`, `frontend`)
**Scope:** `synth_ai/mcp/research/tools/research_intern.py` — 22 string literals
**Estimated size:** ~30 minutes including verification

---

## 0. Why this is urgent

This is not a cosmetic naming cleanup. It is a live outage in the MCP surface.

`ResearchMcpServer.__init__` calls `build_tool_registry(self._build_tools())`
unconditionally (`synth_ai/mcp/research/server.py:385`), and `_build_tools()`
unconditionally includes `*build_research_intern_tools(...)`
(`server.py:452`). The registry rejects any tool declared with the legacy
`smr_` prefix, so **the server raises at construction time**:

```
ValueError: MCP tool 'smr_provision_research_intern' uses the legacy smr_ prefix;
declare it as 'research_provision_research_intern' -- the smr_ spelling is a
generated alias.
```

Everything that constructs the server is therefore broken, not just naming tests:

| Call site | Consequence |
|---|---|
| `backend/app/api/v1/routes_mcp.py:1766, 1832, 1861` | The hosted `/mcp` route 500s on **every** `tools/list` and `tools/call`. Construction is not wrapped in try/except. |
| `synth_ai/mcp/research/server.py:2829` | The `synth-ai-research-mcp` stdio entrypoint fails at startup. |
| `testing/tests/launch_2026_04_09/test_managed_research_setup_smoke_2026_04_09.py:118` | Smoke test fails. |
| `testing/scripts/prove_cloud_deployment_mcp.py:304, 335` | Deployment proof script fails. |

Phase C of the Research Intern acceptance plan requires **real MCP tool
discovery and real calls**. That gate cannot be attempted until this lands.

---

## 1. The rule, and why the rename is safe

`synth_ai/mcp/research/registry.py` implements a one-way generation, not a
migration you have to coordinate:

- `build_tool_registry()` (`registry.py:232-252`) **requires** every declared
  `ToolDefinition.name` to be `research_*`, and raises on `smr_*`
  (`registry.py:244-248`).
- `resolve_tool()` (`registry.py:255-270`) **generates** the `smr_*` alias at
  lookup time from whatever is declared as `research_*`:

  ```python
  # registry.py:268-269
  if name.startswith("smr_"):
      return tools.get(f"research_{name[4:]}")
  ```

Nothing is stored under the `smr_*` key. Discovery (`tools/list`,
`list_tool_payload`) only ever emits `research_*`.

**Consequence: renaming is wire-compatible.** Any client still calling
`smr_get_research_intern` keeps working through the alias — it simply stops
being the advertised name. There is no deprecation window to manage.

`README.md:36-40` already documents the convention: *"Tool builders in `tools/`
declare names as `research_*` directly... Declaring a new tool under the `smr_`
prefix is a registry build error."*

---

## 2. How this happened

Worth knowing so the fix isn't second-guessed:

- Commit `ff5fddf7` *"refactor: retire the smr_ spelling at source level"*
  (2026-07-30 00:45) removed the old `_advertised_name()` auto-rewrite, added
  the hard `ValueError`, and renamed declarations across ~30 tool files.
- `research_intern.py` was added later, in `3ec5a9f3` *"feat(research): add
  reactive Research Intern client and MCP"* (2026-07-30 15:41) — **~15 hours
  after the rule was already live and documented**.

So this is not a file that predates the rule. It was written against the old
convention and evidently never had `build_tool_registry()` run against it.

Corroborating: `server.py`'s own `_STABLE_TOOL_NAMES` set
(`server.py:96-184`) **already lists all 22 tools under their `research_*`
spellings** (e.g. `"research_provision_research_intern"` at `server.py:176`).
The author knew the target names; only the builder file was left behind.

---

## 3. The change

In `synth_ai/mcp/research/tools/research_intern.py`, replace the `smr_` prefix
with `research_` in these 22 `name=` literals. Nothing else in the file changes.

| Line | Current | Target |
|---|---|---|
| 419 | `smr_provision_research_intern` | `research_provision_research_intern` |
| 434 | `smr_get_research_intern` | `research_get_research_intern` |
| 441 | `smr_update_research_intern` | `research_update_research_intern` |
| 457 | `smr_attach_research_intern_factory` | `research_attach_research_intern_factory` |
| 467 | `smr_list_research_intern_factories` | `research_list_research_intern_factories` |
| 474 | `smr_create_research_intern_session` | `research_create_research_intern_session` |
| 503 | `smr_list_research_intern_sessions` | `research_list_research_intern_sessions` |
| 513 | `smr_get_research_intern_session` | `research_get_research_intern_session` |
| 520 | `smr_append_research_intern_event` | `research_append_research_intern_event` |
| 535 | `smr_list_research_intern_events` | `research_list_research_intern_events` |
| 553 | `smr_watch_research_intern_events` | `research_watch_research_intern_events` |
| 588 | `smr_sync_research_intern_session` | `research_sync_research_intern_session` |
| 601 | `smr_run_research_intern_turn` | `research_run_research_intern_turn` |
| 654 | `smr_exchange_research_intern_turn` | `research_exchange_research_intern_turn` |
| 708 | `smr_close_research_intern_session` | `research_close_research_intern_session` |
| 741 | `smr_publish_research_intern_session_trace` | `research_publish_research_intern_session_trace` |
| 769 | `smr_record_research_intern_decision` | `research_record_research_intern_decision` |
| 779 | `smr_list_research_intern_decisions` | `research_list_research_intern_decisions` |
| 789 | `smr_get_research_intern_decision` | `research_get_research_intern_decision` |
| 796 | `smr_publish_research_intern_acceptance_receipt` | `research_publish_research_intern_acceptance_receipt` |
| 815 | `smr_get_research_intern_acceptance_receipt` | `research_get_research_intern_acceptance_receipt` |
| 822 | `smr_list_research_intern_acceptance_receipts` | `research_list_research_intern_acceptance_receipts` |

A blanket `name="smr_` → `name="research_` substitution scoped to this one file
is sufficient — every `smr_`-prefixed literal in it is a tool name. Verify the
count is exactly 22 before and after, and confirm no other file changes.

---

## 4. Blast radius — verified empty

Searched `synth-ai`, `testing`, `backend`, `evals`, and `frontend` for hardcoded
`smr_*research_intern*` tool-name strings.

**`research_intern.py` is the only file containing them.** No test, eval suite,
doc, config, prompt, or manifest anywhere hardcodes the `smr_` wire form of
these names.

The only existing consumers already assume the **post-rename** names:

- `synth_ai/mcp/research/server.py:98, 163-183` — the stable-tool allowlist.
- `testing/backend/unit/synth_ai_sdk/unit/test_research_intern_contract.py:200-206`
- `testing/backend/unit/synth_ai_sdk/unit/test_research_intern_event_stream.py:993-1006`

Those last two are currently red and will go green with this change.

**Backend needs no parallel rename.** `routes_mcp.py` has its own legacy
`_MCP_TOOLS` dict (`routes_mcp.py:1153-1679`) but it covers a disjoint older set
(`smr_capabilities_get`, `smr_projects_list`, …) with no Intern tools. For
everything Intern it merges `ResearchMcpServer().list_tool_payload()` and
delegates `tools/call`, so it inherits whatever synth-ai advertises.

**No tool-manifest digest exists.** The Research Intern release plan mentions an
"MCP tool-manifest digest", but nothing in any repo hashes or snapshots
`available_tool_names()` / `list_tool_payload()`. Two similarly-named artifacts
are unrelated false positives: `codex_tool_manifest_digest()`
(`backend/packages/horizons/actors/codex/app_server.py:404-408`, a Codex CLI
tool-policy hash) and `manifest_digest` in
`synth_ai/mcp/research/tools/environments.py:33-34` (a container image digest).
Nothing downstream needs regenerating.

---

## 5. Verification

```bash
# 1. Server constructs at all (this is the whole bug)
cd synth-ai
uv run python -c "
from synth_ai.mcp.research.server import ResearchMcpServer
s = ResearchMcpServer()
names = set(s.available_tool_names())
intern = {n for n in names if 'research_intern' in n}
assert len(intern) == 22, len(intern)
assert not any(n.startswith('smr_') for n in names), 'smr_ leaked into discovery'
print('OK', len(intern), 'intern tools advertised')
"

# 2. The alias still resolves for old clients
uv run python -c "
from synth_ai.mcp.research.server import ResearchMcpServer
s = ResearchMcpServer()
assert s.get_tool_definition('smr_run_research_intern_turn') is not None
print('OK legacy alias resolves')
"

# 3. The two red tests
cd ../testing
uv run pytest backend/unit/synth_ai_sdk/unit/test_research_intern_contract.py \
              backend/unit/synth_ai_sdk/unit/test_research_intern_event_stream.py -q
```

Note the testing repo resolves sibling repos by path
(`backend/_helpers/source_repo.py`) and expects `synth_ai` installed from the
candidate, not PyPI. Confirm with
`uv run python -c "import synth_ai; print(synth_ai.__version__)"` — it must be
`0.18.1`, not `0.17.0`.

Then re-check the backend `/mcp` route returns a tool list instead of a 500.

---

## 6. Out of scope

- Do **not** add `smr_*` names to `_STABLE_TOOL_NAMES`; they are generated.
- Do **not** touch `registry.py`. The rule and the alias are both correct.
- Do **not** rename the REST routes. The public API stays
  `/smr/research-intern/*`; only MCP tool names change.
- The `slot_id` public-surface failure in
  `testing/backend/unit/synth_ai_sdk/public_surface/test_internal_vocabulary.py`
  is a separate, unrelated decision about `limit_evidence.py`.
