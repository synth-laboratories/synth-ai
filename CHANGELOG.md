# Changelog

All notable changes to the `synth-ai` package are documented here.

## 0.11.7 — 2026-06-24

### Changed

- **Grok catalog consolidation:** `SmrAgentModel` drops `x-ai/grok-4.1-fast` and `x-ai/grok-4.20-beta`; **`x-ai/grok-4.3`** (direct xAI via Codex) is the single public Grok model. The retired ids are aliased to grok-4.3 on the backend catalog, so existing run-policy references still resolve server-side. Vendored `smr_openapi.yaml` re-synced to the backend contract (byte-match).

## 0.11.6 — 2026-06-24

### Added

- **`client.promotions`** namespace — typed SMR promotion registry SDK:
  - `list_public()`, `mine()`, `claim(campaign_id)` for `GET /smr/promotions`,
    `GET /smr/promotions/mine`, and `POST /smr/promotions/{campaign_id}/claim`.
  - Admin helpers: `list_admin_campaigns()`, `upsert_admin_campaign(...)`,
    `retire_admin_campaign(campaign_id)`.

### Notes

- Pair with backend promotions registry routes (Phases 0–3) on the same deploy train.
- OpenAPI contract synced from backend `smr_openapi.yaml`.

## 0.11.5 — 2026-06-24

### Added

- **`client.billing`** namespace — typed SMR Codex-style billing SDK:
  - `get_catalog()`, `get_plan()`, `get_run_drawdown(run_id)`, `preflight_run(...)`,
    factory-effort drawdown/preflight, and admin grant helpers matching
    `GET /smr/billing/plan`, `/smr/billing/catalog`, run drawdown, and preflight routes.

### Notes

- Ships the billing SDK already merged on `dev`; no Factory surface changes.
- Pair with backend billing routes on the same deploy train.

## 0.11.4 — 2026-06-24

### Added

- **`SmrAgentModel.BASETEN_ZAI_ORG_GLM_5_2`** (`baseten/zai-org/GLM-5.2`) — GLM 5.2 via Baseten, added to the public agent-model enum and the shared actor model allowlist.
- Run-policy **`groq`** credential provider and **`openrouter`** inference provider accepted client-side (BYOK / run-policy coercion no longer rejects them).

### Changed

- **Public model catalog sunset** (backend `smr_supported_models.json`): Trinity Large Thinking (`arcee-ai/trinity-large-thinking` and `:free`), `gpt-oss-120b`, `gpt-5.4-nano`, `deepseek/deepseek-v4-flash`, and `deepseek/deepseek-v4-pro` are no longer public and are dropped from `GET /smr/agent-models`. These ids remain importable from `SmrAgentModel` (no breaking enum removal) but fail backend preflight; use `gpt-5.4-mini`, `gpt-5.4`, or GLM 5.2 (entitlement-dependent). DeepSeek×Codex bench profiles remain available internally via `profile_id`.

### Fixed

- Removed the dead **`smr_get_project_economics`** MCP tool. **Errata:** the 0.11.3 notes listed `smr_get_project_economics` as an included tool, but its handler returned a removed-contract error (no backend route). Use **`smr_get_project_usage`** / **`smr_get_run_usage`** / billing entitlements instead (per the 0.11.2 migration table).

## 0.11.3 — 2026-06-22

### Changed

- Prepared a distinct launch package version for the current Managed Research MCP surface so PyPI can ship the 222-tool launch worktree instead of the stale 0.11.2 wheel.
- Included the project economics and GitHub setup MCP tools in the release target: `smr_get_project_economics`, `smr_setup_github_status`, `smr_setup_github_start_oauth`, `smr_setup_github_list_repos`, and `smr_setup_github_disconnect`.

## 0.11.2 — 2026-06-16

### Added

- **`SynthClient().research`** — public Managed Research + Research Factory namespace (`ResearchClient`).
- MCP entrypoint **`synth-ai-managed-research-mcp`** (stdio) alongside hosted HTTP MCP at `https://api.usesynth.ai/mcp`.
- Optional extra **`synth-ai[research]`** for Research/MCP dependencies.

### Changed

- **Canonical install:** `pip install "synth-ai[research]==0.11.2"` (or `uv add "synth-ai[research]"`).
- **Canonical Python entry:** `from synth_ai import SynthClient` → `client.research`.
- MCP tool enumeration and SDK namespaces reconciled to the shipped backend OpenAPI contract.

### Removed (breaking vs 0.11.1)

The following **public SDK/MCP surfaces were removed** because they are not part of the launch contract. Use the replacements below.

| Removed area | Replacement |
| --- | --- |
| Objective / milestone / experiment CRUD helpers | Objectives are **runtime-managed** during runs; inspect with `smr_get_run_work_graph`, `smr_objectives` (read), run observability |
| Project economics convenience wrappers on deprecated paths | **`smr_get_project_usage`**, **`smr_get_run_usage`**, resource-limit progress tools, billing entitlements |
| Connect-git / legacy GitHub setup SDK modules | GitHub OAuth via MCP `smr_setup_github_*` where enabled; attach repos via console or `smr_work_repos_*` |
| Standalone **`managed-research`** PyPI package as the primary story | **`synth-ai[research]`** only; legacy `ManagedResearchClient` import remains under `synth_ai` for advanced/low-level callers |
| Data-factory trigger/publish MCP tools (`smr_trigger_data_factory`, …) | Factory programs via hosted control plane + documented factory MCP tools |

### Migration (0.11.1 → 0.11.2)

```bash
uv add "synth-ai[research]==0.11.2"
export SYNTH_API_KEY="sk_..."
```

```python
# Before (0.11.1 docs / standalone package)
# from managed_research import ManagedResearchClient
# client = ManagedResearchClient()

# After
from synth_ai import SynthClient

research = SynthClient().research
projects = research.projects.list()
run = research.runs.start(
    "Inspect the repo and leave evidence.",
    host_kind="daytona",
    work_mode="directed_effort",
    providers=[{"provider": "openrouter"}],
    runbook="lite",
)
```

- **MCP stdio:** `uv tool install "synth-ai[research]"` then run `synth-ai-managed-research-mcp`.
- **Errors:** launch denials return typed exceptions / MCP tool errors with `402` (insufficient credits) and `429` (limits). See [Preflight and Errors](https://docs.usesynth.ai/managed-research/preflight-and-errors).
