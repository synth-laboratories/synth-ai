# Changelog

All notable changes to the `synth-ai` package are documented here.

## Unreleased

### Added

- **Typed Tag Factory interface** — Tag session creation now requires `TagSessionCreateRequest`; session, watch, receipt, steering, and artifact payloads parse into typed models; and `get_factory_context(scope_id=… | session_id=…)` exposes the current champion, last decision, and candidate counts without raw `/smr/*` reads.
- **Runnable-project runtime artifact authority** — `SmrRunnableProjectRequest` preserves an optional `runtime_artifact_release_id` so callers can bind project creation to an exact backend-registered runtime release without bypassing SDK normalization.
- **Canonical cloud slots** — CloudDeployment creation accepts the typed `slot1-cloud` / `slot2-cloud` identity through the SDK and MCP. The `cloud-slot` CLI now addresses deployments by that identity and exposes status, health, claim, heartbeat, fenced deploy/retire, release, and claim-truth workflows. The MCP live proof records the slot, claim id, fencing token, lifecycle, and release receipts.

## 0.15.0 — 2026-07-13

### Added

- **Research Factory evidence** — typed Factory status now exposes proof readiness, accepted actor outputs, and experiment observability. `FactoriesAPI` adds experiment bundle/history/compare reads, while Tag and CLI projections carry experiment, candidate, and tag evidence.
- **Factory maintenance runs** — `EffortsAPI.launch_maintenance(...)`, `FactoryRunKind.MAINTENANCE`, `FactoryMaintenanceAction`, and the `FactoryActorRole.ORCHESTRATOR` role support typed maintenance cycles without crossing backend authority boundaries.
- **Project-bound Cloud Deployments** — `client.cloud_deployments` exposes exe.dev-backed deployments tied to project Git:
  - Raw methods: `create_cloud_deployment`, `list_cloud_deployments`, `get_cloud_deployment`, `observe_cloud_deployment`, `deploy_cloud_deployment`, `retire_cloud_deployment` over `/smr/v1/deployments`.
  - Convenience: `service_url()`, `wait_until_running()` (`failed` is retryable via `deploy`; `retired` is terminal).
  - Safety: `retire(delete_vm=True)` requires `confirm_vm_name`.
  - MCP tools: `smr_list/create/get/observe/deploy/retire_cloud_deployment` plus `research_*` aliases (list/get READ scope; create/observe/deploy/retire WRITE scope).
  - OpenAPI vendoring: `/smr/v1/deployments` paths and `CloudDeploymentCreateRequest`/`DeployRequest`/`RetireRequest` schemas in `schemas/smr_openapi.yaml`.
- **P8 proof runners**: `scripts/prove_cloud_deployment_client.py`, `scripts/prove_cloud_deployment_mcp.py`.
- **`SmrWorkerSubtype.ARTIFACT_BUILDER`** and **`SmrReviewerSubtype.ARTIFACT_REVIEWER`** — typed actor subtypes for Open Research hosted artifact build and review flows.
- **`SynthClient().research.hosted_artifacts`** — operator API for hosted artifacts:
  - `list(project_id=…)`, `get`, `get_for_run`, `get_content`
  - `update` (PATCH metadata), `publish_public`, `assign_reviewer`, `delete`
  - `list_public`, `get_public` for the Open Research index
- **Run handle:** `handle.hosted_artifact.get()`, `.content()`, `.publish_public()`, `.assign_reviewer()`
- **Mintlify SDK reference:** [Hosted artifacts](/reference/sdk/research/synth_ai-research-hosted-artifacts) (via `make docs-gen`).

### Notes

- Pairs with backend Factory and CloudDeployment routes on the same release train; publish only after the backend routes are deployed for the target audience.
- Pairs with backend hosted-artifacts routes and migration `20260628_add_smr_hosted_artifacts`.
- Creation remains in-run via worker MCP `publish_hosted_artifact`; SDK covers operator read/update/delete after publish.

## 0.14.0 — 2026-06-27

### Added

- **`SynthClient().research.efforts`** — run→Effort graduation surface for promoting related Runs into persistent Research Factory Efforts:
  - `efforts.proposals.list(project_id)` returns Gardener-authored graduation proposals.
  - `efforts.from_runs(project_id, name, run_ids, factory_id=None)` graduates a set of Runs into a new Effort.
- **Graduation SDK models**: `GraduationProposal` and `EffortFromRunsRequest` (with `effort_from_runs_payload`).
- **`TagSession.graduation_proposal`** — optional graduation-proposal payload on the Tag session projection.
- **MCP tools**: `smr_list_graduation_proposals`, `smr_graduate_runs_to_effort`, and `smr_list_runs_by_effort` (reverse index over runs that carry `effort_id`).

### Notes

- Pairs with backend `GET /api/v1/managed_research/efforts/proposals`, `POST /api/v1/managed_research/efforts/from-runs`, and `GET /api/v1/managed_research/efforts/{effort_id}/runs` on the same release train.

## 0.12.0 — 2026-06-25

### Added

- **`SynthClient().research.tag`** — Synth Tag beta session API for delegated Managed Research work:
  - `create_session(...)` starts a Tag session and bound SMR run.
  - `get_session(session_id)` returns coarse status and terminal receipt.
  - `send_message(session_id, ...)` steers the same active run without starting a new run.
  - `get_default_scope()` returns the default organization Tag scope.
- **Tag SDK models**: `TagSession`, `TagSessionCreateRequest`, `TagMessageRequest`, `TagSessionReceipt`, and `TagScope`.
- **MCP tools**: `tag_create_session`, `tag_get_session`, and `tag_send_message`.
- **Example**: `examples/tag_delegate_smoke.py` for delegate, steer, and receipt smoke checks.

### Notes

- Pair with backend `/api/tag/v1` routes and the `tag_steward` runbook on the same release train.
- Production proof: Tag smoke session `b5cd5bf9-c1a7-46db-a5a2-799ca1816c8b`, run `ba255a34-0da3-4007-8ac3-0c677d836d85`, receipt `state=done`, artifact `/smr/work-products/5eae8c7a-6bf4-54e8-93a5-96401fb237f7/content`.

## 0.11.7 — 2026-06-25

### Added

- **`SmrAgentModel.X_AI_GROK_BUILD`** (`x-ai/grok-build`) — typed SDK/MCP support for hosted SMR runs on xAI Grok Build via direct **xAI** routing (`codex_xai_grok_build` platform profile). Pairs with backend catalog row `grok-build-0.1` and metered xAI proxy.

### Changed

- **Grok catalog consolidation:** `SmrAgentModel` drops `x-ai/grok-4.1-fast` and `x-ai/grok-4.20-beta`; **`x-ai/grok-4.3`** (direct xAI via Codex) is the supported public Grok 4.x model. Retired ids are aliased to grok-4.3 on the backend catalog, so existing run-policy references still resolve server-side. Vendored `smr_openapi.yaml` re-synced to the backend contract (byte-match).

### Fixed

- **xAI chat compatibility (backend + contract):** upstream xAI requests no longer send OpenAI-style penalty fields that caused streaming disconnects or hidden provider errors on Grok paths.

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
