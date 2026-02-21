# SMR SDK + MCP Control Proposal (2026-02-19)

## Goal

Add first-class Synth SDK and MCP support for controlling Synth Managed Research (SMR) projects and runs.

This proposal covers:

- a new `synth_ai.sdk.managed_research` client,
- a new MCP server surface that wraps the SDK,
- and rollout sequencing to align backend/frontend/spec route contracts.

## Current State (from code)

### What is already implemented

- Backend SMR control-plane routes exist in `/Users/joshpurtell/Documents/Github/backend/app/api/v1/routes_smr.py`.
- SMR data-plane MCP exists in `/Users/joshpurtell/Documents/Github/synth-managed-research/crates/smr-worker-host/src/main.rs` and serves:
  - `POST /mcp/orchestrator`
  - `POST /mcp/worker/:worker_id`
- Orchestrator and worker MCP tool sets are defined in:
  - `/Users/joshpurtell/Documents/Github/synth-managed-research/crates/smr-worker-host/src/smr_orchestrator_tools.rs`
  - `/Users/joshpurtell/Documents/Github/synth-managed-research/crates/smr-worker-host/src/smr_worker_tools.rs`

### Gaps relevant to SDK + MCP product surface

- `synth-ai` currently has no dedicated SMR control client module under `synth_ai/sdk/`.
- Frontend and specs use project-scoped run routes (`/smr/projects/{project_id}/runs/...`) while backend currently exposes canonical run routes mostly under `/smr/runs/...`.
- Frontend SMR provider-key route currently prefers encrypted payloads (`encrypted_key_b64`) while backend SMR provider-key schema expects plaintext `api_key`.

## Proposed SDK Surface

## Module layout

Add:

- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/sdk/managed_research/__init__.py`
- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/sdk/managed_research/client.py`
- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/sdk/managed_research/models.py`
- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/sdk/managed_research/errors.py`

Export from:

- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/sdk/__init__.py`

### Primary class

`ManagedResearchClient`

Constructor pattern should match existing SDK clients:

- `api_key: Optional[str] = None` (fallback `SYNTH_API_KEY`)
- `backend_base: Optional[str] = None` (fallback Synth backend resolver)
- `timeout_seconds: float = 30.0`

### Method groups

Project lifecycle:

- `create_project(...)`
- `list_projects(include_archived=False)`
- `get_project(project_id)`
- `patch_project(project_id, ...)`
- `get_project_status(project_id)`
- `pause_project(project_id)`
- `resume_project(project_id)`
- `archive_project(project_id)`
- `unarchive_project(project_id)`

Onboarding + keys:

- `onboarding_start(project_id)`
- `onboarding_complete_step(project_id, step, status, detail=None)`
- `onboarding_dry_run(project_id)`
- `onboarding_status(project_id)`
- `set_provider_key(project_id, provider, funding_source, api_key=None, encrypted_key_b64=None, encrypt_before_send=False)`
- `provider_key_status(project_id, provider, funding_source)`

Run controls:

- `trigger_run(project_id, timebox_seconds=None)`
- `list_runs(project_id)`
- `list_active_runs(project_id)`
- `get_run(run_id, project_id=None)`
- `pause_run(run_id)`
- `resume_run(run_id)`
- `stop_run(run_id)`

Human-in-loop:

- `list_project_questions(project_id, status_filter="pending")`
- `list_run_questions(run_id, project_id=None)`
- `respond_question(run_id, question_id, response_text, project_id=None)`
- `list_project_approvals(project_id, status_filter="pending")`
- `list_run_approvals(run_id, project_id=None)`
- `approve(run_id, approval_id, comment=None, project_id=None)`
- `deny(run_id, approval_id, comment=None, project_id=None)`

Artifacts + observability:

- `list_run_artifacts(run_id, project_id=None)`
- `get_artifact(artifact_id)`
- `get_artifact_content_response(artifact_id, disposition="inline", follow_redirects=False)`
- `get_usage(project_id)`
- `get_run_spend_entries(run_id)` (admin spend-ledger rows)
- `get_run_economics(run_id)` (admin run economics summary)
- `get_run_usage_by_actor(run_id, project_id=None, include_done_tasks=True)`
- `get_ops_status(project_id, include_done_tasks=None)`
- `search_victoria_logs(project_id, ...)`

### Compatibility rule

For all run-scoped read/write methods, use project-scoped route first and fallback to canonical `/smr/runs/...` if project-scoped returns `404`.

This gives a stable SDK surface now while backend aliases are added.

### Error model

Define `ManagedResearchApiError` including:

- `status_code`
- `method`
- `path`
- `detail_snippet`

This should mirror existing SDK error ergonomics (clear status + endpoint context).

## Proposed MCP Surface (in `synth-ai`)

## Why separate MCP

SMR already has internal orchestrator/worker MCP for data-plane runtime.
This proposal adds a customer/operator-facing MCP server that exposes control-plane actions via the new SDK client.

### Transport

- Default: `stdio`
- Optional: Streamable HTTP

Rationale: aligns with MCP architecture and transport guidance, where stdio is common local transport and Streamable HTTP is recommended for production deployments.

### Tool namespace

Use explicit names to avoid collision with internal worker tools.

Project tools:

- `smr.project.create`
- `smr.project.list`
- `smr.project.get`
- `smr.project.update`
- `smr.project.status`
- `smr.project.pause`
- `smr.project.resume`
- `smr.project.archive`

Onboarding tools:

- `smr.onboarding.start`
- `smr.onboarding.complete_step`
- `smr.onboarding.dry_run`
- `smr.onboarding.status`
- `smr.provider_key.set`
- `smr.provider_key.status`

Run tools:

- `smr.run.trigger`
- `smr.run.list`
- `smr.run.list_active`
- `smr.run.get`
- `smr.run.pause`
- `smr.run.resume`
- `smr.run.stop`

Approval/question tools:

- `smr.question.list_project`
- `smr.question.list_run`
- `smr.question.respond`
- `smr.approval.list_project`
- `smr.approval.list_run`
- `smr.approval.approve`
- `smr.approval.deny`

Artifact/ops tools:

- `smr.artifact.list_run`
- `smr.artifact.get`
- `smr.artifact.content_link`
- `smr.usage.get`
- `smr.run.spend_entries.get` (admin scope)
- `smr.run.usage_by_actor.get`
- `smr.ops_status.get`
- `smr.logs.search`

### Safety defaults

- No `/smr/internal/*` tools in this MCP.
- Explicitly mark side-effecting tools in tool descriptions.
- Require required IDs (`project_id`, `run_id`, etc.) and reject ambiguous calls.
- Return structured JSON only (no markdown blobs) for machine composability.

## Packaging + CLI integration

Add optional dependency group in `pyproject.toml`:

- `mcp>=1.0.0`

Add CLI command group (flat style, per CLI AGENTS guidance):

- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/cli/commands/mcp/__init__.py`
- `/Users/joshpurtell/Documents/Github/synth-ai/synth_ai/cli/commands/mcp/smr.py`

Suggested command:

- `synth-ai mcp smr --transport stdio`
- `synth-ai mcp smr --transport streamable-http --host 0.0.0.0 --port 8765`

## Rollout Plan

1. Backend contract alignment

- Add project-scoped run aliases in backend.
- Add `encrypted_key_b64` compatibility for SMR provider-key set.
- Keep canonical `/smr/runs/...` routes for backward compatibility.

2. SDK alpha (`synth_ai.sdk.managed_research`)

- Implement methods listed above.
- Add fallback behavior and unit tests.
- Export from `synth_ai.sdk` root.

3. MCP alpha (`synth-ai mcp smr`)

- Implement read-only tools first.
- Add side-effect tools after validation.
- Add tool contract tests (`tools/list`, `tools/call`).

4. GA hardening

- Add end-to-end tests against local backend.
- Publish docs and examples.
- Set stability tag policy in docs (Alpha -> Beta -> Stable).

## Acceptance Criteria

- Users can control full SMR lifecycle from Python without manual REST wiring.
- MCP tools cover all core operator actions and return consistent typed JSON.
- SDK behavior is stable despite backend route transition (project-scoped + canonical).
- Key upload path is explicit and secure, with deterministic fallback behavior.

## Addendum (2026-02-20): Granular usage + dollar-cost semantics

`get_run_usage_by_actor(...)` now defines two explicit output modes:

- `usage_mode="spend_entries"`: exact cost path sourced from `/smr/admin/runs/{run_id}/spend`.
- `usage_mode="logs_thread_totals"`: fallback path sourced from run logs token snapshots.

In exact-cost mode, `summary` and each orchestrator/worker/model/session row includes:

- `total_cost_cents`, `total_cost_usd`
- `meter_quantities` (input, cached input, output, reasoning, and non-token meters when present)
- `meter_cost_cents`, `meter_cost_usd`
- `token_usage` split (`input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens`)
- `token_cost_cents`, `token_cost_usd` split by:
  - input
  - cached input
  - output
  - reasoning output
- `cost_data_available=true`

In fallback mode (admin spend unavailable), output includes model attribution + split token quantities but marks:

- `cost_data_available=false`
- `total_cost_cents=None` / `total_cost_usd=None`

Fallback attempts run-level estimated dollars from project usage rollups:

- `summary.estimated_total_cost_cents` / `summary.estimated_total_cost_usd`
- `summary.estimated_orchestrator_total_cost_cents` / `summary.estimated_worker_total_cost_cents`
- per actor/model/session:
  - `estimated_total_cost_cents`
  - `estimated_total_cost_usd`
- `summary.estimated_cost_source="project_usage_per_run_token_share"`

Allocation rule for estimates: divide run-level project usage cost across actors/models/sessions by `token_usage.total_tokens` share.

## Primary external references

- MCP architecture: [https://modelcontextprotocol.io/docs/learn/architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- MCP transports: [https://modelcontextprotocol.io/docs/concepts/transports](https://modelcontextprotocol.io/docs/concepts/transports)
- MCP tools: [https://modelcontextprotocol.io/docs/concepts/tools](https://modelcontextprotocol.io/docs/concepts/tools)
- MCP Python SDK: [https://github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
- Synth prompt optimization docs: [https://docs.usesynth.ai/prompt-optimization/gepa](https://docs.usesynth.ai/prompt-optimization/gepa)
