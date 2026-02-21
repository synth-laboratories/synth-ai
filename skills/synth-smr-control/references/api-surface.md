# SMR API Surface

This reference is the control-plane route map used by `scripts/smr_control.py`.

## Public project controls

- `POST /smr/projects`
- `GET /smr/projects`
- `GET /smr/projects/{project_id}`
- `PATCH /smr/projects/{project_id}`
- `GET /smr/projects/{project_id}/status`
- `POST /smr/projects/{project_id}/pause`
- `POST /smr/projects/{project_id}/resume`
- `POST /smr/projects/{project_id}/archive`
- `POST /smr/projects/{project_id}/unarchive`
- `POST /smr/projects/archive-oldest`

## Onboarding and provider keys

- `POST /smr/projects/{project_id}/onboarding/start`
- `POST /smr/projects/{project_id}/onboarding/complete_step`
- `POST /smr/projects/{project_id}/onboarding/dry_run`
- `GET /smr/projects/{project_id}/onboarding/status`
- `POST /smr/projects/{project_id}/provider_keys`
- `GET /smr/projects/{project_id}/provider_keys/{provider}/{funding_source}/status`

## Starting data

- `POST /smr/projects/{project_id}/starting-data/upload-urls`

## Runs and human-in-the-loop surfaces

Canonical run routes (always available):

- `POST /smr/projects/{project_id}/trigger`
- `GET /smr/runs?project_id=...`
- `GET /smr/runs/{run_id}`
- `POST /smr/runs/{run_id}/pause`
- `POST /smr/runs/{run_id}/resume`
- `POST /smr/runs/{run_id}/stop`
- `GET /smr/runs/{run_id}/questions`
- `POST /smr/runs/{run_id}/questions/{question_id}/respond`
- `GET /smr/projects/{project_id}/questions?status_filter=...`
- `GET /smr/runs/{run_id}/approvals`
- `GET /smr/projects/{project_id}/approvals?status_filter=...`
- `POST /smr/runs/{run_id}/approvals/{approval_id}/approve`
- `POST /smr/runs/{run_id}/approvals/{approval_id}/deny`
- `GET /smr/runs/{run_id}/artifacts`
- `GET /smr/artifacts/{artifact_id}`
- `GET /smr/artifacts/{artifact_id}/content`

Project-scoped run aliases (enforce `project_id` + `org_id` scoping):

- `GET /smr/projects/{project_id}/runs`
- `GET /smr/projects/{project_id}/runs/active`
- `GET /smr/projects/{project_id}/runs/{run_id}`
- `POST /smr/projects/{project_id}/runs/{run_id}/pause`
- `POST /smr/projects/{project_id}/runs/{run_id}/resume`
- `POST /smr/projects/{project_id}/runs/{run_id}/stop`
- `GET /smr/projects/{project_id}/runs/{run_id}/questions`
- `POST /smr/projects/{project_id}/runs/{run_id}/questions/{question_id}/respond`
- `GET /smr/projects/{project_id}/runs/{run_id}/approvals`
- `POST /smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/approve`
- `POST /smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/deny`
- `GET /smr/projects/{project_id}/runs/{run_id}/artifacts`

## Results, logs, and orchestrator status (run-scoped)

- `GET /smr/projects/{project_id}/runs/{run_id}/results` — outcome + artifacts-by-type + log debug hint
- `GET /smr/projects/{project_id}/runs/{run_id}/orchestrator` — orchestrator phase, heartbeat, turn count, and full turn history (`phase`, `started_at`, `finished_at`, `completed`, `error`, `duration_seconds` per turn)
- `GET /smr/projects/{project_id}/runs/{run_id}/logs` — structured VictoriaLogs query (task_key, component, limit, start, end)
- `GET /smr/projects/{project_id}/victoria-logs/search` — free-text LogSQL search across a project

## Ops and observability

- `GET /smr/projects/{project_id}/usage`
- `GET /smr/projects/{project_id}/ops_status`

## Runtime surfaces

- SDK client: `synth_ai.sdk.managed_research.SmrControlClient`
- CLI group: `synth-ai managed-research ...`
- MCP server: `synth-ai-mcp-managed-research` or `synth-ai managed-research mcp-server`

## Workspace git status

- `GET /smr/projects/{project_id}/workspace/git` — read-only git status: `configured`, `commit_sha`, `last_pushed_at`, `default_branch`, `vcs_provider`, `remote_repo`. Storage internals (bucket, archive key) are intentionally omitted.

## Agent model and kind selection

Three levels of override, in priority order (highest first):

| Level | How to set | Scope |
|---|---|---|
| Per-run override | `trigger_run(agent_model=..., agent_kind=...)` or `POST /trigger` body | Single run only |
| Per-project default | `set_agent_config(project_id, model=..., agent_kind=...)` → writes `execution.agent_model` / `execution.agent_kind` | All future runs |
| Server default | `SMR_AGENT_KIND` + `SMR_AGENT_MODEL` env vars on orchestrator host | Process-wide |

Valid `agent_kind` values: `codex` (default, uses OpenAI), `claude` (Claude Code), `opencode`.

When `agent_kind` changes, the orchestrator rebuilds its runtime for that run (MCP server is reused). The same model and kind is forwarded to all dispatched workers.

## MCP tools (31 total)

| Tool | Description |
|---|---|
| `smr_list_projects` | List managed research projects |
| `smr_get_project` | Fetch a project by id |
| `smr_get_project_status` | Get project status |
| `smr_create_project` | Create a new project |
| `smr_pause_project` | Pause a project |
| `smr_resume_project` | Resume a paused project |
| `smr_archive_project` | Archive a project |
| `smr_unarchive_project` | Unarchive a project |
| `smr_get_starting_data_upload_urls` | Get presigned upload URLs for starting data |
| `smr_upload_starting_data` | Upload starting data files (text) |
| `smr_trigger_run` | Trigger a new run (supports `agent_model` + `agent_kind` per-run override) |
| `smr_set_agent_config` | Set default agent model / kind for all future runs of a project |
| `smr_list_runs` | List runs (all or active-only) |
| `smr_get_run` | Fetch a run by id |
| `smr_pause_run` | Pause a run |
| `smr_resume_run` | Resume a paused run |
| `smr_stop_run` | Stop a run |
| `smr_list_project_questions` | List project-level questions |
| `smr_respond_question` | Respond to a question |
| `smr_list_project_approvals` | List project-level approvals |
| `smr_resolve_approval` | Approve or deny an approval |
| `smr_get_usage` | Fetch project usage metrics |
| `smr_get_ops_status` | Fetch ops/task status |
| `smr_get_run_logs` | Query VictoriaLogs for a run (structured) |
| `smr_search_project_logs` | Free-text LogSQL search across a project |
| `smr_list_run_artifacts` | List run artifacts |
| `smr_get_artifact` | Fetch artifact metadata |
| `smr_get_run_results` | Run result summary: outcome, artifacts, log hint |
| `smr_get_project_git_status` | Read-only workspace git: commit SHA, last push, branch, remote repo |
| `smr_get_orchestrator_status` | Orchestrator phase, heartbeat, turn count, and full turn history |
