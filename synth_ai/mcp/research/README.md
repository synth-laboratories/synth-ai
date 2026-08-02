# MCP

This package owns the canonical MCP surface for Managed Research.

Surface note: MCP tools call the authenticated private-beta Managed Research
API. When tool or schema descriptions say public, they mean the stable API
contract. Managed Research beta access is an account/org entitlement enforced by
the backend through entitlement checks and launch preflight, not by narrowing
the MCP tool list.

What belongs here:
- tool registration, schemas, and scope metadata
- shared tool-list / call-tool primitives
- stdio JSON-RPC/MCP transport handling
- MCP-specific request parsing at the boundary
- translation from MCP tool calls into SDK client calls

What does not belong here:
- general SDK request construction
- broad API response-model ownership
- backend contract decisions

Primary entrypoints:
- `server.py`: stdio server and tool dispatch
- `tools/`: tool definitions and input schemas
- `request_models.py`: typed MCP request parsing helpers
- `registry.py`: shared MCP tool registration, metadata, and call primitives

Boundary rule:
- parse untyped JSON-RPC payloads once near the transport boundary
- pass normalized typed values or request objects into handlers
- do not carry ad hoc `.get()` / `isinstance()` branching deep into tool logic

## Tool names

Tool builders in `tools/` declare names as `research_*` directly, and that is
what discovery advertises. For backward compatibility, every `research_*` tool
also answers to the legacy `smr_*` spelling: `resolve_tool` treats `smr_<x>` as
a generated wire alias for `research_<x>`, so both work on the wire and only
`research_*` appears in `tools/list`. Declaring a new tool under the `smr_`
prefix is a registry build error.

Every tool must declare required scopes, either on the `ToolDefinition` or in
`registry._DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME`, keyed on the advertised
`research_*` name. A tool with neither raises at registry build rather than
becoming callable without a scope.

## What the entrypoint advertises

`synth-ai-research-mcp` advertises the stable subset (64 of 254 tools). The rest
are built but hidden, and because `call_tool` resolves against the advertised
set they are also uncallable. Set `SYNTH_RESEARCH_MCP_ADVANCED_TOOLS=1` to
advertise the full tree.

Stability rule:
- keep advertised tool names and wire payload shapes stable unless a deliberate
  migration is planned
- fail loudly on malformed input instead of silently defaulting to
  success-shaped values

Canonical launch flow, all in the stable subset:
- `research_create_runnable_project`
- `research_get_project_setup`
- `research_prepare_project_setup`
- `research_get_launch_preflight`
- `research_trigger_run`
- `research_get_run`

Noun reads and run-control tools below need
`SYNTH_RESEARCH_MCP_ADVANCED_TOOLS=1`:
- `research_get_project_workspace`, `research_objectives` with `operation=list`,
  `research_list_run_objective_events`, `research_list_run_questions`,
  `research_get_run_work_graph`, `research_get_run_traces`
- `research_get_run_logical_timeline` for operator-facing
  checkpoint/message/branch chronology
- `research_get_run_actor_usage` for truthful per-actor usage attribution
- `research_runtime_message_queue`, the live steering tool, deliberately
  separate from branching

`research_branch_run_from_checkpoint`, for exact branches and
branch-with-message, is in the stable subset.

Sync Intern control plane:
- `intern_sync_create`, `intern_sync_list`, and `intern_sync_get` address
  durable operator-present sessions
- `intern_sync_send`, `intern_sync_intervene`, `intern_sync_answer`,
  `intern_sync_pause`, `intern_sync_resume`, and `intern_sync_close` are typed
  generation-fenced controls over the same command inbox
- `intern_sync_command` preserves the exact command envelope, while
  `intern_sync_events` and `intern_sync_tail` provide reconnectable ledger access

These tools are thin adapters over `client.intern.sync_`. Sync session identity
and event cursors are explicit because one organization may have multiple Sync
sessions; they never address the singleton Async resource implicitly.

Async Intern control plane:
- `intern_async_ensure` and `intern_async_get` address the organization's one
  long-lived Async Intern runtime
- `intern_async_command` preserves the exact durable command envelope
- `intern_async_send`, `intern_async_intervene`,
  `intern_async_redirect_objective`, and `intern_async_request_checkpoint`
  submit typed instructions to the same backend inbox
- `intern_async_pause`, `intern_async_resume`, `intern_async_cancel`, and
  `intern_async_provide_input` expose explicit state-machine controls
- `intern_async_events` replays a bounded event page and `intern_async_tail`
  waits for a bounded number of SSE events

These tools are thin adapters over `client.intern.async_`; they do not call
HTTP directly and do not contain reducer logic. A command receipt proves
durable admission, not completion. Callers retain command/idempotency identity
across retries and use the event cursor for progress after disconnecting. This
MCP surface is the external client control plane. The Intern's own
capability-gated Factory/Swarm MCP execution happens behind Temporal and is not
routed through these tools. Manderqueue remains actor transport for a bound Run
and is never the client mailbox.

Provider-wrapper note:
- OpenRouter, Tinker, and Modal wrapper usage should still be read through canonical
  run usage and actor-usage surfaces, not wrapper-specific payloads
