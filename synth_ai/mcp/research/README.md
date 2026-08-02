# MCP

This package owns the canonical MCP surface for `managed-research`.

Surface note: MCP tools call the authenticated private-beta Managed Research
API. When tool or schema descriptions say public, they mean the stable API
contract. Managed Research beta access is an account/org entitlement enforced by
the backend through entitlement checks and launch preflight, not by narrowing
the MCP tool list.

What belongs here:
- tool registration, schemas, and scope metadata
- shared tool-list / call-tool primitives used by both stdio and hosted transport
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

Stability rule:
- keep MCP tool names and wire payload shapes stable unless a deliberate migration is planned
- fail loudly on malformed input instead of silently defaulting to success-shaped values
- tool names retain the stable `smr_` wire prefix; descriptions and docs should
  call the product Managed Research.

Canonical launch flow:
- `smr_create_runnable_project`
- `smr_get_project_setup`
- `smr_prepare_project_setup`
- `smr_get_launch_preflight`
- `smr_trigger_run`
- `smr_get_run`
- noun reads such as `smr_get_project_workspace`, `smr_objectives` with
  `operation=list`, `smr_list_run_objective_events`, `smr_list_run_questions`,
  `smr_get_run_work_graph`, and `smr_get_run_traces`

Legacy readiness/blocker aliases are intentionally removed from the maintained surface.

Run-control additions:
- `smr_get_run_logical_timeline` for operator-facing checkpoint/message/branch chronology
- `smr_get_run_actor_usage` for truthful per-actor usage attribution
- `smr_get_run_traces` for persisted downloadable run traces
- `smr_branch_run_from_checkpoint` for exact branches and branch-with-message
- `smr_runtime_message_queue` remains the live steering tool and is intentionally separate from branching

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
