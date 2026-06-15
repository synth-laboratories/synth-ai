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
- noun reads such as `smr_list_run_questions`, `smr_get_run_primary_parent`,
  `smr_open_ended_questions`, and `smr_directed_effort_outcomes`

Legacy readiness/blocker aliases are intentionally removed from the maintained surface.

Run-control additions:
- `smr_get_run_logical_timeline` for operator-facing checkpoint/message/branch chronology
- `smr_get_run_actor_usage` for truthful per-actor usage attribution
- `smr_get_run_traces` for persisted downloadable run traces
- `smr_branch_run_from_checkpoint` for exact branches and branch-with-message
- `smr_runtime_message_queue` remains the live steering tool and is intentionally separate from branching

Provider-wrapper note:
- OpenRouter, Tinker, and Modal wrapper usage should still be read through canonical
  run usage and actor-usage surfaces, not wrapper-specific payloads
