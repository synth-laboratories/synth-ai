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

Tool builders in `tools/` still spell names `smr_*`, but nothing is advertised
under that prefix: `build_tool_registry` rewrites every `smr_` to `research_`
before discovery. `resolve_tool` keeps accepting the `smr_` spelling from
callers, so both work on the wire and only `research_*` appears in `tools/list`.

Every tool must declare required scopes, either on the `ToolDefinition` or in
`registry._DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME`, keyed on the advertised
`research_*` name. A tool with neither raises at registry build rather than
becoming callable without a scope.

## What the entrypoint advertises

`synth-ai-research-mcp` advertises the stable subset (64 of 296 tools). The rest
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

Provider-wrapper note:
- OpenRouter, Tinker, and Modal wrapper usage should still be read through canonical
  run usage and actor-usage surfaces, not wrapper-specific payloads
