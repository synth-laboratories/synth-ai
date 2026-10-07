# Forge discrepancy acceptance

Known-red `*_law.py` assertions describe the desired contract and identify the
2026-10-06 finding. They fail normally; no xfail or CI wiring hides the defects.
Move a green law into its contracts suite after a product fix is verified.

From this worktree, using the existing backend test interpreter:

```sh
PYTHONPATH="$PWD" /Users/joshuapurtell/GitHub/backend/.venv/bin/python -m pytest tests/contract tests/mcp -q --tb=short
```

The socket guard prevents network/provider calls. HTTP responses come only from
committed producer schemas through an in-process fake transport. All advertised
MCP input schemas are checked; 212 response fixtures have validated synthetic
inputs, including the 14 confirmed serialization defects. The other audit probe
inputs require local I/O, fail input validation, or have synthetic response noise;
those are not promoted to product acceptance failures.

The generated bounded OpenAPI, full OpenAPI, hidden-route table and public enums
are fixtures; `fixtures/provenance.json` records source commits, generator version
and hashes. Refresh all fixtures from the same backend source pin. The authority
is backend source, not an SDK spec copy. Use the audit generator and route dump,
then export enums from `packages/smr/contracts/public_api/v1/run_state.py` and
`packages/smr/control/public_api/run_controls.py`. Generator scripts and exact
commands are in the dated audit's `scripts/` directory. Ignore only the handoff's
FastAPI-generated visual-body and ValidationError noise; semantic schema changes
must fail. AST route resolution includes raw calls beyond the bounded registry.

First-class errors are acceptance requirements: stable codes, dedicated public
error classification, retryability, retained intent identity and causal chains.
Do not make deterministic rejection look uncertain, or integrity damage look like
a writable revision conflict. Do not weaken these assertions to get a green run.
See backend `tigerstyle.md` and
`notes/specifications/tanha/current/systems/platform/forge_scientific_delivery.md`.
