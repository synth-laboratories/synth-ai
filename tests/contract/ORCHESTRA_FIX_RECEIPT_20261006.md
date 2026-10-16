# Orchestra discrepancy fixes — 2026-10-06

Scope: SYN-4001–4004. Provider spend $0; no slot leases, CI wiring, deploys or package publishing. Built on SDK PR #422 at `349d9887e81ccf4eb5e6c20573ccd609328e9847`. Backend authority is current dev `304f00b8690e7da152450833a16630926cfb6c1d` in an isolated clean worktree.

## Decisions and behavior

- SYN-4002 / RW-02: SDK recognizes `already_terminal`, `cleanup_in_progress`, `run_finalizing`. The existing typed error retains the backend retry posture: terminal is final; cleanup/finalizing may be retried later. Eighteen real httpx transport cases exercise stop/pause/resume under both scopes, preserve details/cause, and require exactly one HTTP request. Source enum parity is checked without importing backend services. No automatic mutation retry was introduced.
- SYN-4001 / RW-01: reviewed PR #422's four MCP launch schemas and forwarding of pins/provenance plus all three explicit inventories. No competing launch implementation. Its merge/publish still depends on Forge SYN-3988 and SYN-3989; do not mark deployed acceptance complete from these offline tests.
- SYN-4003 / MX-01: owner approved using existing data and removing unsupported methods. Removed the three nonexistent diagnostics, their RunHandle/RunsAPI conveniences, MCP tools, handlers and scope registry entries. Existing `/activity` uses scoped authoritative execution; `/evidence` serves durable artifacts/WorkProducts; project execution/actor trace reads remain available. These cannot truthfully supply the removed contracts' participant usage status, required/optional artifact staging counts, or redacted paginated exec stdout/stderr. We did not invent aliases, fabricate empty values or access Orchestra's private journal. Removed models remain import-compatible.
- SYN-4003 / MX-03: both task-read public operation IDs already exist in current backend dev (`openapi_contract.py`). Fresh generated OpenAPI confirms their exact IDs and paths. No backend source change or duplicate fix is needed.
- SYN-4004 / MX-02/MX-06: deleted the invalid status patch and the two nonexistent TAG factory-context calls, including all direct wrappers. No MCP tool calls these Factory methods. Current testing source contains no remaining direct consumer of the deleted names.
- Latest owner direction: Factory remains an optional separate module and is excluded from this release. RunResultsAPI's FactoryResult/Effort adapter was removed from run handles; Factory results remain on `factories.results.list(factory_id, run_id=...)`. Sync/async Swarm handles no longer return Factory trace-store types or query those stores; callers explicitly use the optional Factory module's trace-store API with a run filter. Swarm effort IDs remain optional wire-compatible strings rather than EffortId types. Factory and Factory-Effort MCP tools require the existing advanced-tools opt-in instead of appearing in default discovery. Other Factory/Effort implementations and contracts were preserved. Directed outcome contracts are local Swarm data, not Factory module imports. Intern planner-effort discovery remains available; it is a separate model.

## Regression evidence

Local raw receipts: `/Users/joshuapurtell/GitHub/artifacts/orchestra-discrepancy-fixes-20261006/`.

- `red-controls-factory.log`: 25 failures for missing refusal codes/enum entries and reachable dead Factory methods, before product edits.
- `red-observability.log`: 11 failures for reachable unsupported SDK/MCP diagnostics.
- `red-factory-coupling.log`: boundary test fails on EffortId coupling before the modularity change.
- Original RW-02 and MX-01/MX-02/MX-09 laws were promoted to `test_orchestra_route_contracts.py` with matching/assertions unchanged. The MX-06 body law has no callable subject after deletion; explicit absence assertions cover every former wrapper instead. Unrelated known-red laws remain in the law suite.
- Fresh backend app dump: 1538 routes, including 132 hidden routes. No external-route allowlist needed. Pinned route and refusal provenance is in `fixtures/orchestra_provenance.json` and `run_control_enum.generated.json`; live producer overrides are supported.
- Final acceptance: **69 passed**, using live backend enum, full routes and generated bounded OpenAPI; includes the launch review.
- MCP registry/entrypoint: **11 passed**.
- Scoped compatibility: 31 control/typed-run-read checks and 13 optional Factory trace-store checks passed. The latter used `--confcutdir` to prevent the testing repo's parent conftest from injecting the unrelated primary SDK checkout.
- Ruff E4/E7/E9/F/I and formatter pass on touched files. Existing E501 long-line findings on untouched code were not expanded into unrelated cleanup.

## Reproduction

From the GitHub workspace, set `SDK` to this worktree, `BACKEND` to the pinned producer worktree, and `PYTHON` to `artifacts/discrepancy-tests-20261006/venv/bin/python`:

```sh
DISCREPANCY_BACKEND_ROOT="$BACKEND" \
DISCREPANCY_BACKEND_ROUTES="$PWD/artifacts/orchestra-discrepancy-fixes-20261006/backend_all_routes.json" \
DISCREPANCY_BACKEND_OPENAPI="$PWD/artifacts/orchestra-discrepancy-fixes-20261006/research_openapi.generated.json" \
PYTHONPATH="$SDK" "$PYTHON" -m pytest -q \
  "$SDK/tests/contract/test_orchestra_control_contracts.py" \
  "$SDK/tests/contract/test_orchestra_route_contracts.py" \
  "$SDK/tests/contract/test_factory_removal_contracts.py" \
  "$SDK/tests/contract/test_observability_removal_contracts.py" \
  "$SDK/tests/contract/test_discrepancy_contracts.py" \
  "$SDK/tests/mcp/test_launch_schema_contracts.py" \
  "$SDK/tests/mcp/test_launch_contracts.py"
```

Without producer overrides the portable tests compare the retained source-pinned fixture. Refresh authority explicitly when backend changes; fixture success alone is not deployed acceptance.

## Release constraint

This is a dependent SDK change on PR #422, which is still open. SYN-4001 is blocked until the Forge owner integrates/qualifies SYN-3988/SYN-3989 and deploys the backend before SDK publishing. This session did not mutate that owner's branch or slot2. Exact deployed backend identity and passing launch acceptance must be supplied by that release lane before the combined SDK dev push (which publishes a prerelease). No unattended work is running after qualification.
