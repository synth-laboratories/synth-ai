# Launch and usage integration corrections

Backend source commit: `57552b47b6`. SDK source commit: `faf29a4a9594fb52a35ed199c3e6189d0174b8a7`.

The ResourceUsage union now represents the existing granted CPU/GPU dimensions. Public synchronous/asynchronous Swarms preflight/create and both MCP launch adapters preserve the already-defined strict `parent_hold` reservation reference. The new regressions exercise both adapters and measured/unknown resource states; they make no network or provider calls.

Retained contract generation used Pydantic 2.12.5 in both worktrees. Backend and SDK match:

- core.v1: `b13ac8b7f87f8b3e51a9e4cea8d6eab37184be71377cd9383a59977529a27b5b`
- index.v1: `973d74b553b1cfbeaac9100403f8fc75897aafc3f63d30ed8b6f3a046b4f4289`
- swarms.v1: `f65b870cf100e142e6b7740e5084306a7ba5383aeeb0dca192479ca254578a91`

SDK source verification: **1005 passed, 1 skipped**. The skip is the wheel-content check because this run preceded the paired dev03 wheel build. `sdk-launch-usage-corrected.json` retains the exact command, pinned Pydantic version, exit code and elapsed time; its matching log retains pytest output. `sdk-launch-usage-final` retains the preceding 1003-pass/2-failure run that exposed the two stale reviewed Swarms hash fixtures; those pins were updated to the paired generated hash before the corrected run.

Backend fixture verification: **66 passed**, including 41 PostgreSQL approval/wait/control-grant cases plus native preview and resource dimensions. The wrapper was corrected to supply asynchronous PostgreSQL URLs and the missing CONTROL_GRANT_TEST_DATABASE_URL; the approvals fixture now explicitly supplies its no-run-gates owner port. Native preview fixtures contain genuine canonical seed profile bytes and a valid ProgramSelection, without bypassing effective-limit admission. The earlier wrapper selected the separate `v023_backend_test` database even when given the owner fixture; these 66 cases do not claim verification against `v023_owner_test`. The integration owner subsequently amended the wrapper to preserve the fixture's actual database and is running final installed-wheel/API verification there after the paired wheel and pending capture epoch migration.

No deployment, publication, paid calls, original-run recovery operations or Rust compilation occurred in this bounded package. Pressure papercut `387ba68b-2306-47ea-899c-aabd8e0282cd` records the paired contract gaps as fixed_in_source pending installed dev03 adoption.
