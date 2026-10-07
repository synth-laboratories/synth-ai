# Launch-path fix receipt — 2026-10-06

The launch-path lane is implemented on isolated, unmerged
`codex/forge-launch-contract-fixes-20261006` branches in backend and synth-ai.
Backend base: 203ace371. SDK base: dev 67bc4c9f plus the retained discrepancy
acceptance branch cc1ced4d. No deployment, publishing, CI wiring, provider/model
calls or paid effects. Other fix groups remain outside this lane.

## Result

- Public launch models require non-empty deployment pins, provenance mode, and
  all three explicit resource inventories. MCP exposes and forwards those fields
  on start, trigger, one-off and dev-environment launch paths. Empty inventories
  survive typed serialization, including model_file_ids.
- Preflight uses the trigger provenance validator and scientific writer authority;
  transferred scopes return the same typed code. Factory-linked preflight uses
  the same read-only trace-store binder. Checks allocate no run or hold.
- Native aggregate hold capacity derives from immutable selected role caps.
  Preflight and trigger share the inference-admission budget gate: three $0.25
  holds require a minimum $0.75 ceiling, so $0.50 refuses before allocation.
  The minimum and per-role micros are exposed as typed preflight fields. This is
  a launch minimum, not an actual spend estimate or a guarantee of run completion.
- Selected file IDs survive preflight projections. Verifier-only projects can
  explicitly select no model files. Preflight, snapshot admission and native
  delivery use the same 128 KiB/file and 192 KiB aggregate bounds and typed 413
  resource_delivery_limit_exceeded refusal, without writing a snapshot.
- ResearchLaunchRefusalError preserves codes, operation/intent identity, retry
  posture, limit requirements and authority details. Required-input HTTP 422
  shape failures are classified too. Core replay honors its first-class false
  directive instead of overriding it with an HTTP status fallback (EX-01).
- Backend generated/committed and SDK vendored contracts are synchronized.
  Verified green laws moved to contracts with their assertions and finding IDs
  retained. Known-red laws for other groups were not weakened.

## Evidence

77 primary checks pass: SDK 43, backend 33, disposable PG16 writer-authority 1.
The PG case uses valid pins and explicitly asserts scientific_writer_transferred,
so another gate cannot mask the writer result. Explicit retryable transient
controls also pass. Existing provenance/resource/execution-binding authorities:
82 passed / 1 environment skip. Existing native/readiness/trace suites: 51 passed
/ 5 explicit environment skips. Provider-selection compatibility: 22 passed.
These suite counts overlap and are not presented as a unique combined total.

Freshly built/installed synth-ai 0.22.2 passed bindings and required MCP launch
schema smoke outside PYTHONPATH. Version alone is not the candidate identity;
source commits and fixture hashes pin the change. Raw JUnit, logs, generator
output and wheel-install receipt are retained in
`artifacts/forge-launch-fixes-20261006/`. The disposable PG16 container was
stopped and removed. No slot was claimed and no unattended work remains running.

Ruff F/I checks pass for changed files; the new pure contracts and shared launch
schemas also pass E/F/I. Existing long-line diagnostics in older modules remain;
no exception-handling or type errors were waived. Formatting applied; diff checks
pass. The handoff's FastAPI 0.128 vs locked 0.141 generator-noise policy remains.

## Review and remaining work

This receipt qualifies source/unit/scratch-Postgres behavior. The deployed slot
has not received these commits; its earlier worker journey 503 remains unqualified.
SDK/MCP broken calls, scientific records/writer changes beyond launch refusal,
and the remaining pagination/streaming findings still require their separate
fix lanes. Publishing/deployment and any live candidate qualification are
separate owner-controlled actions; no release or production completion is claimed.

Review entry points: backend packages/smr/contracts/run_start/v1/hold_requirements.py,
resource_delivery.py, preflight_service.py, resource_catalog_authority.py, and
launch_service.py; SDK launch_schemas.py, request_models.py, contracts/types.py,
errors.py and transport/http.py. Original assertions are retained in
backend tests/units/test_launch_discrepancy_contracts.py and SDK
 tests/contract/test_launch_acceptance_contracts.py,
 tests/contract/test_launch_retry_contracts.py,
 tests/mcp/test_launch_schema_contracts.py. New end-to-adapter and boundary checks
are test_launch_contract_fixes.py and tests/mcp/test_launch_contracts.py.

Backend implementation commit: `d2b11ef07f727565ce69504ea106843b97a931ff`.

SDK implementation commit: `dc7e0571ee51d77a2449baf08dafb41414cb4890`. Both source candidates must be coordinated at any later release; published SDK/backend environments have not received these commits.
