# Decision Record: Turso-Native Storage Migration

## Status
- Proposed – awaiting sign-off from tracing, infrastructure, and data engineering leads.
- Owners – tracing platform team (primary), infra reliability (secondary).
- Date – 2024-10-16.

## Context
- The v3 tracing stack historically relied on `AsyncSQLTraceManager`, an async SQLAlchemy adapter targeting SQLite (`sqlite+aiosqlite`) locally and libsql via sqld when available. We have since replaced it with a libsql-native implementation.
- Operationally we toggle between direct-file SQLite and the sqld-managed libsql endpoint using configuration flags, but both paths still flow through SQLAlchemy.
- We recently landed a real-trace regression suite plus fixture generation tooling (`scripts/build_trace_fixtures.py`, `tests/artifacts/traces/*`) to make backend swaps testable.
- Product direction prefers leaning into Turso/libsql primitives and retiring the SQLAlchemy dependency to simplify deployment, reduce ORM overhead, and align with Turso-hosted infrastructure.

## Drivers
- **Consistency** – eliminate divergent behaviour between SQLAlchemy+SQLite and libsql by standardising on a single client stack.
- **Operational simplicity** – reduce sqld orchestration quirks and simplify local/CI bootstrapping by using libsql directly.
- **Performance** – avoid SQLAlchemy translation overhead on hot paths (SessionTracer incremental inserts, replica sync) and leverage libsql-specific optimisations.
- **Long-term support** – Turso-native tooling (auth, replication, hosted backups) lines up with roadmap requirements; SQLAlchemy is increasingly incidental complexity.

## Goals
1. Provide a libsql-backed manager that maintains the existing storage contract (`initialize`, `ensure_*`, `insert_*`, query/reporting helpers).
2. Ensure parity across key workflows validated by the real-trace regression suite and fixture manifests.
3. Preserve sqld-powered replication semantics so local/offline flows keep working.
4. Introduce CI gating that runs regression suites against both the legacy SQLAlchemy path and the proposed Turso-native path.
5. Document operator workflows (fixture regeneration, rollback, troubleshooting) before flipping defaults.

## Non-Goals
- Schema redesign or table consolidation beyond compatibility fixes required by libsql.
- Changing higher-level tracing APIs, CLI UX, or exporter outputs.
- Altering how remote Turso sync and auth are configured (environment variable contract remains the same).
- Migrating unrelated storage consumers outside `synth_ai/tracing_v3`.

## Decision
- Build a new libsql-native manager (`synth_ai/tracing_v3/turso/native_manager.py`) that mirrors the public interface of the SQLAlchemy implementation.
- Gate selection of the implementation via configuration (`STORAGE_BACKEND` or `TURSO_NATIVE=1`) so we can run both backends in parallel during validation.
- Keep the SQLAlchemy adapter temporarily to serve as a fast rollback lever until post-migration verification is complete, after which we will remove it.

## Implementation Plan (summary)
1. **Adapter spike** – implement the libsql manager, smoke test via the existing unit suite and the real-trace replay tests.
2. **Interface hardening** – abstract `SessionTracer`, incremental insert helpers, and ReplicaSync to remove direct SQLAlchemy references.
3. **Dual-path CI** – add matrix jobs executing `make verify-trace-fixtures` and the real-trace pytest bundle for both backends.
4. **Data tooling** – provide migration/rollback scripts (`scripts/migrate_traces_to_turso.py`) and capture verification commands.
5. **Rollout** – canary with `TURSO_NATIVE=1`, monitor, then flip defaults and clean up legacy code.

Detailed steps and owners live in `turso_sqlite.txt` under “Step-by-step migration checklist”.

## Fallback Plan
- Retain the SQLAlchemy manager guarded by `LEGACY_SQL_BACKEND=1` until after the production cutover warms for two releases.
- Maintain the golden fixture baseline (`fixtures/baseline.sqlite3`) so we can diff outputs between implementations.
- If issues surface in canary or post-rollout, toggle back to SQLAlchemy, rerun integrity checks, and file regression bugs referencing fixture diffs.

## Impact & Risks
- **Testing load** – running dual-path regression suites increases CI wall time; we will stage gating in a new job to avoid blocking unrelated work.
- **Operational** – libsql client errors differ from SQLAlchemy’s; observability dashboards and runbooks must be updated.
- **Compatibility** – libsql may have slight behavioural differences (busy timeouts, WAL pragmas); we mitigate by codifying behaviours in the new adapter and covering them with fixtures.
- **Data integrity** – migration scripts must be idempotent and reversible; we will only modify data after automated integrity checks pass.

## Open Questions
1. Do we require hosted Turso features (branching, point-in-time recovery) on day one, or can they remain optional?
2. Should ReplicaSync continue to manage file-based replicas, or can it target Turso directly once the backend is native?
3. What timeline do downstream consumers (analytics, exporters) need for testing against the new backend?

## Next Actions
- Secure approvals on this record.
- Execute the pre-flight checklist: regenerate fixtures, snapshot the baseline database, capture schema fingerprints, and open the migration tracking issue with assigned owners/dates.
