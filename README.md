# Synth Python SDK

A typed Python client for Synth Index search with citations, Swarms research
runs, account access, and hosted or local MCP. Python 3.11 or newer is required.

Swarms requires organization admission. An unavailable operation raises
`UnsupportedOperationError` with code `operation_unavailable`. Inspect
`client.account.entitlements()` for your organization's access conditions.

```sh
pip install synth-ai
pip install 'synth-ai[mcp]'
synth-ai login
```

Use browser login or set `SYNTH_API_KEY`. Stored login credentials are reloaded
on every authenticated call. `SYNTH_API_URL` overrides the default API origin,
`https://api.usesynth.ai`. Anonymous Index search needs no login or API key.

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import PublicSearchOptions

with SynthClient.anonymous() as client:
    result = client.index.search("reproducible retrieval", options=PublicSearchOptions())
    print(result.response)
```

Authenticated search has an explicit maximum charge:

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import KeyedSearchOptions
from synth_ai.types.money import Money

with SynthClient() as client:
    result = client.index.search(
        "reproducible retrieval",
        options=KeyedSearchOptions(maximum_charge=Money(coefficient=5, scale=2)),
    )
    print(result.response)
```

Create a Swarms project, launch bounded work, wait, and download its results:

```python
from synth_ai import SynthClient
from synth_ai.options import ObservationOptions
from synth_ai.sdk.swarms import Limits
from synth_ai.types.money import Money

with SynthClient() as client:
    project = client.swarms.projects.create("Research", objective="Compare retrieval methods")
    run = client.swarms.create(message="Compare retrieval methods", project_id=project.project_id,
        limits=Limits(maximum_charge=Money(coefficient=2, scale=0), wall_time_seconds=600,
                      maximum_tasks=4, maximum_parallel_tasks=2))
    result = run.wait(observation=ObservationOptions(timeout_seconds=660, idle_timeout_seconds=120,
        reconnect_deadline_seconds=30, retry_count_max=3, maximum_events=2000, poll_interval_seconds=2))
    bundle = run.download("research-results")
```

The download verifies exact repository and artifact digests before publishing a
new destination directory. Its manifest records the run reference, final commit,
artifacts, objective verdicts, and answers. Existing destinations are refused.
See [Swarms](docs/swarms.md) for input artifacts, guidance, resources, controls,
and usage. Both `SynthClient` and `AsyncSynthClient` expose the same operations;
async methods are awaited and clients support managed contexts.

Catch `SynthError` for API and transport failures. Common types are
`AuthenticationError`, `RateLimitError`, `BudgetExceededError`,
`RequestTimeoutError`, and `OutcomeUncertainError`.

**After `OutcomeUncertainError`, resolve the retained identity before deciding
whether to submit new work:** `client.resolve_mutation(error.identity)`.
`MutationCancelledError` derives from `asyncio.CancelledError` and also retains
its mutation identity. A local timeout or closing a client does not cancel a run.

Requests have total deadlines. Reads retry transient failures with bounded
jitter and honor `Retry-After`. Mutations are never automatically replayed.
Call `run.cancel()`, `run.pause()`, or `run.resume()` for remote controls.
Use a stable `MutationOptions(idempotency_key=...)` for recoverable writes.

Errors carry operation and request identifiers. Responses accept additive
fields; unknown enum values remain visible. `ClientOptions(strict_contract=True)`
requires exact schema hashes. Client options can borrow caller-owned HTTPX
clients; closing the SDK leaves a borrowed client open. File-based login requires
POSIX ownership and permissions; explicit API keys work on other platforms.

- [Documentation](https://docs.usesynth.ai)
- [Swarms guide](docs/swarms.md)
- [Changelog and migration](CHANGELOG.md)
- [Release policy](RELEASING.md)
- [Issues](https://github.com/synth-laboratories/synth-ai/issues)

## Account companion

The source candidate extends the existing `synth-ai` CLI with read-only cloud
account commands. Its version follows the SDK and MCP package:

```sh
synth-ai --version
synth-ai login
synth-ai auth status --json
synth-ai account context --json
synth-ai account plan
synth-ai account balance
synth-ai account alerts
synth-ai account usage --days 7 --json
synth-ai account logs --days 7 --limit 50 --json
synth-ai account billing
synth-ai account keys
synth-ai account apps
```

Billing, API-key lifecycle and connected-app management open the dashboard;
these commands never purchase, change subscriptions or run workloads. Use
`--no-browser` for a URL. Set `SYNTH_DASHBOARD_URL` to the dashboard origin when
using a custom API origin. API-key setup also supports the existing
`synth-ai login --with-token` stdin flow. Private-file storage and canonical
`SYNTH_API_KEY` / `SYNTH_API_URL` precedence remain unchanged.

The new account reads require a backend serving `account.v1` and explicit
account scopes for financial data. Self-identity/context inspection is available
to every valid customer credential, without workload capability discovery.
`account:read` permits limits together with billing-read; `account:billing:read` permits
plan/balance/alert settings, and usage/logs require both usage-read and billing-read. Existing
credentials do not gain these grants automatically. Fresh OAuth consent names
the account permissions. The existing `account limits` discovery command keeps
its negotiated product-policy contract.

SDK sync/async clients expose `account.context()`, `plan()`, `balance()`,
`alerts()`, `limits()`, `usage(start, end)` and `events(start, end)`. Event reads return one
bounded page and an explicit cursor. Dates must be timezone-aware; windows are
bounded to 366 days. The CLI also accepts matching `--from` / `--to` timestamps
and an optional `--product` filter for usage/logs.

JSON retains exact money coefficients/scales, original event IDs and string
sequence numbers, pending/unknown values, source coverage and observation times.
Successful reads exit 0, including explicitly partial observations; refused or
unavailable operations exit 1, and invalid command/input exits 2. Account reads
use their own observational protocol and do not claim Index/Swarms execution
manifest qualification. Native issuer, installed-package and deployed-cohort
acceptance remain required before release.
## Forge research workspaces (v0.24 candidate)

Forge workspaces organize scientific work independently of executable Swarms
projects. The selected backend must admit these operations and the credential
must carry explicit `forge:read`/`forge:write` scopes.

```python
from synth_ai import SynthClient
from synth_ai.options import MutationOptions

with SynthClient() as client:
    created = client.forge.projects.create(
        title="Craftax comparison",
        objective="Compare the two approaches using the pinned evaluation protocol",
        mutation=MutationOptions(idempotency_key="craftax-study-1"),
    )
    project = client.forge.projects.get(created.receipt.project_id)
```

Async clients expose the same methods. Retain the project ID, original mutation
identity and expected revision for retries and edits. Archive/unarchive preserves
research history; membership changes require current workspace ownership.

The MCP catalogue includes `forge_project_get`, `forge_project_create`,
`forge_project_update`, `forge_project_archive` and `forge_project_members`.
Scientific tools are `forge_record_get`, `forge_record_write` and
`forge_receipt_get`; sync and async clients expose `client.forge.records.get`,
`client.forge.records.write` and `client.forge.receipts.get`. Writes accept typed
canonical records with an expected revision and stable mutation key. Recover
the original receipt after response loss or admission expiry before another
attempt. Manual results retain unverified status.
Typed inputs are available from `synth_ai.sdk.forge`, including `Protocol`,
`Experiment`, `Trial`, `Review`, and protocol/input observation types. Register a
trial before execution; v2 state observations append immutable revisions with
the registered parameters fixed. A rerun uses a new trial and exact `parent_trial`.
Legacy records keep their original versions; native records carry explicit v2
envelopes, and execution-bound input claims need their owning verifier.
Connections with admitted Forge tools can discover the packaged `forge-research`
skill as a content-addressed MCP resource. Research runs pin that skill together
with their admitted scientific tools and method/program versions. Workspace
operations alone do not qualify a complete scientific Swarm or evaluator path.


Private Index handoff uses `client.forge.index_drafts.create(project_id,
export_revision, mutation=...)` to allocate custody for an exact frozen export.
Retain the original draft operation ID. Then call
`client.forge.index_drafts.upload(project_id, draft_operation_id, descriptor,
mutation=...)`, or the `forge_index_draft_upload` MCP tool. The author supplies
research metadata and an asset ID/role/path layout; the backend resolves frozen
native Artifact bytes, digests, licenses and full credit. Current `forge:read`,
`index:intake` and private Artifact grants are required. Transfer credentials
stay inside the native upload service.

Preserve each mutation key on retry. Recover allocation with `index_drafts.get`
and upload custody with `index_drafts.upload_get(project_id,
draft_operation_id, upload_operation_id)`. MCP exposes matching create/get and
upload/upload_get tools. `prepared`, `received` and `finalized` distinguish
reservation, received bytes and an exact native package reference. Finalized
packaging leaves the Contribution private and in draft state; validation,
scientific qualification, rights attestation and publication remain separate.
