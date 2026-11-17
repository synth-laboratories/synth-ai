# J1 customer drivers (Tier 1 tooling)

SYN-5692. SDKDriver uses public SDK methods; CLIDriver invokes the actual CLI
subprocess through its explicit local environment profile. Operator credentials,
cache namespaces and the private qualification route belong exclusively to the
W06 transport adapter. They are never public launch fields or CLI arguments.

Supported source: upload, optimizer=auto launch, milestone wait, run/report/result
reads, paginated output listing and pinned downloads. LaunchJournal fsyncs an
exclusive intent before the POST and retains the accepted run before inspecting
its optimizer receipt. Existing intent refuses relaunch, including after an
unknown outcome. Read/reconcile an accepted run by its retained ID instead.

An observation compares identities and exact customer charges, verifies output
pins, reads result.md when published, and preserves unavailable finality/charge/
attempt-tree/result.md as named missing evidence. Available public attempt trees are checked for run identity, exact organizing
references, parent revisions/digests, duplicate trials and cycles. Unavailable
projections remain missing evidence. This does not prove retention at Forge
or assert supplier forwarding is zero.
Gate reduction requires W06 grant/counter/closure evidence separately.

Read-only invocation from this checkout, using the manager-leased local replay
adapter origin and authorized customer key in the environment:

```
python -m scripts.acceptance --driver sdk --run-id RUN --receipt-dir OWNED_NEW_DIR
python -m scripts.acceptance --driver cli --run-id RUN --receipt-dir OWNED_NEW_DIR
```

Required environment names: SYNTH_API_KEY, SYNTH_LOCAL_API_BASE_URL. No values go
in arguments/logs. The entrypoint never launches. It currently exits 2 with an
Inconclusive receipt, preserves raw public-model evidence with private file
permissions, and creates no launch-accepted artifact for an observation.

T0 tests use real public OpenAPI models and local fixture HTTP only. CLI tests
spawn the real CLI; they are not installed slot, replay or provider proof.

### Mid-run live limits (SYN-5663)

`live_limits.raise_live_limits` takes an authorized `SynthClient`, a running run
ID, four explicit `LimitTargets`, an owned mutation prefix and a fresh receipt
directory. Invoke it from the cached journey before waiting for completion. All
targets must exceed the original caps. Each field changes in its own revision;
the next limits read must show that revision and preserve every other cap.
Tokens use `requested_cap` when their effective cap is intentionally null.

The scenario fsyncs each intent before mutation and retains the response and
next view. An existing directory refuses execution: reconcile an uncertain
mutation with its saved idempotency key instead of creating a fresh retry.
The caller retains cohort/cache-grant identity and the provider-spend gate;
this scenario's receipt proves only the four limit updates and budget views.

CLIDriver accepts an optional owned Python `entrypoint` path for operator replay
composition. The real child invokes `synth_ai.cli.main(argv, http_client=http)`
through the public injection seam; the child owns closing the borrowed client.
W06 LocalReplayTransport grant/credential/binding custody stays in that harness,
with no public cache fields or operator credentials in argv. The default remains
`python -m synth_ai.cli`. Injected-child tests prove source composition only.

## MCP drivers (SYN-5693)

`MCPDriver` runs the same journey through MCP `tools/call` only (catalog names in
`contracts/mcp-tools.json`: `swarms_upload_file`, `swarms_run_launch`,
`swarms_run_get`, `swarms_run_wait`, `swarms_run_report`, `swarms_run_result`,
`swarms_run_outputs`, `swarms_run_output_content`, `account_usage_for_run`).
It never calls SDK client methods; launch bodies use public generated request models. Two transports share
one blocking `SessionToolCaller`:

- `mcp_stdio`: spawns `<venv>/bin/synth-ai-mcp --env local`. The stdio server
  has no client-injection seam, so origin and key reach it via the child
  environment only (SYNTH_LOCAL_API_BASE_URL, SYNTH_API_KEY).
- `mcp_hosted`: streamable HTTP to `--mcp-url` with `Authorization: Bearer`.
  The entrypoint reads an opaque bearer from SYNTH_MCP_ACCESS_TOKEN (a harness
  name introduced here). Core mcpoauth authenticates it via `s.Auth`; whether an
  API key or only an MCP OAuth access token is accepted is not verified yet.
  `mcp` is an optional extra, imported only for the MCP drivers.

```
python -m scripts.acceptance --driver mcp_stdio --run-id RUN --receipt-dir OWNED_NEW_DIR
python -m scripts.acceptance --driver mcp_hosted --mcp-url http://127.0.0.1:PORT/mcp --run-id RUN --receipt-dir OWNED_NEW_DIR
```

The output-content tool returns the canonical UTF-8 or base64 envelope for at most 256 KiB
of exact bytes. The driver sends the pinned revision and verifies the declared size, digest and decoded bytes before
creating an exclusive evidence file. Larger outputs require the REST download
path and cannot qualify MCP parity through this tool. Tool-list gaps,
isError results (stdio Problem, hosted `{"error","data":{"response"}}`),
unstructured errors, wrong result models and transport failures are typed
`AdapterError` refusals whose cause never includes response bodies.

Every entrypoint run also writes `parity_view.json`
(`synth.j1.parity_view.v1`): run id, state, customer charge, finality flags and
sorted canonical trial ancestry. `attempt_id` is the public trial record ID and
`parent_id` is its parent trial record ID; internal worker IDs are not inferred.
`attempts: null` means unavailable; an authoritative empty tree is `[]`. synth-dev
`local_dev/release_tests/parity.py` loads one per driver for the same-run diff.

## J2 money, J3 access and J-refusals (SYN-5695)

synth-dev `local_dev/release_tests/journeys.py` defines the journeys as data:
step order, drivers, expected codes, preconditions and named product gaps. This
checkout executes one step on one surface (`surfaces.py`: public SDK methods,
the real `synth --json` child, or MCP `tools/call` by catalog operationId) and
returns a typed step receipt. The synth-dev orchestrator invokes it per driver:

```
python -m scripts.acceptance.journey_main --capabilities
python -m scripts.acceptance.journey_main --driver sdk --steps STEP[,STEP] --receipt-dir NEW_DIR \
  [--run-id RUN] [--limit-run-id RUN] [--queued-journal DIR] [--foreign-ids FILE] [--mcp-url URL]
```

Environment names (values never printed): SYNTH_API_KEY and
SYNTH_LOCAL_API_BASE_URL (loopback only); SYNTH_JOURNEY_SESSION_TOKEN (keys,
PKCE, MCP consent: the contract makes those routes `clerkSession`-only),
SYNTH_JOURNEY_MEMBER_API_KEY, SYNTH_JOURNEY_EXPIRED_API_KEY,
SYNTH_JOURNEY_FOREIGN_API_KEY (org B for the 404 matrix) and, for mcp_hosted,
SYNTH_MCP_ACCESS_TOKEN. An absent name makes its steps Inconclusive.

Steps never launch an uncached run. `queued_launch` observes a run that the
cached harness launched (its launch journal) while capacity was saturated.
Refusal probes send launches Core must refuse, capped at $0.50; an unexpected
acceptance is cancelled at once and fails. `raise_limit` on the SDK uses
`live_limits.raise_live_limits`; CLI and MCP raise the spend cap once and read
the next budget view. The cross-org matrix covers every ID-bearing contract
operation; operations a surface cannot express stay uncovered (Inconclusive).

Known client findings, retained as failing/Inconclusive steps rather than stubs:
the SDK refuses a contract-invalid launch (oversize message, unknown topology)
with a raw `pydantic.ValidationError` that has no public code, and `synth run`
accepts the goal only as argv, so a message over the 1 MiB contract limit cannot
be sent from the CLI (E2BIG). `tests/acceptance/test_journeys.py` pins both.
