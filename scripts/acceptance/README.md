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
attempt-tree/result.md as named missing evidence. This source does not implement
W08's still-pending attempt-tree projection or assert supplier forwarding is zero.
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
