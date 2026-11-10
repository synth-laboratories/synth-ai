# `synth` command line

`synth` (also installed as `synth-ai`) is a thin layer over the typed client.
Every command accepts `--json`; the global `--base-url` selects the API.

| Step | Command | SDK record under `--json` |
| --- | --- | --- |
| Sign in | `synth login [--no-browser]` | `AuthStatus` |
| | `synth whoami` | `Account` |
| | `synth logout` | `AuthStatus` |
| Discover | `synth profiles [--research-task T]` | `RunProfileCatalog` |
| Inputs | `synth upload PATH...` | array of `File` |
| Launch | `synth run "goal" --budget 2.00 [--profile P] [--research-task T] [--file PATH ...] [--wall SECONDS (1..3600)] [--label k=v ...] [--wait [--timeout S]]` | `RunSnapshot` (final snapshot with `--wait`; human output ends with the answer text when there is one) |
| Watch | `synth status RUN` | `RunSnapshot` |
| | `synth wait RUN [--timeout S]` | `RunSnapshot` |
| | `synth logs RUN [-f] [--limit N] [--cursor C]` | `EventPage`; with `-f`, one `RunEvent` per line |
| | `synth tasks RUN [--limit N] [--cursor C]` | `RunTaskPage` |
| | `synth ls [--state S] [--limit N] [--cursor C]` | `RunPage` |
| Steer | `synth pause RUN`, `synth resume RUN` | `RunControlReceipt` |
| | `synth cancel RUN` | `RunSnapshot` |
| Collect | `synth outputs RUN [--limit N] [--cursor C]` | `OutputPage` |
| | `synth download RUN [OUTPUT] [-o DIR]` | array of `{output, path}` |
| | `synth report RUN` | `RunReport` |
| | `synth result RUN` | `RunResult` as `{run, outputs, answer_output, content, answer, objective_answer}`; human output is the answer text (a text output, or an objective answer's text) |
| Account | `synth usage` | `UsageSummary` |
| | `synth usage RUN` | `UsageRecordPage` |
| | `synth billing` | `Balance` |
| | `synth limits` | `LimitList` |

`synth run` requires the goal; it becomes the launch message. `--file` uploads
each file (prepare, transfer, finalize in one step) and attaches its reference.
The default `--timeout` for `wait` and `run --wait` is 3600 seconds.

`synth result RUN` reads a finished run with `swarms.result`: it prints the
answer (the last available `text/*` output, digest-verified) or, when there is
none, the run, state and output count. It never waits; on an unfinished run it
exits 1 with `result_unavailable`. See [errors.md](errors.md) for every code.

`synth login --no-browser` prints the authorization URL to stderr and reads the
redirect URL that the browser reached (`http://127.0.0.1:PORT/callback?...`) from
stdin. The state and issuer are verified as in the browser flow.

API key minting, rotation and revocation are not offered here; keys are managed
from a signed-in session.

## Output

Without `--json`, output is short aligned text. With `--json`, stdout carries
exactly one JSON document: the SDK record serialised with
`model_dump_json(exclude_unset=True)`. `logs -f --json` is the one stream: one
`RunEvent` document per line. Errors go to stderr: `code: detail`, or the typed
problem JSON under `--json`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Client or usage error, including transport failures |
| 2 | `wait` (or `run --wait`) reached its timeout; the run is still running |
| 3 | The run finished `failed` or `cancelled` (`wait`, `run --wait`) |
| 4 | Server refusal (typed problem from the API) |

The constants are `EXIT_*` in `synth_ai/cli.py`.

## Event checkpoint

`synth logs RUN -f` stores its `StreamCheckpoint` at
`~/.config/synth/streams/RUN.json` (mode 0600, directory 0700). Set
`SYNTH_STATE_DIR` to use another directory. The checkpoint advances only after
an event has been written to stdout, so a restarted `logs -f` resumes after the
last printed event without repeating it.
