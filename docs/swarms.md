# Swarms

`SynthClient().swarms` exposes launch, run, get, wait, cancel, list, events,
iter_events, follow, outputs, output, output_content, create_project, project,
projects and families. `AsyncSynthClient` exposes the same methods with `await`;
its iter_events and follow methods are async iterators.

Supply a project ID or `SYNTH_PROJECT_ID`. Budget is a decimal USD string,
for example `budget="0.10"`. Explicit launch limits default to 3600 seconds,
100 tasks and four concurrent tasks. Profile overrides pass through the generated
schema and their requested/resolved values are returned by the core.

Each mutation uses an Idempotency-Key. Supply `idempotency_key` to preserve the
identity across process retries. Bounded 429/503 retries keep the same key and
body. An unknown outcome does not mean the effect failed; inspect the run and
retry with the same identity. Replay lasts 24 hours after completed admission.

Pages expose `items`, `has_more` and an optional `next_cursor`. Pass the cursor
verbatim. `follow(run_id, last_event_id="12")` resumes SSE after that sequence.
Output bytes require the revision and SHA256 from the output reference.

CLI examples:

```sh
synth swarms create-project --name Example
synth swarms launch 'Compare retrieval methods' --project-id "$SYNTH_PROJECT_ID" --budget 0.10
synth swarms run --task ./task.md --project-id "$SYNTH_PROJECT_ID" --budget 0.10 --program swarms.general@1
synth swarms events RUN_ID --limit 20
synth swarms attach RUN_ID --last-event-id 12
synth billing balance
synth billing checkout --json '{"credit":{"amount":"20.00","currency":"USD"}}'
```

Task directories must contain one task.md, TASK.md or task.txt file, and any other
file in the directory is refused. Task text is the launch message. To send files with
a run, upload them with `synth swarms upload` or `upload_file` and pass the references
in `inputs`; see the README.

## Execution methods

Three methods are available. Status Quo is the default program and needs no setup.

| Method | CLI | Python `method=` |
| --- | --- | --- |
| Status Quo | `--method status-quo` | omit `method` |
| Fork & Flush | `--method fork-flush` | `{"release": "research.fork_flush@1"}` |
| Forgetting | `--method forgetting` | `{"release": "research.forgetting@1"}` |

Fork & Flush and Forgetting need a published pairing for the project, which pins the
evaluator, baseline, context and actor profile. A launch on an unpaired project is
refused. Pass `overrides` to change a method's published settings, for example
`{"release": "research.fork_flush@1", "overrides": {"branches": 2}}`, and read
`run.method` for the release and execution pins retained with the run.

`run.selection` reports what a research run selected. Only `selected` has an incumbent
and a downloadable `selected-snapshot.tar.gz`, a gzipped tar whose SHA-256 equals
`selection.incumbent.artifact_digest`. `candidate_files` lists the files the score
covers. Other statuses are `not_applicable` (Status Quo), `pending`, `no_selection`
and `unavailable`; a `failed` run can still read `selected`, so check both. List pages
omit `selection`, so read a single run.

## Providers and owner resources

Choose a provider with `profile_overrides={"launch": {"preset":
"hosted.openrouter_zdr_us", "effort": "low"}}`. Role-specific choices belong
under `profile_overrides["roles"]`. Hosted presets are
`hosted.openrouter_zdr_us` (same-model OpenAI fallback) and
`hosted.openai_direct`. `test.codex_pool` is restricted to local slots and
staging; production returns `provider_preset_not_allowed` (403).
All three currently serve `gpt-6-luna`.

Pass owner resource intents in `resources`, for example:

```python
resources = [
    {
        "plugin": "rhodes.pool",
        "roles": ["worker"],
        "limits": {"spend_micros": 100000, "rollouts": 1},
        "selection": {"pool_ids": ["my-pool"]},
    }
]
```

The core verifies ownership and freezes grants. `role_limits` can narrow a
resource's run limits; it cannot widen them. The pinned launch contract has no top-level `preset` or `max_output_tokens`
field.

`synth swarms watch RUN_ID --timeout 600` emits JSON events followed by the
terminal run snapshot. A failed run exits 1; completed and cancelled runs exit 0.
The timeout bounds both streaming and final polling, and does not cancel the run.
