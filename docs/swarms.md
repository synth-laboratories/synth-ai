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

Task directories must contain one task.md, TASK.md or task.txt file; other files
are refused because this API has no upload endpoint. Task text is the launch message.
