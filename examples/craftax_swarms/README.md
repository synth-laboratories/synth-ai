# Craftax through Swarms

Reliability tier: 2. This example launches one bounded run through the public
`synth-ai` SDK, retains its admission identity, and records lifecycle metadata.
The run command never registers a project, pool, task, policy, evaluator or dataset. The
swarm receives an empty pool selection and must create its own resources through
its granted tools. It never calls an internal service or writes a database.

Use the exact installed release wheel in a separate environment. Copy only the `examples` directory into a separate runner directory and run
the modules there. Do not run them from a source root that also contains
`synth_ai`, which would shadow the installed wheel. Check
`python -c 'import synth_ai; print(synth_ai.__version__, synth_ai.__file__)'`.

Prepare the pinned public benchmark through the public SDK:

```sh
python -m examples.craftax_swarms.prepare \
  --base-url http://127.0.0.1:8000 --provider docker --region local \
  --api-key-env SYNTH_API_KEY --output /absolute/path/craftax-inputs
```

This validates locally by default. Add `--execute` to register immutable task,
pool configuration, evaluator and dataset revisions. It starts no compute and
creates no execution pool or project. For staging and production select the
explicit HTTPS origin, `--provider modal --region us`, and that environment's
API key. Reuse prepared templates only in their original environment. The
registration keys are content-bound, so an interrupted preparation can be
repeated without new input identities.

The preparation writes `template.json` and `task.txt`. The task requires the
planner to create a fresh project and call `evaluation_pool_register` to clone
the granted template into a fresh run-owned pool/evaluator. It registers policies
through agent tools. The public field is the first 100 instances of Craftax
public-development-v3, 250 steps each; screening uses the first eight. The
baseline is the repository's pinned noop starter. This is public-development
qualification, not a held-out benchmark or a numerical comparison with older
staging experiments that used another baseline. Exact source and image pins
are in `sources.json`; qualify registry platforms and serving behavior before
use. A manifest pin alone does not attest a successful runtime.

An explicit cohort file binds the run to intended release inputs. It is a
reference, not an installed service identity attestation. Verify deployed
revisions separately before launching.

```sh
python -m examples.craftax_swarms.run \
  --base-url http://127.0.0.1:8000 \
  --task-file /absolute/path/craftax-inputs/task.txt \
  --template-file /absolute/path/craftax-inputs/template.json \
  --cohort-file /absolute/path/cohort.json \
  --budget 14.00 --evaluation-micros 8650000 \
  --wall-seconds 2400 --timeout 3600 \
  --api-key-env SYNTH_API_KEY \
  --receipt /absolute/path/local-craftax.json
```

Without `--execute`, the command validates locally and makes no network calls.
The example cap is $14, including model and evaluation compute; the evaluation
allowance is part of that cap, not an additional charge. Use an explicitly funded
test org and record the reservation in the release ledger before `--execute`.
The operator must bound aggregate spend across local, staging, production and
retries; the per-receipt cap does not coordinate different processes or runs.

Use a new receipt for each environment and its own API-key environment variable.
Staging and production require HTTPS. Reuse exactly the same flags and task file
with `--execute --resume` after interruption. Resume attaches to a known run or
replays an unknown admission with its original idempotency key. It refuses an
unknown admission older than 23 hours: reconcile that operation before the
server's 24-hour replay window expires. Never delete a receipt to retry. The
file lock prevents simultaneous writers to one receipt.

Timeout does not cancel the run. Resume observation; use public `synth cancel`
only when cancellation is intended, then resume until accounting and cleanup
finish. The harness never cancels a run merely because observation failed.

The receipt contains run id, state, closure/accounting/cleanup flags, final
charge, failed model call count, elapsed execution time and evaluation ids.
It contains no prompt, model answer, tool payload, credential, or raw error body.
`lifecycle_clean` requires completed execution, all three finality flags, observed
empty problems, a final charge, and zero failed model calls.

`benchmark_pair_verified` and `installed_cohort_verified` deliberately remain
false. Full release acceptance additionally requires a trusted baseline/candidate
pair with matching evaluator/dataset/instance set, signed evidence and policy
identity, expected episodes and zero policy errors; per-service live revisions
and health; and provider resource absence. Preserve those separate receipts and
join them by this run id. A successful exit proves the lifecycle check only.

Verify the pair after lifecycle closure:

```sh
python -m examples.craftax_swarms.verify \
  --base-url http://127.0.0.1:8000 --api-key-env SYNTH_API_KEY \
  --receipt /absolute/path/local-craftax.json \
  --template-file /absolute/path/craftax-inputs/template.json \
  --output /absolute/path/local-pair.json
```

This read-only checker uses the public run report, evaluator registry metadata
and compact rollout summaries. It never requests policy source, answers, raw
rollout reports or tool traces. It requires the exact source-bound baseline
policy revision, a distinct candidate, the same evaluator and dataset, a fresh
pool with the granted evaluator configuration, exactly episode-0 through
episode-99, no policy errors/timeouts, and matching attestation references.
It records the mean scores supplied by the trusted service; a non-improving
candidate remains an honestly reported result. It does not verify Ed25519
signatures locally, so `signatures_verified_locally` is false. Public SDK
methods do not expose per-service deployed SHA/health or provider resource
absence; the release operator must retain those receipts separately.
