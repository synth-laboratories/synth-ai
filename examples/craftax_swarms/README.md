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

The public evaluation registration and rollout routes have been removed. The
former prepare/verify helpers depended on those routes and are no longer shipped.
Supply a task and template receipt from the current release owner's authorized
preparation path. This run harness does not prepare or verify benchmark inputs.

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
