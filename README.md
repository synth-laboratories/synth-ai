# synth-ai

Typed Python 3.11+ client for the Synth public API v1. This checkout declares SDK 0.41.0; the exact Core contract pin is recorded in [contracts/SOURCE.toml](https://github.com/synth-laboratories/synth-ai/blob/main/contracts/SOURCE.toml). A source version does not attest PyPI publication or the deployed service revision.

Only the latest release is supported; upgrade with `pip install -U synth-ai`.

Start with the [quickstart](https://github.com/synth-laboratories/synth-ai/blob/main/docs/quickstart.md) (SDK, CLI and MCP) and the
[error guide](https://github.com/synth-laboratories/synth-ai/blob/main/docs/errors.md).

1. Install the qualified package: `pip install synth-ai`.
2. Sign in: `synth login`.
3. Launch and wait for a run:

```python
from synth_ai import SynthClient

with SynthClient() as client:
    run = client.swarms.run("Compare retrieval methods", budget="2.00")
    print(run.run_id, run.state)
    result = client.swarms.result(run)  # the final answer, its source and the charge
    print(result.answer)
```

Set `SYNTH_API_KEY` and optionally `SYNTH_API_BASE_URL`.
`AsyncSynthClient` provides the same methods with `await`.

`pip install synth-ai[mcp]` adds the `synth-ai-mcp` stdio server.
The `synth` command line is documented in [docs/cli.md](https://github.com/synth-laboratories/synth-ai/blob/main/docs/cli.md).

Upload a task file and include its immutable reference at launch:

```python
from synth_ai.models import InputFile, RunInputs

with SynthClient() as client:
    file = client.swarms.upload_file("task.md")
    run = client.swarms.run(
        "Follow the uploaded task", budget="2.00",
        inputs=RunInputs(files=[InputFile(
            file_id=file.file_id,
            revision_id=file.revision_id,
            digest_sha256=file.digest_sha256,
        )]),
    )
```

Uploads accept binary files with bounded ASCII media types, at most 16 MiB per
file and 64 MiB per launch, with at most 10,000 references. `upload_text` and MCP `swarms_upload_file`
accept up to 32 KiB inline. `synth upload task.md` uses
the full upload protocol. Pool resources use the public selector `pool`.
`upload_directory` preserves relative names through that same protocol; see
[directory uploads](https://github.com/synth-laboratories/synth-ai/blob/main/docs/swarms.md#directory-uploads) for snapshot and recovery semantics.
`update_labels(run_id, labels, expected_revision=...)` refuses stale revisions.


Choose a published run profile. Omit `profile` to let Core use its default.

```python
with SynthClient() as client:
    print(client.swarms.profiles())
    run = client.swarms.run("Solve this task", budget="2.00", profile="swarms-v2")
```

```sh
synth profiles
synth run "Solve this task" --budget 2.00 --profile swarms-v2 --wait
synth profiles --research-task published-task
synth logs RUN_ID -f
synth download RUN_ID -o results/
```

The catalog reports each profile's description, revision, supported routing and
caller-specific availability. Research profiles require a published task and
its approval. Every run retains its exact resolved profile in `run.profile`.
New configurations are published profiles; arbitrary actor overrides are not
part of the public API.

See [Swarms](https://github.com/synth-laboratories/synth-ai/blob/main/docs/swarms.md) for routing, lifecycle and transfer details.
Version 0.29.0 replaces the public `program`, `method` and `profile_overrides`
launch fields with `profile` and, for research profiles, `research_task`.

Since 0.31.0, the SDK uses strict, frozen records, named nested models, and enum values.
Use `CompletionMilestone.FINISHED` for a fully settled wait, `run.finished` to
inspect it, and `run.known_state` when a known state enum is useful. Unknown server
states remain available as `run.state`. See [typing and protocol standards](https://github.com/synth-laboratories/synth-ai/blob/main/docs/protocols.md)
for migration details and the `ty`/Ruff checks.

To check the documented SDK, CLI and stdio MCP examples locally, use a Python
environment with the SDK's development dependencies and MCP extra:

```sh
python scripts/check_quickstart_examples.py --wheel PATH --wheel-sha256 SHA256 --output RECEIPT.json
```

Use the SHA-256 from the selected wheel's qualification receipt. The command
loads the wheel directly, checks its generated contract against this checkout,
and runs the examples and error checks against a loopback fixture.

For release qualification, `python scripts/quickstart_smoke.py --execute --fixture-org ORG_ID --cohort-file COHORT.json --output PATH`
exercises SDK, CLI and stdio MCP against an explicitly configured API. It has
three $4 run caps ($12 aggregate), retains mutation and cancellation identities,
and requires `--resume` with the same receipt after an interruption. Add
`--cancel-outstanding` to reconcile known runs without admitting new ones. A
launch with an unknown run ID must be reconciled using its retained key. Use a
dedicated test organization; uploaded fixtures require separate disposition.
The cohort file's digest binds the receipt but does not attest what is deployed.
Record the spend reservation and actual charges in the release ledger before and
after use. This opt-in smoke does not establish hosted MCP OAuth, provider-invoice
finality, load or capacity acceptance, and must not run automatically in CI without
shared budget admission.

Runnable workflows are in [examples](https://github.com/synth-laboratories/synth-ai/blob/main/examples/README.md); [protocol details](https://github.com/synth-laboratories/synth-ai/blob/main/docs/protocols.md) define compatibility, recovery, pagination and diagnostic guarantees.
