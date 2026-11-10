This is a small customer-authored Harbor task with a separate verifier, plus a task-owned streaming adapter for Rhodes. The Harbor layout follows https://docs.harborframework.com/tasks/overview and https://docs.harborframework.com/tasks/configuration. Harbor grades the exported result file; the adapter applies the same pure verifier to the response from a separately isolated candidate.

Build `environment/Dockerfile` with the environment directory as context and `tests/Dockerfile` with the tests directory as context. Publish both images and use their immutable repository digests below. The candidate image contains no verifier code.

```sh
SYNTH_API_KEY=... python examples/customer_evaluation/run.py \
  --url https://api.example.com \
  --candidate-image registry.example/customer-candidate@sha256:... \
  --evaluator-image registry.example/customer-verifier@sha256:... \
  --provider modal --region us-east --maximum-charge-micros 100000 \
  --operation customer-integer-proof-1
```

The script uses only the public API through the SDK. It registers five immutable inputs, obtains a preflight quote, submits three concurrent instances, and reads the signed aggregate after usage closure. It does not need an organization ID, operator token, SSH access, or database connection. The returned signature is retained; this example checks the evidence bindings and score but does not independently verify the signer key.
