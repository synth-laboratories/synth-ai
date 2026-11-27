# SDK Stateful Hypothesis PBT + `everystate conform` (SYN-6050)

Hypothesis `RuleBasedStateMachine` suites exercising the typed `SynthClient` SDK surface
in both deterministic `httpx.MockTransport` mode and live `--base-url` local-slot mode:

- `test_sdk_run_control_pbt.py`: run launch, idempotent replay vs mutated-payload `409 ConflictError`,
  `ETag` + `304`, pause/resume/cancel, monotonic `limits_revision`, messages, and absorbing terminal finality.
- `test_sdk_hosted_options_pbt.py`: randomized launch-choice queries verifying zero upstream vendor leakage
  and exact agreement across `launch_choices -> preflight_request -> launch_request`.
- `test_sdk_billing_wallet_pbt.py`: wallet `available == ledger - held`, idempotent Stripe webhooks and promos,
  auto-reload caps, plan switches, and Free (`$5/day`) vs PAYG (`$100/day`, `$500/week`) cap transitions.

## Running

```bash
uv run --locked --extra mcp pytest -v tests/pbt --pbt-examples=50 --pbt-steps=20 --trace-out=.pytest_tmp/sdk-pbt-traces
```

Live local-slot mode:

```bash
uv run --locked --extra mcp pytest -v tests/pbt --base-url "$SLOT_CORE_API_URL" --api-key "$SYNTH_API_KEY"
```
