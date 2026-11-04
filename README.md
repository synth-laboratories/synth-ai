# Synth Python SDK

A typed Python client, CLI and MCP server for Synth Swarms. Python 3.11 or newer
is required.

```sh
pip install synth-ai
pip install 'synth-ai[mcp]'
synth login
```

Use browser login or set `SYNTH_API_KEY`. `SYNTH_API_BASE_URL` overrides the
default API origin, `https://api.usesynth.ai`. `SYNTH_PROJECT_ID` sets the
default Swarms project.

## Quickstart

```python
from synth_ai import SynthClient

with SynthClient() as client:
    result = client.swarms.run("Compare retrieval methods for legal search", budget="5.00")
    print(result.run.state)
    for output in result.outputs:
        print(output.output_id, output.filename)
```

`swarms.run` launches a run, waits for it to close and returns its outputs.
`budget` is the maximum charge in USD.

The same steps one at a time:

```python
run = client.swarms.launch("Compare retrieval methods", budget="5.00")
client.swarms.events(run.run_id)
closed = client.swarms.wait(run.run_id, timeout_seconds=3600)
outputs = client.swarms.outputs(run.run_id)
record = client.swarms.output(run.run_id, outputs.items[0].output_id)
client.swarms.list(limit=20)
client.swarms.cancel(run.run_id)
```

## Account and billing

```python
client.account.get()       # org, principal, limits
client.account.keys()      # API keys (session credential)
client.billing.balance()   # integer microcents
client.billing.plans()
client.billing.checkout(plan_id="standard.monthly")
client.billing.checkout(credit_cents=2000)
```

## CLI

```sh
synth swarms run "Compare retrieval methods" --budget 5.00
synth swarms launch|get|wait|events|cancel|outputs|output|list
synth account
synth billing balance
synth billing plans
```

## Errors

Every refusal is `{"error": {code, message, service, stage, upstream_status, cause}}`
and raises a subclass of `synth_ai.public.errors.SwarmsAPIError`:
`InvalidRequestError`, `UnauthenticatedError`, `PaymentRequiredError`,
`ForbiddenError`, `NotFoundError`, `ConflictError`, `RateLimitedError`,
`UpstreamUnavailableError`, `ServerError`; Swarms refusals also carry
`outcome` (`not_accepted` or `unknown`). `swarms.run` raises `RunFailedError`
when a run fails.

## MCP

`synth-ai-mcp` serves the Swarms and account tools locally; the hosted MCP at
`mcp.usesynth.ai` serves the same catalog.

- [Documentation](https://docs.usesynth.ai)
- [Swarms](docs/swarms.md)
- [MCP](docs/mcp.md)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/synth-laboratories/synth-ai/issues)
