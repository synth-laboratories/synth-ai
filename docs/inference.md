# Inference sessions

`client.inference.session(project_id=..., preset="hosted.openrouter_zdr_us",
budget="2.00", ttl=3600, idempotency_key="trial-001")` returns a project-scoped
session. The synchronous and asynchronous clients share this API.

```sh
eval "$(synth inference session --project-id "$SYNTH_PROJECT_ID" --budget 2 --ttl 3600 --shell)"
```

The command prints shell-quoted `SYNTH_INFERENCE_BASE_URL` and
`SYNTH_INFERENCE_SESSION_TOKEN` exports. Treat the token as a credential;
keep command output out of logs and receipts. Provider credentials remain in core.

Budget is a positive USD decimal string, at most $50, in whole microdollars.
TTL is 60 through 86,400 seconds. Reuse the idempotency key for retries of the
same admission; a new trial needs a new key and budget. The gateway ledger
accounts for the session's shared aggregate allowance.

Presets: `hosted.openrouter_zdr_us` (same-model OpenAI fallback),
`hosted.openai_direct`, and `test.codex_pool`. The pool is local/staging only;
production refuses it with `provider_preset_not_allowed`. Its token allowance
is bounded and non-chargeable. The catalogue serves `gpt-6-luna` only.
Use the returned `max_output_tokens` and the session `/models` catalogue for
output limits. The public contract currently exposes creation only; there is
no public session revoke endpoint. Expiry and owner cleanup are core duties.

See [errors](errors.md) for typed admission refusals.
