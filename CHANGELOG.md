# Changelog

## Unreleased

Changes here describe the new Swarms product. Historical releases remain in Git history.

- Contract 1.35.0: `account.redeem_promo`, `synth promo redeem` and the
  `account_promo_redeem` MCP tool redeem a promotional code (SYN-5752). A trial
  campaign's redemption carries `trial` (plan and fixed start/end).

## 0.41.0rc1 — source candidate

This checkout does not attest publication or installed acceptance. The exact Core
revision and digest are recorded in [contracts/SOURCE.toml](contracts/SOURCE.toml).

- Typed synchronous and asynchronous Swarms clients, CLI and MCP tools share the
  public Core contract. Files and a message launch a run; result reads preserve
  its answer, exact charge, output references and canonical attempt tree.
- Named `local`, `staging` and `prod` profiles select the target explicitly. The
  customer default is production; credentials remain bound to their host.
- Retryable 429 and 503 responses honor `Retry-After` with bounded jittered
  retries. Nonretryable refusals surface immediately. Queued runs expose their
  position, estimate and expiry while waiting for admission.
- Special Free requests, self-serve billing, run-limit updates and per-role
  launch choices use the same typed public API.
- Contract 1.32.0 adds queued run standing. Source and fixture qualification do
  not establish acceptance of a deployed cohort.
