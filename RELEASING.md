# Releasing the SDK

Publication requires explicit owner GO after the production public routes are
live. Creating or pushing a release tag is part of that gated publication.
Use the version in `pyproject.toml`; do not create its release tag before approval.

1. Verify the backend commit and contract SHA256 in `contracts/SOURCE.toml`.
2. Run the complete CI matrix, generation checks, wheel and sdist builds,
   `twine check`, package-size checks and fresh-venv install smoke.
3. Install the exact wheel in a fresh venv with `uv pip install --no-cache`; verify
   `RunEvent.model_fields["observed_at"].annotation == str | None`.
   With authorized credentials and spend, run `scripts/smoke.py --execute --output
   <receipt.json>` against the qualified local slot, then staging. The v4 Swarms
   smoke caps two runs at $0.10 each ($0.20 aggregate). It requires successful
   execution and digest-verified output from the first run, a cancellation request
   for the second, and final accounting and cleanup for both. Retain receipts;
   resume only a v4 receipt against its original API with `--resume`. Resume reuses
   the retained run IDs and launch/cancel idempotency keys. Old receipts are
   preserved and refused; they must not be converted to v4 to start new runs.
   `smoke_complete` records only this Swarms journey. Full public API, hosted MCP,
   console contract parity and sustained gateway acceptance require their own
   qualification evidence before release.
4. After live acceptance, merge the update into dev, then promote dev to main
   per the SDK consolidated handoff (this repository does not auto-deploy). Preserve
   the `legacy/0.24.0` and `legacy/dev-20261104` history tags.
5. After owner GO and production parity, tag the qualified commit with `v` followed
   by the qualified package version. The tag workflow builds qualified bytes and
   publishes through the protected `pypi` environment using trusted publishing.
6. Install the published version in a fresh environment and repeat imports,
   auth status and an authenticated read. Record the package URL and hashes.

Never force-push history or replace an existing release artifact. Rollback means
pinning a previous compatible package; a yank is not a service rollback.
