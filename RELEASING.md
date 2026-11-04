# Releasing 0.26.0

Publication requires explicit owner GO after the production public routes are
live. Creating or pushing a release tag is part of that gated publication.
Do not create `v0.26.0` before approval.

1. Verify the backend commit and contract SHA256 in `contracts/SOURCE.toml`.
2. Run the complete CI matrix, generation checks, wheel and sdist builds,
   `twine check`, package-size checks and fresh-venv install smoke.
3. Run `scripts/smoke.py` against staging; retain redacted statuses. Verify the
   console uses the same contract and its sustained gateway acceptance passed.
4. Merge the clean tree into dev, then main by ordinary merges. Preserve the
   `legacy/0.24.0` and `legacy/dev-20261104` history tags.
5. After owner GO and production parity, tag the qualified commit `v0.26.0`.
   The tag workflow builds qualified bytes and publishes through the protected
   `pypi` environment using trusted publishing.
6. Install the published version in a fresh environment and repeat imports,
   auth status and an authenticated read. Record the package URL and hashes.

Never force-push history or replace an existing release artifact. Rollback means
pinning a previous compatible package; a yank is not a service rollback.
