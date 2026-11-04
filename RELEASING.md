# Releasing 0.26.0

Publication requires explicit owner GO after the production public routes are
live. Creating or pushing a release tag is part of that gated publication.
Do not create `v0.26.0` before approval.

1. Verify the backend commit and contract SHA256 in `contracts/SOURCE.toml`.
2. Run the complete CI matrix, generation checks, wheel and sdist builds,
   `twine check`, package-size checks and fresh-venv install smoke.
3. Install the exact wheel in a fresh venv with `uv pip install --no-cache`; verify
   `RunEvent.model_fields["observed_at"].annotation == str | None`.
   Run `scripts/smoke.py` against the qualified local slot, then staging; retain
   redacted statuses. The v2 runner caps two runs at $0.10 each plus one $0.10
   inference reservation. It verifies same-session replay and requires a completed
   first run. Retain separate expiry/revocation and final hold-release evidence;
   full `complete` stays false until the inference cleanup proof is recorded.
   Existing v1 receipts retain their old budget and cannot be reused by the v2
   runner. Verify the
   console uses the same contract and its sustained gateway acceptance passed.
4. After live acceptance, merge the update into dev, then promote dev to main
   per the SDK consolidated handoff (this repository does not auto-deploy). Preserve the
   `legacy/0.24.0` and `legacy/dev-20261104` history tags.
5. After owner GO and production parity, tag the qualified commit `v0.26.0`.
   The tag workflow builds qualified bytes and publishes through the protected
   `pypi` environment using trusted publishing.
6. Install the published version in a fresh environment and repeat imports,
   auth status and an authenticated read. Record the package URL and hashes.

Never force-push history or replace an existing release artifact. Rollback means
pinning a previous compatible package; a yank is not a service rollback.
