# Releasing the SDK

Stable publication requires explicit owner GO after the production public routes
are live. A release-authorized prerelease may be published after local qualification
to qualify staging with the published package; this never authorizes stable
publication or proves production parity. Creating or pushing a release tag is
part of that gated publication.
Use the version in `pyproject.toml`; do not create its release tag before approval.

1. Verify the backend commit and contract SHA256 in `contracts/SOURCE.toml`.
   Run `uv run python scripts/check_cohort.py --core-checkout <core>
   --core-revision <frozen-commit>`; do not promote a mismatched cohort.
2. Run the complete CI matrix, generation checks, wheel and sdist builds,
   `twine check`, package-size checks and fresh-venv install smoke.
3. Install the exact wheel in a fresh venv with `uv pip install --no-cache`; verify
   `RunEvent.model_fields["observed_at"].annotation == str | None`.
   With authorized credentials and spend, run `scripts/smoke.py --execute --output
   <receipt.json>` against the qualified local slot, then staging. New v5 Swarms
   smoke receipts cap two runs at $4 each ($8 aggregate). It requires successful
   execution and digest-verified output from the first run, a cancellation request
   for the second, and final accounting and cleanup for both. Retain receipts;
   resume a retained v4/v5 receipt against its original API with `--resume`. Resume reuses
   the retained run IDs and launch/cancel idempotency keys. Existing v4 receipts
   retain their original $0.20 aggregate cap; unsupported receipts are preserved
   and refused. Never convert a receipt to start new runs.
   `smoke_complete` records only this Swarms journey. Full public API, hosted MCP,
   console contract parity and sustained gateway acceptance require their own
   qualification evidence before release.
4. After live acceptance, merge the update into dev, then promote dev to staging
   and staging to main per the workspace release policy (this repository does not
   auto-deploy). Preserve
   the `legacy/0.24.0` and `legacy/dev-20261104` history tags.
5. After owner GO and production parity, tag the qualified commit with `v` followed
   by the qualified package version. The tag workflow builds qualified bytes and
   publishes through the protected `pypi` environment using trusted publishing.
6. Install the published version in a fresh environment and repeat imports,
   auth status and an authenticated read. Record the package URL and hashes.

Never force-push history or replace an existing release artifact. Rollback means
pinning a previous compatible package; a yank is not a service rollback.

## Staging prerelease

The 2026-11-16 train uses `0.41.0rc1`; planned stable is `0.41.0`.
Verify PyPI and Git tags immediately before each publication; never reuse a
version. The existing `publish-dev.yml` `v*` tag workflow supports PEP 440
prereleases and asserts the exact tag equals `v` plus the package version.
Tag `v0.41.0rc1` only after the local candidate gate. It uses the same protected
`pypi` environment and trusted publisher as stable. Satisfy the configured
GitHub environment review through the authorized owner identity; do not bypass
it or use a separate token publisher. Install staging's exact prerelease with
`pip install synth-ai==0.41.0rc1` and retain its package hashes and live receipts.

After staging acceptance and production routes are live, change the version to
`0.41.0`, regenerate the version module, rebuild and repeat the isolated wheel
smoke. Candidate and stable have distinct artifact bytes and hashes. Promote
source through dev → staging → main and publish/tag the qualified stable commit
only at the stable gate. PyPI's default stable selection excludes the candidate.
