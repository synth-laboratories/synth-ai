# Releasing

Use the locked development group to generate and check contract fingerprints:

```sh
uv sync --locked --group dev --extra mcp
uv run --no-sync python -m synth_ai.contracts.fingerprint --write
uv run --no-sync pytest -q tests/test_fingerprints_current.py
```

Generation requires the qualification pin, pydantic 2.12.5. Runtime clients read
`synth_ai/contracts/fingerprints.json` and never regenerate public schemas.
Regenerate the retained contract JSON objects before updating the fingerprint
table after a wire change. Review all contract diffs together.

Run the offline suite on Python 3.11, 3.12 and 3.13 with the supported pydantic
floor and latest 2.x release. The live-export oracle runs with the qualification
pin; all versions check the shipped hashes against the retained contract objects
and exercise real capability negotiation through MockTransport.
The supported floor is pydantic 2.11.0. Versions 2.9.2 and 2.10.6 cannot import
the existing generic `Page` type with its default cursor parameter. Test core
and MCP environments separately so the optional extra stays optional.

Build the wheel and source distribution with `python -m build`, run
`twine check dist/*`, and verify both in fresh environments. Retain the exact
artifacts and their qualification receipts. Publication uses the existing
qualified artifact workflow after branch promotion; source merges alone do not
authorize publication.


Public API changes use minor versions; fixes without API changes use patch
versions. During 0.x, explicitly documented breaking changes may ship in a
minor release. Retire public names only after a release with a deprecation
warning. Environment aliases are removed in 0.25.0.

Use tags vX.Y.Z for releases and vX.Y.Z.devN for sequential development
candidates, for example v0.23.4.dev1. Qualification evidence belongs under
refs/qualification/ and is not a public release tag. Retain both wheel and
sdist digests in the release attestation. Do not rewrite published tags.

The next source candidate is 0.23.5. Swarms is included, so all seven retained
qualification checks are required. Its generated contract files are source
artifacts; they do not attest an installed wheel, service cohort, or runtime
qualification. Publish only after the release owner accepts those receipts.

A check the release owner explicitly waives is recorded as `owner_waived`, never
as `passed`. The waiver authority lives in `.github/release-policy.json` under the
version's `owner_waivers`: the directive text, who approved it, when, the ticket
and the exact checks it covers. Each waived check carries a concrete reason and an
`evidence/<check>/owner-waiver.json` record that must match that authority. A
bundle with any waiver has manifest status `passed_with_owner_waivers`, and the
publish receipt lists `passed` and `owner_waived` checks separately.
