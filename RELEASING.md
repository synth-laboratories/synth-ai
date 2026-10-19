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
